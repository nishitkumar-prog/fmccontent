import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
import time
from datetime import datetime
import PyPDF2
import io

# DOCX support
try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("⚠️ Install: `pip install python-docx` for Word export")

# App config
st.set_page_config(page_title="Qforia Research Platform", layout="wide")
st.title("FMC Content Guide")

# API URLs
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# Session State
if 'fanout_results' not in st.session_state:
    st.session_state.fanout_results = None
if 'generation_details' not in st.session_state:
    st.session_state.generation_details = None
if 'research_results' not in st.session_state:
    st.session_state.research_results = {}
if 'selected_queries' not in st.session_state:
    st.session_state.selected_queries = set()
if 'pdf_analysis' not in st.session_state:
    st.session_state.pdf_analysis = None
if 'enhanced_topics' not in st.session_state:
    st.session_state.enhanced_topics = []
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = {}
if 'content_structure' not in st.session_state:
    st.session_state.content_structure = []
if 'user_query' not in st.session_state:
    st.session_state.user_query = ''

# Sidebar
st.sidebar.header("Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password")
grok_key = st.sidebar.text_input("Grok API Key", type="password", help="xAI Grok API key")

# Gemini Model
st.sidebar.subheader("Gemini Model")
gemini_model = st.sidebar.selectbox(
    "Select Model",
    ["gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    index=0
)

if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(gemini_model)
    st.sidebar.success(f"✅ {gemini_model}")
else:
    model = None

# === UTILITY FUNCTIONS ===
def call_perplexity(query, system_prompt="Provide clear, factual, and data-rich answers with examples."):
    if not perplexity_key:
        return {"error": "Missing Perplexity API key"}
    headers = {"Authorization": f"Bearer {perplexity_key}", "Content-Type": "application/json"}
    data = {
        "model": "sonar",
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
        "max_tokens": 600
    }
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def call_grok(messages, max_tokens=4000, temperature=0.7):
    if not grok_key:
        return None, "Missing Grok API key"
    headers = {"Authorization": f"Bearer {grok_key}", "Content-Type": "application/json"}
    payload = {
        "messages": messages,
        "model": "grok-2-1212",
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'], None
    except Exception as e:
        return None, f"Grok API error: {str(e)}"

# === CONTENT GENERATION USING QFORIA OUTLINE ===
def generate_content_from_outline(outline, research_context, tone="professional"):
    """Generate full article using Qforia outline with clean, structured, bullet-heavy content"""
    if not grok_key:
        return None, "Grok API key required"

    # Compile research summary
    research_summary = ""
    for data in list(st.session_state.research_results.values())[:12]:
        research_summary += f"Q: {data['query']}\nA: {data['result'][:400]}...\n\n"

    # Build full article
    full_content = []
    messages = []

    # Title & Meta
    title = outline.get("article_title", "Article Title")
    meta = outline.get("meta_description", "")
    full_content.append(f"{title}\n")
    full_content.append(f"{meta}\n")

    # Generate each section
    for idx, section in enumerate(outline.get("sections", [])):
        prompt = f"""Write a complete section in SIMPLE, DIRECT English.

SECTION TITLE: {section['title']}
DESCRIPTION: {section['description']}
KEY POINTS: {', '.join(section.get('key_points', []))}
TARGET WORDS: {section.get('estimated_words', 400)}
TONE: {tone}

RESEARCH CONTEXT:
{research_summary[:3000]}

INSTRUCTIONS:
- Use SHORT sentences.
- Use BULLET POINTS and NUMBERED LISTS heavily.
- NO asterisks (*), NO hashtags (#), NO markdown.
- Use tables when comparing data.
- Include 3–5 specific examples or stats.
- End with a clear action step.

{section.get('table_description', '') and f"INCLUDE TABLE: {section['table_description']}" or ''}

Write the full section now in plain text:
"""
        messages = [{"role": "user", "content": prompt}]
        content, error = call_grok(messages, max_tokens=2000, temperature=0.6)
        if error:
            full_content.append(f"[Error generating section: {section['title']}]")
        else:
            # Clean any accidental markdown
            clean_content = content.replace("*", "").replace("#", "")

            # Generate table if needed
            table_data = None
            if section.get('needs_table'):
                table_prompt = f"""Create a comparison table for: {section['title']}
Show: {section.get('table_description', '')}
Return ONLY JSON:
{{
  "table_title": "Title",
  "headers": ["Col1", "Col2"],
  "rows": [["A", "B"], ["C", "D"]]
}}
"""
                table_resp, _ = call_grok([{"role": "user", "content": table_prompt}], max_tokens=1000)
                try:
                    import re
                    json_str = re.search(r'\{.*\}', table_resp, re.DOTALL)
                    if json_str:
                        table_data = json.loads(json_str.group())
                except:
                    pass

            full_content.append(clean_content.strip())
            if table_data:
                full_content.append(f"\n{table_data['table_title']}")
                headers = table_data['headers']
                rows = table_data['rows']
                col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
                header_row = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
                separator = "-|-".join("-" * w for w in col_widths)
                table_lines = [header_row, separator] + [
                    " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)) for row in rows
                ]
                full_content.append("\n".join(table_lines))

            if section.get('needs_infographic'):
                full_content.append(f"\nInfographic: {section.get('infographic_description', 'Visual recommended')}")

        full_content.append("\n" + "-" * 60 + "\n")

    return "\n".join(full_content), None

# === QFORIA OUTLINE GENERATION (UNCHANGED) ===
def generate_qforia_outline(query, mode="AI Mode (complex)"):
    if not model:
        return None, "Gemini not configured"
    prompt = f"""
Analyze: "{query}"
Mode: {mode}

Generate a detailed SEO article outline in JSON.

Requirements:
- H1 title (50–60 chars)
- Meta description (150–160 chars)
- 8–12 H2 sections
- Each section has: title, description, key_points, needs_table, table_description, needs_infographic
- Focus on user intent, data, comparisons

Return ONLY valid JSON:
{{
  "article_title": "...",
  "meta_description": "...",
  "sections": [{{...}}]
}}
"""
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        return json.loads(text.strip()), None
    except Exception as e:
        return None, f"Outline error: {e}"

# === EXPORT FUNCTIONS (CLEAN TEXT) ===
def export_to_txt(content, title):
    buffer = io.BytesIO()
    buffer.write(content.encode('utf-8'))
    buffer.seek(0)
    return buffer

# === TABS ===
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Query Research", "PDF Analyzer", "Fact Checker", "Dashboard", "Content Generator"])

with tab1:
    st.header("Qforia Query Fan-Out")
    col1, col2 = st.columns([2, 1])
    with col1:
        user_query = st.text_area("Research Topic", value="JEE Main How to Prepare", height=100)
    with col2:
        mode = st.selectbox("Depth", ["AI Overview (simple)", "AI Mode (complex)"])
        if st.button("Generate Queries", type="primary"):
            if gemini_key and user_query:
                with st.spinner("Generating..."):
                    st.session_state.user_query = user_query
                    st.session_state.fanout_results = generate_fanout(user_query, mode)
                st.rerun()

    # Research execution (same as before, simplified)
    if st.session_state.fanout_results:
        queries = st.session_state.fanout_results.get('expanded_queries', [])
        for q in queries:
            qid = f"q_{hash(q['query'])}"
            col1, col2 = st.columns([1, 6])
            with col1:
                st.checkbox("Select", key=f"sel_{qid}")
            with col2:
                st.write(f"**{q['query']}**")
                if st.button("Research", key=f"res_{qid}") and perplexity_key:
                    res = call_perplexity(q['query'])
                    if 'choices' in res:
                        st.session_state.research_results[qid] = {
                            'query': q['query'],
                            'result': res['choices'][0]['message']['content']
                        }

with tab5:
    st.header("AI Content Generator")
    if not grok_key:
        st.warning("Enter Grok API key")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            topic = st.text_input("Article Topic", value=st.session_state.user_query)
        with col2:
            tone = st.selectbox("Tone", ["Professional", "Simple", "Educational"])

        if st.button("Generate Qforia Outline", type="primary"):
            if topic and gemini_key:
                with st.spinner("Creating outline..."):
                    outline, err = generate_qforia_outline(topic)
                    if outline:
                        st.session_state.content_structure = outline
                        st.success("Outline ready!")
                        st.rerun()

        if st.session_state.content_structure:
            st.success("Outline Generated")
            outline = st.session_state.content_structure
            st.subheader("Article Preview")
            st.write(f"**Title:** {outline['article_title']}")
            st.caption(outline['meta_description'])

            if st.button("Generate Full Article (Clean Text)", type="primary"):
                with st.spinner("Writing article..."):
                    research_ctx = "\n".join([f"{v['query']}: {v['result'][:300]}" for v in st.session_state.research_results.values()])
                    article, err = generate_content_from_outline(outline, research_ctx, tone.lower())
                    if article:
                        st.session_state.generated_content = {"full": article}
                        st.rerun()

        if st.session_state.generated_content.get("full"):
            st.subheader("Final Article")
            article = st.session_state.generated_content["full"]
            st.text_area("Article", article, height=600)

            # Export
            txt_buffer = export_to_txt(article, topic)
            st.download_button(
                "Download .txt",
                data=txt_buffer,
                file_name=f"{topic.replace(' ', '_')}.txt",
                mime="text/plain"
            )

# Clear Data
if st.sidebar.button("Clear All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Cleared!")
    st.rerun()

st.markdown("---")
st.markdown("**Qforia Research Platform** | Powered by Gemini, Perplexity, Grok")
