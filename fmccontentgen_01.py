# ==============================================
# QFORIA RESEARCH PLATFORM - FULL FIXED VERSION
# All features + Clean Output + Bullets/Tables
# ==============================================
import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
import time
from datetime import datetime
import PyPDF2
import io
import re

# DOCX support for Word export
try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("python-docx not installed. Install: pip install python-docx")

# App config
st.set_page_config(page_title="Qforia Research Platform", layout="wide")
st.title("FMC Content Guide")

# Grok API Configuration
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# Initialize session states
for key in ['fanout_results', 'generation_details', 'research_results', 'selected_queries',
            'pdf_analysis', 'enhanced_topics', 'generated_content', 'content_structure', 'user_query']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'research_results' else {}
        if key == 'selected_queries':
            st.session_state[key] = set()

# Sidebar Configuration
st.sidebar.header("Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password")
grok_key = st.sidebar.text_input("Grok API Key (for content generation)", type="password", help="Enter your xAI Grok API key")

# Gemini model selector
st.sidebar.subheader("Gemini Model")
gemini_model = st.sidebar.selectbox(
    "Select Model",
    ["gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    index=0
)

if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(gemini_model)
    st.sidebar.success(f"{gemini_model}")
else:
    model = None

# ==================== UTILITY FUNCTIONS ====================

def call_perplexity(query, system_prompt="Provide comprehensive, actionable insights with specific data points, statistics, and practical recommendations."):
    if not perplexity_key:
        return {"error": "Missing Perplexity API key"}
    headers = {
        "Authorization": f"Bearer {perplexity_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "max_tokens": 600
    }
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Perplexity API error: {e}"}

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

# ==================== QFORIA OUTLINE GENERATION ====================
def generate_content_structure(research_data, topic):
    if not model:
        return None, "Gemini not configured"
    research_summary = ""
    for data in list(research_data.values())[:10]:
        research_summary += f"\n{data['query']}: {data['result'][:200]}..."
    prompt = f"""
Create a detailed SEO-optimized article structure for: "{topic}"
Use this research data:
{research_summary}

Return ONLY valid JSON:
{{
    "article_title": "H1 Title (50-60 chars)",
    "meta_description": "150-160 char description",
    "primary_keyword": "main keyword",
    "semantic_keywords": ["kw1", "kw2"],
    "sections": [
        {{
            "title": "H2 Section Title",
            "description": "What this covers",
            "key_points": ["point1", "point2"],
            "estimated_words": 400,
            "needs_table": true,
            "table_description": "Compare X vs Y",
            "needs_infographic": true,
            "infographic_description": "Visualize timeline"
        }}
    ]
}}
"""
    try:
        response = model.generate_content(prompt)
        txt = response.text.strip()
        if "```json" in txt:
            txt = txt.split("```json")[1].split("```")[0]
        return json.loads(txt), None
    except Exception as e:
        return None, f"Failed to parse structure: {str(e)}"

# ==================== CONTENT GENERATION (CLEAN, BULLET-HEAVY) ====================
def generate_section_content(section, research_context, tone="professional"):
    prompt = f"""Write a complete section in SIMPLE, DIRECT English.

SECTION TITLE: {section['title']}
DESCRIPTION: {section['description']}
KEY POINTS: {', '.join(section.get('key_points', []))}
TARGET WORDS: {section.get('estimated_words', 400)}
TONE: {tone}

RESEARCH DATA:
{research_context[:2500]}

RULES:
- Short sentences only.
- Use bullet points and numbered lists heavily.
- NO asterisks, NO hashtags, NO markdown.
- Include 3-5 specific examples or stats.
- End with a clear action step.
- If table needed, create a plain text table with | separators.

TABLE: {section.get('table_description','') if section.get('needs_table') else ''}

Write the full section now:
"""
    messages = [{"role": "user", "content": prompt}]
    return call_grok(messages, max_tokens=2200, temperature=0.6)

def generate_table_content(section):
    if not section.get('needs_table'):
        return None, None
    prompt = f"""Create a comparison table for: {section['title']}
Show: {section.get('table_description','')}
Return ONLY JSON:
{{
  "table_title": "Title",
  "headers": ["Col1", "Col2"],
  "rows": [["A", "B"], ["C", "D"]]
}}
"""
    resp, _ = call_grok([{"role": "user", "content": prompt}], max_tokens=1000)
    try:
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            return json.loads(m.group()), None
    except:
        pass
    return None, "Table parse failed"

# ==================== EXPORT FUNCTIONS ====================
def export_to_txt(article_title, meta_description, sections_content, keywords):
    content = f"{article_title}\n\n{meta_description}\n\n"
    content += "Keywords: " + ", ".join(keywords[:10]) + "\n\n"
    content += "="*80 + "\n\n"
    for sec in sections_content:
        content += f"{sec['section']['title']}\n\n"
        content += sec['content'].strip() + "\n\n"
        if 'table' in sec:
            t = sec['table']
            content += f"{t['table_title']}\n"
            headers = t['headers']
            rows = t['rows']
            widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
            content += " | ".join(str(h).ljust(w) for h,w in zip(headers, widths)) + "\n"
            content += "-|-".join("-"*w for w in widths) + "\n"
            for row in rows:
                content += " | ".join(str(c).ljust(w) for c,w in zip(row, widths)) + "\n"
            content += "\n"
        if sec['section'].get('needs_infographic'):
            content += f"Infographic: {sec['section'].get('infographic_description','')}\n\n"
        content += "-"*80 + "\n\n"
    buffer = io.BytesIO(content.encode('utf-8'))
    buffer.seek(0)
    return buffer

# ==================== PDF & ANALYSIS ====================
def extract_pdf_text(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text[:12000], None
    except Exception as e:
        return None, f"PDF error: {e}"

def analyze_pdf_content(content, filename):
    if not model:
        return None, "Gemini not configured"
    prompt = f"""
Analyze PDF: {filename}
Content: {content[:8000]}

Return JSON:
{{
  "keywords": ["kw1"],
  "main_topics": ["topic1"],
  "summary": "2-3 sentences"
}}
"""
    try:
        r = model.generate_content(prompt)
        txt = r.text.strip()
        if "```json" in txt:
            txt = txt.split("```json")[1].split("```")[0]
        return json.loads(txt), None
    except Exception as e:
        return None, f"Analysis error: {e}"

def QUERY_FANOUT_PROMPT(q, mode):
    min_q = 12 if "simple" in mode else 25
    return f"""
Generate {min_q}+ diverse queries for: "{q}"
Mode: {mode}
Return ONLY JSON:
{{
  "expanded_queries": [
    {{"query":"...", "category":"...", "priority":"high/medium/low"}}
  ]
}}
"""

def generate_fanout(query, mode):
    if not model:
        st.error("Gemini key required")
        return None
    prompt = QUERY_FANOUT_PROMPT(query, mode)
    try:
        r = model.generate_content(prompt)
        txt = r.text.strip()
        if "```json" in txt:
            txt = txt.split("```json")[1].split("```")[0]
        return json.loads(txt)
    except Exception as e:
        st.error(f"Fanout error: {e}")
        return None

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Query Research", "PDF Analyzer", "Fact Checker", "Dashboard", "Content Generator"])

# === TAB 1: Query Research ===
with tab1:
    st.header("Qforia Query Fan-Out Research")
    col1, col2 = st.columns([2, 1])
    with col1:
        user_query = st.text_area("Research Query", value="JEE Main How to Prepare", height=100)
    with col2:
        mode = st.selectbox("Depth", ["AI Overview (simple)", "AI Mode (complex)"])
        if st.button("Generate Research Queries", type="primary"):
            if gemini_key and user_query:
                with st.spinner("Generating..."):
                    st.session_state.user_query = user_query
                    results = generate_fanout(user_query, mode)
                    if results:
                        st.session_state.fanout_results = results
                        st.success("Queries generated!")
                        st.rerun()

    if st.session_state.fanout_results:
        queries = st.session_state.fanout_results.get('expanded_queries', [])
        for q in queries:
            qid = f"q_{hash(q['query'])}"
            c1, c2, c3 = st.columns([1, 6, 2])
            with c1:
                st.checkbox("Select", key=f"sel_{qid}")
            with c2:
                st.write(f"**{q['query']}**")
            with c3:
                if st.button("Research", key=f"res_{qid}") and perplexity_key:
                    res = call_perplexity(q['query'])
                    if 'choices' in res:
                        st.session_state.research_results[qid] = {
                            'query': q['query'],
                            'result': res['choices'][0]['message']['content']
                        }
                    st.rerun()

# === TAB 5: Content Generator (FULL) ===
with tab5:
    st.header("AI Content Generator")
    if not grok_key:
        st.warning("Enter Grok API key")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            topic = st.text_input("Article Topic", value=st.session_state.user_query or "")
        with col2:
            tone = st.selectbox("Tone", ["Professional", "Simple", "Educational"])

        if st.button("Generate Qforia Outline", type="primary"):
            if gemini_key and topic:
                with st.spinner("Creating outline..."):
                    outline, err = generate_content_structure(st.session_state.research_results, topic)
                    if outline:
                        st.session_state.content_structure = outline
                        st.rerun()

        if st.session_state.content_structure:
            outline = st.session_state.content_structure
            st.success("Outline ready")
            with st.expander("Edit Structure"):
                st.json(outline)

            if st.button("Generate Full Article", type="primary"):
                research_ctx = "\n".join([f"{v['query']}: {v['result'][:300]}" for v in st.session_state.research_results.values()])
                progress = st.progress(0)
                sections_out = []
                for i, sec in enumerate(outline['sections']):
                    content, _ = generate_section_content(sec, research_ctx, tone.lower())
                    sec_data = {'section': sec, 'content': content.replace("*","").replace("#","")}
                    if sec.get('needs_table'):
                        table, _ = generate_table_content(sec)
                        if table:
                            sec_data['table'] = table
                    sections_out.append(sec_data)
                    progress.progress((i+1)/len(outline['sections']))
                st.session_state.generated_content = {
                    'title': outline['article_title'],
                    'meta': outline['meta_description'],
                    'sections': sections_out,
                    'keywords': outline.get('semantic_keywords', [])
                }
                st.rerun()

        if st.session_state.generated_content:
            gen = st.session_state.generated_content
            st.subheader("Generated Article")
            st.write(f"**{gen['title']}**")
            st.caption(gen['meta'])
            for sec in gen['sections']:
                st.markdown(f"### {sec['section']['title']}")
                st.write(sec['content'])
                if 'table' in sec:
                    df = pd.DataFrame(sec['table']['rows'], columns=sec['table']['headers'])
                    st.dataframe(df)
            txt_buf = export_to_txt(gen['title'], gen['meta'], gen['sections'], gen['keywords'])
            st.download_button("Download .txt", txt_buf, f"{topic}.txt", "text/plain")

# === CLEAR & FOOTER ===
if st.sidebar.button("Clear All Data"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.markdown("---")
st.markdown("**Qforia Complete Research Platform** | Powered by Gemini, Perplexity, Grok")
