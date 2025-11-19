import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
import time
from datetime import datetime, timedelta
import re

# --- APP CONFIGURATION ---
st.set_page_config(page_title="SEO Content Generator - Simple & Clean", layout="wide")
st.title("SEO Content Generator")

# --- SESSION STATE ---
if 'research_results' not in st.session_state: st.session_state.research_results = {}
if 'selected_queries' not in st.session_state: st.session_state.selected_queries = set()
if 'content_outline' not in st.session_state: st.session_state.content_outline = None
if 'generated_content' not in st.session_state: st.session_state.generated_content = None
if 'main_topic' not in st.session_state: st.session_state.main_topic = ""
if 'target_country' not in st.session_state: st.session_state.target_country = "India"

# --- API CONFIGURATION ---
st.sidebar.header("API Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password")
grok_key = st.sidebar.text_input("Grok API Key", type="password")

context_date = st.sidebar.date_input("Content Date", value=datetime.now())
formatted_date = context_date.strftime("%B %d, %Y")
current_year = context_date.year

if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
else:
    model = None

# --- FUNCTIONS ---

def call_perplexity(query):
    if not perplexity_key: return None
    headers = {"Authorization": f"Bearer {perplexity_key}", "Content-Type": "application/json"}
    data = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": f"Current date: {formatted_date}. Provide factual, specific data with numbers and examples. No fluff."},
            {"role": "user", "content": query}
        ]
    }
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except:
        return None

def call_grok(prompt, max_tokens=1200):
    if not grok_key: return None
    headers = {"Authorization": f"Bearer {grok_key}", "Content-Type": "application/json"}
    payload = {
        "model": "grok-2-latest",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    try:
        response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=120)
        result = response.json()
        return result['choices'][0]['message']['content']
    except:
        return None

def generate_research_queries(topic):
    if not model: return None
    prompt = f"""Generate 12 research queries for: "{topic}"
    Make queries specific and detailed. Ask for exact data, breakdowns, comparisons.
    Return ONLY JSON: {{"queries": ["query1", "query2", ...]}}"""
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        return json.loads(text)['queries']
    except:
        return None

def generate_outline(topic, research_data):
    if not model: return None
    research_summary = "\n".join([f"- {q}: {r[:100]}..." for q, r in list(research_data.items())[:10]])
    
    prompt = f"""Create content outline for: "{topic}"

Research summary:
{research_summary}

CRITICAL INSTRUCTIONS:

1. H1 STRUCTURE (Semantic Heading):
   - Format: "[Main Topic] - [Key Aspects Covered]"
   - Example: "Machine Learning - Applications, Algorithms and Career Path"
   - Example: "Real Estate Investment - Types, Returns and Risk Analysis"
   - Identify 3-4 main aspects from research and add after hyphen
   - DO NOT just use topic name alone
   
2. H2 STRUCTURE (Short Keywords):
   - Each H2 must be 2-5 words maximum
   - Use exact keywords/phrases people search for
   - Examples: "Eligibility Criteria", "Fee Structure", "Course Duration", "Career Prospects"
   - NOT full questions or long descriptions
   - Extract from research what people want to know

3. CONTENT PLANNING:
   - For each H2, decide: needs_table (true/false)
   - If data/comparison/numbers → table
   - If list/steps/points → bullets
   - Specify what table should contain

Return ONLY JSON:
{{
  "h1": "[Topic] - [Aspect 1], [Aspect 2] and [Aspect 3]",
  "headings": [
    {{
      "h2": "Short Keyword",
      "needs_table": true,
      "table_desc": "what data table should show",
      "content_focus": "what paragraph should cover"
    }}
  ]
}}

Generate 6-8 H2 headings based on research content."""
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        result = json.loads(text)
        # Add year to H1
        if " - " in result['h1']:
            result['h1'] = f"{result['h1']} ({current_year})"
        return result
    except Exception as e:
        st.error(f"Error generating outline: {e}")
        return None

def generate_section(heading, research_data):
    if not grok_key: return {"paragraph": "", "table": None}
    
    research_text = "\n\n".join([f"Q: {q}\nA: {r}" for q, r in list(research_data.items())[:10]])
    
    prompt = f"""Write content for heading: "{heading['h2']}"

Research data:
{research_text}

CRITICAL RULES:
1. Write ONE paragraph (4-6 sentences, 15-20 words each)
2. Be factual - only use research data
3. No fluff words: "comprehensive", "crucial", "it's important"
4. Direct present tense statements
5. {"Then create a data table" if heading.get('needs_table') else "Then create 4-6 bullet points"}

{"TABLE: " + heading.get('table_desc', '') if heading.get('needs_table') else "BULLETS: Key facts from research"}

Return ONLY JSON:
{{
  "paragraph": "one paragraph text",
  {"table": {"headers": ["col1", "col2"], "rows": [["data1", "data2"]]} if heading.get('needs_table') else "bullets": ["point 1", "point 2"]}
}}"""
    
    try:
        response = call_grok(prompt, 1500)
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip())
    except:
        return {"paragraph": "", "table": None, "bullets": []}

def export_html(h1, sections):
    html = [f"<h1>{h1}</h1>", ""]
    
    for section in sections:
        html.append(f"<h2>{section['h2']}</h2>")
        html.append(f"<p>{section['content']['paragraph']}</p>")
        
        if section['content'].get('table'):
            table = section['content']['table']
            html.append("<table>")
            html.append("  <tr>")
            for h in table['headers']:
                html.append(f"    <th>{h}</th>")
            html.append("  </tr>")
            for row in table['rows']:
                html.append("  <tr>")
                for cell in row:
                    html.append(f"    <td>{cell}</td>")
                html.append("  </tr>")
            html.append("</table>")
        
        if section['content'].get('bullets'):
            html.append("<ul>")
            for bullet in section['content']['bullets']:
                html.append(f"  <li>{bullet}</li>")
            html.append("</ul>")
        
        html.append("")
    
    return '\n'.join(html)

# --- UI ---

tab1, tab2, tab3 = st.tabs(["1. Setup & Research", "2. Outline", "3. Content"])

with tab1:
    st.header("Step 1: Topic & Research")
    
    topic = st.text_input("Main Topic:", value=st.session_state.main_topic)
    if topic: st.session_state.main_topic = topic
    
    st.session_state.target_country = st.selectbox("Target Audience:", ["India", "United States", "United Kingdom", "Global"])
    
    if st.button("Generate Research Queries"):
        if not topic or not gemini_key:
            st.error("Enter topic and Gemini API key")
        else:
            with st.spinner("Generating..."):
                queries = generate_research_queries(topic)
                if queries:
                    st.session_state.queries = queries
                    st.success(f"Generated {len(queries)} queries")
                    st.rerun()
    
    if 'queries' in st.session_state:
        st.markdown("---")
        st.subheader("Select Queries to Research")
        
        for i, q in enumerate(st.session_state.queries):
            qid = f"q_{i}"
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.checkbox(q, value=qid in st.session_state.selected_queries, key=qid):
                    st.session_state.selected_queries.add(qid)
                else:
                    st.session_state.selected_queries.discard(qid)
            with col2:
                if q in st.session_state.research_results:
                    st.success("✓")
        
        if st.session_state.selected_queries and perplexity_key:
            unresearched = [st.session_state.queries[int(qid.split('_')[1])] 
                          for qid in st.session_state.selected_queries 
                          if st.session_state.queries[int(qid.split('_')[1])] not in st.session_state.research_results]
            
            if unresearched and st.button(f"Research {len(unresearched)} Queries"):
                progress = st.progress(0)
                for i, q in enumerate(unresearched):
                    st.text(f"Researching: {q[:60]}...")
                    result = call_perplexity(q)
                    if result:
                        st.session_state.research_results[q] = result
                    progress.progress((i + 1) / len(unresearched))
                    time.sleep(1)
                st.success("Done!")
                st.rerun()

with tab2:
    st.header("Step 2: Generate Outline")
    
    if not st.session_state.research_results:
        st.warning("Complete research first")
    else:
        st.success(f"{len(st.session_state.research_results)} queries researched")
        
        if st.button("Generate Outline"):
            with st.spinner("Creating outline..."):
                outline = generate_outline(st.session_state.main_topic, st.session_state.research_results)
                if outline:
                    st.session_state.content_outline = outline
                    st.rerun()
        
        if st.session_state.content_outline:
            st.subheader(st.session_state.content_outline['h1'])
            for h in st.session_state.content_outline['headings']:
                with st.expander(f"H2: {h['h2']}"):
                    st.write(f"**Content:** {h['content_focus']}")
                    if h.get('needs_table'):
                        st.info(f"Table: {h.get('table_desc')}")
                    else:
                        st.info("Bullets")

with tab3:
    st.header("Step 3: Generate Content")
    
    if not st.session_state.content_outline:
        st.warning("Generate outline first")
    else:
        if st.button("Generate Article"):
            sections = []
            progress = st.progress(0)
            
            for i, heading in enumerate(st.session_state.content_outline['headings']):
                st.text(f"Writing: {heading['h2']}...")
                content = generate_section(heading, st.session_state.research_results)
                sections.append({"h2": heading['h2'], "content": content})
                progress.progress((i + 1) / len(st.session_state.content_outline['headings']))
            
            st.session_state.generated_content = {
                "h1": st.session_state.content_outline['h1'],
                "sections": sections
            }
            st.success("Done!")
            st.rerun()
        
        if st.session_state.generated_content:
            st.markdown(f"## {st.session_state.generated_content['h1']}")
            
            for section in st.session_state.generated_content['sections']:
                st.markdown(f"### {section['h2']}")
                st.write(section['content']['paragraph'])
                
                if section['content'].get('table'):
                    st.dataframe(pd.DataFrame(
                        section['content']['table']['rows'],
                        columns=section['content']['table']['headers']
                    ))
                
                if section['content'].get('bullets'):
                    for b in section['content']['bullets']:
                        st.markdown(f"- {b}")
            
            html = export_html(
                st.session_state.generated_content['h1'],
                st.session_state.generated_content['sections']
            )
            
            st.download_button(
                "Download HTML",
                html,
                file_name=f"{st.session_state.main_topic.replace(' ', '_')}.html",
                mime="text/html"
            )
