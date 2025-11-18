import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
import time
from datetime import datetime
import io

# App config
st.set_page_config(page_title="Content Generator", layout="wide")
st.title("SEO Content Generator")

# API Configuration
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# Initialize session states
if 'fanout_results' not in st.session_state:
    st.session_state.fanout_results = None
if 'research_results' not in st.session_state:
    st.session_state.research_results = {}
if 'selected_queries' not in st.session_state:
    st.session_state.selected_queries = set()
if 'content_outline' not in st.session_state:
    st.session_state.content_outline = None
if 'paa_keywords' not in st.session_state:
    st.session_state.paa_keywords = []
if 'keyword_combinations' not in st.session_state:
    st.session_state.keyword_combinations = []
if 'google_ads_keywords' not in st.session_state:
    st.session_state.google_ads_keywords = []
if 'generated_sections' not in st.session_state:
    st.session_state.generated_sections = []
if 'generated_faqs' not in st.session_state:
    st.session_state.generated_faqs = []
if 'main_topic' not in st.session_state:
    st.session_state.main_topic = ""
if 'focus_keyword' not in st.session_state:
    st.session_state.focus_keyword = ""
if 'target_country' not in st.session_state:
    st.session_state.target_country = ""
if 'required_tables' not in st.session_state:
    st.session_state.required_tables = []
if 'bulk_research_running' not in st.session_state:
    st.session_state.bulk_research_running = False
if 'bulk_research_progress' not in st.session_state:
    st.session_state.bulk_research_progress = {}

# Sidebar
st.sidebar.header("API Keys")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password")
grok_key = st.sidebar.text_input("Grok API Key", type="password")

gemini_model = st.sidebar.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)

if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(gemini_model)
        st.sidebar.success(f"✓ {gemini_model}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        model = None
else:
    model = None

# Utility Functions
def call_perplexity(query, system_prompt=None, max_retries=2):
    if not perplexity_key:
        return {"error": "Missing API key"}
    
    if not system_prompt:
        system_prompt = "Provide factual data with specific numbers, dates, and examples. Be direct and accurate."
    
    headers = {"Authorization": f"Bearer {perplexity_key}", "Content-Type": "application/json"}
    data = {"model": "sonar", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]}
    
    for attempt in range(max_retries):
        try:
            response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=data, timeout=45)
            response.raise_for_status()
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result
            return {"error": "Invalid response"}
        except:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"error": "Failed"}
    return {"error": "Max retries"}

def call_grok(messages, max_tokens=4000, temperature=0.6):
    if not grok_key:
        return None, "Missing API key"
    headers = {"Authorization": f"Bearer {grok_key}", "Content-Type": "application/json"}
    payload = {"messages": messages, "model": "grok-3", "stream": False, "temperature": temperature, "max_tokens": max_tokens}
    try:
        response = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content'], None
        return None, "Error"
    except Exception as e:
        return None, str(e)

def generate_research_queries(topic, mode="AI Overview (simple)"):
    if not model:
        return None, "Gemini not configured"
    min_queries = 12 if mode == "AI Overview (simple)" else 25
    prompt = f"""Generate {min_queries}+ research queries for: "{topic}"
Return ONLY valid JSON:
{{"queries": [{{"query": "question", "category": "category", "priority": "high/medium/low", "purpose": "purpose"}}]}}"""
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        return json.loads(json_text.strip()), None
    except Exception as e:
        return None, str(e)

def generate_outline_from_research(focus_keyword, target_country, required_tables, research_data):
    if not model:
        return None, "Gemini not configured"
    
    research_summary = "\n".join([f"- {data['query']}: {data['result'][:150]}..." for data in list(research_data.values())[:15]])
    
    tables_context = ""
    if required_tables:
        tables_context = f"\n\nMANDATORY TABLES:\n" + "\n".join([f"- {t}" for t in required_tables])
    
    prompt = f"""Create article outline for: "{focus_keyword}"

TARGET: Students in {target_country}
RESEARCH:
{research_summary}
{tables_context}

CRITICAL - AVOID REPETITION:
Each heading must cover UNIQUE aspect. NO overlap between sections.

Example BAD (repetitive):
- What is CMA? 
- CMA Overview
- About CMA Certification

Example GOOD (unique):
- CMA 2025 Exam Dates & Schedule
- CMA Eligibility Criteria
- CMA Exam Pattern & Syllabus
- CMA Application Process

CREATE 8-10 UNIQUE HEADINGS:
- First heading: Overview with latest dates/news
- Each heading = one specific topic only
- Topics: dates, eligibility, fees, process, syllabus, exam pattern, results, career, salary, comparison
- Use year (2025/2026) in date-related headings
- Be specific, not general

Return ONLY valid JSON:
{{
  "article_title": "Simple H1 with {focus_keyword} and year",
  "meta_description": "150-160 chars with key facts for students",
  "headings": [
    {{
      "h2_title": "Specific heading (not generic)",
      "student_question": "What unique info this provides",
      "key_facts": ["fact 1", "fact 2", "fact 3"],
      "needs_table": true/false,
      "table_purpose": "what comparison/data"
    }}
  ]
}}"""
    
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        return json.loads(json_text.strip()), None
    except Exception as e:
        return None, str(e)

def get_latest_alerts(focus_keyword, target_country):
    """Get latest news/updates/alerts for the topic"""
    if not perplexity_key:
        return ""
    
    current_date = datetime.now().strftime("%B %Y")
    query = f"Latest news, updates, deadlines, or important announcements about {focus_keyword} in {target_country} in {current_date}. Include exam dates, fee changes, new regulations, or recent updates."
    
    system_prompt = "Provide only the most recent and factual updates from 2024-2025. Include specific dates and changes. Be direct and concise."
    
    res = call_perplexity(query, system_prompt=system_prompt)
    if res and 'choices' in res and len(res['choices']) > 0:
        return res['choices'][0]['message'].get('content', '')[:500]
    return ""

def generate_section_content(heading, focus_keyword, target_country, research_context, is_first_section=False, latest_alerts=""):
    if not grok_key:
        return None, "Grok required"
    
    if is_first_section:
        alerts_context = f"\n\nLATEST UPDATES:\n{latest_alerts}" if latest_alerts else ""
        
        prompt = f"""Write opening content for: "{heading['h2_title']}"

TOPIC: {focus_keyword}
COUNTRY: {target_country}
RESEARCH: {research_context[:2500]}
{alerts_context}

STRUCTURE:

PART 1 - Latest Updates (if available):
**Latest About {focus_keyword}:**
- First update/announcement
- Second update/date
- Third update/change (if available)

PART 2 - Definition & Summary:
Write 180-220 words using SHORT SENTENCES (max 12-15 words each).

CRITICAL SENTENCE RULES:
- Break long sentences into 2-3 shorter ones
- One idea per sentence
- Use periods frequently
- No complex clauses

CONTENT STRUCTURE:
1. Opening definition (1-2 sentences, what it is)
2. Who offers/conducts it (1 sentence)
3. Core purpose (2-3 short sentences)
4. Key components (use bullet points if 3+ items)
5. Target audience (1-2 sentences)
6. Basic requirements (use bullet points)
7. Why it matters (1-2 sentences)

Use bullet points for:
- Lists of 3+ items
- Key features
- Requirements
- Benefits

Example SHORT sentence style:
❌ BAD: "The CMA certification, globally recognized and offered by IMA in the United States, is a prestigious professional credential that signifies advanced expertise in management accounting and strategic financial management."

✅ GOOD: "The CMA is a global professional certification. IMA in the United States offers this credential. It validates expertise in management accounting. Professionals gain skills in strategic financial management."

Write content now. Use markdown bullets (- ) for lists."""
    else:
        prompt = f"""Write content for: "{heading['h2_title']}"

TOPIC: {focus_keyword}
COUNTRY: {target_country}
KEY ASPECTS: {', '.join(heading.get('key_facts', []))}
RESEARCH: {research_context[:1500]}

CRITICAL RULES:
1. SHORT SENTENCES ONLY (max 12-15 words)
2. One idea per sentence
3. Use bullet points for any list of 3+ items
4. NO repetition of info from previous sections
5. 80-120 words total

USE BULLETS FOR:
- Lists (steps, requirements, documents, etc.)
- Multiple items or options
- Key points or features
- Any enumeration

Example structure for "Eligibility":
"CMA Foundation requires Class 12 completion. Students need a recognized board certificate. Enrollment possible after Class 10. However, exams allowed only after Class 12 results.

CMA Intermediate eligibility includes:
- Foundation course completion, OR
- Graduation degree (except fine arts), OR
- Professional qualifications (CA Inter, CS Foundation)

CMA Final requires both Intermediate groups cleared. Minimum 40% needed in each paper."

Write content now. Short sentences. Use markdown bullets (- ) for lists."""
    
    messages = [{"role": "user", "content": prompt}]
    max_tokens = 900 if is_first_section else 500
    content, error = call_grok(messages, max_tokens=max_tokens, temperature=0.5)
    return content, error

def generate_data_table(heading, target_country, research_context):
    if not grok_key or not heading.get('needs_table'):
        return None, "No table needed"
    
    prompt = f"""Create table for: "{heading['h2_title']}"

PURPOSE: {heading.get('table_purpose', '')}
COUNTRY: {target_country}
RESEARCH: {research_context[:1500]}

SHIKSHA.COM TABLE STYLE - Clean, professional, complete:
- Only REAL data from research
- Common formats: Exam dates, Fees, Eligibility, Pattern
- Use {target_country} currency
- Date format: DD MMM 'YY
- 4-8 rows, 3-5 columns max

Return ONLY valid JSON:
{{
  "table_title": "CMA [Topic] [Year]",
  "headers": ["Column1", "Column2", "Column3"],
  "rows": [["data", "data", "data"]]
}}"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=1000, temperature=0.4)
    if error:
        return None, error
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except:
        return None, "Parse error"

def generate_faqs(focus_keyword, target_country, paa_keywords, research_context):
    if not grok_key:
        return None, "Grok required"
    
    paa_text = "\n".join([f"- {kw}" for kw in paa_keywords[:30]])
    
    prompt = f"""Generate FAQs for: "{focus_keyword}"

AUDIENCE: Students in {target_country}
PAA: {paa_text}
RESEARCH: {research_context[:2000]}

Answer questions:
- Active voice
- Simple language
- Specific facts
- 40-60 words each

Return ONLY valid JSON:
{{"faqs": [{{"question": "Question?", "answer": "Direct answer with facts"}}]}}"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=3000, temperature=0.6)
    if error:
        return None, error
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except:
        return None, "Parse error"

def export_to_html(article_title, meta_description, sections, faqs):
    html = []
    html.append(f'<h1>{article_title}</h1>')
    html.append(f'<p><em>{meta_description}</em></p>')
    html.append('')
    
    for section in sections:
        html.append(f'<h2>{section["heading"]["h2_title"]}</h2>')
        
        if section.get('content'):
            content = section['content']
            
            # Process content with bold and bullets
            import re
            
            # Split into blocks (paragraphs and bullet lists)
            blocks = content.split('\n\n')
            
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                
                # Check if block contains bullet points
                lines = block.split('\n')
                
                # If multiple lines start with "- ", it's a bullet list
                bullet_count = sum(1 for line in lines if line.strip().startswith('- '))
                
                if bullet_count >= 2:  # It's a list
                    html.append('<ul>')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('- '):
                            # Remove "- " and handle bold
                            item = line[2:].strip()
                            item = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', item)
                            html.append(f' <li>{item}</li>')
                        elif line and not line.startswith('- '):
                            # Text before list - output as paragraph
                            html.append('</ul>')
                            line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
                            html.append(f'<p>{line}</p>')
                            html.append('<ul>')
                    html.append('</ul>')
                else:
                    # Regular paragraph - handle bold
                    para = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', block)
                    html.append(f'<p>{para}</p>')
        
        if section.get('table'):
            table = section['table']
            html.append(f'<h3>{table.get("table_title", "Data")}</h3>')
            html.append('<table>')
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            if headers:
                html.append(' <thead>')
                html.append('  <tr>')
                for header in headers:
                    html.append(f'   <th>{header}</th>')
                html.append('  </tr>')
                html.append(' </thead>')
            if rows:
                html.append(' <tbody>')
                for row in rows:
                    html.append('  <tr>')
                    for cell in row:
                        html.append(f'   <td>{cell}</td>')
                    html.append('  </tr>')
                html.append(' </tbody>')
            html.append('</table>')
        html.append('')
    
    if faqs:
        html.append('<h2>FAQs</h2>')
        for faq in faqs:
            html.append(f'<h3>{faq["question"]}</h3>')
            # Handle bullets in FAQ answers too
            import re
            answer = faq["answer"]
            if '\n- ' in answer:
                lines = answer.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('- '):
                        if '<ul>' not in html[-1] if html else False:
                            html.append('<ul>')
                        item = line[2:].strip()
                        html.append(f' <li>{item}</li>')
                    else:
                        if html and '<ul>' in html[-1]:
                            html.append('</ul>')
                        if line:
                            html.append(f'<p>{line}</p>')
                if html and '<ul>' in html[-1]:
                    html.append('</ul>')
            else:
                html.append(f'<p>{answer}</p>')
        html.append('')
    
    return '\n'.join(html)

# Main Interface
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Research", "2. Settings", "3. Results", "4. Outline", "5. Content"])

with tab1:
    st.header("Research")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Topic:", placeholder="e.g., CMA certification")
        if topic:
            st.session_state.main_topic = topic
    with col2:
        mode = st.selectbox("Depth", ["AI Overview (simple)", "AI Mode (complex)"])
    
    if st.button("Generate Queries", type="primary", use_container_width=True):
        if not topic.strip():
            st.error("Enter topic")
        elif not gemini_key:
            st.error("Enter Gemini key")
        else:
            with st.spinner("Generating..."):
                result, error = generate_research_queries(topic, mode)
                if result:
                    st.session_state.fanout_results = result
                    st.session_state.selected_queries = set()
                    st.success(f"✓ {len(result['queries'])} queries")
                    st.rerun()
                else:
                    st.error(error)
    
    if st.session_state.fanout_results and 'queries' in st.session_state.fanout_results:
        st.markdown("---")
        queries = st.session_state.fanout_results['queries']
        st.subheader(f"Queries ({len(queries)})")
        
        categories = sorted(list(set(q.get('category', 'Unknown') for q in queries)))
        selected_cats = st.multiselect("Filter:", categories, default=categories)
        
        filtered = [q for q in queries if q.get('category', 'Unknown') in selected_cats]
        filtered_ids = {f"q_{queries.index(q)}" for q in filtered}
        all_selected = all(qid in st.session_state.selected_queries for qid in filtered_ids) if filtered_ids else False
        
        select_all = st.checkbox("Select All", value=all_selected, key="select_all")
        if select_all and not all_selected:
            st.session_state.selected_queries.update(filtered_ids)
            st.rerun()
        elif not select_all and all_selected:
            st.session_state.selected_queries.difference_update(filtered_ids)
            st.rerun()
        
        for q in filtered:
            qid = f"q_{queries.index(q)}"
            col1, col2 = st.columns([4, 1])
            with col1:
                is_selected = qid in st.session_state.selected_queries
                selected = st.checkbox(f"**{q['query']}**", value=is_selected, key=f"cb_{qid}")
                if selected != is_selected:
                    if selected:
                        st.session_state.selected_queries.add(qid)
                    else:
                        st.session_state.selected_queries.discard(qid)
                    st.rerun()
            with col2:
                if qid in st.session_state.research_results:
                    st.success("✓")
                elif perplexity_key:
                    if st.button("Research", key=f"btn_{qid}"):
                        with st.spinner("..."):
                            res = call_perplexity(q['query'])
                            if 'choices' in res and res['choices']:
                                st.session_state.research_results[qid] = {
                                    'query': q['query'],
                                    'category': q.get('category', 'Unknown'),
                                    'result': res['choices'][0]['message']['content']
                                }
                                st.rerun()
        
        if st.session_state.selected_queries and perplexity_key:
            st.markdown("---")
            unreserached = [qid for qid in st.session_state.selected_queries if qid not in st.session_state.research_results]
            
            if len(unreserached) > 0:
                if not st.session_state.bulk_research_running:
                    if st.button(f"Research {len(unreserached)} Queries", type="secondary", use_container_width=True):
                        st.session_state.bulk_research_running = True
                        st.session_state.bulk_research_progress = {
                            'current': 0, 'total': len(unreserached), 'queries': unreserached, 'success': 0, 'errors': 0
                        }
                        st.rerun()
                else:
                    prog = st.session_state.bulk_research_progress
                    if prog['current'] < prog['total']:
                        progress_bar = st.progress(prog['current'] / prog['total'])
                        status = st.empty()
                        
                        qid = prog['queries'][prog['current']]
                        q_idx = int(qid.split('_')[1])
                        q = queries[q_idx]
                        status.info(f"({prog['current']+1}/{prog['total']}): {q['query'][:50]}...")
                        
                        res = call_perplexity(q['query'])
                        if res and 'choices' in res and len(res['choices']) > 0:
                            st.session_state.research_results[qid] = {
                                'query': q['query'],
                                'category': q.get('category', 'Unknown'),
                                'result': res['choices'][0]['message'].get('content', '')
                            }
                            prog['success'] += 1
                        else:
                            prog['errors'] += 1
                        
                        prog['current'] += 1
                        st.session_state.bulk_research_progress = prog
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.success(f"✅ Done! Success: {prog['success']}, Errors: {prog['errors']}")
                        if st.button("Close"):
                            st.session_state.bulk_research_running = False
                            st.session_state.bulk_research_progress = {}
                            st.rerun()

with tab2:
    st.header("Settings")
    
    st.subheader("Required Settings")
    col1, col2 = st.columns(2)
    with col1:
        focus_keyword = st.text_input("Focus Keyword *", placeholder="e.g., CMA certification")
        if focus_keyword:
            st.session_state.focus_keyword = focus_keyword
    with col2:
        target_country = st.selectbox("Country *", ["India", "United States", "United Kingdom", "Canada", "Australia", "Global"])
        st.session_state.target_country = target_country
    
    st.markdown("---")
    st.subheader("Mandatory Tables")
    st.caption("Specify tables that MUST be included in the article")
    
    table_input = st.text_area(
        "Enter table topics (one per line):",
        placeholder="CMA exam dates 2025\nCMA fees India\nCMA vs CPA comparison\nCMA eligibility requirements",
        height=120
    )
    
    if st.button("Save Table Requirements"):
        if table_input.strip():
            tables = [t.strip() for t in table_input.split('\n') if t.strip()]
            st.session_state.required_tables = tables
            st.success(f"✓ {len(tables)} tables required")
        else:
            st.session_state.required_tables = []
            st.info("No tables required")
    
    if st.session_state.required_tables:
        st.write("**Required Tables:**")
        for i, t in enumerate(st.session_state.required_tables, 1):
            st.write(f"{i}. {t}")
    
    st.markdown("---")
    st.subheader("Upload Keywords")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("**PAA Keywords**")
        paa_file = st.file_uploader("CSV", type=['csv'], key="paa")
        if paa_file:
            try:
                try:
                    df = pd.read_csv(paa_file, encoding='utf-8')
                except:
                    df = pd.read_csv(paa_file, encoding='utf-16')
                col_name = df.columns[0] if len(df.columns) == 1 else st.selectbox("Column:", df.columns.tolist(), key="paa_col")
                if st.button("Load", key="load_paa"):
                    paa_list = [q.strip() for q in df[col_name].dropna().astype(str).tolist() if '?' in q or len(q.split()) > 3]
                    st.session_state.paa_keywords = paa_list
                    st.success(f"✓ {len(paa_list)}")
            except Exception as e:
                st.error(str(e))
    
    with col2:
        st.caption("**Keywords**")
        kw_file = st.file_uploader("CSV", type=['csv'], key="kw")
        if kw_file:
            try:
                df = pd.read_csv(kw_file, encoding='utf-8')
                col_name = 'Suggestion' if 'Suggestion' in df.columns else st.selectbox("Column:", df.columns.tolist(), key="kw_col")
                if st.button("Load", key="load_kw"):
                    kw_list = df[col_name].dropna().astype(str).tolist()
                    st.session_state.keyword_combinations = kw_list
                    st.success(f"✓ {len(kw_list)}")
            except Exception as e:
                st.error(str(e))
    
    with col3:
        st.caption("**Google Ads**")
        gads_file = st.file_uploader("CSV", type=['csv'], key="gads")
        if gads_file:
            try:
                try:
                    df = pd.read_csv(gads_file, encoding='utf-16', sep='\t')
                except:
                    df = pd.read_csv(gads_file, encoding='utf-8')
                if 'Keyword' not in df.columns and len(df) > 3:
                    df = pd.read_csv(gads_file, encoding='utf-16', sep='\t', skiprows=2)
                if 'Keyword' in df.columns and st.button("Load", key="load_gads"):
                    keywords = [str(row['Keyword']).strip() for _, row in df.iterrows() if str(row['Keyword']) != 'nan']
                    st.session_state.google_ads_keywords = keywords
                    st.success(f"✓ {len(keywords)}")
            except Exception as e:
                st.error(str(e))

with tab3:
    st.header("Research Results")
    
    if not st.session_state.research_results:
        st.info("No research yet")
    else:
        total = len(st.session_state.research_results)
        st.metric("Completed", total)
        
        for qid, data in st.session_state.research_results.items():
            with st.expander(f"{data['query']}", expanded=False):
                st.write(data['result'])

with tab4:
    st.header("Generate Outline")
    
    if not st.session_state.research_results:
        st.warning("Complete research first")
    elif not st.session_state.focus_keyword:
        st.warning("Set focus keyword")
    else:
        st.success(f"✓ {len(st.session_state.research_results)} research | Focus: {st.session_state.focus_keyword}")
        
        if st.session_state.required_tables:
            st.info(f"Required tables: {len(st.session_state.required_tables)}")
        
        if st.button("Generate Outline", type="primary", use_container_width=True):
            with st.spinner("Creating..."):
                outline, error = generate_outline_from_research(
                    st.session_state.focus_keyword,
                    st.session_state.target_country,
                    st.session_state.required_tables,
                    st.session_state.research_results
                )
                if outline:
                    st.session_state.content_outline = outline
                    st.success("✓ Ready!")
                    st.rerun()
                else:
                    st.error(error)
        
        if st.session_state.content_outline:
            st.markdown("---")
            outline = st.session_state.content_outline
            st.text_input("Title:", value=outline['article_title'], disabled=True)
            st.text_area("Meta:", value=outline['meta_description'], height=60, disabled=True)
            
            for i, h in enumerate(outline['headings']):
                with st.expander(f"{i+1}. {h['h2_title']}", expanded=False):
                    st.write(f"**Q:** {h.get('student_question', 'N/A')}")
                    if h.get('needs_table'):
                        st.info(f"📊 {h.get('table_purpose', 'Table')}")

with tab5:
    st.header("Generate Content")
    
    if not st.session_state.content_outline:
        st.warning("Generate outline first")
    else:
        outline = st.session_state.content_outline
        total = len(outline['headings'])
        completed = len(st.session_state.generated_sections)
        
        st.metric("Progress", f"{completed}/{total}")
        
        if st.button("Generate", type="primary", use_container_width=True):
            research_context = "\n".join([f"{d['query']}: {d['result']}" for d in list(st.session_state.research_results.values())[:15]])
            
            progress = st.progress(0)
            status = st.empty()
            
            st.session_state.generated_sections = []
            
            # Get latest alerts for first section
            status.text("Checking latest updates...")
            latest_alerts = get_latest_alerts(st.session_state.focus_keyword, st.session_state.target_country)
            
            for idx, heading in enumerate(outline['headings']):
                status.text(f"Writing: {heading['h2_title'][:50]}...")
                
                # First section gets special treatment - summary with news
                is_first = (idx == 0)
                content, _ = generate_section_content(
                    heading, 
                    st.session_state.focus_keyword, 
                    st.session_state.target_country, 
                    research_context,
                    is_first_section=is_first,
                    latest_alerts=latest_alerts if is_first else ""
                )
                
                table, _ = generate_data_table(heading, st.session_state.target_country, research_context) if heading.get('needs_table') else (None, None)
                
                st.session_state.generated_sections.append({'heading': heading, 'content': content, 'table': table})
                progress.progress((idx + 1) / total)
                time.sleep(1)
            
            if st.session_state.paa_keywords:
                status.text("FAQs...")
                faqs, _ = generate_faqs(st.session_state.focus_keyword, st.session_state.target_country, st.session_state.paa_keywords, research_context)
                if faqs:
                    st.session_state.generated_faqs = faqs['faqs']
            
            status.success("Done!")
            st.rerun()
        
        if st.session_state.generated_sections:
            st.markdown("---")
            st.markdown(f"# {outline['article_title']}")
            
            total_words = 0
            for section in st.session_state.generated_sections:
                st.markdown(f"## {section['heading']['h2_title']}")
                if section['content']:
                    st.markdown(section['content'])
                    total_words += len(section['content'].split())
                if section.get('table'):
                    table = section['table']
                    st.markdown(f"### {table.get('table_title')}")
                    if table.get('rows') and table.get('headers'):
                        df = pd.DataFrame(table['rows'], columns=table['headers'])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                st.markdown("---")
            
            if st.session_state.generated_faqs:
                st.markdown("## FAQs")
                for faq in st.session_state.generated_faqs:
                    st.markdown(f"### {faq['question']}")
                    st.markdown(faq['answer'])
            
            st.success(f"{total_words:,} words")
            
            html = export_to_html(outline['article_title'], outline['meta_description'], st.session_state.generated_sections, st.session_state.generated_faqs)
            st.download_button("Download HTML", data=html.encode('utf-8'), file_name=f"{st.session_state.focus_keyword.replace(' ', '_')}.html", mime="text/html", use_container_width=True)

st.sidebar.markdown("---")
if st.sidebar.button("Clear All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
