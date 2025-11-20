import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
import time
from datetime import datetime, timedelta
import re

# --- APP CONFIGURATION ---
st.set_page_config(page_title="SEO Content Generator - Expert Quality", layout="wide")
st.title("🎯 SEO Content Generator - Expert Quality")
st.caption("Publication-ready articles with strict quality control & Coherent Flow")

# --- SESSION STATE INITIALIZATION ---
if 'research_results' not in st.session_state: st.session_state.research_results = {}
if 'selected_queries' not in st.session_state: st.session_state.selected_queries = set()
if 'fanout_results' not in st.session_state: st.session_state.fanout_results = None
if 'content_outline' not in st.session_state: st.session_state.content_outline = None
if 'paa_keywords' not in st.session_state: st.session_state.paa_keywords = []
if 'keyword_planner_data' not in st.session_state: st.session_state.keyword_planner_data = None
if 'generated_sections' not in st.session_state: st.session_state.generated_sections = []
if 'generated_faqs' not in st.session_state: st.session_state.generated_faqs = []
if 'main_topic' not in st.session_state: st.session_state.main_topic = ""
if 'focus_keyword' not in st.session_state: st.session_state.focus_keyword = ""
if 'target_country' not in st.session_state: st.session_state.target_country = "India"
if 'latest_updates' not in st.session_state: st.session_state.latest_updates = []
if 'custom_headings' not in st.session_state: st.session_state.custom_headings = []
if 'selected_custom_headings' not in st.session_state: st.session_state.selected_custom_headings = set()
if 'previous_section_context' not in st.session_state: st.session_state.previous_section_context = ""

# --- API CONFIGURATION SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
with st.sidebar.expander("🔑 API Keys", expanded=True):
    gemini_key = st.text_input("Gemini API Key", type="password", help="Get from Google AI Studio")
    perplexity_key = st.text_input("Perplexity API Key", type="password", help="Get from perplexity.ai/settings")
    grok_key = st.text_input("Grok API Key", type="password", help="Get from x.ai")

gemini_model = st.sidebar.selectbox("Gemini Model", ["gemini-2.0-flash-exp", "gemini-2.0-flash"], index=0)

# Reset Button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset All Data", type="secondary", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --- CONTEXT DATE SELECTOR ---
st.sidebar.markdown("---")
st.sidebar.header("📅 Content Context")
context_date = st.sidebar.date_input("Content Date", value=datetime.now())
formatted_date = context_date.strftime("%B %d, %Y")
current_year = context_date.year

# Initialize Gemini
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(gemini_model)
    except Exception as e:
        st.sidebar.error(f"Gemini Error: {e}")
        model = None
else:
    model = None

# --- UTILITY FUNCTIONS ---

def generate_research_queries(topic, mode="AI Overview (simple)"):
    """FANOUT: Generate research queries using Gemini"""
    if not model: return None, "Gemini not configured"
    
    min_queries = 12 if mode == "AI Overview (simple)" else 25
    
    prompt = f"""Generate {min_queries}+ DEEP research queries for the topic: "{topic}"
    Context Date: {formatted_date}
    
    QUERY QUALITY REQUIREMENTS:
    - Ask for SPECIFIC, DETAILED information (not general overviews)
    - Include queries that demand: exact data, complete information, step-by-step details
    - Cover ALL major aspects of this topic: Definition, Eligibility, Fees, Process, Pros/Cons, Latest Updates.
    
    Return ONLY valid JSON:
    {{"queries": [{{"query": "specific detailed question", "category": "category", "priority": "high/medium/low"}}]}}"""
    
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        return json.loads(json_text.strip()), None
    except Exception as e:
        return None, str(e)

def call_perplexity(query, system_prompt=None, max_retries=2):
    """Call Perplexity with deep research instructions and better error handling"""
    if not perplexity_key: return {"error": "Missing API key"}
    
    # Updated model to 'sonar' (latest standard)
    model_name = "sonar" 
    
    if not system_prompt:
        system_prompt = f"""Current Date: {formatted_date}
CRITICAL DATA COLLECTION RULES:
- Provide COMPLETE, FACTUAL data with exact numbers.
- Search official websites, government portals.
- If exact data unavailable, say "Data not found".
- NO approximations. Return comprehensive data."""
    
    headers = {
        "Authorization": f"Bearer {perplexity_key}", 
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name, 
        "messages": [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": query}
        ]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post("https://api.perplexity.ai/chat/completions", 
                                   headers=headers, json=data, timeout=60)
            
            # Return specific error message if API fails (e.g. 401 Unauthorized)
            if response.status_code != 200:
                return {"error": f"API Error {response.status_code}: {response.text}"}
                
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1: time.sleep(2); continue
            return {"error": f"Connection Failed: {str(e)}"}
            
    return {"error": "Max retries exceeded"}
def call_grok(messages, max_tokens=4000, temperature=0.3):
    """Call Grok with strict quality controls"""
    if not grok_key: return None, "Missing API key"
    GROK_API_URL = "[https://api.x.ai/v1/chat/completions](https://api.x.ai/v1/chat/completions)"
    
    headers = {"Authorization": f"Bearer {grok_key}", "Content-Type": "application/json"}
    payload = {
        "messages": messages, 
        "model": "grok-2-latest", 
        "stream": False, 
        "temperature": temperature, 
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content'], None
        return None, "No response"
    except Exception as e:
        return None, str(e)

def parse_keyword_planner_csv(file_path):
    """Parse CSV keywords"""
    keywords = []
    try:
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        keyword_col = df.columns[0] # Assuming first column
        for idx, row in df.iterrows():
            kw = str(row[keyword_col]).strip()
            if kw and len(kw) > 2:
                keywords.append({'keyword': kw, 'suitable_for_heading': True})
        return keywords, None
    except:
        return None, "Unable to read file"

def generate_semantic_h1(focus_keyword, research_data):
    """Generate H1"""
    if not model: return f"{focus_keyword} - Complete Guide {current_year}"
    prompt = f"""Create a semantic H1 for "{focus_keyword}" using year {current_year}. 
    Format: "[Main Topic] - [Key Aspect 1], [Aspect 2] and [Aspect 3] ({current_year})"
    Example: "MBA in India - Fees, Eligibility and Top Colleges (2025)"
    Return ONLY the H1."""
    try:
        response = model.generate_content(prompt)
        return response.text.strip().strip('"')
    except:
        return f"{focus_keyword} - Complete Guide {current_year}"

def convert_queries_to_h2_headings(research_data, focus_keyword):
    """Convert queries to SEO H2s using Grok"""
    if not grok_key: return None
    
    queries_list = "\n".join([f"{idx+1}. {data['query']}" for idx, data in enumerate(research_data.values())])
    
    prompt = f"""Convert these research queries into Short, Punchy, SEO-Optimized H2 Headings.
    Topic: {focus_keyword}
    Queries: {queries_list}
    
    RULES:
    1. HEADINGS MUST BE KEYWORDS or QUESTIONS.
    2. MAX 6-8 words.
    3. NATURAL phrasing (e.g., "Eligibility Criteria" instead of "What is the eligibility...").
    4. NO duplicates.
    
    Return JSON: {{"headings": [{{"original_query": "query", "h2": "Better Heading"}}]}}"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=2500)
    
    try:
        if "```json" in response: response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()).get('headings', [])
    except: return None

def generate_section_content(heading, research_context, previous_context="", is_first_section=False, latest_updates=None):
    """Generate content with flow and strict rules"""
    if not grok_key: return None, "Grok required"
    
    # Structure definition
    structure_instruction = ""
    if heading.get('needs_bullets'):
        structure_instruction = "Format: Introductory paragraph followed by a detailed Bulleted List."
    elif heading.get('needs_table'):
        structure_instruction = "Format: detailed Introductory paragraph only. (The table will be added separately, do not write a text table)."
    else:
        structure_instruction = "Format: 2-3 cohesive paragraphs."

    prompt = f"""You are an expert SEO Writer. Write the section: "{heading['h2_title']}".
    Context Date: {formatted_date}
    
    PREVIOUS SECTION SUMMARY (For Flow):
    {previous_context}
    
    INSTRUCTIONS:
    1. TRANSITION: Start by bridging from the previous section (if any) to this one naturally.
    2. NO LINKS: Do NOT include external links or [text](url) or markdown links.
    3. NO REPETITION: Do not repeat data presented in previous sections.
    4. STYLE: Professional, direct, data-driven. No fluff ("In this section we will explore...").
    5. {structure_instruction}
    
    RESEARCH DATA TO USE:
    {research_context[:4000]}
    
    If specific data is missing, write generally about the process/standard norms, do not say "Data not available".
    """

    if is_first_section and latest_updates:
        prompt += f"\n\nInclude these Latest Updates at the very top: {latest_updates}"

    messages = [{"role": "user", "content": prompt}]
    content, error = call_grok(messages, max_tokens=1200, temperature=0.3)
    return content, error

def generate_intelligent_table(heading, research_context):
    """Generate validated table"""
    if not grok_key or not heading.get('needs_table'): return None, "No table needed"
    
    prompt = f"""Create a DATA TABLE for: "{heading['h2_title']}" based on this research:
    {research_context[:3000]}
    
    RULES:
    - Return JSON: {{ "table_title": "Title", "headers": ["Col1", "Col2"], "rows": [["Data1", "Data2"]] }}
    - NO empty cells. If data missing, remove the row.
    - NO "N/A" or "Varies".
    - Specific data only."""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=2000, temperature=0.2)
    
    try:
        if "```json" in response: response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except: return None, "Parse error"

def generate_faqs(focus_keyword, paa_keywords, research_context):
    """Generate FAQs (2-3 lines)"""
    if not grok_key: return None, "Grok required"
    
    prompt = f"""Generate 8 FAQs for "{focus_keyword}".
    Context: {research_context[:3000]}
    Questions to cover: {paa_keywords[:10]}
    
    RULES:
    1. Answer length: 2-3 sentences (40-60 words). Detailed but crisp.
    2. Factual answers based on research.
    3. Return JSON: {{ "faqs": [{{"question": "Q", "answer": "A"}}] }}"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=2500)
    try:
        if "```json" in response: response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except: return None, "Parse error"

def export_to_html(article_title, sections, faqs, latest_updates):
    """Export clean HTML"""
    html = [f'<h1>{article_title}</h1>']
    
    if latest_updates:
        html.append('<div style="background:#fff3cd;padding:15px;border-left:4px solid #ffc107;margin-bottom:20px">')
        html.append('<strong>Latest Updates:</strong><ul>')
        for u in latest_updates: html.append(f'<li>{u}</li>')
        html.append('</ul></div>')
    
    for sec in sections:
        html.append(f'<h2>{sec["heading"]["h2_title"]}</h2>')
        # Clean Markdown links if any slipped through
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', sec['content']) 
        content = content.replace('\n\n', '</p><p>')
        html.append(f'<p>{content}</p>')
        
        if sec.get('table'):
            t = sec['table']
            html.append(f'<h3>{t.get("table_title","")}</h3><table border="1" style="border-collapse:collapse;width:100%">')
            html.append('<thead><tr>' + ''.join([f'<th style="padding:8px;background:#eee">{h}</th>' for h in t['headers']]) + '</tr></thead><tbody>')
            for row in t['rows']:
                html.append('<tr>' + ''.join([f'<td style="padding:8px">{c}</td>' for c in row]) + '</tr>')
            html.append('</tbody></table><br>')
            
    if faqs:
        html.append('<h2>Frequently Asked Questions</h2>')
        for f in faqs:
            html.append(f'<h3>{f["question"]}</h3><p>{f["answer"]}</p>')
            
    return '\n'.join(html)

# --- MAIN UI ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Setup & Keywords", "2. Research Queries", "3. Outline Structure", "4. Generate Content"])

# TAB 1: SETUP
with tab1:
    st.header("Step 1: Topic & Keyword Setup")
    col1, col2 = st.columns(2)
    with col1:
        focus_keyword = st.text_input("Main Topic *", value=st.session_state.main_topic)
        if focus_keyword: st.session_state.main_topic = focus_keyword; st.session_state.focus_keyword = focus_keyword
    with col2:
        st.session_state.target_country = st.selectbox("Target Audience", ["India", "USA", "Global"], index=0)
    
    st.markdown("### Uploads")
    k_file = st.file_uploader("Keywords CSV (becomes H2s)", key="kw")
    if k_file:
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False) as tmp: tmp.write(k_file.getvalue()); t_path=tmp.name
        kws, _ = parse_keyword_planner_csv(t_path)
        if kws: st.session_state.keyword_planner_data = kws; st.success(f"{len(kws)} keywords loaded")
        os.unlink(t_path)

# TAB 2: RESEARCH
with tab2:
    st.header("Step 2: Research")
    if not st.session_state.main_topic: 
        st.warning("⚠️ Please set a Main Topic in Tab 1 first.")
        st.stop()
    
    # 1. Generate Queries Section
    if st.button("Generate Research Queries", type="primary"):
        if not gemini_key:
            st.error("❌ Gemini API Key is missing in the sidebar.")
        else:
            with st.spinner("🤖 Generating research queries..."):
                res, err = generate_research_queries(st.session_state.main_topic)
                if res: 
                    st.session_state.fanout_results = res
                    st.rerun()
                else:
                    st.error(f"Error generating queries: {err}")
    
    # 2. Display and Select Queries
    if st.session_state.fanout_results:
        queries = st.session_state.fanout_results['queries']
        st.write(f"Generated {len(queries)} queries. Select the ones you want to research:")
        
        # "Select All" button
        if st.button("Select All"):
            for i in range(len(queries)):
                st.session_state.selected_queries.add(f"q_{i}")
            st.rerun()
            
        # Selection Checkboxes
        for i, q in enumerate(queries):
            qid = f"q_{i}"
            # Checkbox logic
            is_checked = qid in st.session_state.selected_queries
            if st.checkbox(f"{q['query']} ({q.get('category', 'General')})", value=is_checked, key=qid):
                st.session_state.selected_queries.add(qid)
            else:
                st.session_state.selected_queries.discard(qid)
        
        st.markdown("---")
        
        # 3. Run Research Button
        count_selected = len(st.session_state.selected_queries)
        st.write(f"**Selected: {count_selected} queries**")
        
        if st.button(f"🚀 Run Research on {count_selected} Queries", type="primary", disabled=(count_selected==0)):
            if not perplexity_key:
                st.error("❌ Perplexity API Key is missing in the sidebar.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                success_count = 0
                errors = []
                
                # Create a list to iterate so we can track index
                selected_ids = list(st.session_state.selected_queries)
                
                for idx, qid in enumerate(selected_ids):
                    # Get query text
                    q_index = int(qid.split('_')[1])
                    q_text = queries[q_index]['query']
                    
                    status_text.text(f"Researching ({idx+1}/{count_selected}): {q_text[:50]}...")
                    
                    # Call API
                    res = call_perplexity(q_text)
                    
                    # Handle Response
                    if 'choices' in res:
                        st.session_state.research_results[qid] = {
                            'query': q_text, 
                            'result': res['choices'][0]['message']['content']
                        }
                        success_count += 1
                    elif 'error' in res:
                        errors.append(f"Query '{q_text[:20]}...': {res['error']}")
                    else:
                        errors.append(f"Query '{q_text[:20]}...': Unknown API response format")
                        
                    progress_bar.progress((idx + 1) / count_selected)
                
                # Final Report
                if errors:
                    st.error(f"❌ Encountered {len(errors)} errors:")
                    for e in errors:
                        st.warning(e)
                    if success_count > 0:
                        st.info(f"✅ Successfully researched {success_count} queries despite errors.")
                        time.sleep(2) # Give user time to read errors
                        st.rerun()
                else:
                    st.success(f"✅ Research Complete! {success_count}/{count_selected} queries saved.")
                    time.sleep(1)
                    st.rerun()

# TAB 3: OUTLINE (HEAVILY MODIFIED)
with tab3:
    st.header("Step 3: Structure & Order")
    if not st.session_state.research_results: st.warning("Complete Tab 2 first"); st.stop()
    
    # Generate Initial Outline if empty
    if not st.session_state.content_outline:
        if st.button("Generate Initial Outline"):
            h1 = generate_semantic_h1(st.session_state.focus_keyword, st.session_state.research_results)
            h2s = convert_queries_to_h2_headings(st.session_state.research_results, st.session_state.focus_keyword)
            
            headings = []
            for i, h in enumerate(h2s):
                headings.append({
                    'id': f"h_{i}",
                    'order': i + 1,
                    'h2_title': h['h2'],
                    'original_query': h['original_query'],
                    'needs_table': False,
                    'needs_bullets': False,
                    'is_manual': False
                })
            st.session_state.content_outline = {'article_title': h1, 'headings': headings}
            st.rerun()

    if st.session_state.content_outline:
        # H1 Editor
        st.session_state.content_outline['article_title'] = st.text_input("Article H1 Title", st.session_state.content_outline['article_title'])
        
        st.markdown("### Organize Sections")
        st.info("🔢 Change the 'Order' number to rearrange. Check 'Table' or 'Bullets' to define format.")
        
        # Manual Add
        with st.expander("➕ Add Custom Manual Section"):
            c_title = st.text_input("New H2 Title")
            c_order = st.number_input("Position", value=len(st.session_state.content_outline['headings'])+1)
            if st.button("Add Section"):
                st.session_state.content_outline['headings'].append({
                    'id': f"man_{len(st.session_state.content_outline['headings'])}",
                    'order': c_order,
                    'h2_title': c_title,
                    'original_query': c_title, # Will be researched on fly
                    'needs_table': False,
                    'needs_bullets': False,
                    'is_manual': True
                })
                st.rerun()

        # List Editor with Order Input
        headings = st.session_state.content_outline['headings']
        # Sort by order for display
        headings.sort(key=lambda x: x['order'])
        
        for i, h in enumerate(headings):
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([1, 4, 1, 1, 0.5])
                with c1:
                    new_order = st.number_input(f"Order", value=h['order'], key=f"ord_{h['id']}", label_visibility="collapsed")
                    if new_order != h['order']:
                        h['order'] = new_order
                        # We don't rerun immediately to allow multiple edits, user must click 'Refresh Order' or just switch tabs
                with c2:
                    new_title = st.text_input("Title", h['h2_title'], key=f"tit_{h['id']}", label_visibility="collapsed")
                    h['h2_title'] = new_title
                    if h.get('is_manual'): st.caption(f"🔷 Manual Section (Will be researched on-the-fly)")
                with c3:
                    h['needs_table'] = st.checkbox("Table", h['needs_table'], key=f"tab_{h['id']}")
                with c4:
                    h['needs_bullets'] = st.checkbox("Bullets", h['needs_bullets'], key=f"bul_{h['id']}")
                with c5:
                    if st.button("❌", key=f"del_{h['id']}"):
                        st.session_state.content_outline['headings'].pop(i)
                        st.rerun()
                st.markdown("---")
        
        if st.button("Refresh Order"): st.rerun()

# TAB 4: GENERATE (ENHANCED)
with tab4:
    st.header("Step 4: Generate Content")
    if not st.session_state.content_outline: st.stop()
    
    headings = sorted(st.session_state.content_outline['headings'], key=lambda x: x['order'])
    
    st.markdown("### Final Plan")
    for h in headings:
        st.text(f"{h['order']}. {h['h2_title']} [{'Table' if h['needs_table'] else ''} {'Bullets' if h['needs_bullets'] else ''}]")
        
    if st.button("🚀 Generate Article", type="primary"):
        progress = st.progress(0)
        st.session_state.generated_sections = []
        previous_context = "This is the start of the article."
        
        # 1. Get Updates
        updates = [] # Placeholder for call_perplexity(news) if needed
        
        total = len(headings)
        existing_research_text = "\n".join([r['result'] for r in st.session_state.research_results.values()])
        
        for idx, h in enumerate(headings):
            st.caption(f"Processing: {h['h2_title']}...")
            
            # A. Dynamic Research Logic
            section_research = existing_research_text
            
            # If manual or we want fresh data for this specific H2
            if h.get('is_manual') or len(section_research) < 500: 
                with st.spinner(f"🔍 Researching fresh data for: {h['h2_title']}..."):
                    fresh_data = call_perplexity(f"Detailed data and facts about: {h['h2_title']} related to {st.session_state.focus_keyword}")
                    if 'choices' in fresh_data:
                        section_research += f"\n\nSpecific Data for {h['h2_title']}:\n{fresh_data['choices'][0]['message']['content']}"
            
            # B. Write Content (Passing previous context for flow)
            content, err = generate_section_content(
                h, 
                section_research, 
                previous_context=previous_context,
                is_first_section=(idx==0),
                latest_updates=None # Can implement updates logic here
            )
            
            # C. Generate Table if requested
            table = None
            if h['needs_table']:
                table, _ = generate_intelligent_table(h, section_research)
            
            # D. Update Context for next section
            # Keep last 300 words as context
            previous_context = content[-1000:] if content else ""
            
            st.session_state.generated_sections.append({
                'heading': h,
                'content': content,
                'table': table
            })
            progress.progress((idx+1)/total)
            
        # Generate FAQs
        faqs, _ = generate_faqs(st.session_state.focus_keyword, [], existing_research_text)
        st.session_state.generated_faqs = faqs.get('faqs', []) if faqs else []
        
        st.success("Done!")
        st.rerun()

    # Display Output
    if st.session_state.generated_sections:
        h1 = st.session_state.content_outline['article_title']
        html = export_to_html(h1, st.session_state.generated_sections, st.session_state.generated_faqs, [])
        
        col1, col2 = st.columns(2)
        col1.download_button("Download HTML", html, "article.html", "text/html")
        
        st.markdown(html, unsafe_allow_html=True)
