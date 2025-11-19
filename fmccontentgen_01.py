import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
import time
from datetime import datetime
import re

# --- APP CONFIGURATION ---
st.set_page_config(page_title="SEO Content Generator", layout="wide")
st.title("SEO Content Generator (Dense-Table Protocol)")

# --- SESSION STATE INITIALIZATION ---
if 'fanout_results' not in st.session_state: st.session_state.fanout_results = None
if 'research_results' not in st.session_state: st.session_state.research_results = {}
if 'selected_queries' not in st.session_state: st.session_state.selected_queries = set()
if 'content_outline' not in st.session_state: st.session_state.content_outline = None
if 'paa_keywords' not in st.session_state: st.session_state.paa_keywords = []
if 'keyword_combinations' not in st.session_state: st.session_state.keyword_combinations = []
if 'google_ads_keywords' not in st.session_state: st.session_state.google_ads_keywords = []
if 'generated_sections' not in st.session_state: st.session_state.generated_sections = []
if 'generated_faqs' not in st.session_state: st.session_state.generated_faqs = []
if 'main_topic' not in st.session_state: st.session_state.main_topic = ""
if 'focus_keyword' not in st.session_state: st.session_state.focus_keyword = ""
if 'target_country' not in st.session_state: st.session_state.target_country = ""
if 'required_tables' not in st.session_state: st.session_state.required_tables = []
if 'bulk_research_running' not in st.session_state: st.session_state.bulk_research_running = False
if 'bulk_research_progress' not in st.session_state: st.session_state.bulk_research_progress = {}

# --- API CONFIGURATION SIDEBAR ---
st.sidebar.header("Configuration")
with st.sidebar.expander("API Keys", expanded=True):
    gemini_key = st.text_input("Gemini API Key", type="password")
    perplexity_key = st.text_input("Perplexity API Key", type="password")
    grok_key = st.text_input("Grok API Key", type="password")

gemini_model = st.sidebar.selectbox("Gemini Model", ["gemini-2.0-flash", "gemini-1.5-pro"], index=0)

# --- CONTEXT DATE SELECTOR ---
st.sidebar.markdown("---")
st.sidebar.header("Temporal Context")
st.sidebar.info("Select the 'Current Date' to anchor the AI's research and content generation.")
context_date = st.sidebar.date_input("Anchor Date", value=datetime.now())
formatted_date = context_date.strftime("%B %d, %Y")
current_year = context_date.year

# Initialize Gemini
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(gemini_model)
        st.sidebar.success(f"✓ Gemini Active")
    except Exception as e:
        st.sidebar.error(f"Gemini Error: {e}")
        model = None
else:
    model = None

# --- UTILITY FUNCTIONS ---

def call_perplexity(query, system_prompt=None, max_retries=2):
    if not perplexity_key: return {"error": "Missing API key"}
    
    if not system_prompt:
        system_prompt = f"Current Date: {formatted_date}. Provide factual data with specific numbers, dates, and examples. Be direct."
    
    headers = {"Authorization": f"Bearer {perplexity_key}", "Content-Type": "application/json"}
    data = {
        "model": "sonar", 
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    }
    
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
    if not grok_key: return None, "Missing API key"
    # Use Grok API URL (Verify specifically if it is x.ai or standard OpenAI compatible endpoint)
    GROK_API_URL = "https://api.x.ai/v1/chat/completions" 
    
    headers = {"Authorization": f"Bearer {grok_key}", "Content-Type": "application/json"}
    payload = {"messages": messages, "model": "grok-2-latest", "stream": False, "temperature": temperature, "max_tokens": max_tokens}
    
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
    if not model: return None, "Gemini not configured"
    min_queries = 12 if mode == "AI Overview (simple)" else 25
    prompt = f"""Generate {min_queries}+ research queries for: "{topic}"
    Context Date: {formatted_date}
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

# --- OPTIMIZED OUTLINE GENERATOR (STRICT KEYWORDS) ---
def generate_outline_from_research(focus_keyword, target_country, required_tables, research_data):
    if not model: return None, "Gemini not configured"
    
    research_summary = "\n".join([f"- {data['query']}: {data['result'][:150]}..." for data in list(research_data.values())[:15]])
    
    prompt = f"""
    ROLE: SEO Technical Architect.
    TASK: Structuring a content outline for "{focus_keyword}".
    TARGET: {target_country} | ANCHOR DATE: {formatted_date}

    ### HEADING PROTOCOL (STRICT):
    1. **EXACT KEYWORDS ONLY:** Headings (H2) must be raw search terms or direct questions.
       - ❌ NO Creative/Fluff: "Unlock Your Potential," "The Comprehensive Guide to..."
       - ❌ NO Dates in H2: Do not put "{current_year}" in the heading text itself (keep it timeless).
       - ✅ YES: "CMA Eligibility," "CMA Exam Fees," "CMA vs CPA," "CMA Salary."
    
    2. **LOGICAL FLOW:** - Start with Definition/Eligibility.
       - Move to Technical Details (Fees, Syllabus, Pattern).
       - End with Outcomes (Salary, Job Roles).

    ### MANDATORY TABLES:
    Ensure every data-heavy section (Eligibility, Fees, Dates, Syllabus) is marked as 'needs_table'.

    RESEARCH CONTEXT:
    {research_summary}

    RETURN JSON ONLY:
    {{
      "article_title": "{focus_keyword}: [Main Benefit/Result] ({current_year} Update)",
      "meta_description": "Detailed breakdown of {focus_keyword} for {target_country}. Covers eligibility, fees, and exam pattern.",
      "headings": [
        {{
          "h2_title": "[Insert Exact Keyword]",
          "content_focus": "Specific technical details to cover",
          "needs_table": true,
          "table_purpose": "Full breakdown of data including [Metric 1] and [Metric 2]"
        }}
      ]
    }}
    """
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        return json.loads(json_text.strip()), None
    except Exception as e:
        return None, str(e)

def get_latest_alerts(focus_keyword, target_country):
    if not perplexity_key: return ""
    query = f"Latest news, updates, deadlines, or important announcements about {focus_keyword} in {target_country} as of {formatted_date}. Include exam dates, fee changes, new regulations."
    system_prompt = f"Provide only the most recent and factual updates relevant to {formatted_date}. Be direct and concise."
    res = call_perplexity(query, system_prompt=system_prompt)
    if res and 'choices' in res and len(res['choices']) > 0:
        return res['choices'][0]['message'].get('content', '')[:500]
    return ""

# --- OPTIMIZED CONTENT GENERATOR (NO FLUFF, NATURAL DATE) ---
def generate_section_content(heading, focus_keyword, target_country, research_context, is_first_section=False, latest_alerts=""):
    if not grok_key: return None, "Grok required"
    
    # Base System Instruction for Dense Information
    system_instruction = f"""
    You are a technical documentation expert.
    CONTEXT: Anchor Date is {formatted_date}. Target Country: {target_country}.
    
    ### STYLE RULES (CRITICAL):
    1. **DEFINITIVE PRESENT TENSE:** Do not keep saying "As of {formatted_date}" or "According to latest info." Just state the facts.
       - ❌ Avoid: "As per the latest update, the deadline is May."
       - ✅ Use: "The registration deadline closes in May."
    2. **NO GENERIC FILLER:** - If writing about 'Eligibility', do NOT list "Good communication skills." List specific degrees, CPA credits, or years of experience.
       - If writing about 'Fees', list the exact currency amount.
    3. **NO REPEATING HEADINGS:** Do not start the text with the heading itself.
    4. **NO INTROS:** Start immediately with the core answer.
    """

    if is_first_section:
        prompt = f"""
        {system_instruction}
        
        TASK: Write the specific opening paragraph for "{focus_keyword}".
        LATEST NEWS: {latest_alerts}
        
        INSTRUCTIONS:
        1. **Immediate Context:** If there is a specific deadline or exam window open RIGHT NOW ({formatted_date}), mention it in the first sentence naturally.
        2. **Technical Definition:** Define exactly what it is, who issues it, and the primary requirement.
        
        RESEARCH: {research_context[:2500]}
        """
    else:
        prompt = f"""
        {system_instruction}
        
        TASK: Write high-density technical content for the keyword: "{heading['h2_title']}"
        FOCUS: {heading.get('content_focus', 'Technical details')}
        
        DATA REQUIREMENTS:
        - If covering **Fees**: Include Entrance Fee, Exam Fee, and Membership Fee separately.
        - If covering **Eligibility**: Mention specific degree types and experience months.
        - If covering **Syllabus**: Mention specific topic weights.
        
        RESEARCH: {research_context[:2000]}
        """
    
    messages = [{"role": "user", "content": prompt}]
    # Lower temperature (0.3) for higher factual adherence
    content, error = call_grok(messages, max_tokens=700, temperature=0.3)
    return content, error

# --- OPTIMIZED TABLE GENERATOR (TOTALITY CHECK) ---
def generate_data_table(heading, target_country, research_context):
    if not grok_key or not heading.get('needs_table'): return None, "No table needed"
    
    prompt = f"""
    TASK: Generate a "Totality Data Table" for "{heading['h2_title']}".
    CONTEXT: {target_country} | Anchor Date: {formatted_date}
    RESEARCH: {research_context[:2500]}

    ### TABLE RULES:
    1. **NO SUMMARIES:** Do not summarize. If there are different fees for Students vs Professionals, create separate rows.
    2. **MANDATORY COLUMNS:**
       - **Metric/Category:** (e.g., "Entrance Fee", "Part 1 Exam")
       - **Value ({target_country} Currency):** Exact numbers.
       - **Validity/Condition:** (e.g., "Valid for 3 years", "Non-refundable")
    3. **COVERAGE:** The table must cover the topic in totality. (Minimum 5 rows unless impossible).
    4. **FRESHNESS:** If a deadline has passed relative to {formatted_date}, mark it as "Closed".
    
    Return ONLY valid JSON:
    {{
      "table_title": "Detailed Breakdown: {heading['h2_title']}",
      "headers": ["Category", "Value/Date", "Important Condition"],
      "rows": [
          ["Item 1", "Value 1", "Condition 1"],
          ["Item 2", "Value 2", "Condition 2"]
      ]
    }}
    """
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=1000, temperature=0.2)
    
    if error: return None, error
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except:
        return None, "Parse error"

def generate_faqs(focus_keyword, target_country, paa_keywords, research_context):
    if not grok_key: return None, "Grok required"
    paa_text = "\n".join([f"- {kw}" for kw in paa_keywords[:30]])
    prompt = f"""Generate FAQs for: "{focus_keyword}"
    AUDIENCE: Students in {target_country} | DATE: {formatted_date}
    PAA: {paa_text}
    RESEARCH: {research_context[:2000]}
    Answer questions with specific facts/numbers. No fluff.
    Return ONLY valid JSON:
    {{"faqs": [{{"question": "Question?", "answer": "Direct answer with facts"}}]}}"""
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=3000, temperature=0.6)
    if error: return None, error
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
            # Process content blocks for basic formatting
            blocks = content.split('\n\n')
            for block in blocks:
                block = block.strip()
                if not block: continue
                # Simple bold replacement
                block = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', block)
                # Check if it's a list
                if block.strip().startswith('- '):
                    html.append('<ul>')
                    lines = block.split('\n')
                    for line in lines:
                        if line.strip().startswith('- '):
                            html.append(f' <li>{line.strip()[2:]}</li>')
                    html.append('</ul>')
                else:
                    html.append(f'<p>{block}</p>')
        
        if section.get('table'):
            table = section['table']
            html.append(f'<h3>{table.get("table_title", "Data")}</h3>')
            html.append('<table border="1" style="border-collapse: collapse; width: 100%;">')
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            if headers:
                html.append(' <thead><tr>')
                for header in headers: html.append(f'  <th style="padding: 8px; background-color: #f2f2f2;">{header}</th>')
                html.append(' </tr></thead>')
            if rows:
                html.append(' <tbody>')
                for row in rows:
                    html.append('  <tr>')
                    for cell in row: html.append(f'   <td style="padding: 8px;">{cell}</td>')
                    html.append('  </tr>')
                html.append(' </tbody>')
            html.append('</table>')
        html.append('')
    
    if faqs:
        html.append('<h2>FAQs</h2>')
        for faq in faqs:
            html.append(f'<h3>{faq["question"]}</h3>')
            html.append(f'<p>{faq["answer"]}</p>')
    
    return '\n'.join(html)

# --- MAIN UI LAYOUT ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Research", "2. Settings", "3. Results", "4. Outline", "5. Content"])

with tab1:
    st.header("Research Phase")
    st.info(f"Current Context Date: **{formatted_date}**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Topic:", placeholder="e.g., CMA certification")
        if topic: st.session_state.main_topic = topic
    with col2:
        mode = st.selectbox("Depth", ["AI Overview (simple)", "AI Mode (complex)"])
    
    if st.button("Generate Queries", type="primary", use_container_width=True):
        if not topic.strip() or not gemini_key:
            st.error("Please enter topic and Gemini API Key")
        else:
            with st.spinner("Generating..."):
                result, error = generate_research_queries(topic, mode)
                if result:
                    st.session_state.fanout_results = result
                    st.session_state.selected_queries = set()
                    st.rerun()
                else:
                    st.error(error)
    
    if st.session_state.fanout_results and 'queries' in st.session_state.fanout_results:
        st.markdown("---")
        queries = st.session_state.fanout_results['queries']
        
        categories = sorted(list(set(q.get('category', 'Unknown') for q in queries)))
        selected_cats = st.multiselect("Filter by Category:", categories, default=categories)
        filtered = [q for q in queries if q.get('category', 'Unknown') in selected_cats]
        
        # Batch Selection
        filtered_ids = {f"q_{queries.index(q)}" for q in filtered}
        all_selected = all(qid in st.session_state.selected_queries for qid in filtered_ids) if filtered_ids else False
        if st.checkbox("Select All Visible", value=all_selected):
            st.session_state.selected_queries.update(filtered_ids)
        else:
            st.session_state.selected_queries.difference_update(filtered_ids)
            
        for q in filtered:
            qid = f"q_{queries.index(q)}"
            col1, col2 = st.columns([4, 1])
            with col1:
                is_selected = qid in st.session_state.selected_queries
                if st.checkbox(f"**{q['query']}**", value=is_selected, key=f"cb_{qid}"):
                    st.session_state.selected_queries.add(qid)
                else:
                    st.session_state.selected_queries.discard(qid)
            with col2:
                if qid in st.session_state.research_results:
                    st.success("✓ Done")
        
        if st.session_state.selected_queries and perplexity_key:
            unresearched = [qid for qid in st.session_state.selected_queries if qid not in st.session_state.research_results]
            if unresearched:
                if st.button(f"Research {len(unresearched)} Queries", type="secondary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status = st.empty()
                    for i, qid in enumerate(unresearched):
                        q_idx = int(qid.split('_')[1])
                        q_text = queries[q_idx]['query']
                        status.text(f"Researching: {q_text}...")
                        
                        res = call_perplexity(q_text)
                        if res and 'choices' in res:
                            st.session_state.research_results[qid] = {
                                'query': q_text,
                                'result': res['choices'][0]['message']['content']
                            }
                        progress_bar.progress((i + 1) / len(unresearched))
                    st.success("Research Complete!")
                    st.rerun()

with tab2:
    st.header("Target Settings")
    col1, col2 = st.columns(2)
    with col1:
        focus_keyword = st.text_input("Focus Keyword *", value=st.session_state.main_topic)
        if focus_keyword: st.session_state.focus_keyword = focus_keyword
    with col2:
        target_country = st.selectbox("Target Audience/Country *", ["India", "United States", "United Kingdom", "Canada", "Global"])
        st.session_state.target_country = target_country
    
    st.subheader("Keyword Uploads (Optional)")
    paa_file = st.file_uploader("Upload PAA (CSV)", type=['csv'])
    if paa_file:
        df = pd.read_csv(paa_file)
        st.session_state.paa_keywords = df.iloc[:, 0].dropna().astype(str).tolist()
        st.success(f"Loaded {len(st.session_state.paa_keywords)} PAA keywords")

with tab3:
    st.header("Raw Research Data")
    if st.session_state.research_results:
        for qid, data in st.session_state.research_results.items():
            with st.expander(data['query']):
                st.write(data['result'])
    else:
        st.info("No research data yet. Go to Tab 1.")

with tab4:
    st.header("Content Outline")
    if st.button("Generate Outline", type="primary"):
        if not st.session_state.research_results:
            st.error("Please complete research first.")
        else:
            with st.spinner("Architecting Outline..."):
                outline, error = generate_outline_from_research(
                    st.session_state.focus_keyword,
                    st.session_state.target_country,
                    [],
                    st.session_state.research_results
                )
                if outline:
                    st.session_state.content_outline = outline
                    st.rerun()
                else:
                    st.error(error)

    if st.session_state.content_outline:
        outline = st.session_state.content_outline
        st.subheader(outline['article_title'])
        st.caption(outline['meta_description'])
        for h in outline['headings']:
            with st.expander(f"H2: {h['h2_title']}"):
                st.write(f"**Focus:** {h['content_focus']}")
                if h.get('needs_table'):
                    st.info(f"📊 Table Required: {h.get('table_purpose')}")

with tab5:
    st.header("Final Content Generation")
    if st.button("Generate Article", type="primary"):
        if not st.session_state.content_outline:
            st.error("No outline found.")
        else:
            progress = st.progress(0)
            status = st.empty()
            st.session_state.generated_sections = []
            
            # Context Preparation
            research_context = "\n".join([f"{d['query']}: {d['result']}" for d in list(st.session_state.research_results.values())[:15]])
            latest_alerts = get_latest_alerts(st.session_state.focus_keyword, st.session_state.target_country)
            
            total_sections = len(st.session_state.content_outline['headings'])
            
            for idx, heading in enumerate(st.session_state.content_outline['headings']):
                status.text(f"Writing Section {idx+1}/{total_sections}: {heading['h2_title']}...")
                
                is_first = (idx == 0)
                content, _ = generate_section_content(
                    heading, 
                    st.session_state.focus_keyword, 
                    st.session_state.target_country, 
                    research_context, 
                    is_first_section=is_first, 
                    latest_alerts=latest_alerts if is_first else ""
                )
                
                table = None
                if heading.get('needs_table'):
                    status.text(f"Generating Data Table for {heading['h2_title']}...")
                    table, _ = generate_data_table(heading, st.session_state.target_country, research_context)
                
                st.session_state.generated_sections.append({'heading': heading, 'content': content, 'table': table})
                progress.progress((idx + 1) / total_sections)
            
            # FAQs
            if st.session_state.paa_keywords:
                status.text("Generating FAQs...")
                faqs, _ = generate_faqs(st.session_state.focus_keyword, st.session_state.target_country, st.session_state.paa_keywords, research_context)
                if faqs: st.session_state.generated_faqs = faqs['faqs']
            
            status.success("Generation Complete!")
            st.rerun()

    if st.session_state.generated_sections:
        st.markdown("## Preview")
        for section in st.session_state.generated_sections:
            st.markdown(f"### {section['heading']['h2_title']}")
            st.markdown(section['content'])
            if section.get('table'):
                st.dataframe(pd.DataFrame(section['table']['rows'], columns=section['table']['headers']), hide_index=True)
            st.markdown("---")
        
        if st.session_state.generated_faqs:
            st.markdown("### FAQs")
            for faq in st.session_state.generated_faqs:
                with st.expander(faq['question']):
                    st.write(faq['answer'])

        # Export
        html_content = export_to_html(
            st.session_state.content_outline['article_title'],
            st.session_state.content_outline['meta_description'],
            st.session_state.generated_sections,
            st.session_state.generated_faqs
        )
        st.download_button("Download HTML", data=html_content, file_name="article.html", mime="text/html")
