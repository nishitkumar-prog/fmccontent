import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
import time
from datetime import datetime, timedelta
import re

# --- APP CONFIGURATION ---
st.set_page_config(page_title="SEO Content Generator", layout="wide")
st.title("SEO Content Generator (Keyword-Driven)")

# --- SESSION STATE INITIALIZATION ---
if 'fanout_results' not in st.session_state: st.session_state.fanout_results = None
if 'research_results' not in st.session_state: st.session_state.research_results = {}
if 'selected_queries' not in st.session_state: st.session_state.selected_queries = set()
if 'content_outline' not in st.session_state: st.session_state.content_outline = None
if 'paa_keywords' not in st.session_state: st.session_state.paa_keywords = []
if 'keyword_planner_data' not in st.session_state: st.session_state.keyword_planner_data = None
if 'google_ads_keywords' not in st.session_state: st.session_state.google_ads_keywords = []
if 'generated_sections' not in st.session_state: st.session_state.generated_sections = []
if 'generated_faqs' not in st.session_state: st.session_state.generated_faqs = []
if 'main_topic' not in st.session_state: st.session_state.main_topic = ""
if 'focus_keyword' not in st.session_state: st.session_state.focus_keyword = ""
if 'target_country' not in st.session_state: st.session_state.target_country = ""
if 'latest_updates' not in st.session_state: st.session_state.latest_updates = []

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
        system_prompt = f"""Current Date: {formatted_date}
        
CRITICAL INSTRUCTIONS:
- Provide DEEP, COMPREHENSIVE data - not surface-level summaries
- Include ALL specific numbers, exact amounts, precise dates
- List complete breakdowns (all fee components, all eligibility criteria, all steps)
- Include source citations where possible
- Provide context: WHY these numbers matter, HOW they compare
- If discussing fees: break down EVERY component (application, exam, membership, renewal)
- If discussing eligibility: list EVERY requirement with exact specifications
- If discussing process: detail EVERY step with timelines
- Be ACCURATE - verify facts, don't approximate
- Go DEEP - provide expert-level detail, not beginner overviews"""
    
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
    prompt = f"""Generate {min_queries}+ DEEP research queries for: "{topic}"
    Context Date: {formatted_date}
    
    QUERY QUALITY REQUIREMENTS:
    - Ask for SPECIFIC, DETAILED information (not general overviews)
    - Include queries that demand: exact numbers, complete breakdowns, step-by-step processes
    - Examples of GOOD queries:
      * "What are ALL fee components for {topic} including application, exam, membership, and renewal costs?"
      * "What are the EXACT eligibility criteria with specific degree requirements and work experience details?"
      * "What is the COMPLETE step-by-step process with timeline for each phase?"
    - Examples of BAD queries:
      * "Tell me about {topic}" (too general)
      * "What is {topic}?" (too basic)
    
    Categories should include: Fees, Eligibility, Process, Syllabus, Career, Comparison, Timeline, Requirements
    
    Return ONLY valid JSON:
    {{"queries": [{{"query": "specific detailed question", "category": "category", "priority": "high/medium/low", "purpose": "why this data is needed"}}]}}"""
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        return json.loads(json_text.strip()), None
    except Exception as e:
        return None, str(e)

def parse_keyword_planner_csv(df):
    """Parse Google Keyword Planner CSV and extract relevant keywords"""
    keywords = []
    
    # Common column names in Google Keyword Planner exports
    keyword_col = None
    search_vol_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'keyword' in col_lower and keyword_col is None:
            keyword_col = col
        elif 'avg' in col_lower and 'search' in col_lower:
            search_vol_col = col
    
    if keyword_col:
        for idx, row in df.iterrows():
            kw = str(row[keyword_col]).strip()
            if kw and kw != 'nan' and len(kw) > 2:
                search_vol = row[search_vol_col] if search_vol_col else 0
                keywords.append({
                    'keyword': kw,
                    'search_volume': search_vol,
                    'suitable_for_heading': True
                })
    
    return keywords

def get_latest_news_updates(focus_keyword, target_country, days_back=15):
    """Fetch latest news/updates from last X days"""
    if not perplexity_key: return []
    
    cutoff_date = context_date - timedelta(days=days_back)
    cutoff_str = cutoff_date.strftime("%B %d, %Y")
    
    query = f"""Find ONLY the latest news, updates, or announcements about {focus_keyword} in {target_country} published between {cutoff_str} and {formatted_date}. 
    Focus on: exam dates, registration deadlines, fee changes, new regulations, policy updates.
    If nothing significant happened in this period, say 'No major updates'.
    Be specific with dates."""
    
    system_prompt = f"Current date is {formatted_date}. Return only factual updates with exact dates from the last {days_back} days. If no updates exist, clearly state that."
    
    res = call_perplexity(query, system_prompt=system_prompt)
    
    if res and 'choices' in res and len(res['choices']) > 0:
        content = res['choices'][0]['message'].get('content', '')
        
        # Check if there are actual updates
        if any(phrase in content.lower() for phrase in ['no major update', 'no significant', 'no recent', 'no new']):
            return []
        
        # Extract bullet points or create them
        updates = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 20:  # Meaningful update
                # Remove existing bullets
                line = re.sub(r'^[-•*]\s*', '', line)
                if line:
                    updates.append(line)
        
        # Return max 2 most relevant updates
        return updates[:2]
    
    return []

def generate_outline_with_keywords(focus_keyword, target_country, keyword_data, research_data):
    """Generate outline using keywords from Google Keyword Planner"""
    if not model: return None, "Gemini not configured"
    
    research_summary = "\n".join([f"- {data['query']}: {data['result'][:150]}..." for data in list(research_data.values())[:15]])
    
    # Prepare keyword list
    keyword_list = "\n".join([f"- {kw['keyword']} (Search Vol: {kw.get('search_volume', 'N/A')})" for kw in keyword_data[:50]])
    
    prompt = f"""
    ROLE: SEO Content Architect
    TASK: Create a content outline for "{focus_keyword}"
    TARGET: {target_country} | DATE: {formatted_date}

    ### AVAILABLE KEYWORDS (from Google Keyword Planner):
    {keyword_list}

    ### HEADING PROTOCOL (CRITICAL):
    1. **USE EXACT KEYWORDS AS H2 HEADINGS:** Pick keywords from the list above that match user search intent.
       - ✅ YES: "CMA Eligibility Criteria", "CMA Exam Fees", "CMA vs CPA Comparison"
       - ❌ NO: Generic phrases like "Getting Started", "What You Need to Know"
    
    2. **LOGICAL FLOW:**
       - Start: Definition/Overview (using main keyword)
       - Middle: Technical Details (Eligibility, Fees, Process, Syllabus)
       - End: Outcomes (Salary, Career, Comparison)
    
    3. **TABLE REQUIREMENTS:**
       - Mark sections with data-heavy content as 'needs_table'
       - Table purpose should specify exact data points to include
       - Tables should present ALL relevant data points comprehensively

    4. **CONTENT STRUCTURE:**
       - Each section should have: Paragraph → Table → Paragraph pattern where relevant
       - First paragraph introduces the topic
       - Table presents all data
       - Final paragraph provides context/implications

    RESEARCH CONTEXT:
    {research_summary}

    RETURN ONLY VALID JSON:
    {{
      "article_title": "{focus_keyword} in {target_country}: Complete Guide {current_year}",
      "meta_description": "Comprehensive guide on {focus_keyword} covering eligibility, fees, exam pattern, and career prospects in {target_country}.",
      "headings": [
        {{
          "h2_title": "[Exact Keyword from List]",
          "content_focus": "Specific technical details to cover",
          "needs_table": true,
          "table_purpose": "Present complete breakdown of [specific data]",
          "table_type": "comparison|fees|timeline|requirements|other"
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

def generate_section_content(heading, focus_keyword, target_country, research_context, is_first_section=False, latest_updates=None):
    """Generate direct, simple content for each section"""
    if not grok_key: return None, "Grok required"
    
    system_instruction = f"""
You are a technical writer creating PRACTICAL, CONTEXTUAL, and EXTENSIVE content.
Context Date: {formatted_date} | Target: {target_country}

CRITICAL CONTENT RULES:
1. **BE PRACTICAL & CONTEXTUAL:**
   - Ground every statement in real-world application
   - Explain WHY information matters to the reader
   - Provide context: how does this compare? what does this mean practically?
   - Example: Don't just say "Fee is ₹15,000" - say "The registration fee is ₹15,000, which includes study materials and first exam attempt"

2. **GO DEEP, NOT WIDE:**
   - Cover fewer points but explain each thoroughly
   - Provide complete context for every fact
   - Include implications and practical consequences
   - Example: "Eligibility requires a bachelor's degree in any discipline with 50% marks. This inclusive approach means both commerce and non-commerce students qualify, unlike CA which requires commerce background"

3. **ACCURACY FIRST:**
   - ONLY include information directly from research data
   - Never invent examples, dates, or numbers
   - If research lacks specific data, focus on what IS known
   - Better to cover 3 points deeply than 10 points superficially

4. **DIRECT STYLE:**
   - ❌ Avoid: "According to latest data", "As of {formatted_date}", "It's important to note"
   - ✅ Use: Direct present tense statements with practical context

5. **STRUCTURE:**
   - Paragraphs should be 4-6 sentences (not 2-3)
   - Each paragraph should fully explore one aspect
   - Connect ideas: show relationships between concepts
   - NO HEADING REPETITION in opening line
"""

    if is_first_section and latest_updates:
        updates_text = "\n".join([f"• {update}" for update in latest_updates])
        prompt = f"""
{system_instruction}

TASK: Write COMPREHENSIVE opening section for "{focus_keyword}"

LATEST UPDATES (include these at the very top as bullet points):
{updates_text}

Then write 3-4 SUBSTANTIAL paragraphs that:
1. **Define & Contextualize**: What exactly is {focus_keyword}? Who issues/governs it? Why does it matter?
2. **Target Audience**: Who should pursue this? What specific benefits do they gain?
3. **Key Requirements Overview**: Briefly mention main prerequisites (detailed section will follow)
4. **Current Relevance**: Why is this particularly important in {target_country} right now?

DEPTH REQUIREMENTS:
- Each paragraph should be 4-6 sentences
- Include specific facts from research (numbers, institutions, timeframes)
- Provide practical context for every major point
- Connect concepts: show relationships and implications

RESEARCH DATA:
{research_context[:2500]}

Write extensively but stay focused. Make every sentence count.
"""
    else:
        table_hint = ""
        if heading.get('needs_table'):
            table_hint = f"\n\nIMPORTANT: A comprehensive data table will follow this content showing {heading.get('table_purpose', 'detailed information')}. Your content should INTRODUCE and CONTEXTUALIZE the data - explain what the table will show and WHY it matters. After the table concept, provide IMPLICATIONS and PRACTICAL ADVICE."
        
        prompt = f"""
{system_instruction}

TASK: Write EXTENSIVE, PRACTICAL content for: "{heading['h2_title']}"
Focus Area: {heading.get('content_focus', 'Technical details')}
{table_hint}

STRUCTURE YOUR RESPONSE:
1. **Introduction (1 paragraph)**: Set context - why does this aspect matter? What will this section cover?

2. **Core Content (2-3 paragraphs)**: Deep dive into specifics
   - Include ALL relevant details from research
   - Provide practical context for each point
   - Explain implications and consequences
   - Show relationships between different aspects

3. **Practical Guidance (1 paragraph)**: If applicable, provide actionable insights
   - What should readers specifically do or know?
   - Common mistakes to avoid
   - Best practices or recommendations

CONTENT DEPTH REQUIREMENTS:
- If discussing FEES: Break down every component, explain what each covers, mention payment options, discuss refund policies
- If discussing ELIGIBILITY: List every requirement with exact specs, explain WHY each requirement exists, discuss exceptions or alternatives
- If discussing PROCESS: Detail every step with specific timelines, explain what happens at each stage, mention common delays or issues
- If discussing SYLLABUS: Cover topics with weightage, explain difficulty levels, mention how topics connect
- If discussing CAREER/SALARY: Provide specific ranges, factors affecting compensation, career progression paths, market demand

CRITICAL:
- Write 3-4 substantial paragraphs (4-6 sentences each)
- Every fact must be grounded in the research data provided
- Provide context and practical implications for all data points
- NO generic advice - be specific to {focus_keyword} in {target_country}

RESEARCH DATA:
{research_context[:2500]}
"""
    
    messages = [{"role": "user", "content": prompt}]
    content, error = call_grok(messages, max_tokens=1200, temperature=0.3)
    return content, error

def generate_intelligent_table(heading, target_country, research_context):
    """Generate well-structured table with LLM deciding format"""
    if not grok_key or not heading.get('needs_table'): return None, "No table needed"
    
    table_type = heading.get('table_type', 'general')
    
    prompt = f"""
TASK: Create a COMPREHENSIVE, PRECISE data table for "{heading['h2_title']}"
Context: {target_country} | Date: {formatted_date}
Table Type: {table_type}

RESEARCH DATA:
{research_context[:3000]}

CRITICAL TABLE GENERATION RULES:

1. **PRECISION & ACCURACY FIRST:**
   - ONLY include data that is EXPLICITLY mentioned in the research data
   - NEVER invent or assume data points
   - If research lacks specific information, DO NOT include that row/column
   - Every number, date, amount must be traceable to research data
   - Better to have 3 accurate rows than 10 rows with guessed data

2. **BE EXTENSIVE WITHIN BOUNDS:**
   - Include EVERY data point available in research
   - Break down complex information into granular rows
   - Example: Don't just say "Exam Fee: ₹10,000" - break it into "Part 1 Fee", "Part 2 Fee", "Retake Fee" if research provides this
   - Aim for comprehensive coverage but ONLY from actual research data

3. **CONTEXTUAL & PRACTICAL:**
   - Add a "Notes/Context" column to explain what data means practically
   - Include validity periods, conditions, prerequisites
   - Show relationships: "Required before Part 2" or "One-time only"
   - Make table actionable - readers should understand HOW to use this information

4. **CHOOSE APPROPRIATE FORMAT BASED ON CONTENT:**
   
   For FEES/COSTS:
   - Columns: [Fee Component, Amount ({target_country} Currency), Validity/When Payable, Refund Policy/Notes]
   - Break down into: Application, Entrance, Each Exam Part, Membership, Annual, Re-examination
   
   For TIMELINE/DEADLINES:
   - Columns: [Phase/Event, Date/Timeline, Duration, Prerequisites/Next Steps]
   - Include: Registration windows, exam dates, result dates, validity periods
   
   For COMPARISONS:
   - Columns: [Aspect/Feature, Option 1, Option 2, Key Difference/Recommendation]
   - Compare meaningfully: qualifications, career paths, institutions
   
   For REQUIREMENTS/ELIGIBILITY:
   - Columns: [Requirement Type, Specific Criteria, Mandatory/Optional, Acceptable Alternatives]
   - Break by: Education, Experience, Other qualifications, Documents
   
   For SYLLABUS/TOPICS:
   - Columns: [Section/Paper, Topics Covered, Weightage/Marks, Difficulty Level/Focus Areas]
   - Detail each subject/paper separately

5. **COMPLETENESS CHECK:**
   - For fees: Have you included ALL fee components mentioned in research?
   - For eligibility: Have you listed ALL criteria from research?
   - For process: Have you covered ALL steps mentioned?
   - Minimum 5 rows ONLY if research data supports it
   - If research only provides 3 data points, create 3 excellent rows

6. **FORMATTING:**
   - Use clear, specific headers
   - Include units in headers (₹, $, %, months, years)
   - Keep cells concise but complete
   - Use "N/A" sparingly - only when truly not applicable
   - Add footer note for data source date or important caveats

7. **RECENCY & RELEVANCE:**
   - Mark outdated information clearly
   - If deadline has passed relative to {formatted_date}, note "Closed" or show next occurrence
   - Prioritize current/upcoming information over historical

QUALITY OVER QUANTITY:
A table with 4 accurate, well-contextualized rows is FAR BETTER than a table with 8 rows containing vague or invented data.

Return ONLY valid JSON:
{{
  "table_title": "Specific, Descriptive Title (e.g., 'CMA Exam Fee Breakdown 2025')",
  "headers": ["Column 1", "Column 2", "Column 3", "Column 4"],
  "rows": [
      ["Precise Data 1A", "Precise Data 1B", "Context 1C", "Notes 1D"],
      ["Precise Data 2A", "Precise Data 2B", "Context 2C", "Notes 2D"]
  ],
  "footer_note": "Source date, important caveat, or clarification about the data"
}}
"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=1500, temperature=0.2)
    
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
    prompt = f"""Generate 8-10 FAQs for: "{focus_keyword}"
    Target: {target_country} | Date: {formatted_date}
    
    PAA Keywords to address:
    {paa_text}
    
    RESEARCH: {research_context[:2000]}
    
    RULES:
    - Answer with specific facts and numbers
    - Keep answers direct and concise (2-3 sentences)
    - No filler phrases
    
    Return ONLY valid JSON:
    {{"faqs": [{{"question": "Question?", "answer": "Direct answer with facts"}}]}}"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=3000, temperature=0.5)
    if error: return None, error
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except:
        return None, "Parse error"

def export_to_html(article_title, meta_description, sections, faqs, latest_updates):
    html = []
    html.append(f'<h1>{article_title}</h1>')
    html.append(f'<p><em>{meta_description}</em></p>')
    
    # Add latest updates if available
    if latest_updates:
        html.append('<div style="background-color: #fff3cd; padding: 15px; margin: 20px 0; border-left: 4px solid #ffc107;">')
        html.append('<h3 style="margin-top: 0;">Latest Updates:</h3>')
        html.append('<ul style="margin-bottom: 0;">')
        for update in latest_updates:
            html.append(f'<li>{update}</li>')
        html.append('</ul>')
        html.append('</div>')
    
    html.append('')
    
    for section in sections:
        html.append(f'<h2>{section["heading"]["h2_title"]}</h2>')
        
        if section.get('content'):
            content = section['content']
            # Remove any leading bullet points from content (updates handled separately)
            content = re.sub(r'^[•\-\*]\s+.*\n', '', content, flags=re.MULTILINE)
            
            blocks = content.split('\n\n')
            for block in blocks:
                block = block.strip()
                if not block: continue
                
                # Format bold text
                block = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', block)
                
                # Check if it's a list
                if block.strip().startswith('- ') or block.strip().startswith('• '):
                    html.append('<ul>')
                    lines = block.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith(('- ', '• ')):
                            html.append(f'  <li>{line[2:]}</li>')
                    html.append('</ul>')
                else:
                    html.append(f'<p>{block}</p>')
        
        if section.get('table'):
            table = section['table']
            html.append(f'<h3>{table.get("table_title", "Data Table")}</h3>')
            html.append('<table border="1" style="border-collapse: collapse; width: 100%; margin: 20px 0;">')
            
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            
            if headers:
                html.append('  <thead>')
                html.append('    <tr>')
                for header in headers:
                    html.append(f'      <th style="padding: 12px; background-color: #2c3e50; color: white; text-align: left;">{header}</th>')
                html.append('    </tr>')
                html.append('  </thead>')
            
            if rows:
                html.append('  <tbody>')
                for row in rows:
                    html.append('    <tr>')
                    for cell in row:
                        html.append(f'      <td style="padding: 10px; border: 1px solid #ddd;">{cell}</td>')
                    html.append('    </tr>')
                html.append('  </tbody>')
            
            html.append('</table>')
            
            if table.get('footer_note'):
                html.append(f'<p style="font-size: 0.9em; color: #666; font-style: italic;">{table["footer_note"]}</p>')
        
        html.append('')
    
    if faqs:
        html.append('<h2>Frequently Asked Questions (FAQs)</h2>')
        for faq in faqs:
            html.append(f'<h3>{faq["question"]}</h3>')
            html.append(f'<p>{faq["answer"]}</p>')
    
    return '\n'.join(html)

# --- MAIN UI LAYOUT ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Research", "2. Settings & Keywords", "3. Research Data", "4. Outline", "5. Content"])

with tab1:
    st.header("Research Phase")
    st.info(f"Current Context Date: **{formatted_date}**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Topic:", placeholder="e.g., CMA certification")
        if topic: st.session_state.main_topic = topic
    with col2:
        mode = st.selectbox("Depth", ["AI Overview (simple)", "AI Mode (complex)"])
    
    if st.button("Generate Research Queries", type="primary", use_container_width=True):
        if not topic.strip() or not gemini_key:
            st.error("Please enter topic and Gemini API Key")
        else:
            with st.spinner("Generating queries..."):
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
                        time.sleep(1)  # Rate limiting
                    st.success("Research Complete!")
                    st.rerun()

with tab2:
    st.header("Target Settings & Keywords")
    
    col1, col2 = st.columns(2)
    with col1:
        focus_keyword = st.text_input("Focus Keyword *", value=st.session_state.main_topic, 
                                     placeholder="e.g., CMA certification")
        if focus_keyword: st.session_state.focus_keyword = focus_keyword
    with col2:
        target_country = st.selectbox("Target Audience/Country *", 
                                     ["India", "United States", "United Kingdom", "Canada", "Global"])
        st.session_state.target_country = target_country
    
    st.markdown("---")
    st.subheader("Upload Keywords")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Google Keyword Planner Data** (For H2 Headings)")
        keyword_planner_file = st.file_uploader("Upload Google Keyword Planner CSV", type=['csv'], key='kp')
        if keyword_planner_file:
            try:
                df = pd.read_csv(keyword_planner_file)
                keywords = parse_keyword_planner_csv(df)
                st.session_state.keyword_planner_data = keywords
                st.success(f"✓ Loaded {len(keywords)} keywords for headings")
                
                # Show preview
                with st.expander("Preview Keywords"):
                    preview_df = pd.DataFrame(keywords[:20])
                    st.dataframe(preview_df, hide_index=True)
            except Exception as e:
                st.error(f"Error parsing file: {e}")
    
    with col2:
        st.markdown("**People Also Ask (PAA)** (For FAQs)")
        paa_file = st.file_uploader("Upload PAA CSV (Optional)", type=['csv'], key='paa')
        if paa_file:
            try:
                df = pd.read_csv(paa_file)
                st.session_state.paa_keywords = df.iloc[:, 0].dropna().astype(str).tolist()
                st.success(f"✓ Loaded {len(st.session_state.paa_keywords)} PAA keywords")
            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    st.header("Research Data")
    if st.session_state.research_results:
        st.info(f"Total Research Queries: {len(st.session_state.research_results)}")
        for qid, data in st.session_state.research_results.items():
            with st.expander(f"📊 {data['query']}", expanded=False):
                st.write(data['result'])
    else:
        st.info("No research data yet. Complete research in Tab 1.")

with tab4:
    st.header("Content Outline Generation")
    
    if not st.session_state.keyword_planner_data:
        st.warning("⚠️ Please upload Google Keyword Planner data in Tab 2 to generate keyword-driven headings.")
    
    if st.button("Generate Outline", type="primary", disabled=not st.session_state.keyword_planner_data):
        if not st.session_state.research_results:
            st.error("Please complete research first (Tab 1).")
        elif not st.session_state.keyword_planner_data:
            st.error("Please upload keyword data (Tab 2).")
        else:
            with st.spinner("Creating keyword-driven outline..."):
                outline, error = generate_outline_with_keywords(
                    st.session_state.focus_keyword,
                    st.session_state.target_country,
                    st.session_state.keyword_planner_data,
                    st.session_state.research_results
                )
                if outline:
                    st.session_state.content_outline = outline
                    st.success("✓ Outline generated!")
                    st.rerun()
                else:
                    st.error(f"Error: {error}")

    if st.session_state.content_outline:
        outline = st.session_state.content_outline
        st.markdown("### 📋 Content Structure")
        st.subheader(outline['article_title'])
        st.caption(outline['meta_description'])
        
        st.markdown("---")
        for idx, h in enumerate(outline['headings'], 1):
            with st.expander(f"**H2 #{idx}: {h['h2_title']}**", expanded=False):
                st.write(f"**Content Focus:** {h['content_focus']}")
                if h.get('needs_table'):
                    st.info(f"📊 **Table Required:** {h.get('table_purpose')}")
                    st.caption(f"Table Type: {h.get('table_type', 'general')}")

with tab5:
    st.header("Final Content Generation")
    
    if not st.session_state.content_outline:
        st.info("Generate an outline in Tab 4 first.")
    else:
        if st.button("🚀 Generate Complete Article", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            
            st.session_state.generated_sections = []
            st.session_state.latest_updates = []
            
            # Get latest updates first
            status.text("Scanning for latest updates...")
            latest_updates = get_latest_news_updates(
                st.session_state.focus_keyword, 
                st.session_state.target_country
            )
            st.session_state.latest_updates = latest_updates
            
            # Prepare research context
            research_context = "\n\n".join([
                f"Q: {d['query']}\nA: {d['result']}" 
                for d in list(st.session_state.research_results.values())[:15]
            ])
            
            total_sections = len(st.session_state.content_outline['headings'])
            
            # Generate each section
            for idx, heading in enumerate(st.session_state.content_outline['headings']):
                status.text(f"Writing section {idx+1}/{total_sections}: {heading['h2_title']}...")
                
                is_first = (idx == 0)
                content, _ = generate_section_content(
                    heading, 
                    st.session_state.focus_keyword, 
                    st.session_state.target_country, 
                    research_context, 
                    is_first_section=is_first, 
                    latest_updates=latest_updates if is_first else None
                )
                
                table = None
                if heading.get('needs_table'):
                    status.text(f"Creating data table for {heading['h2_title']}...")
                    table, _ = generate_intelligent_table(
                        heading, 
                        st.session_state.target_country, 
                        research_context
                    )
                
                st.session_state.generated_sections.append({
                    'heading': heading, 
                    'content': content, 
                    'table': table
                })
                
                progress.progress((idx + 1) / total_sections)
                time.sleep(0.5)
            
            # Generate FAQs if PAA data exists
            if st.session_state.paa_keywords:
                status.text("Generating FAQs...")
                faqs, _ = generate_faqs(
                    st.session_state.focus_keyword, 
                    st.session_state.target_country, 
                    st.session_state.paa_keywords, 
                    research_context
                )
                if faqs: 
                    st.session_state.generated_faqs = faqs.get('faqs', [])
            
            status.success("✅ Article generation complete!")
            time.sleep(1)
            st.rerun()

    # Display generated content
    if st.session_state.generated_sections:
        st.markdown("---")
        st.markdown("## 📄 Article Preview")
        
        # Show latest updates if available
        if st.session_state.latest_updates:
            st.markdown("### 🔔 Latest Updates")
            for update in st.session_state.latest_updates:
                st.markdown(f"• {update}")
            st.markdown("---")
        
        # Show all sections
        for section in st.session_state.generated_sections:
            st.markdown(f"### {section['heading']['h2_title']}")
            
            if section.get('content'):
                st.markdown(section['content'])
            
            if section.get('table'):
                table = section['table']
                st.markdown(f"**{table.get('table_title', 'Data Table')}**")
                
                df = pd.DataFrame(
                    table['rows'], 
                    columns=table['headers']
                )
                st.dataframe(df, hide_index=True, use_container_width=True)
                
                if table.get('footer_note'):
                    st.caption(f"*{table['footer_note']}*")
            
            st.markdown("---")
        
        # Show FAQs
        if st.session_state.generated_faqs:
            st.markdown("### ❓ Frequently Asked Questions")
            for faq in st.session_state.generated_faqs:
                with st.expander(faq['question']):
                    st.write(faq['answer'])

        # Export options
        st.markdown("---")
        st.markdown("### 📥 Export Options")
        
        html_content = export_to_html(
            st.session_state.content_outline['article_title'],
            st.session_state.content_outline['meta_description'],
            st.session_state.generated_sections,
            st.session_state.generated_faqs,
            st.session_state.latest_updates
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📄 Download HTML",
                data=html_content,
                file_name=f"{st.session_state.focus_keyword.replace(' ', '_')}_article.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col2:
            # Create plain text version
            text_content = f"{st.session_state.content_outline['article_title']}\n\n"
            for section in st.session_state.generated_sections:
                text_content += f"{section['heading']['h2_title']}\n\n"
                if section.get('content'):
                    text_content += f"{section['content']}\n\n"
            
            st.download_button(
                "📝 Download Text",
                data=text_content,
                file_name=f"{st.session_state.focus_keyword.replace(' ', '_')}_article.txt",
                mime="text/plain",
                use_container_width=True
            )
