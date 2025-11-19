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
if 'custom_headings' not in st.session_state: st.session_state.custom_headings = []
if 'selected_custom_headings' not in st.session_state: st.session_state.selected_custom_headings = set()

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
            try:
                result = response.json()
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response from API"}
                
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
        try:
            result = response.json()
        except json.JSONDecodeError:
            return None, "Invalid JSON response from Grok"
            
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
        if "\`\`\`json" in json_text:
            json_text = json_text.split("\`\`\`json")[1].split("\`\`\`")[0]
        return json.loads(json_text.strip()), None
    except Exception as e:
        return None, f"JSON Error: {str(e)}"

def parse_keyword_planner_csv(file_path):
    """Parse Google Keyword Planner CSV with multiple encoding support"""
    keywords = []
    df = None
    
    # Try different encodings commonly used by Google Keyword Planner
    encodings = ['utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break  # Success, exit loop
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            continue
    
    if df is None:
        # Last resort: try with error handling
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        except:
            try:
                df = pd.read_csv(file_path, encoding='latin-1', errors='ignore')
            except:
                return None, "Unable to read file with any encoding"
    
    # Common column names in Google Keyword Planner exports
    keyword_col = None
    search_vol_col = None
    
    # Look for keyword column (case-insensitive)
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if keyword_col is None and any(term in col_lower for term in ['keyword', 'query', 'search term']):
            keyword_col = col
        elif search_vol_col is None and any(term in col_lower for term in ['avg', 'average', 'monthly', 'searches', 'volume']):
            search_vol_col = col
    
    if not keyword_col:
        # If no keyword column found, use first column
        keyword_col = df.columns[0]
    
    if keyword_col:
        for idx, row in df.iterrows():
            try:
                kw = str(row[keyword_col]).strip()
                if kw and kw.lower() not in ['nan', 'none', ''] and len(kw) > 2:
                    # Try to get search volume
                    search_vol = 0
                    if search_vol_col and search_vol_col in row:
                        try:
                            vol_str = str(row[search_vol_col]).replace(',', '').replace('-', '0')
                            search_vol = int(float(vol_str)) if vol_str.replace('.','').isdigit() else 0
                        except:
                            search_vol = 0
                    
                    keywords.append({
                        'keyword': kw,
                        'search_volume': search_vol,
                        'suitable_for_heading': True
                    })
            except:
                continue
    
    return keywords, None

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

def generate_outline_with_keywords(focus_keyword, target_country, keyword_data, research_data, custom_headings):
    """Generate outline using researched queries as H2 headings with custom structure control"""
    if not model: return None, "Gemini not configured"
    
    research_summary = "\n".join([f"- {data['query']}: {data['result'][:150]}..." for data in list(research_data.values())[:15]])
    
    # Extract researched query topics
    researched_queries = [data['query'] for data in research_data.values()]
    
    # Build custom headings instructions
    custom_instructions = ""
    if custom_headings:
        custom_instructions = "\n### CUSTOM HEADINGS WITH CONTENT CONTROL:\n"
        for ch in custom_headings:
            custom_instructions += f"\n- H2: \"{ch['query']}\"\n"
            custom_instructions += f"  Content Type: {ch.get('content_type', 'Paragraph Only')}\n"
            if ch.get('table_instruction'):
                custom_instructions += f"  Table Requirement: {ch['table_instruction']}\n"
    
    researched_queries_text = "\n".join([f"- {q}" for q in researched_queries])
    
    # Prepare keyword list as backup
    keyword_list = "\n".join([f"- {kw['keyword']}" for kw in keyword_data[:50]]) if keyword_data else "No keyword data"
    
    prompt = f"""
    ROLE: SEO Content Architect
    TASK: Create content outline for "{focus_keyword}"
    TARGET: {target_country} | DATE: {formatted_date}

    ### RESEARCHED QUERIES (use as H2 headings):
    {researched_queries_text}
    {custom_instructions}

    ### CONTENT STRUCTURE RULES (MANDATORY FOR EVERY HEADING):
    
    1. **EVERY SECTION MUST HAVE:**
       - ONE paragraph (4-6 sentences, readable length 12-20 words each)
       - THEN: Table OR Bullet points (never just paragraph alone)
       - NEVER: Multiple long paragraphs without structured data
    
    2. **TABLE vs BULLETS Decision:**
       - If heading has CUSTOM table instruction → MUST have table
       - If data is quantitative (fees, marks, dates) → Table
       - If data is qualitative (steps, features, tips) → Bullet points
       - If data is mixed → Table preferred
    
    3. **TABLE FOCUS:**
       - Each table strictly on its heading topic only
       - Fee table = ONLY fees, no scholarships
       - Course table = ONLY courses, no eligibility
    
    4. **HEADING ORDER:**
       - Use logical flow: Overview → Details → Outcomes
       - Keep researched queries in sensible order
       - Custom headings should be placed where they fit best

    RESEARCH CONTEXT:
    {research_summary}

    RETURN ONLY JSON:
    {{
      "article_title": "{focus_keyword}: Complete Guide {current_year}",
      "meta_description": "Detailed guide on {focus_keyword} with comprehensive data tables and structured information.",
      "headings": [
        {{
          "h2_title": "[Exact researched query or custom heading]",
          "content_focus": "What to cover in the paragraph",
          "needs_table": true,
          "table_purpose": "Exact data to show in table",
          "table_type": "fees|courses|eligibility|timeline|comparison|syllabus",
          "needs_bullets": false,
          "custom_table_instruction": "[Copy from custom heading if provided]"
        }}
      ]
    }}
    
    CRITICAL: EVERY heading must have needs_table=true OR needs_bullets=true. Never both false.
    """
    
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        if "\`\`\`json" in json_text:
            json_text = json_text.split("\`\`\`json")[1].split("\`\`\`")[0]
        return json.loads(json_text.strip()), None
    except Exception as e:
        return None, str(e)

def generate_section_content(heading, focus_keyword, target_country, research_context, is_first_section=False, latest_updates=None):
    """Generate direct, simple content - ALWAYS 1 paragraph + (table OR bullets)"""
    if not grok_key: return None, "Grok required"
    
    system_instruction = f"""
You are a technical writer creating STRUCTURED, SCANNABLE content.
Context Date: {formatted_date} | Target: {target_country}

MANDATORY STRUCTURE FOR EVERY SECTION:
✓ ONE introductory paragraph (4-6 sentences, 12-20 words each)
✓ THEN: Table OR Bullet points (specified in instructions)
✗ NEVER: Multiple paragraphs without structured data
✗ NEVER: Long walls of text

WRITING RULES:

1. **PARAGRAPH (Always exactly ONE):**
   - Introduce the heading topic
   - Set context for what follows (table or bullets)
   - 4-6 sentences, readable length (12-20 words per sentence, flexible)
   - Direct facts only

2. **ONLY EXPLICIT DATA:**
   - Never invent "varies", "usually separate", "typically"
   - If research doesn't state it, don't mention it
   - Example: If no refund policy mentioned → skip it entirely

3. **SENTENCE STYLE:**
   - Direct statements in present tense
   - No connectors: "Additionally", "Furthermore", "Moreover"
   - No fluff: "It's important to note", "Understanding this is crucial"

4. **NO HEADING REPETITION:** Don't start with the heading text
"""

    if is_first_section and latest_updates:
        updates_text = "\n".join([f"• {update}" for update in latest_updates])
        prompt = f"""
{system_instruction}

TASK: Write opening content for "{focus_keyword}"

FORMAT:
1. Latest Updates (bullet points at top)
2. ONE introductory paragraph (4-6 sentences)

LATEST UPDATES:
{updates_text}

Then write ONE paragraph covering:
- Brief definition of {focus_keyword}
- Who governs/issues it
- Main purpose
- Target audience

CRITICAL:
- Just ONE paragraph after updates
- Each sentence: 12-20 words (flexible)
- Only facts from research below

RESEARCH DATA:
{research_context[:2500]}
"""
    else:
        # Determine if bullets needed
        needs_bullets = heading.get('needs_bullets', False)
        has_custom_table = heading.get('custom_table_instruction', '')
        
        structure_instruction = ""
        if needs_bullets:
            structure_instruction = """
STRUCTURE REQUIRED:
1. ONE paragraph (4-6 sentences) introducing the topic
2. THEN: Bullet points with key facts

Bullet points should be:
- Short (under 20 words each)
- Specific and factual
- 5-8 bullets covering main points from research
"""
        else:
            structure_instruction = f"""
STRUCTURE REQUIRED:
1. ONE paragraph (4-6 sentences) introducing the topic
2. THEN: A table will follow (don't list table data in paragraph)

Paragraph should explain:
- What this section covers
- Why it matters
- What the table will show
{f"- Table will show: {has_custom_table}" if has_custom_table else ""}
"""
        
        prompt = f"""
{system_instruction}

TASK: Write content for: "{heading['h2_title']}"
Focus: {heading.get('content_focus', 'Technical details')}

{structure_instruction}

STRICT RULES:
- Write ONLY ONE paragraph (4-6 sentences)
{"- Then write bullet points with factual data" if needs_bullets else "- Table will follow your paragraph"}
- Only use facts explicitly in research
- No invented data, no "varies", no "typically"
- Sentences: 12-20 words (aim for readability)

RESEARCH DATA:
{research_context[:2500]}
"""
    
    messages = [{"role": "user", "content": prompt}]
    content, error = call_grok(messages, max_tokens=900, temperature=0.3)
    return content, error

def generate_intelligent_table(heading, target_country, research_context):
    """Generate well-structured table with LLM deciding format"""
    if not grok_key or not heading.get('needs_table'): return None, "No table needed"
    
    table_type = heading.get('table_type', 'general')
    custom_instruction = heading.get('custom_table_instruction', '')
    
    # Build specific instructions based on custom or standard requirements
    specific_requirement = ""
    if custom_instruction:
        specific_requirement = f"""
### USER'S SPECIFIC TABLE REQUIREMENT:
{custom_instruction}

YOU MUST create a table that exactly matches this requirement. This is a custom heading added by the user with explicit instructions.
"""
    
    prompt = f"""
TASK: Create a FOCUSED data table STRICTLY for "{heading['h2_title']}"
Context: {target_country} | Date: {formatted_date}
Table Type: {table_type}

{specific_requirement}

RESEARCH DATA:
{research_context[:3500]}

CRITICAL TABLE GENERATION RULES:

1. **STRICT TOPIC FOCUS:**
   - Table must be 100% relevant to "{heading['h2_title']}" ONLY
   - ❌ If heading is "B.Tech Courses and Fees", show ALL courses with their fees
   - ❌ If heading is "Scholarships", show ONLY scholarships
   - ✅ Stay laser-focused on the exact heading topic
   - 3-5 focused rows minimum, up to 12 rows if data available

2. **ONLY EXPLICIT DATA - NO INVENTION:**
   - ONLY include data EXPLICITLY stated in research
   - ❌ NEVER: "Varies", "Not specified", "Typically", "Usually"
   - ✅ If research says "CSE - ₹2,50,000", use that
   - ✅ If research doesn't have data for a course, don't add that row
   - Empty cells = remove that row entirely

3. **SCAN FOR OFFICIAL DATA:**
   - Look for data from official university websites in research
   - Use most recent figures mentioned
   - Cross-reference competitor data if available
   - Prioritize {formatted_date} or latest academic year data

4. **FORMAT BASED ON HEADING:**
   
   For "Courses and Fees" headings:
   - Columns: [Course/Specialization, Annual Fee, Duration, Total Fee]
   - List ALL available courses from research
   
   For "Fee Structure" headings:
   - Columns: [Fee Component, Amount, When Payable, Notes]
   - Break down: Tuition, Exam, Lab, Library, etc.
   
   For "Eligibility" headings:
   - Columns: [Requirement, Criteria, Mandatory]
   - Education, Marks, Exam scores
   
   For "Scholarships" headings:
   - Columns: [Scholarship Name, Criteria, Amount/%, Validity]
   
   For "Timeline" headings:
   - Columns: [Phase, Date/Period, Action Required]

5. **COMPREHENSIVE BUT ACCURATE:**
   - Include ALL relevant data points from research
   - If research lists 8 courses, table should have 8 rows
   - If research lists 3 fee components, table has 3 rows
   - Don't pad with generic rows

6. **NO VAGUE LANGUAGE:**
   - ❌ "Varies", "Depends", "Typically", "Usually"
   - ✅ "₹2,50,000", "Bachelor's with 50%", "June 2025"

7. **CLEAR FORMATTING:**
   - Headers with units (₹, %, years)
   - Concise cells (under 15 words)
   - Footer note only if critical

REMEMBER: If this is a custom heading with specific instructions, follow those exactly.

Return ONLY valid JSON:
{{
  "table_title": "{heading['h2_title']} - {current_year}",
  "headers": ["Column 1", "Column 2", "Column 3"],
  "rows": [
      ["Data 1A", "Data 1B", "Data 1C"],
      ["Data 2A", "Data 2B", "Data 2C"]
  ],
  "footer_note": "Only if critical context needed"
}}
"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=1800, temperature=0.2)
    
    if error: return None, error
    try:
        if "\`\`\`json" in response:
            response = response.split("\`\`\`json")[1].split("\`\`\`")[0]
        return json.loads(response.strip()), None
    except:
        return None, "Parse error"

def generate_faqs(focus_keyword, target_country, paa_keywords, research_context, researched_queries):
    if not grok_key: return None, "Grok required"
    
    # Combine researched queries and PAA keywords
    researched_text = "\n".join([f"- {q}" for q in researched_queries[:20]]) if researched_queries else ""
    paa_text = "\n".join([f"- {kw}" for kw in paa_keywords[:20]]) if paa_keywords else ""
    
    combined_questions = f"""
RESEARCHED QUERIES (PRIORITY - use these first):
{researched_text}

PAA Keywords (additional context):
{paa_text}
"""
    
    prompt = f"""Generate 8-10 FAQs for: "{focus_keyword}"
    Target: {target_country} | Date: {formatted_date}
    
    QUESTIONS TO ADDRESS:
    {combined_questions}
    
    RESEARCH DATA:
    {research_context[:2500]}
    
    RULES:
    - Prioritize questions from researched queries (convert them to FAQ format)
    - Answer with specific facts and numbers from research
    - Keep answers direct and concise (2-3 short sentences, max 20 words each)
    - No filler phrases
    - If a researched query is "What are CMA fees?", FAQ should be similar or exactly that
    
    Return ONLY valid JSON:
    {{"faqs": [{{"question": "Question?", "answer": "Direct answer with facts"}}]}}"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=3000, temperature=0.5)
    if error: return None, error
    try:
        if "\`\`\`json" in response:
            response = response.split("\`\`\`json")[1].split("\`\`\`")[0]
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
        html.append('<strong>Latest Updates:</strong>')
        html.append('<ul style="margin: 10px 0 0 0;">')
        for update in latest_updates:
            html.append(f'<li>{update}</li>')
        html.append('</ul>')
        html.append('</div>')
    
    html.append('')
    
    for section in sections:
        # Only add H2 - no other headings
        html.append(f'<h2>{section["heading"]["h2_title"]}</h2>')
        
        if section.get('content'):
            content = section['content']
            
            # Remove any subheadings the LLM might have added
            content = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'\*\*Introduction\*\*|\*\*Core Content\*\*|\*\*Practical Guidance\*\*', '', content, flags=re.IGNORECASE)
            
            # Remove leading bullet points from content (updates handled separately)
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
            html.append(f'<h3 style="font-size: 1.1em; margin-top: 20px;">{table.get("table_title", "Data Table")}</h3>')
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
                html.append(f'<p style="font-size: 0.9em; color: #666; font-style: italic; margin-top: 5px;">{table["footer_note"]}</p>')
        
        html.append('')
    
    if faqs:
        html.append('<h2>Frequently Asked Questions</h2>')
        for faq in faqs:
            html.append(f'<h3 style="font-size: 1.05em; margin-top: 15px;">{faq["question"]}</h3>')
            html.append(f'<p>{faq["answer"]}</p>')
    
    return '\n'.join(html)

# --- MAIN UI LAYOUT ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Research", "2. Settings & Keywords", "3. Research Data", "4. Outline", "5. Content"])

# Debug panel in sidebar
with st.sidebar:
    with st.expander("🔍 Debug: Session State", expanded=False):
        st.write(f"Research Results: {len(st.session_state.research_results)}")
        st.write(f"Custom Headings: {len(st.session_state.custom_headings)}")
        st.write(f"Selected Queries: {len(st.session_state.selected_queries)}")
        st.write(f"Content Outline: {bool(st.session_state.content_outline)}")
        st.write(f"Generated Sections: {len(st.session_state.generated_sections)}")


with tab1:
    st.header("Research Phase")
    st.info(f"Current Context Date: **{formatted_date}**")
    
    st.subheader("AI-Generated Research Queries")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Topic:", placeholder="e.g., Adamas University BTech Fees")
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
    
    # Combined queries display (AI + Custom together)
    if st.session_state.fanout_results and 'queries' in st.session_state.fanout_results:
        st.markdown("---")
        
        # Add custom heading section at top
        with st.expander("➕ Add Your Own Heading/Query", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                custom_heading_input = st.text_input("Heading/Query:", placeholder="e.g., All B.Tech Courses and Fees", key="custom_input")
            with col2:
                content_type = st.selectbox("Content Type:", ["Table Required", "Bullet Points", "Paragraph Only"], key="custom_type")
            
            table_instruction = ""
            if content_type == "Table Required":
                table_instruction = st.text_area(
                    "What should the table show?", 
                    placeholder="e.g., All B.Tech specializations with annual fees, duration, eligibility",
                    height=70,
                    key="custom_table_instr"
                )
            
            if st.button("➕ Add to Research Queries", use_container_width=True, type="secondary"):
                if custom_heading_input.strip():
                    heading_id = f"custom_{len(st.session_state.custom_headings)}"
                    st.session_state.custom_headings.append({
                        'id': heading_id,
                        'query': custom_heading_input.strip(),
                        'category': 'Custom',
                        'content_type': content_type,
                        'table_instruction': table_instruction if content_type == "Table Required" else ""
                    })
                    st.success(f"✓ Added: {custom_heading_input.strip()}")
                    st.rerun()
        
        queries = st.session_state.fanout_results['queries']
        
        # Filter controls
        categories = sorted(list(set(q.get('category', 'Unknown') for q in queries)))
        if st.session_state.custom_headings:
            categories = ['Custom'] + categories
        
        selected_cats = st.multiselect("Filter by Category:", categories, default=categories)
        filtered = [q for q in queries if q.get('category', 'Unknown') in selected_cats]
        
        # Add custom headings to filtered list if Custom is selected
        if 'Custom' in selected_cats:
            for ch in st.session_state.custom_headings:
                filtered.insert(0, ch)  # Add at top
        
        # Batch Selection
        all_ids = {f"q_{queries.index(q)}" if q not in st.session_state.custom_headings else q['id'] for q in filtered}
        combined_selected = st.session_state.selected_queries.union(st.session_state.selected_custom_headings)
        all_selected = all(qid in combined_selected for qid in all_ids) if all_ids else False
        
        if st.checkbox("Select All Visible", value=all_selected):
            for qid in all_ids:
                if qid.startswith('custom_'):
                    st.session_state.selected_custom_headings.add(qid)
                else:
                    st.session_state.selected_queries.add(qid)
        else:
            for qid in all_ids:
                if qid.startswith('custom_'):
                    st.session_state.selected_custom_headings.discard(qid)
                else:
                    st.session_state.selected_queries.discard(qid)
        
        # Display all queries together
        for item in filtered:
            # Check if custom or AI-generated
            if item in st.session_state.custom_headings:
                # Custom heading
                qid = item['id']
                col1, col2, col3 = st.columns([4, 1, 0.5])
                with col1:
                    is_selected = qid in st.session_state.selected_custom_headings
                    label = f"**{item['query']}** 🏷️ Custom"
                    if item.get('content_type'):
                        label += f" [{item['content_type']}]"
                    if st.checkbox(label, value=is_selected, key=f"cb_{qid}"):
                        st.session_state.selected_custom_headings.add(qid)
                    else:
                        st.session_state.selected_custom_headings.discard(qid)
                    
                    if item.get('table_instruction'):
                        st.caption(f"📊 {item['table_instruction'][:80]}...")
                
                with col2:
                    if qid in st.session_state.research_results:
                        st.success("✓ Done")
                
                with col3:
                    if st.button("🗑️", key=f"del_{qid}", help="Delete"):
                        st.session_state.custom_headings = [h for h in st.session_state.custom_headings if h['id'] != qid]
                        st.session_state.selected_custom_headings.discard(qid)
                        st.rerun()
            else:
                # AI-generated query
                qid = f"q_{queries.index(item)}"
                col1, col2 = st.columns([4, 1])
                with col1:
                    is_selected = qid in st.session_state.selected_queries
                    if st.checkbox(f"**{item['query']}**", value=is_selected, key=f"cb_{qid}"):
                        st.session_state.selected_queries.add(qid)
                    else:
                        st.session_state.selected_queries.discard(qid)
                with col2:
                    if qid in st.session_state.research_results:
                        st.success("✓ Done")
        
        # Combined research button
        st.markdown("---")
        all_selected = st.session_state.selected_queries.union(st.session_state.selected_custom_headings)
        if all_selected and perplexity_key:
            unresearched = [qid for qid in all_selected if qid not in st.session_state.research_results]
            if unresearched:
                if st.button(f"🔍 Research {len(unresearched)} Selected Queries", type="secondary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status = st.empty()
                    for i, qid in enumerate(unresearched):
                        q_text = None # Initialize q_text to avoid UnboundLocalError
                        
                        # Handle both AI-generated and custom queries
                        if qid.startswith('custom_'):
                            custom_h = next((h for h in st.session_state.custom_headings if h['id'] == qid), None)
                            if custom_h:
                                q_text = custom_h['query']
                                if custom_h.get('table_instruction'):
                                    q_text = f"{q_text}. Specifically: {custom_h['table_instruction']}"
                        else:
                            try: # Add try/except for safe index access
                                q_idx = int(qid.split('_')[1])
                                if 0 <= q_idx < len(queries):
                                    q_text = queries[q_idx]['query']
                            except (ValueError, IndexError):
                                pass
                        
                        if q_text: # Only proceed if we have a valid query
                            status.text(f"Researching: {q_text[:80]}...")
                            
                            res = call_perplexity(q_text)
                            if res and 'choices' in res:
                                st.session_state.research_results[qid] = {
                                    'query': q_text,
                                    'result': res['choices'][0]['message']['content']
                                }
                        progress_bar.progress((i + 1) / len(unresearched))
                        time.sleep(1)
                    st.success("Research Complete!")
                    st.rerun()
    elif st.session_state.custom_headings:
        # Show custom headings even if no AI queries generated
        st.markdown("---")
        st.markdown("**Your Custom Queries:**")
        for heading in st.session_state.custom_headings:
            col1, col2, col3 = st.columns([4, 1, 0.5])
            with col1:
                is_selected = heading['id'] in st.session_state.selected_custom_headings
                label = f"**{heading['query']}** [{heading.get('content_type', 'N/A')}]"
                if st.checkbox(label, value=is_selected, key=f"custom_{heading['id']}"):
                    st.session_state.selected_custom_headings.add(heading['id'])
                else:
                    st.session_state.selected_custom_headings.discard(heading['id'])
            with col2:
                if heading['id'] in st.session_state.research_results:
                    st.success("✓ Done")
            with col3:
                if st.button("🗑️", key=f"del_{heading['id']}", help="Delete"):
                    st.session_state.custom_headings = [h for h in st.session_state.custom_headings if h['id'] != heading['id']]
                    st.session_state.selected_custom_headings.discard(heading['id'])
                    st.rerun()
        
        if st.session_state.selected_custom_headings and perplexity_key:
            unresearched = [qid for qid in st.session_state.selected_custom_headings if qid not in st.session_state.research_results]
            if unresearched:
                if st.button(f"🔍 Research {len(unresearched)} Custom Queries", type="secondary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status = st.empty()
                    for i, qid in enumerate(unresearched):
                        custom_h = next((h for h in st.session_state.custom_headings if h['id'] == qid), None)
                        if custom_h:
                            q_text = custom_h['query']
                            if custom_h.get('table_instruction'):
                                q_text = f"{q_text}. Specifically: {custom_h['table_instruction']}"
                            status.text(f"Researching: {q_text[:80]}...")
                            
                            res = call_perplexity(q_text)
                            if res and 'choices' in res:
                                st.session_state.research_results[qid] = {
                                    'query': q_text,
                                    'result': res['choices'][0]['message']['content']
                                }
                            progress_bar.progress((i + 1) / len(unresearched))
                            time.sleep(1)
                    st.success("Research Complete!")
                    st.rerun()

with tab2:
    st.header("Target Settings & Keywords")
    
    # Show research status
    if st.session_state.research_results:
        st.success(f"✅ Research Data Loaded: {len(st.session_state.research_results)} queries researched")
        with st.expander("📊 View Research Summary"):
            for qid, data in list(st.session_state.research_results.items())[:5]:
                st.markdown(f"**{data['query']}**")
                st.caption(data['result'][:100] + "...")
            if len(st.session_state.research_results) > 5:
                st.info(f"... and {len(st.session_state.research_results) - 5} more")
    else:
        st.warning("⚠️ No research data found. Complete research in Tab 1 first.")
    
    st.markdown("---")
    
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
                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(keyword_planner_file.getvalue())
                    tmp_path = tmp_file.name
                
                keywords, error = parse_keyword_planner_csv(tmp_path)
                
                # Clean up temp file
                import os
                os.unlink(tmp_path)
                
                if error:
                    st.error(f"Error parsing file: {error}")
                elif keywords:
                    st.session_state.keyword_planner_data = keywords
                    st.success(f"✓ Loaded {len(keywords)} keywords for headings")
                    
                    # Show preview
                    with st.expander("Preview Keywords"):
                        preview_df = pd.DataFrame(keywords[:20])
                        st.dataframe(preview_df, hide_index=True)
                else:
                    st.warning("No keywords found in file. Please check the file format.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
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
    
    # Debug: Show research data status
    if st.session_state.research_results:
        st.success(f"✅ {len(st.session_state.research_results)} researched queries available")
    else:
        st.error("❌ No research data found!")
    
    has_research = bool(st.session_state.research_results)
    has_keywords = bool(st.session_state.keyword_planner_data)
    
    if not has_research:
        st.warning("⚠️ Please complete research in Tab 1 first.")
    elif not has_keywords:
        st.info("💡 Keyword Planner data not uploaded. Outline will use researched queries as headings.")
    
    if st.button("Generate Outline", type="primary", disabled=not has_research):
        if not st.session_state.research_results:
            st.error("Please complete research first (Tab 1).")
        else:
            with st.spinner("Creating outline from researched queries..."):
                outline, error = generate_outline_with_keywords(
                    st.session_state.focus_keyword,
                    st.session_state.target_country,
                    st.session_state.keyword_planner_data or [],
                    st.session_state.research_results,
                    st.session_state.custom_headings
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
                
                # Show structure
                structure_tags = []
                if h.get('needs_table'):
                    structure_tags.append("📊 Table")
                if h.get('needs_bullets'):
                    structure_tags.append("📝 Bullets")
                structure_tags.append("📄 Paragraph")
                
                st.info(f"Structure: {' + '.join(structure_tags)}")
                
                if h.get('needs_table'):
                    st.write(f"**Table Purpose:** {h.get('table_purpose')}")
                    if h.get('custom_table_instruction'):
                        st.success(f"**Custom Requirement:** {h['custom_table_instruction']}")
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
            
            # Generate FAQs - always generate, prioritizing researched queries
            status.text("Generating FAQs...")
            researched_queries = [data['query'] for data in st.session_state.research_results.values()]
            faqs, _ = generate_faqs(
                st.session_state.focus_keyword, 
                st.session_state.target_country, 
                st.session_state.paa_keywords, 
                research_context,
                researched_queries
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
