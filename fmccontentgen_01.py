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
st.title("SEO Content Generator - Dense Table Protocol")

# API Configuration
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# Initialize session states
if 'fanout_results' not in st.session_state:
    st.session_state.fanout_results = None
if 'research_results' not in st.session_state:
    st.session_state.research_results = {}
if 'competitor_research' not in st.session_state:
    st.session_state.competitor_research = {}
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
            response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result
            return {"error": "Invalid response"}
        except:
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            return {"error": "Failed"}
    return {"error": "Max retries"}

def call_grok(messages, max_tokens=4000, temperature=0.6):
    if not grok_key:
        return None, "Missing API key"
    headers = {"Authorization": f"Bearer {grok_key}", "Content-Type": "application/json"}
    payload = {"messages": messages, "model": "grok-3", "stream": False, "temperature": temperature, "max_tokens": max_tokens}
    try:
        response = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=150)
        response.raise_for_status()
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content'], None
        return None, "Error"
    except Exception as e:
        return None, str(e)

def analyze_competitor_coverage(focus_keyword, target_country):
    """Analyze what competitors are covering for the topic"""
    if not perplexity_key:
        return ""
    
    current_date = datetime.now().strftime("%B %d, %Y")
    
    query = f"""CURRENT DATE: {current_date}

Analyze comprehensive coverage of "{focus_keyword}" for {target_country} students on:
- Shiksha.com
- CollegeDunia.com  
- Careers360.com
- CollegeDekho.com
- AglaSem.com
- Vedantu.com

List ALL sections with specific details:

1. TABLES THEY USE:
   - Eligibility table: what columns? (e.g., Level, Age, Qualification, Experience)
   - Fees table: what breakdown? (e.g., Registration, Exam, Late fee, Membership)
   - Exam pattern: what details? (e.g., Paper, Questions, Marks, Duration, Negative marking)
   - Dates table: what sessions/phases?
   - Comparison table: compared with what alternatives?
   - College/Institute listings: what metrics shown?
   - Salary data: how segmented? (Experience, Location, Role)

2. UNIQUE SECTIONS:
   - What competitor-specific headings they use
   - Special calculators or tools
   - Regional variations covered

3. DATA GRANULARITY:
   - How many tiers/levels they show
   - Whether they show state-wise variations
   - Time-based comparisons (year over year)

Provide specific table structures and column names they use."""
    
    system_prompt = "Analyze competitor websites in forensic detail. List exact table structures, column headers, data segmentation strategies. Be comprehensive and specific about how they organize information."
    
    res = call_perplexity(query, system_prompt=system_prompt, max_retries=3)
    if res and 'choices' in res and len(res['choices']) > 0:
        return res['choices'][0]['message'].get('content', '')
    return ""

def generate_research_queries(topic, mode="AI Overview (simple)"):
    if not model:
        return None, "Gemini not configured"
    min_queries = 18 if mode == "AI Overview (simple)" else 35
    prompt = f"""Generate {min_queries}+ deep research queries for: "{topic}"

MANDATORY QUERY TYPES:

1. COMPETITOR ANALYSIS (3-4 queries):
   - "Exact table structures used by Shiksha.com for {topic}"
   - "Column headers in CollegeDunia {topic} eligibility tables"
   - "Data granularity in Careers360 {topic} fee breakdowns"

2. EXHAUSTIVE CRITERIA (4-5 queries):
   - "Complete eligibility criteria with age limits, qualifications, experience for {topic}"
   - "All fee components including hidden charges for {topic}"
   - "Detailed exam pattern with section-wise marks distribution for {topic}"
   - "Edge cases and special categories in {topic} eligibility"

3. TEMPORAL DATA (3-4 queries):
   - "Current year exam dates and deadlines for {topic}"
   - "Year-over-year changes in {topic} fees and pattern"
   - "Latest policy updates and notifications for {topic}"

4. COMPARATIVE ANALYSIS (3-4 queries):
   - "Tier-wise comparison in {topic}"
   - "{topic} vs alternative certifications detailed comparison"
   - "State-wise or regional variations in {topic}"

5. DEEP SPECIFICS (remaining queries):
   - Technical specifications, syllabus weightage
   - College/institute listings with placement data
   - Salary trends by experience and location
   - Pass percentages and cutoff trends
   - Application process step-by-step with timelines
   - Document requirements and verification process

Return ONLY valid JSON:
{{"queries": [{{"query": "detailed question", "category": "category", "priority": "high/medium/low", "purpose": "specific data need"}}]}}"""
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        return json.loads(json_text.strip()), None
    except Exception as e:
        return None, str(e)

def research_table_data_dense(heading, target_country, focus_keyword):
    """DENSE-TABLE DEEP DIVE research protocol"""
    if not perplexity_key:
        return ""
    
    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year
    
    query = f"""### DENSE-TABLE DEEP DIVE RESEARCH
**CURRENT DATE:** {current_date}
**TOPIC:** {heading['h2_title']} for {focus_keyword} in {target_country}
**TABLE PURPOSE:** {heading.get('table_purpose', 'exhaustive data')}

STRICT REQUIREMENTS:

1. TEMPORAL ANCHOR:
   - Provide {current_year} data ONLY
   - If {current_year} data unavailable, write "No {current_year} Data Available" 
   - Include exact dates: DD MMM YYYY format
   - Flag expired policies/criteria as "EXPIRED as of [date]"

2. BANNED PHRASES (Auto-fail if used):
   ❌ "Generally", "Typically", "Usually", "Standard"
   ❌ "Various factors", "Depends on", "May vary"
   ❌ "Around", "Approximately", "About"
   ❌ "Requirements vary", "Subject to change"

3. DATA DENSITY REQUIREMENTS:
   - Minimum 8-10 distinct data rows
   - Cover ALL tiers/levels/categories
   - Include edge cases and exceptions
   - Show regional variations within {target_country}

4. SPECIFICITY FILTERS:

For ELIGIBILITY topics:
- NOT "Must be graduate" → "Bachelor's degree (10+2+3 pattern) with minimum 50% marks (45% for reserved categories)"
- NOT "Age limit applies" → "Minimum 18 years, Maximum 35 years (40 for OBC, 45 for SC/ST as per Govt. norms dated 15 Jan 2024)"
- NOT "Work experience required" → "2 years post-qualification experience in accounts/finance domain, verified through Form 16 or salary slips"

For FEES topics:
- NOT "Registration fee required" → "Registration Fee: ₹1,500 (Non-refundable), Exam Fee: ₹3,500 per paper, Late Fee: ₹500 (applies after 15 Mar 2025), Reappearance Fee: ₹2,800"
- Include GST breakdowns, payment modes, refund policies

For EXAM PATTERN topics:
- NOT "Multiple choice questions" → "100 MCQs (4 options each), 1 mark per question, Negative marking: -0.25 for wrong answer, Total marks: 100, Duration: 120 minutes"
- Include section-wise breakup, marking scheme details, calculator/chart allowance

For DATES topics:
- NOT "Exam in March" → "Session 1: 15-18 March 2025, Session 2: 22-25 May 2025, Results: Within 45 days of exam, Admit Card: 10 days before exam"

5. MULTI-DIMENSIONAL COVERAGE:
- Break down by: Level/Stage, Category (General/OBC/SC/ST), Location, Experience tier
- Include: Minimum requirements, Preferred criteria, Exceptional cases
- Add: Source references (circular numbers, notification dates, official URLs)

6. COMPARATIVE METRICS:
- Before vs After policy changes
- Different institutes/boards if applicable
- Year-over-year trends
- Tier-wise fee structures

OUTPUT FORMAT:
Provide structured data ready for table conversion:
Row 1: [Column 1 data] | [Column 2 data] | [Column 3 data] | [Column 4 data]
Row 2: [Column 1 data] | [Column 2 data] | [Column 3 data] | [Column 4 data]
...minimum 8 rows

Focus on OBSCURE, TECHNICAL, EXCLUSIONARY details that competitors miss."""
    
    system_prompt = f"You are a forensic data specialist. Provide ONLY specific, verifiable {current_year} data. No generic statements. Every number must be exact. Every date must be precise. Include policy references and edge cases. Prioritize technical details over obvious information."
    
    res = call_perplexity(query, system_prompt=system_prompt, max_retries=3)
    if res and 'choices' in res and len(res['choices']) > 0:
        return res['choices'][0]['message'].get('content', '')
    return ""

def generate_outline_from_research(focus_keyword, target_country, required_tables, research_data, competitor_analysis):
    if not model:
        return None, "Gemini not configured"
    
    current_year = datetime.now().year
    research_summary = "\n".join([f"- {data['query']}: {data['result'][:200]}..." for data in list(research_data.values())[:20]])
    
    tables_context = ""
    if required_tables:
        tables_context = f"\n\nMANDATORY TABLES:\n" + "\n".join([f"- {t}" for t in required_tables])
    
    competitor_context = ""
    if competitor_analysis:
        competitor_context = f"\n\nCOMPETITOR COVERAGE:\n{competitor_analysis[:1800]}"
    
    prompt = f"""Create comprehensive article outline for: "{focus_keyword}" ({current_year})

TARGET: Students in {target_country}
CURRENT YEAR: {current_year}
RESEARCH:
{research_summary}
{tables_context}
{competitor_context}

CRITICAL REQUIREMENT: Each section MUST have a DENSE data table with 8-10+ rows covering the topic exhaustively.

HEADING RULES:
1. Direct, factual headings with year - NO questions, NO suffixes
2. Format: "[Topic] {current_year}" or "[Topic] [Specific Aspect]"
3. Examples:
   ✅ "CMA Eligibility {current_year}"
   ✅ "CMA Fees Breakdown"
   ✅ "CMA vs CPA Comparison"
   ❌ "CMA Eligibility: Everything You Need to Know"
   ❌ "How to Check CMA Fees?"

MANDATORY SECTIONS (adapt to topic):
1. Eligibility Criteria → Table: Level-wise requirements with age, qualification, experience, categories
2. Fees Structure → Table: Component-wise breakdown (Registration, Exam, Late, Membership, GST)
3. Exam Pattern → Table: Paper-wise marks, duration, negative marking, passing criteria
4. Important Dates {current_year} → Table: Session-wise registration, exam, result dates
5. Application Process → Table: Step-by-step with timelines and documents
6. Syllabus Overview → Table: Subject-wise topics with weightage
7. Comparison with Alternatives → Table: Side-by-side comparison (fees, duration, difficulty, value)
8. Exam Centers → Table: State/city-wise centers with codes
9. Pass Percentage Trends → Table: Year-wise statistics
10. Career Prospects → Table: Role-wise salary by experience
11. Top Colleges/Institutes → Table: Institute, location, fees, placements
12. [Additional unique sections from competitor analysis]

CREATE 12-15 UNIQUE HEADINGS.
Each heading MUST specify exact table structure.

Return ONLY valid JSON:
{{
  "article_title": "{focus_keyword} {current_year}: [Simple subtitle]",
  "meta_description": "150-160 chars with {current_year} key facts",
  "headings": [
    {{
      "h2_title": "Direct heading with year if relevant",
      "student_question": "What specific info this provides",
      "key_facts": ["specific fact 1", "specific fact 2", "specific fact 3"],
      "needs_table": true,
      "table_purpose": "Exact data to show",
      "table_type": "comparison/timeline/breakdown/listing/statistics",
      "table_columns": ["Column 1", "Column 2", "Column 3", "Column 4"],
      "min_rows": 8
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
    
    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year
    
    query = f"""CURRENT DATE: {current_date}

Latest updates for {focus_keyword} in {target_country} in {current_year}:

PROVIDE ONLY:
- Exam date notifications (with exact dates DD MMM YYYY)
- Fee changes (with new amounts and effective dates)
- New regulations/policies (with circular numbers and dates)
- Deadline extensions (with old and new dates)
- Pattern changes (with specifics)

EXCLUDE:
- Speculative information
- Unverified news
- Generic announcements

Format: [Update Type] - [Specific Detail] - [Date/Source Reference]"""
    
    system_prompt = f"Provide ONLY verified {current_year} updates with exact dates. Include notification/circular numbers. If no recent updates exist, state 'No recent updates available'."
    
    res = call_perplexity(query, system_prompt=system_prompt, max_retries=3)
    if res and 'choices' in res and len(res['choices']) > 0:
        return res['choices'][0]['message'].get('content', '')[:600]
    return ""

def generate_section_content_dense(heading, focus_keyword, target_country, research_context, table_research, is_first_section=False, latest_alerts=""):
    """Generate content using DENSE-TABLE protocol"""
    if not grok_key:
        return None, "Grok required"
    
    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year
    
    if is_first_section:
        alerts_context = f"\n\nLATEST UPDATES ({current_year}):\n{latest_alerts}" if latest_alerts else ""
        
        prompt = f"""### DENSE-TABLE PROTOCOL: OPENING SECTION
**CURRENT DATE:** {current_date}
**HEADING:** {heading['h2_title']}
**TOPIC:** {focus_keyword}
**COUNTRY:** {target_country}
**RESEARCH:** {research_context[:2500]}
{alerts_context}

STRUCTURE:

**Part 1: Latest Updates (if available)**
Start with: **{current_year} Updates on {focus_keyword}:**
- 2-3 sentences with EXACT dates and specific changes
- Use format: "As of DD MMM YYYY, [specific change]"
- If no updates: skip this part

**Part 2: Technical Deep Dive (220-280 words)**
NO introduction phrases. Start directly with definition.

BANNED WORDS: Generally, Typically, Prestigious, Renowned, Comprehensive, Deep, Various, Standard

INCLUDE:
- Exact definition with technical terms
- Specific structure (e.g., "3 levels: Foundation, Intermediate, Final")
- Precise requirements (e.g., "Bachelor's degree with 50% marks")
- Governing body with full name
- What it covers (list 4-5 specific subjects/areas)
- Who conducts it (institute/board name)
- Duration in months/years
- Recognition scope (national/international/industry-specific)
- Prerequisites with specifics

WRITE SHORT SENTENCES (10-14 words max).
End naturally: "The table below shows [what table covers]."

Write content now. Direct facts only."""
    else:
        prompt = f"""### DENSE-TABLE PROTOCOL: SECTION CONTENT
**CURRENT DATE:** {current_date}
**HEADING:** {heading['h2_title']}
**TOPIC:** {focus_keyword}
**COUNTRY:** {target_country}
**KEY FACTS:** {', '.join(heading.get('key_facts', []))}
**RESEARCH:** {research_context[:1200]}
**TABLE RESEARCH:** {table_research[:1500]}

CRITICAL RULES:
1. NO heading repetition. Start directly with content.
2. Write ONLY about this specific heading topic
3. 120-180 words
4. SHORT sentences (10-14 words max)
5. NO generic eligibility like "must be 18" - focus on SPECIFIC, TECHNICAL criteria
6. End with: "The table below provides [specific data type] in detail."

BANNED WORDS: Generally, Typically, Usually, Standard, Various, Approximately, Around

SPECIFICITY REQUIREMENT:
- NOT "Multiple levels exist" → "Foundation: Class 12 pass; Intermediate: Graduate; Final: CA Inter pass"
- NOT "Fees apply" → "Registration: ₹1,500; Exam: ₹3,500 per paper; Late fee: ₹500 after 15 Mar"
- NOT "Experience needed" → "2 years post-qualification in accounts/audit domain with Form 16 proof"

If heading is about ELIGIBILITY:
- Skip obvious criteria (age 18+, citizenship)
- Focus on: Specific qualification patterns, percentage requirements by category, experience domain, document verification process, exceptions/exemptions

If heading is about FEES:
- Skip "fees required"
- Focus on: Each component separately, GST breakdown, payment deadlines with penalties, refund policy specifics, payment modes

If heading is about EXAM PATTERN:
- Skip "exam has questions"
- Focus on: Exact question distribution, marking scheme with negative marking ratio, time per section, allowances (calculator/charts), passing criteria variations

If heading is about DATES:
- List specific date ranges for {current_year}
- Mention session/phase differences
- Note admit card timeline, result declaration timeline

Write content immediately. Reference table at end."""
    
    messages = [{"role": "user", "content": prompt}]
    max_tokens = 1100 if is_first_section else 700
    content, error = call_grok(messages, max_tokens=max_tokens, temperature=0.4)
    return content, error

def generate_data_table_dense(heading, target_country, research_context, table_research):
    """Generate DENSE table using DENSE-TABLE protocol"""
    if not grok_key:
        return None, "No Grok"
    
    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year
    
    combined_research = f"{research_context[:800]}\n\n=== DETAILED TABLE RESEARCH ===\n{table_research}"
    
    prompt = f"""### DENSE-TABLE GENERATION PROTOCOL
**CURRENT DATE:** {current_date}
**HEADING:** {heading['h2_title']}
**PURPOSE:** {heading.get('table_purpose', 'exhaustive data')}
**TYPE:** {heading.get('table_type', 'listing')}
**COUNTRY:** {target_country}
**SUGGESTED COLUMNS:** {', '.join(heading.get('table_columns', ['Column1', 'Column2', 'Column3']))}
**MINIMUM ROWS:** {heading.get('min_rows', 8)}

**RESEARCH DATA:**
{combined_research[:2500]}

STRICT REQUIREMENTS:

1. TEMPORAL ANCHOR:
   - Use ONLY {current_year} data
   - If {current_year} data unavailable, write "No {current_year} Data" in that cell
   - Include year in table title if time-sensitive

2. DATA DENSITY:
   - Minimum {heading.get('min_rows', 8)} rows (aim for 10-12)
   - Cover ALL tiers/levels/categories
   - Include edge cases and exceptions
   - Show variations (by state, category, level)

3. CELL CONTENT RULES:
   - NO boolean ticks/crosses alone → "Yes (Cap: ₹50,000)" not just "Yes"
   - NO generic terms → "Registration Fee: ₹1,500" not "Registration: Applicable"
   - Include units → "120 minutes" not "120", "₹3,500" not "3500"
   - Add qualifiers → "50% (45% for reserved)" not just "50%"

4. BANNED CONTENT IN CELLS:
   ❌ "Varies", "Depends", "TBD", "NA", "—"
   ❌ "As per norms", "Check official site"
   ❌ "To be announced", "Will be notified"
   Use: "Not Applicable for {current_year}" or "No {current_year} Data"

5. SPECIFIC TABLE FORMATS:

ELIGIBILITY TABLE:
Headers: ["Level/Stage", "Educational Qualification", "Age Limit", "Work Experience", "Additional Criteria"]
Min 8 rows covering: Foundation, Intermediate, Final (or equivalent tiers), Special categories, Exemptions

FEES TABLE:
Headers: ["Fee Component", "Amount (₹)", "GST (₹)", "Total (₹)", "Due Date", "Late Fee"]
Min 8 rows: Registration, Exam (per paper/level), Membership, Certificate, Reappearance, Late payment, Exemption cases

EXAM PATTERN TABLE:
Headers: ["Paper/Subject", "Questions", "Marks", "Duration", "Negative Marking", "Passing %"]
Min 8 rows covering all papers, sections, optional subjects

DATES TABLE:
Headers: ["Event", "Start Date", "End Date", "Late Date (if any)", "Remarks"]
Min 8 rows: Registration phases, Exam sessions, Admit card, Results, Document verification

COMPARISON TABLE:
Headers: ["Parameter", "{focus_keyword}", "Alternative 1", "Alternative 2", "Alternative 3"]
Min 8 rows: Duration, Fees, Eligibility, Difficulty, Recognition, Career scope, Salary prospects, Pass rate

SALARY TABLE:
Headers: ["Experience", "Job Role", "Avg Salary (₹ LPA)", "Top Companies", "Growth Rate"]
Min 8 rows: Fresher, 2-5 years, 5-10 years, 10+ years, different roles

INSTITUTE TABLE:
Headers: ["Institute", "Location", "Fees (₹)", "Duration", "Placement %", "Avg Package"]
Min 10 rows: List top institutes with exact data

6. MANDATORY ADDITIONS:
- Add a "note" field if disclaimer needed
- Include source references if available
- Flag expired data: "EXPIRED: Was ₹1,000 till Dec 2024"

Generate table from research. ONLY real data, no placeholders.

Return ONLY valid JSON:
{{
  "table_title": "{heading['h2_title']} ({current_year} Data)",
  "headers": ["Column1", "Column2", "Column3", "Column4"],
  "rows": [
    ["specific data", "specific data with units", "specific data with qualifiers", "specific data"],
    ["specific data", "specific data with units", "specific data with qualifiers", "specific data"]
  ],
  "note": "Source: [Official source] | Last updated: [Date] | Optional disclaimer"
}}"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=2000, temperature=0.3)
    if error:
        return None, error
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except Exception as e:
        return None, f"Parse error: {str(e)}"

def generate_faqs(focus_keyword, target_country, paa_keywords, research_context):
    if not grok_key:
        return None, "Grok required"
    
    current_year = datetime.now().year
    paa_text = "\n".join([f"- {kw}" for kw in paa_keywords[:40]])
    
    prompt = f"""Generate 10-15 FAQs for: "{focus_keyword}" ({current_year})

AUDIENCE: Students in {target_country}
CURRENT YEAR: {current_year}
PAA KEYWORDS: {paa_text}
RESEARCH: {research_context[:2500]}

REQUIREMENTS:
- Active voice, simple language
- Specific facts with EXACT numbers
- 60-80 words per answer
- Use {current_year} data
- NO promotional tone
- Include source/reference where possible

COVER:
- Eligibility specifics (age ranges, qualification patterns, exceptions)
- Fees breakdown (with GST, late fees, refunds)
- Exam pattern details (marking, duration, calculators allowed)
- Important dates for {current_year}
- Career prospects (specific roles, salary ranges)
- Comparison with alternatives (specific differences)
- Application process (step-by-step)
- Common concerns (attempts allowed, validity, recognition)
- Pass percentages and trends
- Document requirements

ANSWER FORMAT:
- Start with direct answer
- Add specific data (numbers, dates, percentages)
- Mention variations if applicable (by category, level, state)
- End with source reference if available

Return ONLY valid JSON:
{{"faqs": [{{"question": "Specific question?", "answer": "Direct answer with {current_year} data, exact numbers, and specific details. Include variations if applicable. [Source: Official source]"}}]}}"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=4000, temperature=0.6)
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
            import re
            blocks = content.split('\n\n')
            
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                
                lines = block.split('\n')
                bullet_count = sum(1 for line in lines if line.strip().startswith('- '))
                
                if bullet_count >= 2:
                    html.append('<ul>')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('- '):
                            item = line[2:].strip()
                            item = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', item)
                            html.append(f'  <li>{item}</li>')
                        elif line and not line.startswith('- '):
                            html.append('</ul>')
                            line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
                            html.append(f'<p>{line}</p>')
                            html.append('<ul>')
                    html.append('</ul>')
                else:
                    para = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', block)
                    html.append(f'<p>{para}</p>')
        
        if section.get('table'):
            table = section['table']
            html.append(f'<h3>{table.get("table_title", "Data")}</h3>')
            html.append('<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%; margin: 20px 0;">')
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            if headers:
                html.append('  <thead>')
                html.append('    <tr style="background-color: #2c3e50; color: white;">')
                for header in headers:
                    html.append(f'      <th style="text-align: left; padding: 12px; font-weight: bold;">{header}</th>')
                html.append('    </tr>')
                html.append('  </thead>')
            if rows:
                html.append('  <tbody>')
                for idx, row in enumerate(rows):
                    bg_color = '#f8f9fa' if idx % 2 == 0 else '#ffffff'
                    html.append(f'    <tr style="background-color: {bg_color};">')
                    for cell in row:
                        html.append(f'      <td style="padding: 10px; border: 1px solid #ddd;">{cell}</td>')
                    html.append('    </tr>')
                html.append('  </tbody>')
            html.append('</table>')
            if table.get('note'):
                html.append(f'<p style="font-size: 0.9em; color: #666; margin-top: 10px;"><em>Note: {table["note"]}</em></p>')
        html.append('')
    
    if faqs:
        html.append('<h2>Frequently Asked Questions (FAQs)</h2>')
        for faq in faqs:
            html.append(f'<h3 style="color: #2c3e50; margin-top: 20px;">{faq["question"]}</h3>')
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
                        html.append(f'  <li>{item}</li>')
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
    st.header("Research Phase")
    st.caption("Generate comprehensive research queries and execute them")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Topic:", placeholder="e.g., CMA certification India")
        if topic:
            st.session_state.main_topic = topic
    with col2:
        mode = st.selectbox("Depth", ["AI Overview (simple)", "AI Mode (complex)"])
    
    if st.button("🔍 Generate Research Queries", type="primary", use_container_width=True):
        if not topic.strip():
            st.error("Please enter a topic")
        elif not gemini_key:
            st.error("Gemini API key required")
        else:
            with st.spinner("Generating comprehensive research queries..."):
                result, error = generate_research_queries(topic, mode)
                if result:
                    st.session_state.fanout_results = result
                    st.session_state.selected_queries = set()
                    st.success(f"✅ Generated {len(result['queries'])} research queries")
                    st.rerun()
                else:
                    st.error(f"Error: {error}")
    
    if st.session_state.fanout_results and 'queries' in st.session_state.fanout_results:
        st.markdown("---")
        queries = st.session_state.fanout_results['queries']
        st.subheader(f"Research Queries ({len(queries)})")
        
        categories = sorted(list(set(q.get('category', 'Unknown') for q in queries)))
        selected_cats = st.multiselect("Filter by category:", categories, default=categories, key="cat_filter")
        
        filtered = [q for q in queries if q.get('category', 'Unknown') in selected_cats]
        filtered_ids = {f"q_{queries.index(q)}" for q in filtered}
        all_selected = all(qid in st.session_state.selected_queries for qid in filtered_ids) if filtered_ids else False
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            select_all = st.checkbox("Select All Visible", value=all_selected, key="select_all")
            if select_all and not all_selected:
                st.session_state.selected_queries.update(filtered_ids)
                st.rerun()
            elif not select_all and all_selected:
                st.session_state.selected_queries.difference_update(filtered_ids)
                st.rerun()
        with col2:
            completed = sum(1 for qid in filtered_ids if qid in st.session_state.research_results)
            st.metric("Completed", f"{completed}/{len(filtered)}")
        
        for q in filtered:
            qid = f"q_{queries.index(q)}"
            col1, col2, col3 = st.columns([5, 1, 1])
            with col1:
                is_selected = qid in st.session_state.selected_queries
                category_emoji = "🔍" if q.get('priority') == 'high' else "📊"
                selected = st.checkbox(
                    f"{category_emoji} **{q['query']}**  \n`{q.get('category', 'Unknown')}` • Priority: {q.get('priority', 'medium')}",
                    value=is_selected,
                    key=f"cb_{qid}"
                )
                if selected != is_selected:
                    if selected:
                        st.session_state.selected_queries.add(qid)
                    else:
                        st.session_state.selected_queries.discard(qid)
                    st.rerun()
            with col2:
                if qid in st.session_state.research_results:
                    st.success("✅")
            with col3:
                if qid not in st.session_state.research_results and perplexity_key:
                    if st.button("Run", key=f"btn_{qid}"):
                        with st.spinner("Researching..."):
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
            unresearched = [qid for qid in st.session_state.selected_queries if qid not in st.session_state.research_results]
            
            if len(unresearched) > 0:
                if not st.session_state.bulk_research_running:
                    if st.button(f"🚀 Research {len(unresearched)} Selected Queries", type="secondary", use_container_width=True):
                        st.session_state.bulk_research_running = True
                        st.session_state.bulk_research_progress = {
                            'current': 0, 'total': len(unresearched), 'queries': unresearched, 'success': 0, 'errors': 0
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
                        status.info(f"🔍 ({prog['current']+1}/{prog['total']}): {q['query'][:60]}...")
                        
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
                        st.success(f"✅ Bulk research completed! Success: {prog['success']}, Errors: {prog['errors']}")
                        if st.button("Close"):
                            st.session_state.bulk_research_running = False
                            st.session_state.bulk_research_progress = {}
                            st.rerun()

with tab2:
    st.header("Article Settings")
    st.caption("Configure focus keyword, target country, and mandatory tables")
    
    st.subheader("Core Settings")
    col1, col2 = st.columns(2)
    with col1:
        focus_keyword = st.text_input("Focus Keyword *", placeholder="e.g., CMA certification", help="Main topic for the article")
        if focus_keyword:
            st.session_state.focus_keyword = focus_keyword
    with col2:
        target_country = st.selectbox("Target Country *", ["India", "United States", "United Kingdom", "Canada", "Australia", "Global"])
        st.session_state.target_country = target_country
    
    st.markdown("---")
    st.subheader("Mandatory Tables")
    st.caption("Specify tables that MUST be included. Each heading will automatically get a detailed table.")
    
    table_input = st.text_area(
        "Enter table topics (one per line):",
        placeholder="CMA exam dates 2025\nCMA fees breakdown\nCMA vs CPA comparison\nCMA eligibility by level\nCMA exam centers statewise",
        height=150,
        help="These tables will be prioritized in the outline"
    )
    
    if st.button("💾 Save Table Requirements", use_container_width=True):
        if table_input.strip():
            tables = [t.strip() for t in table_input.split('\n') if t.strip()]
            st.session_state.required_tables = tables
            st.success(f"✅ {len(tables)} mandatory tables saved")
        else:
            st.session_state.required_tables = []
            st.info("No mandatory tables set")
    
    if st.session_state.required_tables:
        with st.expander("📋 View Required Tables", expanded=True):
            for i, t in enumerate(st.session_state.required_tables, 1):
                st.write(f"{i}. {t}")
    
    st.markdown("---")
    st.subheader("Optional: Upload Keywords")
    st.caption("Upload CSV files with keywords for FAQ generation")
    
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
                col_name = df.columns[0] if len(df.columns) == 1 else st.selectbox("Select column:", df.columns.tolist(), key="paa_col")
                if st.button("Load PAA", key="load_paa"):
                    paa_list = [q.strip() for q in df[col_name].dropna().astype(str).tolist() if '?' in q or len(q.split()) > 3]
                    st.session_state.paa_keywords = paa_list
                    st.success(f"✅ {len(paa_list)} keywords")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        st.caption("**Keyword Combinations**")
        kw_file = st.file_uploader("CSV", type=['csv'], key="kw")
        if kw_file:
            try:
                df = pd.read_csv(kw_file, encoding='utf-8')
                col_name = 'Suggestion' if 'Suggestion' in df.columns else st.selectbox("Select column:", df.columns.tolist(), key="kw_col")
                if st.button("Load Keywords", key="load_kw"):
                    kw_list = df[col_name].dropna().astype(str).tolist()
                    st.session_state.keyword_combinations = kw_list
                    st.success(f"✅ {len(kw_list)} keywords")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col3:
        st.caption("**Google Ads Keywords**")
        gads_file = st.file_uploader("CSV", type=['csv'], key="gads")
        if gads_file:
            try:
                try:
                    df = pd.read_csv(gads_file, encoding='utf-16', sep='\t')
                except:
                    df = pd.read_csv(gads_file, encoding='utf-8')
                if 'Keyword' not in df.columns and len(df) > 3:
                    df = pd.read_csv(gads_file, encoding='utf-16', sep='\t', skiprows=2)
                if 'Keyword' in df.columns and st.button("Load Ads", key="load_gads"):
                    keywords = [str(row['Keyword']).strip() for _, row in df.iterrows() if str(row['Keyword']) != 'nan']
                    st.session_state.google_ads_keywords = keywords
                    st.success(f"✅ {len(keywords)} keywords")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab3:
    st.header("Research Results")
    st.caption("View completed research data")
    
    if not st.session_state.research_results:
        st.info("No research completed yet. Go to Research tab to start.")
    else:
        total = len(st.session_state.research_results)
        categories = {}
        for qid, data in st.session_state.research_results.items():
            cat = data.get('category', 'Unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Completed", total)
        with col2:
            st.metric("Categories", len(categories))
        
        st.markdown("---")
        
        for cat, items in categories.items():
            with st.expander(f"📁 {cat} ({len(items)} queries)", expanded=False):
                for data in items:
                    st.markdown(f"**Q:** {data['query']}")
                    st.markdown(data['result'][:500] + "..." if len(data['result']) > 500 else data['result'])
                    st.markdown("---")

with tab4:
    st.header("Generate Article Outline")
    st.caption("Create comprehensive outline with dense table requirements")
    
    current_year = datetime.now().year
    
    if not st.session_state.research_results:
        st.warning("⚠️ Complete research first (Tab 1)")
    elif not st.session_state.focus_keyword:
        st.warning("⚠️ Set focus keyword in Settings (Tab 2)")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Research Queries", len(st.session_state.research_results))
        with col2:
            st.metric("Focus Keyword", st.session_state.focus_keyword)
        with col3:
            st.metric("Required Tables", len(st.session_state.required_tables))
        
        st.markdown("---")
        
        # Competitor analysis
        if 'competitor_analysis_done' not in st.session_state or not st.session_state.competitor_research:
            if st.button("🔍 Analyze Competitor Coverage", type="secondary", use_container_width=True):
                with st.spinner("Analyzing Shiksha, CollegeDunia, Careers360, and other competitors..."):
                    competitor_data = analyze_competitor_coverage(
                        st.session_state.focus_keyword,
                        st.session_state.target_country
                    )
                    if competitor_data:
                        st.session_state.competitor_research = competitor_data
                        st.session_state.competitor_analysis_done = True
                        st.success("✅ Competitor analysis complete!")
                        st.rerun()
        else:
            st.success("✅ Competitor analysis complete")
            if st.button("👁️ View Competitor Analysis"):
                with st.expander("Competitor Coverage Analysis", expanded=True):
                    st.markdown(st.session_state.competitor_research)
        
        st.markdown("---")
        
        if st.button(f"📋 Generate {current_year} Outline", type="primary", use_container_width=True):
            with st.spinner("Creating comprehensive outline with dense table specifications..."):
                outline, error = generate_outline_from_research(
                    st.session_state.focus_keyword,
                    st.session_state.target_country,
                    st.session_state.required_tables,
                    st.session_state.research_results,
                    st.session_state.competitor_research
                )
                if outline:
                    st.session_state.content_outline = outline
                    st.success(f"✅ Outline ready with {len(outline.get('headings', []))} sections!")
                    st.rerun()
                else:
                    st.error(f"Error: {error}")
        
        if st.session_state.content_outline:
            st.markdown("---")
            outline = st.session_state.content_outline
            
            st.subheader("Article Structure")
            st.text_input("📰 Title:", value=outline['article_title'], disabled=True)
            st.text_area("📝 Meta Description:", value=outline['meta_description'], height=60, disabled=True)
            
            st.markdown("### Sections")
            for i, h in enumerate(outline['headings'], 1):
                with st.expander(f"{i}. {h['h2_title']}", expanded=False):
                    st.write(f"**Student Question:** {h.get('student_question', 'N/A')}")
                    st.write(f"**Key Facts:** {', '.join(h.get('key_facts', []))}")
                    st.info(f"📊 **Table:** {h.get('table_purpose', 'Data')} ({h.get('table_type', 'listing')})  \nMin rows: {h.get('min_rows', 8)}")
                    if h.get('table_columns'):
                        st.write(f"**Columns:** {', '.join(h['table_columns'])}")

with tab5:
    st.header("Generate Content")
    st.caption("Generate article with DENSE-TABLE protocol")
    
    current_year = datetime.now().year
    
    if not st.session_state.content_outline:
        st.warning("⚠️ Generate outline first (Tab 4)")
    else:
        outline = st.session_state.content_outline
        total_sections = len(outline['headings'])
        completed = len(st.session_state.generated_sections)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sections", total_sections)
        with col2:
            st.metric("Completed", completed)
        with col3:
            progress_pct = (completed / total_sections * 100) if total_sections > 0 else 0
            st.metric("Progress", f"{progress_pct:.0f}%")
        
        st.markdown("---")
        
        if st.button(f"🚀 Generate Complete {current_year} Article", type="primary", use_container_width=True):
            research_context = "\n\n".join([f"**{d['query']}**\n{d['result']}" for d in list(st.session_state.research_results.values())[:25]])
            
            progress = st.progress(0)
            status = st.empty()
            
            st.session_state.generated_sections = []
            
            # Get latest alerts
            status.text(f"🔍 Checking {current_year} updates...")
            latest_alerts = get_latest_alerts(st.session_state.focus_keyword, st.session_state.target_country)
            
            for idx, heading in enumerate(outline['headings']):
                # Phase 1: Deep table research
                status.text(f"📊 ({idx+1}/{total_sections}) Researching dense table data: {heading['h2_title'][:45]}...")
                
                table_research = research_table_data_dense(
                    heading,
                    st.session_state.target_country,
                    st.session_state.focus_keyword
                )
                
                time.sleep(1)
                
                # Phase 2: Generate content
                status.text(f"✍️ ({idx+1}/{total_sections}) Writing content: {heading['h2_title'][:45]}...")
                
                is_first = (idx == 0)
                content, _ = generate_section_content_dense(
                    heading,
                    st.session_state.focus_keyword,
                    st.session_state.target_country,
                    research_context,
                    table_research,
                    is_first_section=is_first,
                    latest_alerts=latest_alerts if is_first else ""
                )
                
                time.sleep(1)
                
                # Phase 3: Generate dense table
                status.text(f"📋 ({idx+1}/{total_sections}) Creating dense table: {heading['h2_title'][:45]}...")
                
                table, table_error = generate_data_table_dense(
                    heading,
                    st.session_state.target_country,
                    research_context,
                    table_research
                )
                
                if table_error:
                    st.warning(f"Table generation issue for {heading['h2_title']}: {table_error}")
                
                st.session_state.generated_sections.append({
                    'heading': heading,
                    'content': content,
                    'table': table
                })
                
                progress.progress((idx + 1) / total_sections)
                time.sleep(2)
            
            # Phase 4: Generate FAQs
            if st.session_state.paa_keywords:
                status.text("❓ Generating FAQs...")
                faqs, _ = generate_faqs(
                    st.session_state.focus_keyword,
                    st.session_state.target_country,
                    st.session_state.paa_keywords,
                    research_context
                )
                if faqs:
                    st.session_state.generated_faqs = faqs['faqs']
            
            status.success(f"✅ Article generation complete!")
            time.sleep(1)
            st.rerun()
        
        if st.session_state.generated_sections:
            st.markdown("---")
            st.markdown(f"# {outline['article_title']}")
            st.caption(outline['meta_description'])
            
            total_words = 0
            table_count = 0
            total_rows = 0
            
            for section in st.session_state.generated_sections:
                st.markdown(f"## {section['heading']['h2_title']}")
                
                if section['content']:
                    st.markdown(section['content'])
                    total_words += len(section['content'].split())
                
                if section.get('table'):
                    table = section['table']
                    table_count += 1
                    rows = len(table.get('rows', []))
                    total_rows += rows
                    
                    st.markdown(f"### {table.get('table_title')}")
                    if table.get('rows') and table.get('headers'):
                        df = pd.DataFrame(table['rows'], columns=table['headers'])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.caption(f"*{rows} rows*")
                        if table.get('note'):
                            st.info(table['note'])
                
                st.markdown("---")
            
            if st.session_state.generated_faqs:
                st.markdown("## Frequently Asked Questions (FAQs)")
                for faq in st.session_state.generated_faqs:
                    st.markdown(f"### {faq['question']}")
                    st.markdown(faq['answer'])
                    total_words += len(faq['answer'].split())
                st.markdown("---")
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📝 Words", f"{total_words:,}")
            with col2:
                st.metric("📊 Tables", table_count)
            with col3:
                st.metric("📋 Table Rows", total_rows)
            with col4:
                avg_rows = total_rows / table_count if table_count > 0 else 0
                st.metric("📈 Avg Rows/Table", f"{avg_rows:.1f}")
            
            st.markdown("---")
            
            # Export
            html = export_to_html(
                outline['article_title'],
                outline['meta_description'],
                st.session_state.generated_sections,
                st.session_state.generated_faqs
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.download_button(
                    "📥 Download HTML Article",
                    data=html.encode('utf-8'),
                    file_name=f"{st.session_state.focus_keyword.replace(' ', '_')}_{current_year}.html",
                    mime="text/html",
                    use_container_width=True
                )
            with col2:
                if st.button("🔄 Regenerate", use_container_width=True):
                    st.session_state.generated_sections = []
                    st.session_state.generated_faqs = []
                    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Current Year: {datetime.now().year}")
if st.sidebar.button("🗑️ Clear All Data"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
