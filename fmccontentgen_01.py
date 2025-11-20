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
st.caption("Publication-ready articles with structured outline and strict quality control")

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
if 'h2_headings' not in st.session_state: st.session_state.h2_headings = []
if 'used_table_content' not in st.session_state: st.session_state.used_table_content = set()
if 'seo_intro' not in st.session_state: st.session_state.seo_intro = ""

# --- API CONFIGURATION SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
with st.sidebar.expander("🔑 API Keys", expanded=True):
    gemini_key = st.text_input("Gemini API Key", type="password", help="Get from Google AI Studio")
    perplexity_key = st.text_input("Perplexity API Key", type="password", help="Get from perplexity.ai/settings")
    grok_key = st.text_input("Grok API Key", type="password", help="Get from x.ai")
    
    if perplexity_key:
        if perplexity_key.startswith('pplx-'):
            st.success("✓ Perplexity key format valid")
        else:
            st.error("⚠️ Perplexity key should start with 'pplx-'")
    
    if grok_key:
        if grok_key.startswith('xai-'):
            st.success("✓ Grok key format valid")
        else:
            st.warning("⚠️ Grok key usually starts with 'xai-'")

gemini_model = st.sidebar.selectbox("Gemini Model", ["gemini-2.0-flash-exp", "gemini-2.0-flash"], index=0)

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
        st.sidebar.success(f"✓ Gemini Active")
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
    - Cover ALL major aspects of this topic:
      * Definition and overview
      * Key features or characteristics  
      * Requirements or prerequisites
      * Process or methodology
      * Costs or pricing (if applicable)
      * Benefits and outcomes
      * Comparisons with alternatives
      * Timeline or duration (if applicable)
      * Latest updates or changes
    
    EXAMPLES OF GOOD QUERIES:
    * "What are ALL the components/aspects of {topic} with complete details?"
    * "What is the EXACT process/methodology for {topic} with step-by-step breakdown?"
    * "What are the COMPLETE requirements or prerequisites for {topic}?"
    
    EXAMPLES OF BAD QUERIES:
    * "Tell me about {topic}" (too general)
    * "What is {topic}?" (too basic)
    
    DO NOT assume this is about fees/costs unless the topic explicitly mentions it.
    Adapt questions to fit the topic domain naturally.
    
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

def call_perplexity(query, system_prompt=None, max_retries=2):
    """Call Perplexity with deep research instructions"""
    if not perplexity_key: return {"error": "Missing API key"}
    
    if not system_prompt:
        system_prompt = f"""Current Date: {formatted_date}

CRITICAL DATA COLLECTION RULES:
- Provide COMPLETE, FACTUAL data with exact numbers and specifics
- Search official websites, university sites, government portals
- Include ALL components (if fees: list EVERY fee component)
- NO external links or URLs in your response
- Return data in structured format
- If exact data unavailable, say "Data not found in sources"
- NO approximations, NO "typically", NO "usually"

SEARCH PRIORITY:
1. Official institutional websites
2. Government/regulatory body sites  
3. Ranking websites (NIRF, QS, etc.)
4. Recent news from last 6 months
5. Competitor analysis sites

Return comprehensive data with:
- Exact figures with currency symbols
- Complete lists (not samples)
- Date ranges and validity periods
- Eligibility with specific percentages/marks
- NO URLs or external links"""
    
    headers = {
        "Authorization": f"Bearer {perplexity_key}", 
        "Content-Type": "application/json"
    }
    data = {
        "model": "sonar", 
        "messages": [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": query}
        ]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post("https://api.perplexity.ai/chat/completions", 
                                   headers=headers, json=data, timeout=60)
            
            if response.status_code == 401:
                return {"error": "Invalid API key - Check your Perplexity API key at perplexity.ai/settings"}
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result
            return {"error": "Invalid response format"}
            
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"error": f"HTTP Error: {str(e)}"}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"error": f"Failed: {str(e)}"}
    
    return {"error": "Max retries exceeded"}

def call_grok(messages, max_tokens=4000, temperature=0.3):
    """Call Grok with strict quality controls"""
    if not grok_key: return None, "Missing API key"
    GROK_API_URL = "https://api.x.ai/v1/chat/completions"
    
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
    """Parse Google Keyword Planner CSV with multiple encoding support"""
    keywords = []
    df = None
    
    encodings = ['utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except:
            continue
    
    if df is None:
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        except:
            return None, "Unable to read file"
    
    keyword_col = None
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if any(term in col_lower for term in ['keyword', 'query', 'search term']):
            keyword_col = col
            break
    
    if not keyword_col:
        keyword_col = df.columns[0]
    
    if keyword_col:
        for idx, row in df.iterrows():
            try:
                kw = str(row[keyword_col]).strip()
                if kw and kw.lower() not in ['nan', 'none', ''] and len(kw) > 2:
                    keywords.append({'keyword': kw, 'suitable_for_heading': True})
            except:
                continue
    
    return keywords, None

def get_latest_news_updates(focus_keyword, target_country, days_back=30):
    """Fetch latest news/updates"""
    if not perplexity_key: return []
    
    cutoff_date = context_date - timedelta(days=days_back)
    cutoff_str = cutoff_date.strftime("%B %d, %Y")
    
    query = f"""Find ONLY latest news, announcements, or updates about {focus_keyword} in {target_country} 
    published between {cutoff_str} and {formatted_date}.
    
    Focus on:
    - Admission notifications
    - Registration deadlines
    - Fee changes
    - New programs launched
    - Policy updates
    - Exam date announcements
    
    DO NOT include any URLs or external links.
    If no major updates in this period, return "No significant updates found"."""
    
    system_prompt = f"Current date: {formatted_date}. Return ONLY factual updates with exact dates. Be specific. NO URLs."
    
    res = call_perplexity(query, system_prompt=system_prompt)
    
    if res and 'choices' in res:
        content = res['choices'][0]['message'].get('content', '')
        
        if any(phrase in content.lower() for phrase in ['no major update', 'no significant', 'no recent']):
            return []
        
        updates = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 20:
                line = re.sub(r'^[-•*]\s*', '', line)
                # Remove URLs
                line = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', line)
                if line and not line.startswith(('No ', 'There are no')):
                    updates.append(line)
        
        return updates[:3]
    
    return []

def generate_semantic_h1(focus_keyword, research_data):
    """Generate semantic H1 from research insights"""
    if not model: return f"{focus_keyword} - Complete Guide {current_year}"
    
    research_summary = "\n".join([f"- {d['query']}: {d['result'][:100]}..." 
                                 for d in list(research_data.values())[:10]])
    
    prompt = f"""Analyze research and create semantic H1 for: "{focus_keyword}"

Research insights:
{research_summary}

TASK: Create H1 in format: "[Main Topic] - [Key Aspect 1], [Key Aspect 2] and [Key Aspect 3] ({current_year})"

RULES:
- Keep aspects concise (2-4 words each)
- Focus on what users search for: comparison, fees, eligibility, scope, career, process, benefits, requirements
- Make it generic and universally applicable
- DO NOT include specific institution names in H1
- Identify 3-4 main aspects from research

FORMAT EXAMPLES (adapt pattern to any topic):
✅ "[Topic A vs Topic B] - [Aspect 1], [Aspect 2] and [Aspect 3] (2025)"
✅ "[Main Topic] - [Feature 1], [Feature 2] and [Feature 3] (2025)"
✅ "[Program/Course Name] - [Info 1], [Info 2] and [Info 3] (2025)"

PATTERN EXAMPLES:
- "Topic - Comparison, Cost and Career Path (2025)"
- "Topic - Requirements, Process and Benefits (2025)"  
- "Topic - Features, Pricing and Use Cases (2025)"
- "Topic - Overview, Options and Selection Guide (2025)"

REMEMBER: 
- Use actual topic name: "{focus_keyword}"
- Extract 3-4 key aspects from research
- Keep it natural and searchable

Return ONLY the H1 title, nothing else."""
    
    try:
        response = model.generate_content(prompt)
        h1 = response.text.strip().strip('"\'')
        if f"({current_year})" not in h1:
            h1 = f"{h1} ({current_year})"
        return h1
    except:
        return f"{focus_keyword} - Complete Guide {current_year}"

def convert_queries_to_crisp_h2_headings(research_data, focus_keyword, keyword_list=None):
    """Convert research queries into CRISP, KEYWORD-FOCUSED H2 headings (3-5 words max)"""
    if not grok_key: return None
    
    queries_list = "\n".join([f"{idx+1}. {data['query']}" 
                             for idx, data in enumerate(research_data.values())])
    
    keyword_context = ""
    if keyword_list:
        keyword_context = f"\n\nAVAILABLE KEYWORDS (prioritize using these as H2s):\n" + "\n".join([f"- {kw['keyword']}" for kw in keyword_list[:30]])
    
    prompt = f"""Convert these research queries into CRISP, SEO-OPTIMIZED H2 headings.

Topic: {focus_keyword}

Research Queries:
{queries_list}
{keyword_context}

CRITICAL H2 HEADING RULES:

1. **LENGTH: 3-5 WORDS MAXIMUM** (This is non-negotiable)
   ✅ "Fee Structure Breakdown"
   ✅ "Eligibility Criteria"
   ✅ "Career Opportunities"
   ✅ "Top Colleges Comparison"
   ❌ "What is the Fee Structure for {focus_keyword}?" (too long)
   ❌ "Career Opportunities and Job Prospects in {focus_keyword}" (too long)

2. **KEYWORD-FOCUSED** (Make them searchable)
   - Use exact keywords from keyword list when possible
   - Focus on search intent: "fees", "eligibility", "scope", "colleges", "salary", "courses"
   - Make them Google-search friendly
   ✅ "MBA Fees India"
   ✅ "Engineering Entrance Exams"
   ✅ "Data Science Courses"

3. **NATURAL & CONTEXTUAL**
   - Include topic context only when needed for clarity
   - Don't force topic name in every heading
   ✅ "Admission Process" (clear from article context)
   ✅ "CEED Exam Pattern" (needs specificity)

4. **REMOVE ALL FLUFF**
   - No questions marks
   - No "How to", "What are", "Complete Guide to"
   - No dates/years
   - Just the core keyword phrase

5. **VARIETY IN FORMAT**
   - Statements: "Course Curriculum"
   - Comparisons: "Online vs Offline"
   - Lists: "Top 10 Institutes"
   - Processes: "Application Procedure"

6. **NO DUPLICATES** - Each heading must be unique

PATTERN EXAMPLES (3-5 words each):

Single Topic:
✅ "Fee Structure"
✅ "Eligibility Requirements"
✅ "Career Options"
✅ "Course Duration"

Comparisons:
✅ "IIT vs NIT"
✅ "Online vs Classroom"
✅ "MBA vs PGDM"

Lists/Rankings:
✅ "Top Engineering Colleges"
✅ "Best Career Paths"
✅ "Popular Specializations"

Process/Action:
✅ "Application Process"
✅ "Exam Preparation Tips"
✅ "Admission Procedure"

Return ONLY JSON:
{{
  "headings": [
    {{"original_query": "full original query", "h2": "Crisp 3-5 Word Heading"}},
    ...
  ]
}}

REMEMBER: Every H2 must be 3-5 words maximum and keyword-focused!"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=2500, temperature=0.4)
    
    if error: return None
    
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        result = json.loads(response.strip())
        return result.get('headings', [])
    except:
        return None

def create_structured_outline_with_hierarchy(h2_headings, research_data):
    """Create structured outline with content hierarchy for each H2"""
    if not grok_key: return None
    
    h2_list = "\n".join([f"{idx+1}. {h['h2']}" for idx, h in enumerate(h2_headings)])
    
    prompt = f"""Create a STRUCTURED CONTENT OUTLINE for each H2 heading.

H2 Headings:
{h2_list}

For EACH H2, define the content structure following this hierarchy:

CONTENT STRUCTURE OPTIONS:
1. **Intro Paragraph** - Always include a 1-2 paragraph introduction
2. **Main Content Type** - Choose ONE:
   - Table (for comparative data, lists with multiple attributes)
   - Bullet Points (for lists, steps, features)
   - Paragraph Only (for explanatory content)
3. **H3 Subsections** - 0-3 subsections under this H2, each with:
   - H3 title (3-4 words)
   - Content type (paragraph/bullets)

EXAMPLES OF GOOD STRUCTURE:

Example 1 - "Fee Structure":
- Intro: 1 paragraph explaining fee components
- Main: Table (showing different fee types)
- H3s: 
  * "Payment Schedule" (bullets)
  * "Refund Policy" (paragraph)

Example 2 - "Eligibility Criteria":
- Intro: 1 paragraph overview
- Main: Bullet points (listing criteria)
- H3s: None (content is straightforward)

Example 3 - "Career Opportunities":
- Intro: 1 paragraph about career scope
- Main: Paragraph only
- H3s:
  * "Top Job Roles" (bullets)
  * "Salary Expectations" (table)
  * "Career Growth" (paragraph)

Return JSON for ALL headings:
{{
  "structured_outline": [
    {{
      "h2": "Heading Title",
      "intro_paragraphs": 1 or 2,
      "main_content_type": "table" or "bullets" or "paragraph",
      "main_content_description": "What the main content will cover",
      "h3_subsections": [
        {{
          "h3": "Subsection Title",
          "content_type": "paragraph" or "bullets" or "table",
          "content_focus": "What this H3 will cover"
        }}
      ]
    }}
  ]
}}

RULES:
- Every H2 MUST have intro paragraph(s)
- Main content type should match the H2 topic (fees=table, steps=bullets, etc.)
- H3s are optional but recommended for complex topics
- Keep H3 titles to 3-4 words
- Total structure should feel like a professional article"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=3500, temperature=0.3)
    
    if error: return None
    
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        result = json.loads(response.strip())
        return result.get('structured_outline', [])
    except:
        return None

def validate_table_data(table):
    """Validate table has no empty cells or placeholder text"""
    if not table or 'rows' not in table:
        return False
    
    invalid_phrases = ['n/a', 'not available', 'not specified', 'data not found', 
                      'not mentioned', 'varies', 'typically', '', '-']
    
    for row in table['rows']:
        for cell in row:
            cell_lower = str(cell).strip().lower()
            if not cell_lower or cell_lower in invalid_phrases:
                return False
    
    return True

def is_table_duplicate(table, used_tables):
    """Check if table content is duplicate"""
    if not table or 'rows' not in table:
        return False
    
    # Create signature from table content
    table_sig = str(sorted([str(row) for row in table['rows']]))
    
    if table_sig in used_tables:
        return True
    
    return False

def generate_section_content(heading_structure, research_context, is_first_section=False, latest_updates=None):
    """Generate expert-level content following strict hierarchical structure"""
    if not grok_key: return None, "Grok required"
    
    system_instruction = f"""You are an EXPERT TECHNICAL WRITER creating publication-ready content.
Context Date: {formatted_date}

ABSOLUTE RULES - ZERO TOLERANCE:

1. DATA INTEGRITY:
   - Write ONLY about data explicitly in research
   - If research lacks data for a point → skip completely (write nothing)
   - NEVER use: "not available", "not specified", "data not found"
   - NEVER use: "typically", "usually", "generally", "varies"
   - NO external links or URLs anywhere in content

2. EXPERT WRITING STYLE:
   - Direct, authoritative tone
   - Vary sentence structure (8-25 words, mix of simple and complex)
   - Use active voice predominantly
   - No fluff words: "comprehensive", "crucial", "important to note"
   - No meta-commentary: "let's explore", "it's worth mentioning"
   - Present tense for facts, past tense for events

3. FORBIDDEN PHRASES:
   ❌ "Understanding this is important"
   ❌ "It should be noted that"
   ❌ "Additionally, it's worth mentioning"
   ❌ "Let's look at" / "Let's explore"
   ❌ Any sentence starting with "It is important"
   ❌ Any URLs or web links
"""

    if is_first_section and latest_updates:
        updates_text = "\n".join([f"• {update}" for update in latest_updates])
        prompt = f"""{system_instruction}

TASK: Write opening section with latest updates

FORMAT:
**Latest Updates** (bold heading)
{updates_text}

Then {heading_structure.get('intro_paragraphs', 1)} paragraph(s) covering:
- Definition/overview
- Governing body/authority (if applicable)
- Primary purpose
- Target audience

RESEARCH DATA:
{research_context[:2500]}

CRITICAL: Only write about facts in research. Skip missing information silently. NO URLs."""

    else:
        # Regular section
        h3_info = ""
        if heading_structure.get('h3_subsections'):
            h3_list = "\n".join([f"- H3: {h3['h3']} [{h3['content_type']}]" 
                                for h3 in heading_structure['h3_subsections']])
            h3_info = f"\n\nH3 SUBSECTIONS TO INCLUDE:\n{h3_list}\n(Write content for each H3 as well)"
        
        main_content_instruction = {
            'table': "After intro, mention that detailed data is shown in the table below (don't list the data in paragraph)",
            'bullets': "After intro, present key points as bullet points",
            'paragraph': "Write 2-3 detailed paragraphs explaining the topic thoroughly"
        }.get(heading_structure.get('main_content_type', 'paragraph'), '')
        
        prompt = f"""{system_instruction}

TASK: Write content for: "{heading_structure['h2']}"

STRUCTURE TO FOLLOW:
1. Introduction: {heading_structure.get('intro_paragraphs', 1)} paragraph(s)
2. Main Content: {heading_structure.get('main_content_type', 'paragraph')} - {main_content_instruction}
{h3_info}

Focus: {heading_structure.get('main_content_description', '')}

RESEARCH DATA:
{research_context[:3500]}

REMEMBER: 
- Follow the structure exactly
- Only facts from research
- Skip missing data silently
- Vary sentence structure
- Expert authoritative tone
- NO URLs or external links"""

    messages = [{"role": "user", "content": prompt}]
    content, error = call_grok(messages, max_tokens=1500, temperature=0.3)
    
    if content:
        # Remove any URLs that might have slipped through
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
    
    return content, error

def generate_intelligent_table(heading_structure, research_context):
    """Generate validated table with complete data only"""
    if not grok_key or heading_structure.get('main_content_type') != 'table': 
        return None, "No table needed"
    
    custom_instruction = heading_structure.get('main_content_description', '')
    
    prompt = f"""Create focused data table for: "{heading_structure['h2']}"

Content Focus: {custom_instruction}

RESEARCH DATA:
{research_context[:3500]}

CRITICAL TABLE RULES:

1. ZERO TOLERANCE FOR INCOMPLETE DATA:
   - ONLY include rows with COMPLETE data in ALL cells
   - If ANY cell would be empty → remove entire row
   - NEVER use: "N/A", "Not specified", "Varies", "-", or empty cells
   - 3-5 focused rows minimum with complete data

2. DATA EXTRACTION:
   - Extract ONLY explicit data from research
   - Use exact numbers, dates, amounts from sources
   - Don't pad with generic/incomplete rows

3. STAY ON TOPIC:
   - Table 100% focused on heading topic only
   - "Fee Structure" = fees only, no scholarships
   - "Courses" = courses only, no eligibility

4. FORMATTING:
   - Headers with units: ₹, %, years
   - Concise cells (under 15 words)
   - No vague language
   - NO URLs or external links

Return ONLY valid JSON:
{{
  "table_title": "{heading_structure['h2']} - {current_year}",
  "headers": ["Column 1", "Column 2", "Column 3"],
  "rows": [
    ["Complete data 1A", "Complete data 1B", "Complete data 1C"],
    ["Complete data 2A", "Complete data 2B", "Complete data 2C"]
  ],
  "footer_note": "Optional brief note if critical context needed"
}}

REMEMBER: Every cell must have complete, factual data. Remove row if any cell incomplete. NO URLs."""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=2000, temperature=0.2)
    
    if error: return None, error
    
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        table = json.loads(response.strip())
        
        # Validate table
        if validate_table_data(table):
            # Check for duplicates
            if is_table_duplicate(table, st.session_state.used_table_content):
                return None, "Duplicate table detected - skipped"
            
            # Mark as used
            table_sig = str(sorted([str(row) for row in table['rows']]))
            st.session_state.used_table_content.add(table_sig)
            
            return table, None
        else:
            return None, "Table contains incomplete data - rejected"
    except:
        return None, "Parse error"

def fetch_additional_research(query_text, focus_keyword):
    """Fetch additional research data from Perplexity during article generation"""
    if not perplexity_key:
        return None
    
    enhanced_query = f"{query_text} for {focus_keyword} - provide specific, factual data with exact numbers and details. NO URLs."
    
    system_prompt = f"""Current Date: {formatted_date}

Provide SPECIFIC, FACTUAL data for this query. Include:
- Exact numbers, percentages, amounts
- Complete lists (not samples)
- Official data from authoritative sources
- NO approximations or vague terms
- NO external links or URLs

Be comprehensive and data-focused."""
    
    res = call_perplexity(enhanced_query, system_prompt=system_prompt)
    
    if res and 'choices' in res and len(res['choices']) > 0:
        content = res['choices'][0]['message'].get('content', '')
        # Remove URLs
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        return content
    
    return None

def generate_faqs_detailed(focus_keyword, paa_keywords, research_context):
    """Generate FAQs with detailed 2-3 line answers"""
    if not grok_key: return None, "Grok required"
    
    paa_text = "\n".join([f"- {kw}" for kw in paa_keywords[:15]]) if paa_keywords else "Generate relevant FAQs"
    
    prompt = f"""Generate 8-10 FAQs for: "{focus_keyword}"

QUESTIONS (use these as basis):
{paa_text}

RESEARCH DATA:
{research_context[:3500]}

RULES FOR ANSWERS:
- Each answer MUST be 2-3 sentences (40-60 words)
- Provide specific facts and numbers from research
- Be comprehensive but concise
- Direct, informative tone
- NO fluff or meta-commentary
- NO external links or URLs
- If research lacks answer → skip that question entirely

EXAMPLE FORMAT:
Q: "What is the eligibility for XYZ?"
A: "Candidates must have completed 10+2 with 50% marks in Physics, Chemistry, and Mathematics from a recognized board. General category students need minimum 50% while reserved categories require 45%. Age limit is 17-25 years as of the exam date."

Return ONLY valid JSON:
{{"faqs": [{{"question": "Question?", "answer": "Detailed 2-3 sentence answer with specific facts"}}]}}"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=3000, temperature=0.4)
    
    if error: return None, error
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        result = json.loads(response.strip())
        
        # Remove URLs from FAQ answers
        if result and 'faqs' in result:
            for faq in result['faqs']:
                if 'answer' in faq:
                    faq['answer'] = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', faq['answer'])
        
        return result, None
    except:
        return None, "Parse error"

def generate_seo_introduction(h1_title, focus_keyword, research_context, latest_updates=None):
    """Generate comprehensive SEO introduction paragraph after H1"""
    if not grok_key: return None, "Grok required"
    
    prompt = f"""Write a comprehensive SEO INTRODUCTION PARAGRAPH that appears immediately after the H1 title.

H1: {h1_title}
Focus Keyword: {focus_keyword}

RESEARCH DATA:
{research_context[:2500]}

PURPOSE OF THIS PARAGRAPH:
This introduction serves as the article's foundation - it must:
1. Define what {focus_keyword} is (clear, concise definition)
2. Explain its purpose/importance
3. Preview what the article covers
4. Include target keywords naturally
5. Hook the reader to continue reading

STRUCTURE (150-200 words):
- Opening sentence: Clear definition of {focus_keyword}
- 2-3 sentences: Context, importance, who it's for
- 1-2 sentences: What readers will learn in this article
- Closing sentence: Brief mention of latest updates (if any)

TONE:
- Authoritative and informative
- Natural keyword integration
- Engaging but professional
- Present tense

SEO OPTIMIZATION:
- Include "{focus_keyword}" in first 100 characters
- Use semantic variations (synonyms, related terms)
- Answer implicit "what is" query
- Include year ({current_year}) for freshness signal

EXAMPLE STRUCTURE:
"{focus_keyword} is [clear definition]. [Purpose/importance sentence]. [Who it's for sentence]. This comprehensive guide covers [topic 1], [topic 2], [topic 3], and [topic 4], providing aspiring candidates with everything needed to [goal]. [Latest update reference if available]."

CRITICAL RULES:
- Write ONLY the introduction paragraph
- 150-200 words exactly
- NO heading (just the paragraph text)
- NO bullet points or lists
- Natural, flowing prose
- Data-backed where possible
- NO external links or URLs

Return ONLY the paragraph text, nothing else."""
    
    messages = [{"role": "user", "content": prompt}]
    intro, error = call_grok(messages, max_tokens=500, temperature=0.4)
    
    if intro:
        # Remove any URLs
        intro = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', intro)
        # Remove any accidental headings
        intro = re.sub(r'^#+\s+.*$', '', intro, flags=re.MULTILINE)
        return intro.strip(), None
    
    return intro, error

def final_seo_quality_check(h1, seo_intro, sections, faqs, focus_keyword):
    """Comprehensive SEO and quality check by Grok acting as SEO expert"""
    if not grok_key: return True, "Skipped"
    
    # Compile full article structure
    article_preview = f"H1: {h1}\n\n"
    article_preview += f"INTRO: {seo_intro[:200]}...\n\n"
    
    for sec in sections[:8]:  # First 8 sections for review
        article_preview += f"H2: {sec['heading']['h2']}\n"
        article_preview += f"Content preview: {sec.get('content', '')[:150]}...\n"
        if sec.get('table'):
            article_preview += f"[Table: {sec['table'].get('table_title', 'N/A')}]\n"
        article_preview += "\n"
    
    faq_preview = "\n".join([f"Q: {faq['question']}" for faq in faqs[:5]]) if faqs else "No FAQs"
    
    prompt = f"""You are an EXPERT SEO CONSULTANT reviewing this article for publication.

ARTICLE STRUCTURE:
{article_preview}

FAQS:
{faq_preview}

TARGET KEYWORD: {focus_keyword}
CURRENT YEAR: {current_year}

COMPREHENSIVE REVIEW CHECKLIST:

1. SEO OPTIMIZATION (Critical):
   ✓ H1 includes target keyword naturally?
   ✓ Introduction paragraph includes keyword in first 100 chars?
   ✓ H2 headings are keyword-focused and searchable?
   ✓ Semantic keywords and variations used throughout?
   ✓ Meta-relevant content (year, location, specific data)?
   ✓ No keyword stuffing?

2. CONTENT STRUCTURE (Critical):
   ✓ Clear hierarchy: H1 → Intro → H2s → Content?
   ✓ Logical flow between sections?
   ✓ Each section serves a purpose?
   ✓ No orphaned or out-of-place content?
   ✓ Proper transitions between topics?

3. CONTENT QUALITY (Critical):
   ✓ Introduction summarizes entire article?
   ✓ No placeholder text ("not available", "data not found")?
   ✓ No repetitive content across sections?
   ✓ Tables add value (not duplicates)?
   ✓ Factual, specific data (no vague statements)?
   ✓ NO external URLs or links?

4. READABILITY (Important):
   ✓ Varied sentence structures?
   ✓ Natural language (not robotic)?
   ✓ Clear, concise explanations?
   ✓ Appropriate paragraph lengths?
   ✓ Professional tone throughout?

5. TECHNICAL SEO (Important):
   ✓ Proper heading hierarchy (no skipped levels)?
   ✓ Descriptive table titles with year?
   ✓ FAQ answers are comprehensive (2-3 lines)?
   ✓ Content length appropriate (not too thin)?

6. USER INTENT (Critical):
   ✓ Article answers the main query about {focus_keyword}?
   ✓ Provides actionable, useful information?
   ✓ Covers topic comprehensively?
   ✓ Satisfies search intent?

SCORING CRITERIA:
- 10/10: Perfect, publication-ready
- 8-9/10: Very good, minor tweaks needed
- 6-7/10: Good, but has notable issues
- <6/10: Needs significant improvement

Return ONLY valid JSON:
{{
  "overall_score": 1-10,
  "status": "EXCELLENT" or "GOOD" or "NEEDS_IMPROVEMENT",
  "seo_score": 1-10,
  "structure_score": 1-10,
  "content_quality_score": 1-10,
  "readability_score": 1-10,
  "strengths": ["list of 2-3 major strengths"],
  "issues": ["list of specific issues found - empty if none"],
  "seo_recommendations": ["list of 2-3 SEO improvements if needed"],
  "verdict": "One sentence professional verdict on publication readiness"
}}

BE CRITICAL. Only rate 9-10 if truly exceptional. Identify real issues if they exist."""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=1500, temperature=0.2)
    
    if error: return True, "Check skipped due to error"
    
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        result = json.loads(response.strip())
        
        overall_score = result.get('overall_score', 10)
        status = result.get('status', 'GOOD')
        
        # Format detailed feedback
        feedback = f"""
**Overall Score: {overall_score}/10** - {status}

📊 **Detailed Scores:**
- SEO: {result.get('seo_score', 'N/A')}/10
- Structure: {result.get('structure_score', 'N/A')}/10  
- Content Quality: {result.get('content_quality_score', 'N/A')}/10
- Readability: {result.get('readability_score', 'N/A')}/10

✅ **Strengths:**
{chr(10).join([f'• {s}' for s in result.get('strengths', ['Content meets standards'])])}

{f"⚠️ **Issues Found:**{chr(10)}" + chr(10).join([f'• {i}' for i in result.get('issues', [])]) if result.get('issues') else '✅ No major issues found'}

{f"🎯 **SEO Recommendations:**{chr(10)}" + chr(10).join([f'• {r}' for r in result.get('seo_recommendations', [])]) if result.get('seo_recommendations') else ''}

**Verdict:** {result.get('verdict', 'Article is ready for publication.')}
"""
        
        if overall_score >= 8:
            return True, feedback
        else:
            return False, feedback
            
    except Exception as e:
        return True, f"Quality check completed (parsing error: {str(e)})"

def export_to_html(article_title, seo_intro, sections, faqs, latest_updates):
    """Export clean HTML without external links"""
    html = ['<!DOCTYPE html>', '<html>', '<head>',
            '<meta charset="UTF-8">',
            '<style>',
            'body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }',
            'h1 { font-size: 2em; color: #2c3e50; margin-bottom: 20px; }',
            'h2 { font-size: 1.5em; color: #34495e; margin-top: 30px; margin-bottom: 15px; border-bottom: 2px solid #3498db; padding-bottom: 5px; }',
            'h3 { font-size: 1.2em; color: #555; margin-top: 20px; margin-bottom: 10px; }',
            'p { margin: 15px 0; text-align: justify; }',
            'table { width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            'th { background-color: #2c3e50; color: white; padding: 12px; text-align: left; font-weight: bold; }',
            'td { padding: 10px; border: 1px solid #ddd; }',
            'tr:nth-child(even) { background-color: #f8f9fa; }',
            'tr:hover { background-color: #e8f4f8; }',
            'ul { margin: 15px 0; padding-left: 25px; }',
            'li { margin: 8px 0; }',
            '.update-box { background: #fff3cd; padding: 15px; margin: 20px 0; border-left: 4px solid #ffc107; }',
            '.table-note { font-size: 0.9em; color: #666; font-style: italic; margin-top: 5px; }',
            '</style>',
            '</head>', '<body>']
    
    html.append(f'<h1>{article_title}</h1>')
    
    # Add SEO Introduction Paragraph
    if seo_intro:
        seo_intro_clean = re.sub(r'http[s]?://\S+', '', seo_intro)
        html.append(f'<p><strong>{seo_intro_clean}</strong></p>')
    
    if latest_updates:
        html.append('<div class="update-box">')
        html.append('<strong>Latest Updates</strong>')
        html.append('<ul>')
        for update in latest_updates:
            update_clean = re.sub(r'http[s]?://\S+', '', update)
            html.append(f'<li>{update_clean}</li>')
        html.append('</ul></div>')
    
    for section in sections:
        html.append(f'<h2>{section["heading"]["h2"]}</h2>')
        
        if section.get('content'):
            content = section['content']
            # Clean any LLM artifacts and URLs
            content = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'\*\*Latest Updates\*\*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'http[s]?://\S+', '', content)
            
            blocks = content.split('\n\n')
            for block in blocks:
                block = block.strip()
                if not block: continue
                
                # Check for H3
                if block.startswith('###'):
                    h3_text = block.replace('###', '').strip()
                    html.append(f'<h3>{h3_text}</h3>')
                    continue
                
                # Bold text
                block = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', block)
                
                # Check for bullet points
                if block.startswith(('- ', '• ')):
                    html.append('<ul>')
                    for line in block.split('\n'):
                        line = line.strip()
                        if line.startswith(('- ', '• ')):
                            html.append(f'<li>{line[2:]}</li>')
                    html.append('</ul>')
                else:
                    html.append(f'<p>{block}</p>')
        
        if section.get('table'):
            table = section['table']
            html.append(f'<h3>{table.get("table_title", "")}</h3>')
            html.append('<table>')
            
            if table.get('headers'):
                html.append('<thead><tr>')
                for h in table['headers']:
                    html.append(f'<th>{h}</th>')
                html.append('</tr></thead>')
            
            if table.get('rows'):
                html.append('<tbody>')
                for row in table['rows']:
                    html.append('<tr>')
                    for cell in row:
                        cell_clean = re.sub(r'http[s]?://\S+', '', str(cell))
                        html.append(f'<td>{cell_clean}</td>')
                    html.append('</tr>')
                html.append('</tbody>')
            
            html.append('</table>')
            if table.get('footer_note'):
                footer_clean = re.sub(r'http[s]?://\S+', '', table['footer_note'])
                html.append(f'<p class="table-note">{footer_clean}</p>')
    
    if faqs:
        html.append('<h2>Frequently Asked Questions</h2>')
        for faq in faqs:
            question_clean = re.sub(r'http[s]?://\S+', '', faq["question"])
            answer_clean = re.sub(r'http[s]?://\S+', '', faq["answer"])
            html.append(f'<h3>{question_clean}</h3>')
            html.append(f'<p>{answer_clean}</p>')
    
    html.append('</body></html>')
    return '\n'.join(html)
# --- MAIN UI ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Setup & Keywords", "2. Research Queries", "3. Outline Structure", "4. Generate Content"])

with tab1:
    st.header("Step 1: Topic & Keyword Setup")
    st.info(f"📅 Content Date: **{formatted_date}**")
    
    col1, col2 = st.columns(2)
    with col1:
        focus_keyword = st.text_input(
            "Main Topic *",
            value=st.session_state.main_topic,
            placeholder="e.g., MBA Colleges India, Cloud Computing, Investment Banking",
            key="topic_input"
        )
        if focus_keyword and focus_keyword != st.session_state.main_topic:
            st.session_state.focus_keyword = focus_keyword
            st.session_state.main_topic = focus_keyword
            st.session_state.fanout_results = None
            st.session_state.research_results = {}
            st.session_state.selected_queries = set()
            st.session_state.custom_headings = []
            st.session_state.selected_custom_headings = set()
            st.session_state.generated_sections = []
            st.session_state.content_outline = None
            st.warning("⚠️ Topic changed - all research data cleared. Start fresh in Tab 2.")
        elif focus_keyword:
            st.session_state.focus_keyword = focus_keyword
            st.session_state.main_topic = focus_keyword
    
    with col2:
        target_country = st.selectbox("Target Audience *", 
                                     ["India", "United States", "United Kingdom", "Canada", "Australia", "Global"])
        st.session_state.target_country = target_country
    
    st.markdown("---")
    st.markdown("### Upload Keywords (Become H2 Headings)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Keywords CSV**")
        st.caption("These will become your H2 headings")
        keyword_file = st.file_uploader("Upload Keywords", type=['csv'], key='kw')
        if keyword_file:
            try:
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    tmp.write(keyword_file.getvalue())
                    tmp_path = tmp.name
                
                keywords, error = parse_keyword_planner_csv(tmp_path)
                os.unlink(tmp_path)
                
                if keywords:
                    st.session_state.keyword_planner_data = keywords
                    st.success(f"✓ {len(keywords)} keywords → H2 headings")
                else:
                    st.warning("No keywords found")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.markdown("**PAA Questions CSV**")
        st.caption("For FAQ generation")
        paa_file = st.file_uploader("Upload PAA", type=['csv'], key='paa')
        if paa_file:
            try:
                df = pd.read_csv(paa_file)
                st.session_state.paa_keywords = df.iloc[:, 0].dropna().astype(str).tolist()
                st.success(f"✓ {len(st.session_state.paa_keywords)} questions")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.focus_keyword:
            st.success(f"✓ Topic Set")
        else:
            st.error("✗ No topic")
    with col2:
        if st.session_state.keyword_planner_data:
            st.success(f"✓ {len(st.session_state.keyword_planner_data)} keywords")
        else:
            st.info("○ No keywords")
    with col3:
        if st.session_state.paa_keywords:
            st.success(f"✓ {len(st.session_state.paa_keywords)} PAA")
        else:
            st.info("○ No PAA")

with tab2:
    st.header("Step 2: Research Queries Selection")
    
    if not st.session_state.focus_keyword:
        st.error("⚠️ Set topic in Tab 1 first")
        st.stop()
    
    st.success(f"📌 **{st.session_state.focus_keyword}** | {st.session_state.target_country}")
    
    st.markdown("---")
    st.subheader("🤖 AI-Generated Research Queries (Fanout)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("Generate comprehensive research queries using AI")
    with col2:
        mode = st.selectbox("Depth", ["AI Overview (simple)", "AI Mode (complex)"])
    
    if st.button("Generate Research Queries", type="primary", use_container_width=True):
        if not gemini_key:
            st.error("Gemini API Key required")
        else:
            with st.spinner("Generating queries..."):
                result, error = generate_research_queries(st.session_state.focus_keyword, mode)
                if result:
                    st.session_state.fanout_results = result
                    st.session_state.selected_queries = set()
                    st.success(f"✓ Generated {len(result['queries'])} research queries")
                    st.rerun()
                else:
                    st.error(f"Error: {error}")
    
    if st.session_state.fanout_results and 'queries' in st.session_state.fanout_results:
        queries = st.session_state.fanout_results['queries']
        
        st.markdown("---")
        
        with st.expander("➕ Add Your Own Heading/Query", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                custom_heading_input = st.text_input("Heading/Query:", 
                                                    placeholder="e.g., All B.Tech Courses and Fees")
            with col2:
                content_type = st.selectbox("Content Type:", 
                                          ["Table Required", "Bullet Points", "Paragraph Only"])
            
            table_instruction = ""
            if content_type == "Table Required":
                table_instruction = st.text_area(
                    "What should the table show?", 
                    placeholder="e.g., All B.Tech specializations with annual fees, duration, eligibility",
                    height=70
                )
            
            if st.button("➕ Add Custom Heading", use_container_width=True, type="secondary"):
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
        
        if st.session_state.keyword_planner_data:
            with st.expander("🔄 Add Keywords as H2 Headings", expanded=False):
                st.info(f"You have {len(st.session_state.keyword_planner_data)} keywords loaded")
                num_kw = st.slider("How many keywords to add as H2s?", 5, 20, 10)
                
                if st.button("Add Keywords as Headings", type="secondary"):
                    added = 0
                    for kw_data in st.session_state.keyword_planner_data[:num_kw]:
                        kw = kw_data['keyword']
                        if not any(h['query'] == kw for h in st.session_state.custom_headings):
                            heading_id = f"custom_kw_{len(st.session_state.custom_headings)}"
                            st.session_state.custom_headings.append({
                                'id': heading_id,
                                'query': kw,
                                'category': 'Keyword',
                                'content_type': 'Table Required',
                                'table_instruction': f"Detailed data table for {kw}"
                            })
                            added += 1
                    st.success(f"✓ Added {added} keyword headings")
                    st.rerun()
        
        categories = sorted(list(set(q.get('category', 'Unknown') for q in queries)))
        if st.session_state.custom_headings:
            categories = ['Custom', 'Keyword'] + [c for c in categories if c not in ['Custom', 'Keyword']]
        
        selected_cats = st.multiselect("Filter by Category:", categories, default=categories)
        
        all_items = []
        
        if 'Custom' in selected_cats or 'Keyword' in selected_cats:
            for ch in st.session_state.custom_headings:
                if ch.get('category') in selected_cats:
                    all_items.append(ch)
        
        for q in queries:
            if q.get('category', 'Unknown') in selected_cats:
                all_items.append(q)
        
        st.markdown(f"### 📋 {len(all_items)} Queries to Research")
        
        all_ids = set()
        for item in all_items:
            if item in st.session_state.custom_headings:
                all_ids.add(item['id'])
            else:
                all_ids.add(f"q_{queries.index(item)}")
        
        combined_selected = st.session_state.selected_queries.union(st.session_state.selected_custom_headings)
        all_selected_check = all(qid in combined_selected for qid in all_ids) if all_ids else False
        
        if st.checkbox("Select All Visible", value=all_selected_check):
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
        
        for item in all_items:
            if item in st.session_state.custom_headings:
                qid = item['id']
                col1, col2, col3 = st.columns([4, 1, 0.5])
                with col1:
                    is_selected = qid in st.session_state.selected_custom_headings
                    label = f"**{item['query']}** 🏷️ {item.get('category', 'Custom')}"
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
                        st.session_state.custom_headings = [h for h in st.session_state.custom_headings 
                                                           if h['id'] != qid]
                        st.session_state.selected_custom_headings.discard(qid)
                        st.rerun()
            else:
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
        
        st.markdown("---")
        all_selected = st.session_state.selected_queries.union(st.session_state.selected_custom_headings)
        
        st.caption(f"Selected: {len(all_selected)} | Perplexity Key: {'✓' if perplexity_key else '✗'}")
        
        if all_selected and perplexity_key:
            unresearched = [qid for qid in all_selected if qid not in st.session_state.research_results]
            if unresearched:
                if st.button(f"🔍 Research {len(unresearched)} Selected Queries", 
                           type="primary", use_container_width=True):
                    progress = st.progress(0)
                    status = st.empty()
                    error_container = st.container()
                    start_time = time.time()
                    
                    errors = []
                    successful = 0
                    
                    try:
                        for i, qid in enumerate(unresearched):
                            if i > 0:
                                elapsed = time.time() - start_time
                                avg_time = elapsed / i
                                remaining = avg_time * (len(unresearched) - i)
                                mins = int(remaining // 60)
                                secs = int(remaining % 60)
                                timer = f"⏱️ ~{mins}m {secs}s remaining"
                            else:
                                timer = "⏱️ Starting..."
                            
                            try:
                                if qid.startswith('custom_'):
                                    custom_h = next((h for h in st.session_state.custom_headings 
                                                   if h['id'] == qid), None)
                                    if custom_h:
                                        q_text = custom_h['query']
                                        if custom_h.get('table_instruction'):
                                            q_text = f"{q_text}. Specifically: {custom_h['table_instruction']}"
                                    else:
                                        errors.append(f"Custom heading {qid} not found")
                                        continue
                                else:
                                    q_idx = int(qid.split('_')[1])
                                    q_text = queries[q_idx]['query']
                                
                                status.text(f"{timer} | Researching: {q_text[:60]}...")
                                
                                res = call_perplexity(q_text)
                                
                                if res and isinstance(res, dict):
                                    if 'error' in res:
                                        errors.append(f"API Error for '{q_text[:40]}...': {res['error']}")
                                        continue
                                    elif 'choices' in res and len(res['choices']) > 0:
                                        st.session_state.research_results[qid] = {
                                            'query': q_text,
                                            'result': res['choices'][0]['message']['content']
                                        }
                                        successful += 1
                                    else:
                                        errors.append(f"No data returned for: {q_text[:40]}...")
                                else:
                                    errors.append(f"Invalid response for: {q_text[:40]}...")
                                
                            except Exception as e:
                                errors.append(f"Error processing {qid}: {str(e)}")
                                continue
                            
                            progress.progress((i + 1) / len(unresearched))
                            time.sleep(1.5)
                        
                        elapsed_total = time.time() - start_time
                        mins_total = int(elapsed_total // 60)
                        secs_total = int(elapsed_total % 60)
                        
                        if successful > 0:
                            st.success(f"✅ {successful}/{len(unresearched)} queries researched in {mins_total}m {secs_total}s!")
                        
                        if errors:
                            with error_container:
                                with st.expander("⚠️ Errors Encountered", expanded=True):
                                    for err in errors:
                                        st.warning(err)
                        
                        if successful > 0:
                            st.rerun()
                        else:
                            st.error("❌ Research failed for all queries. Check errors above.")
                    
                    except Exception as e:
                        st.error(f"❌ Critical error during research: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.info("All selected queries already researched!")
        elif not perplexity_key:
            st.error("⚠️ Add Perplexity API key in sidebar")
        elif not all_selected:
            st.info("Select queries above to research")

with tab3:
    st.header("Step 3: Structured Outline Builder")
    
    if not st.session_state.research_results:
        st.warning("⚠️ Complete research in Tab 2 first")
        st.stop()
    
    st.success(f"✅ {len(st.session_state.research_results)} queries researched and ready")
    
    st.markdown("---")
    st.subheader("📝 Generate Article Outline")
    
    if not st.session_state.content_outline:
        if st.button("🎯 Generate Structured Outline", type="primary", use_container_width=True):
            with st.spinner("Step 1/3: Generating semantic H1..."):
                h1 = generate_semantic_h1(st.session_state.focus_keyword, st.session_state.research_results)
            
            with st.spinner("Step 2/3: Converting queries to crisp H2 headings (3-5 words)..."):
                h2_conversions = convert_queries_to_crisp_h2_headings(
                    st.session_state.research_results, 
                    st.session_state.focus_keyword,
                    st.session_state.keyword_planner_data
                )
                
                if not h2_conversions:
                    st.error("Failed to convert queries to H2 headings. Please try again.")
                    st.stop()
            
            with st.spinner("Step 3/3: Creating content hierarchy for each H2..."):
                structured_outline = create_structured_outline_with_hierarchy(
                    h2_conversions,
                    st.session_state.research_results
                )
                
                if not structured_outline or len(structured_outline) == 0:
                    st.warning("⚠️ Grok outline generation failed. Creating basic structure...")
                    # Fallback: Create basic structure from H2 conversions
                    structured_outline = []
                    for conversion in h2_conversions:
                        structured_outline.append({
                            'h2': conversion['h2'],
                            'intro_paragraphs': 1,
                            'main_content_type': 'paragraph',
                            'main_content_description': f"Write about {conversion['h2']}",
                            'h3_subsections': []
                        })
                
                # Validate outline structure
                for section in structured_outline:
                    if 'h2' not in section:
                        section['h2'] = 'Untitled Section'
                    if 'intro_paragraphs' not in section:
                        section['intro_paragraphs'] = 1
                    if 'main_content_type' not in section:
                        section['main_content_type'] = 'paragraph'
                    if 'h3_subsections' not in section:
                        section['h3_subsections'] = []
                
                # Map qids
                query_to_qid = {}
                for qid, data in st.session_state.research_results.items():
                    query_to_qid[data['query']] = qid
                
                # Match original queries to structured outline
                for section in structured_outline:
                    # Find closest matching query
                    best_match_qid = None
                    for conversion in h2_conversions:
                        if conversion['h2'] == section['h2']:
                            original_q = conversion['original_query']
                            best_match_qid = query_to_qid.get(original_q)
                            break
                    
                    section['qid'] = best_match_qid or list(st.session_state.research_results.keys())[0]
                    section['is_manual'] = False
                
                st.session_state.content_outline = {
                    'article_title': h1,
                    'structured_sections': structured_outline
                }
                
                st.success("✓ Structured outline generated with content hierarchy!")
                st.rerun()
    
    if st.session_state.content_outline:
        st.markdown("### 📋 Article Structure")
        
        h1_edit = st.text_input("Edit H1 Title:", 
                               value=st.session_state.content_outline['article_title'],
                               key="h1_editor")
        if h1_edit != st.session_state.content_outline['article_title']:
            st.session_state.content_outline['article_title'] = h1_edit
        
        st.markdown("---")
        st.markdown("### 🏗️ Content Sections (Reorder with buttons)")
        
        st.info("Each section shows: **Intro Paragraph(s)** → **Main Content (Table/Bullets/Paragraph)** → **H3 Subsections (if any)**")
        
        sections = st.session_state.content_outline['structured_sections']
        
        for idx, section in enumerate(sections):
            with st.expander(f"**{idx+1}. {section['h2']}**", expanded=False):
                # Reorder buttons
                col_up, col_down, col_del = st.columns([1, 1, 1])
                with col_up:
                    if idx > 0 and st.button("⬆️ Move Up", key=f"up_{idx}"):
                        sections[idx], sections[idx-1] = sections[idx-1], sections[idx]
                        st.rerun()
                with col_down:
                    if idx < len(sections)-1 and st.button("⬇️ Move Down", key=f"down_{idx}"):
                        sections[idx], sections[idx+1] = sections[idx+1], sections[idx]
                        st.rerun()
                with col_del:
                    if st.button("🗑️ Delete", key=f"del_{idx}"):
                        sections.pop(idx)
                        st.rerun()
                
                # Edit H2
                new_h2 = st.text_input("H2 Title (3-5 words):", value=section['h2'], key=f"h2_{idx}")
                if new_h2 != section['h2']:
                    section['h2'] = new_h2
                
                # Structure display
                st.markdown(f"**Structure:**")
                st.write(f"1️⃣ **Intro:** {section.get('intro_paragraphs', 1)} paragraph(s)")
                st.write(f"2️⃣ **Main Content:** {section.get('main_content_type', 'paragraph').title()}")
                st.caption(f"_Focus: {section.get('main_content_description', 'N/A')}_")
                
                # Edit structure
                col1, col2 = st.columns(2)
                with col1:
                    new_intro = st.selectbox("Intro paragraphs:", [1, 2], 
                                            index=section.get('intro_paragraphs', 1)-1,
                                            key=f"intro_{idx}")
                    section['intro_paragraphs'] = new_intro
                
                with col2:
                    current_content_type = section.get('main_content_type', 'paragraph') if section.get('main_content_type', 'paragraph') in ["table", "bullets", "paragraph"] else 'paragraph'
                    new_main = st.selectbox("Main content type:", ["table", "bullets", "paragraph"], index=["table", "bullets", "paragraph"].index(current_content_type), key=f"main_{idx}")
                    section['main_content_type'] = new_main
                    
                
                # H3 subsections
                if section.get('h3_subsections'):
                    st.markdown("**3️⃣ H3 Subsections:**")
                    for h3_idx, h3 in enumerate(section['h3_subsections']):
                        col1, col2, col3 = st.columns([2, 1, 0.5])
                        with col1:
                            st.write(f"• **{h3['h3']}** [{h3['content_type']}]")
                            st.caption(f"_{h3.get('content_focus', '')}_")
                        with col2:
                            new_h3_type = st.selectbox(f"Type:", ["paragraph", "bullets", "table"],
                                                      index=["paragraph", "bullets", "table"].index(h3['content_type']),
                                                      key=f"h3type_{idx}_{h3_idx}")
                            h3['content_type'] = new_h3_type
                        with col3:
                            if st.button("✖️", key=f"delh3_{idx}_{h3_idx}"):
                                section['h3_subsections'].pop(h3_idx)
                                st.rerun()
                else:
                    st.info("No H3 subsections")
                
                # Add H3
                with st.expander("➕ Add H3 Subsection"):
                    new_h3_title = st.text_input("H3 Title (3-4 words):", key=f"newh3_{idx}")
                    new_h3_type = st.selectbox("Content Type:", ["paragraph", "bullets", "table"], key=f"newh3type_{idx}")
                    new_h3_focus = st.text_input("What will this cover?", key=f"newh3focus_{idx}")
                    
                    if st.button("Add H3", key=f"addh3_{idx}"):
                        if new_h3_title:
                            if 'h3_subsections' not in section:
                                section['h3_subsections'] = []
                            section['h3_subsections'].append({
                                'h3': new_h3_title,
                                'content_type': new_h3_type,
                                'content_focus': new_h3_focus
                            })
                            st.rerun()
        
        # Add new section
        st.markdown("---")
        with st.expander("➕ Add New Section"):
            st.info("Add a custom section (Grok will research this during content generation)")
            new_h2 = st.text_input("H2 Heading (3-5 words):", key="new_section_h2")
            new_research = st.text_area("What should be researched?", 
                                       placeholder="e.g., Common misconceptions and myths",
                                       key="new_section_research")
            col1, col2 = st.columns(2)
            with col1:
                new_intro = st.selectbox("Intro paragraphs:", [1, 2], key="new_intro")
            with col2:
                new_main = st.selectbox("Main content:", ["table", "bullets", "paragraph"], key="new_main")
            
            if st.button("Add Section", type="secondary"):
                if new_h2:
                    sections.append({
                        'h2': new_h2,
                        'qid': f'manual_{len(sections)}',
                        'intro_paragraphs': new_intro,
                        'main_content_type': new_main,
                        'main_content_description': new_research or f"Research and write about {new_h2}",
                        'h3_subsections': [],
                        'is_manual': True
                    })
                    st.success(f"✓ Added '{new_h2}'")
                    st.rerun()
        
        # Preview
        st.markdown("---")
        st.markdown("### 📄 Final Outline Preview")
        
        preview = [f"# {st.session_state.content_outline['article_title']}", ""]
        for idx, sec in enumerate(sections, 1):
            is_manual = sec.get('is_manual', False)
            manual_tag = " 🔷 (Manual - Grok will research)" if is_manual else ""
            
            preview.append(f"## {idx}. {sec['h2']}{manual_tag}")
            preview.append(f"   📝 Intro: {sec.get('intro_paragraphs', 1)} para(s)")
            preview.append(f"   📊 Main: {sec.get('main_content_type', 'paragraph').title()}")
            
            if sec.get('h3_subsections'):
                for h3 in sec['h3_subsections']:
                    preview.append(f"      ### {h3['h3']} [{h3['content_type']}]")
            preview.append("")
        
        st.code("\n".join(preview), language="markdown")
        
        manual_count = sum(1 for s in sections if s.get('is_manual', False))
        if manual_count > 0:
            st.info(f"🔷 {manual_count} manual sections will be researched by Grok during content generation")
        
        st.success("✅ Outline ready! Go to Tab 4 to generate content.")

with tab4:
    st.header("Step 4: Generate SEO-Optimized Article")
    
    if not st.session_state.content_outline:
        st.warning("⚠️ Create outline structure in Tab 3 first")
        st.stop()
    
    # Handle both old and new outline formats
    sections = []
    
    if 'structured_sections' in st.session_state.content_outline:
        sections = st.session_state.content_outline['structured_sections']
    elif 'headings' in st.session_state.content_outline:
        # Migrate old format to new format
        st.warning("⚠️ Old outline format detected. Please regenerate outline in Tab 3 for better structure.")
        old_headings = st.session_state.content_outline['headings']
        
        # Convert old format to new format with default structure
        for old_h in old_headings:
            sections.append({
                'h2': old_h.get('h2_title', old_h.get('original_query', 'Untitled')),
                'qid': old_h.get('qid', f'legacy_{len(sections)}'),
                'intro_paragraphs': 1,
                'main_content_type': 'table' if old_h.get('needs_table') else 'bullets',
                'main_content_description': old_h.get('content_focus', ''),
                'h3_subsections': [],
                'is_manual': old_h.get('is_manual', False)
            })
        
        # Update to new format
        st.session_state.content_outline['structured_sections'] = sections
        del st.session_state.content_outline['headings']
    else:
        st.error("⚠️ Invalid outline format. Please regenerate outline in Tab 3.")
        st.stop()
    
    if len(sections) == 0:
        st.error("⚠️ No sections found in outline. Please regenerate outline in Tab 3.")
        st.stop()
    
    manual_count = sum(1 for s in sections if s.get('is_manual', False))
    
    st.success(f"✅ Outline ready: {len(sections)} sections ({manual_count} will be researched by Grok)")
    
    with st.expander("📋 Article Outline", expanded=False):
        st.markdown(f"**H1:** {st.session_state.content_outline['article_title']}")
        for idx, s in enumerate(sections, 1):
            manual_tag = " 🔷" if s.get('is_manual') else ""
            st.markdown(f"{idx}. **{s['h2']}**{manual_tag} - {s.get('main_content_type', 'paragraph')} + {len(s.get('h3_subsections', []))} H3s")
    
    if st.button("🚀 Generate Publication-Ready Article", type="primary", use_container_width=True):
        # Reset duplicate tracking
        st.session_state.used_table_content = set()
        
        progress = st.progress(0)
        status = st.empty()
        start_time = time.time()
        
        status.text("⏱️ Checking for latest updates...")
        latest_updates = get_latest_news_updates(st.session_state.focus_keyword, st.session_state.target_country)
        st.session_state.latest_updates = latest_updates
        
        # Prepare research context
        existing_research_context = "\n\n".join([f"Q: {d['query']}\nA: {d['result']}" 
                                                for d in st.session_state.research_results.values()])
        
        # Generate SEO Introduction Paragraph
        status.text("⏱️ Generating SEO introduction paragraph...")
        seo_intro, intro_error = generate_seo_introduction(
            h1,
            st.session_state.focus_keyword,
            existing_research_context,
            latest_updates
        )
        
        if not seo_intro:
            st.warning("⚠️ Could not generate SEO intro - using basic intro")
            seo_intro = f"{st.session_state.focus_keyword} is a comprehensive examination covering various aspects. This article provides detailed information to help candidates prepare effectively."
        
        st.session_state.seo_intro = seo_intro
        
        existing_research_context = "\n\n".join([f"Q: {d['query']}\nA: {d['result']}" 
                                                for d in st.session_state.research_results.values()])
        
        st.session_state.generated_sections = []
        total = len(sections)
        
        for idx, section in enumerate(sections):
            if idx > 0:
                elapsed = time.time() - start_time
                avg = elapsed / idx
                remaining = avg * (total - idx)
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                timer = f"⏱️ ~{mins}m {secs}s remaining"
            else:
                timer = "⏱️ Starting..."
            
            # Check if manual (needs research)
            research_context = existing_research_context
            
            if section.get('is_manual', False):
                status.text(f"{timer} | 🔷 Researching: {section['h2'][:50]}...")
                
                # Fetch additional research for this topic
                additional_data = fetch_additional_research(
                    section.get('main_content_description', section['h2']),
                    st.session_state.focus_keyword
                )
                
                if additional_data:
                    research_context += f"\n\nADDITIONAL RESEARCH FOR {section['h2']}:\n{additional_data}"
            
            # Check if we need additional research for H3 subsections
            if section.get('h3_subsections'):
                for h3 in section['h3_subsections']:
                    status.text(f"{timer} | 🔍 Researching H3: {h3['h3'][:40]}...")
                    h3_data = fetch_additional_research(
                        h3.get('content_focus', h3['h3']),
                        st.session_state.focus_keyword
                    )
                    if h3_data:
                        research_context += f"\n\nH3 RESEARCH ({h3['h3']}):\n{h3_data}"
            
            # Generate content
            status.text(f"{timer} | ✍️ Writing: {section['h2'][:50]}...")
            
            is_first = (idx == 0)
            content, _ = generate_section_content(
                section, 
                research_context, 
                is_first_section=is_first,
                latest_updates=latest_updates if is_first else None
            )
            
            # Generate table if needed
            table = None
            if section.get('main_content_type') == 'table':
                status.text(f"{timer} | 📊 Creating table...")
                table, error = generate_intelligent_table(section, research_context)
                if error:
                    if "duplicate" in error.lower():
                        status.warning(f"⚠️ Duplicate table detected - skipped")
                    elif "incomplete" in error.lower():
                        status.warning(f"⚠️ Table skipped - incomplete data")
            
            st.session_state.generated_sections.append({
                'heading': section,
                'content': content,
                'table': table
            })
            
            progress.progress((idx + 1) / total)
        
        # Generate FAQs with detailed answers
        status.text("⏱️ Generating detailed FAQs (2-3 line answers)...")
        
        # Optionally fetch additional research for FAQs
        if st.session_state.paa_keywords:
            faq_research_context = existing_research_context
            # Fetch additional data for top 5 PAA questions
            for paa in st.session_state.paa_keywords[:5]:
                status.text(f"⏱️ Researching FAQ: {paa[:40]}...")
                faq_data = fetch_additional_research(paa, st.session_state.focus_keyword)
                if faq_data:
                    faq_research_context += f"\n\nFAQ RESEARCH ({paa}):\n{faq_data}"
        else:
            faq_research_context = existing_research_context
        
        faqs, _ = generate_faqs_detailed(
            st.session_state.focus_keyword, 
            st.session_state.paa_keywords, 
            faq_research_context
        )
        st.session_state.generated_faqs = faqs.get('faqs', []) if faqs else []
        
        # Final SEO & Quality Check by Expert
        status.text("⏱️ Running comprehensive SEO quality review...")
        article_h1 = st.session_state.content_outline.get('article_title', st.session_state.focus_keyword)
        passed, feedback = final_seo_quality_check(
            h1, 
            st.session_state.seo_intro,
            st.session_state.generated_sections, 
            st.session_state.generated_faqs,
            st.session_state.focus_keyword
        )
        
        elapsed_total = time.time() - start_time
        mins_total = int(elapsed_total // 60)
        secs_total = int(elapsed_total % 60)
        
        if passed:
            st.success(f"✅ Article Ready in {mins_total}m {secs_total}s!")
            st.markdown(feedback)
        else:
            st.warning(f"⚠️ Quality Review Feedback:")
            st.markdown(feedback)
        
        st.rerun()
    
    # Display generated content
    # Display generated content
    if st.session_state.generated_sections:
        st.markdown("---")
        st.markdown("## 📄 Article Preview")
        
        article_h1 = st.session_state.content_outline.get('article_title', st.session_state.focus_keyword)
        st.markdown(f"# {article_h1}")
        
        # Display SEO Introduction
        if st.session_state.get('seo_intro'):
            st.markdown(f"**{st.session_state.seo_intro}**")
            st.markdown("---")
        
        if st.session_state.latest_updates:
            st.markdown("**Latest Updates**")
            for update in st.session_state.latest_updates:
                st.markdown(f"• {update}")
            st.markdown("---")
        
        for section in st.session_state.generated_sections:
            st.markdown(f"## {section['heading']['h2']}")
            
            if section.get('content'):
                # Parse and display content with proper H3 handling
                content = section['content']
                blocks = content.split('\n\n')
                
                for block in blocks:
                    block = block.strip()
                    if not block: continue
                    
                    # Check for H3
                    if block.startswith('###'):
                        h3_text = block.replace('###', '').strip()
                        st.markdown(f"### {h3_text}")
                        continue
                    
                    # Regular content
                    st.markdown(block)
            
            if section.get('table'):
                table = section['table']
                st.markdown(f"**{table.get('table_title', '')}**")
                df = pd.DataFrame(table['rows'], columns=table['headers'])
                st.dataframe(df, hide_index=True, use_container_width=True)
                if table.get('footer_note'):
                    st.caption(f"*{table['footer_note']}*")
            
            st.markdown("---")
        
        if st.session_state.generated_faqs:
            st.markdown("## Frequently Asked Questions")
            for faq in st.session_state.generated_faqs:
                with st.expander(faq['question']):
                    st.write(faq['answer'])
        
        # Export
        st.markdown("---")
        html = export_to_html(article_h1, st.session_state.get('seo_intro', ''),
                            st.session_state.generated_sections, 
                            st.session_state.generated_faqs, st.session_state.latest_updates)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📄 Download HTML", html,
                             file_name=f"{st.session_state.focus_keyword.replace(' ', '_')}.html",
                             mime="text/html", use_container_width=True)
        with col2:
            text = f"{article_h1}\n\n"
            if st.session_state.get('seo_intro'):
                text += f"{st.session_state.seo_intro}\n\n"
            for sec in st.session_state.generated_sections:
                text += f"{sec['heading']['h2']}\n\n{sec.get('content', '')}\n\n"
            st.download_button("📝 Download Text", text,
                             file_name=f"{st.session_state.focus_keyword.replace(' ', '_')}.txt",
                             mime="text/plain", use_container_width=True)
