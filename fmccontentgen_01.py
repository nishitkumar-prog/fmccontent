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
st.caption("Publication-ready articles with strict quality control")

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

# --- API CONFIGURATION SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
with st.sidebar.expander("🔑 API Keys", expanded=True):
    gemini_key = st.text_input("Gemini API Key", type="password")
    perplexity_key = st.text_input("Perplexity API Key", type="password")
    grok_key = st.text_input("Grok API Key", type="password")

gemini_model = st.sidebar.selectbox("Gemini Model", ["gemini-2.0-flash-exp", "gemini-2.0-flash"], index=0)

# Add reset button
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
        if "\`\`\`json" in json_text:
            json_text = json_text.split("\`\`\`json")[1].split("\`\`\`")[0]
        return json.loads(json_text.strip()), None
    except Exception as e:
        return None, str(e)

def optimize_headings_batch(queries):
    """Convert long research queries into short SEO headings using Gemini"""
    if not model: return {q: q for q in queries}
    
    # Process in batches of 10 to avoid token limits
    mappings = {}
    
    # Deduplicate queries
    unique_queries = list(set(queries))
    
    prompt = f"""You are an expert SEO editor. Rewrite these long research queries into short, punchy, engaging H2 headings (max 3-6 words).
    
    RULES:
    - Remove "What is", "Provide a detailed", "Explain", "Compare", etc.
    - Focus on the core topic (e.g., "Fee Structure 2025", "Eligibility Criteria", "Syllabus Comparison")
    - Use Title Case
    - Keep it professional
    - Return JSON mapping original query to new heading
    - IF the query is already short, keep it as is but ensure Title Case.
    
    QUERIES:
    {json.dumps(unique_queries)}
    
    Return ONLY valid JSON: {{ "mappings": {{ "original query text": "Short Heading" }} }}"""
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if "\`\`\`json" in text:
            text = text.split("\`\`\`json")[1].split("\`\`\`")[0]
        elif "\`\`\`" in text: # Handle case where json tag is missing
             text = text.split("\`\`\`")[1].split("\`\`\`")[0]
             
        result = json.loads(text)
        mappings = result.get('mappings', {})
        
        # Fallback: if mapping missing or same as key, try simple heuristic
        final_map = {}
        for q in queries:
            if q in mappings and len(mappings[q].split()) < 10:
                final_map[q] = mappings[q]
            else:
                # Fallback heuristic if AI fails or returns long text
                # Remove common prompt phrases
                clean = re.sub(r'^(What is|Provide|Explain|Describe|Compare|List|Detailed)\s+', '', q, flags=re.IGNORECASE)
                clean = clean.split('?')[0].strip()
                if len(clean.split()) > 8:
                    clean = " ".join(clean.split()[:8]) + "..."
                final_map[q] = clean.title()
                
        return final_map
    except Exception as e:
        print(f"Heading optimization error: {e}")
        # Fallback heuristic
        final_map = {}
        for q in queries:
             clean = re.sub(r'^(What is|Provide|Explain|Describe|Compare|List|Detailed)\s+', '', q, flags=re.IGNORECASE)
             clean = clean.split('?')[0].strip()
             final_map[q] = clean.title()
        return final_map

def generate_optimized_headings(research_results):
    """Convert long research queries into concise SEO H2s"""
    queries_list = [data['query'] for data in research_results.values()]
    
    prompt = f"""You are an SEO Expert. Convert these research queries into concise, engaging H2 headings.

    INPUT QUERIES:
    {json.dumps(queries_list)}

    RULES:
    1. Transform questions into Topics (e.g., "What are the fees?" -> "Fee Structure & Costs")
    2. Remove instructional language (e.g., "Provide a detailed comparison..." -> "Detailed Comparison")
    3. Keep them under 8 words
    4. Use Title Case
    5. Make them sound professional and authoritative
    6. Return a list of strings matching the input order exactly

    Return ONLY valid JSON:
    {{
        "optimized_headings": [
            "Heading 1",
            "Heading 2"
        ]
    }}"""
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if "\`\`\`json" in text:
            text = text.split("\`\`\`json")[1].split("\`\`\`")[0]
        data = json.loads(text.strip())
        return data.get('optimized_headings', [])
    except:
        return None

def call_perplexity(query, system_prompt=None, max_retries=2):
    """Call Perplexity with deep research instructions"""
    if not perplexity_key: return {"error": "Missing API key"}
    
    if not system_prompt:
        system_prompt = f"""Current Date: {formatted_date}

CRITICAL DATA COLLECTION RULES:
- Provide COMPLETE, FACTUAL data with exact numbers and specifics
- Search official websites, university sites, government portals
- Include ALL components (if fees: list EVERY fee component)
- Provide source citations
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
- Authoritative sources"""
    
    headers = {"Authorization": f"Bearer {perplexity_key}", "Content-Type": "application/json"}
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
            response.raise_for_status()
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result
            return {"error": "Invalid response"}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"error": f"Failed: {str(e)}"}
    return {"error": "Max retries"}

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
    
    If no major updates in this period, return "No significant updates found"."""
    
    system_prompt = f"Current date: {formatted_date}. Return ONLY factual updates with exact dates. Be specific."
    
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
                if line and not line.startswith(('No ', 'There are no')):
                    updates.append(line)
        
        return updates[:3]  # Max 3 updates
    
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

Examples:
- "MBA Programs India - Top Colleges, Fees and Placement Records (2025)"
- "BSc vs BTech Biotechnology - Eligibility, Syllabus and Career Path (2025)"
- "Machine Learning - Algorithms, Applications and Career Guide (2025)"

Identify 3-4 most important aspects covered in research.

Return ONLY the H1 title, nothing else."""
    
    try:
        response = model.generate_content(prompt)
        h1 = response.text.strip().strip('"\'')
        if f"({current_year})" not in h1:
            h1 = f"{h1} ({current_year})"
        return h1
    except:
        return f"{focus_keyword} - Complete Guide {current_year}"

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

def generate_section_content(heading, research_context, is_first_section=False, latest_updates=None):
    """Generate expert-level content with strict data-only rules"""
    if not grok_key: return None, "Grok required"
    
    context_topic = heading.get('original_query', heading['h2_title'])
    display_title = heading['h2_title']
    
    system_instruction = f"""You are an EXPERT TECHNICAL WRITER creating publication-ready content.
Context Date: {formatted_date}

ABSOLUTE RULES - ZERO TOLERANCE:

1. DATA INTEGRITY:
   - Write ONLY about data explicitly in research
   - If research lacks data for a point → skip completely (write nothing)
   - NEVER use: "not available", "not specified", "data not found", "information unavailable"
   - NEVER use: "typically", "usually", "generally", "varies", "depends"
   - Write naturally as if you don't know about missing information

2. EXPERT WRITING STYLE:
   - Direct, authoritative tone
   - Vary sentence structure (8-25 words, mix of simple and complex)
   - Use active voice predominantly
   - No fluff words: "comprehensive", "crucial", "important to note"
   - No meta-commentary: "let's explore", "it's worth mentioning"
   - Present tense for facts, past tense for events

3. STRUCTURE (MANDATORY):
   - ONE introductory paragraph (4-6 sentences)
   - Establish context and preview what follows
   - Then: Table OR Bullet points (as specified)

4. SENTENCE CRAFTING:
   - Start sentences differently (avoid repetitive patterns)
   - Use parallel structure for lists
   - Employ transitions sparingly: only when logical flow demands
   - Numbers and data inline: "The program costs ₹2.5 lakh annually"

5. FORBIDDEN PHRASES:
   ❌ "Understanding this is important"
   ❌ "It should be noted that"
   ❌ "Additionally, it's worth mentioning"
   ❌ "Let's look at" / "Let's explore"
   ❌ Any sentence starting with "It is important"
"""

    if is_first_section and latest_updates:
        updates_text = "\n".join([f"• {update}" for update in latest_updates])
        prompt = f"""{system_instruction}

TASK: Write opening section with latest updates

FORMAT:
**Latest Updates** (bold heading)
{updates_text}

Then ONE paragraph (4-6 sentences) covering:
- Definition/overview
- Governing body/authority
- Primary purpose
- Target audience

RESEARCH DATA:
{research_context[:2500]}

CRITICAL: Only write about facts in research. Skip missing information silently."""

    else:
        needs_bullets = heading.get('needs_bullets', False)
        
        if needs_bullets:
            structure = "1. ONE paragraph (4-6 sentences)\n2. Then 5-8 bullet points with specific facts"
        else:
            structure = "1. ONE paragraph (4-6 sentences)\n2. Table will follow (don't list data in paragraph)"
        
        prompt = f"""{system_instruction}

TASK: Write content for section: "{display_title}"
SPECIFIC REQUIREMENT: {context_topic}
Focus: {heading.get('content_focus', 'Technical details')}

STRUCTURE:
{structure}

RESEARCH DATA:
{research_context[:3000]}

REMEMBER: 
- Only facts from research
- Skip missing data silently (no placeholders)
- Vary sentence structure
- Expert authoritative tone"""

    messages = [{"role": "user", "content": prompt}]
    content, error = call_grok(messages, max_tokens=1000, temperature=0.3)
    return content, error

def generate_intelligent_table(heading, research_context):
    """Generate validated table with complete data only"""
    if not grok_key or not heading.get('needs_table'): 
        return None, "No table needed"
    
    custom_instruction = heading.get('custom_table_instruction', '')
    context_topic = heading.get('original_query', heading['h2_title'])
    
    prompt = f"""Create focused data table for: "{heading['h2_title']}"
CONTEXT/REQUIREMENT: {context_topic}

{f"CUSTOM REQUIREMENT: {custom_instruction}" if custom_instruction else ""}

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
   - If research has 8 courses with fees → include all 8
   - If research has 3 courses with fees → include only those 3
   - Don't pad with generic/incomplete rows

3. STAY ON TOPIC:
   - Table 100% focused on heading topic only
   - "Fee Structure" = fees only, no scholarships
   - "Courses" = courses only, no eligibility

4. FORMATTING:
   - Headers with units: ₹, %, years
   - Concise cells (under 15 words)
   - No vague language

Return ONLY valid JSON:
{{
  "table_title": "{heading['h2_title']} - {current_year}",
  "headers": ["Column 1", "Column 2", "Column 3"],
  "rows": [
    ["Complete data 1A", "Complete data 1B", "Complete data 1C"],
    ["Complete data 2A", "Complete data 2B", "Complete data 2C"]
  ],
  "footer_note": "Optional brief note if critical context needed"
}}

REMEMBER: Every cell must have complete, factual data. Remove row if any cell incomplete."""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=2000, temperature=0.2)
    
    if error: return None, error
    
    try:
        if "\`\`\`json" in response:
            response = response.split("\`\`\`json")[1].split("\`\`\`")[0]
        table = json.loads(response.strip())
        
        # Validate table
        if validate_table_data(table):
            return table, None
        else:
            return None, "Table contains incomplete data - rejected"
    except:
        return None, "Parse error"

def generate_faqs(focus_keyword, paa_keywords, research_context):
    """Generate FAQs from PAA keywords"""
    if not grok_key: return None, "Grok required"
    
    paa_text = "\n".join([f"- {kw}" for kw in paa_keywords[:15]]) if paa_keywords else "Generate relevant FAQs"
    
    prompt = f"""Generate 8-10 FAQs for: "{focus_keyword}"

QUESTIONS (use these as basis):
{paa_text}

RESEARCH DATA:
{research_context[:3000]}

RULES:
- Answer with specific facts and numbers from research only
- Keep answers direct (2-3 sentences, 15-20 words each)
- NO fluff or meta-commentary
- If research lacks answer → skip that question entirely

Return ONLY valid JSON:
{{"faqs": [{{"question": "Question?", "answer": "Direct factual answer"}}]}}"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=2500, temperature=0.4)
    
    if error: return None, error
    try:
        if "\`\`\`json" in response:
            response = response.split("\`\`\`json")[1].split("\`\`\`")[0]
        return json.loads(response.strip()), None
    except:
        return None, "Parse error"

def final_coherence_check(h1, sections, faqs):
    """Final quality pass by Grok"""
    if not grok_key: return True, "Skipped"
    
    # Compile full article
    article_text = f"H1: {h1}\n\n"
    for sec in sections:
        article_text += f"H2: {sec['heading']['h2_title']}\n"
        article_text += f"{sec.get('content', '')[:300]}\n\n"
    
    prompt = f"""FINAL QUALITY CHECK

Article: {article_text[:4000]}

CHECK:
1. Any placeholder text? ("not available", "not specified", "data not found")
2. Empty table cells or N/A values?
3. Repetitive sentence structures?
4. Missing transitions between sections?
5. Inconsistent formatting?
6. Generic/vague statements?

Return JSON:
{{
  "status": "PASS" or "ISSUES_FOUND",
  "issues": ["list of issues found"],
  "quality_score": 1-10
}}

If quality_score < 8, list specific issues."""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=1000, temperature=0.1)
    
    if error: return True, "Check skipped"
    
    try:
        if "\`\`\`json" in response:
            response = response.split("\`\`\`json")[1].split("\`\`\`")[0]
        result = json.loads(response.strip())
        if result.get('quality_score', 10) >= 8:
            return True, f"Quality Score: {result['quality_score']}/10"
        else:
            return False, f"Issues: {', '.join(result.get('issues', []))}"
    except:
        return True, "Check completed"

def export_to_html(article_title, sections, faqs, latest_updates):
    """Export clean HTML"""
    html = [f'<h1>{article_title}</h1>', '']
    
    if latest_updates:
        html.append('<div style="background: #fff3cd; padding: 15px; margin: 20px 0; border-left: 4px solid #ffc107;">')
        html.append('<strong>Latest Updates</strong>')
        html.append('<ul style="margin: 10px 0 0 0;">')
        for update in latest_updates:
            html.append(f'<li>{update}</li>')
        html.append('</ul></div>')
        html.append('')
    
    for section in sections:
        html.append(f'<h2>{section["heading"]["h2_title"]}</h2>')
        
        if section.get('content'):
            content = section['content']
            # Clean any LLM artifacts
            content = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'\*\*Latest Updates\*\*', '', content, flags=re.IGNORECASE)
            
            blocks = content.split('\n\n')
            for block in blocks:
                block = block.strip()
                if not block: continue
                
                block = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', block)
                
                if block.startswith(('- ', '• ')):
                    html.append('<ul>')
                    for line in block.split('\n'):
                        line = line.strip()
                        if line.startswith(('- ', '• ')):
                            html.append(f'  <li>{line[2:]}</li>')
                    html.append('</ul>')
                else:
                    html.append(f'<p>{block}</p>')
        
        if section.get('table'):
            table = section['table']
            html.append(f'<h3 style="font-size: 1.1em; margin-top: 20px;">{table.get("table_title", "")}</h3>')
            html.append('<table border="1" style="border-collapse: collapse; width: 100%; margin: 20px 0;">')
            
            if table.get('headers'):
                html.append('  <thead><tr>')
                for h in table['headers']:
                    html.append(f'    <th style="padding: 12px; background: #2c3e50; color: white;">{h}</th>')
                html.append('  </tr></thead>')
            
            if table.get('rows'):
                html.append('  <tbody>')
                for row in table['rows']:
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
        html.append('<h2>Frequently Asked Questions</h2>')
        for faq in faqs:
            html.append(f'<h3 style="font-size: 1.05em; margin-top: 15px;">{faq["question"]}</h3>')
            html.append(f'<p>{faq["answer"]}</p>')
    
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
            # Topic changed - clear all research data
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
    
    # FANOUT: Generate AI research queries
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
    
    # Display fanout queries + custom headings + keyword headings together
    if st.session_state.fanout_results and 'queries' in st.session_state.fanout_results:
        queries = st.session_state.fanout_results['queries']
        
        st.markdown("---")
        
        # Section to add custom headings
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
        
        # Convert keywords to headings option
        if st.session_state.keyword_planner_data:
            with st.expander("🔄 Add Keywords as H2 Headings", expanded=False):
                st.info(f"You have {len(st.session_state.keyword_planner_data)} keywords loaded")
                num_kw = st.slider("How many keywords to add as H2s?", 5, 20, 10)
                
                if st.button("Add Keywords as Headings", type="secondary"):
                    added = 0
                    for kw_data in st.session_state.keyword_planner_data[:num_kw]:
                        kw = kw_data['keyword']
                        # Check if already added
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
        
        # Filter and display all queries (AI + Custom)
        categories = sorted(list(set(q.get('category', 'Unknown') for q in queries)))
        if st.session_state.custom_headings:
            categories = ['Custom', 'Keyword'] + [c for c in categories if c not in ['Custom', 'Keyword']]
        
        selected_cats = st.multiselect("Filter by Category:", categories, default=categories)
        
        # Combine all queries
        all_items = []
        
        # Add custom headings first
        if 'Custom' in selected_cats or 'Keyword' in selected_cats:
            for ch in st.session_state.custom_headings:
                if ch.get('category') in selected_cats:
                    all_items.append(ch)
        
        # Add AI queries
        for q in queries:
            if q.get('category', 'Unknown') in selected_cats:
                all_items.append(q)
        
        st.markdown(f"### 📋 {len(all_items)} Queries to Research")
        
        # Batch select
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
        
        # Display all items
        for item in all_items:
            if item in st.session_state.custom_headings:
                # Custom/Keyword heading
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
        
        st.caption(f"Selected: {len(all_selected)} | Perplexity Key: {'✓' if perplexity_key else '✗'}")
        
        if all_selected and perplexity_key:
            unresearched = [qid for qid in all_selected if qid not in st.session_state.research_results]
            if unresearched:
                if st.button(f"🔍 Research {len(unresearched)} Selected Queries", 
                           type="primary", use_container_width=True):
                    progress = st.progress(0)
                    status = st.empty()
                    start_time = time.time()
                    
                    for i, qid in enumerate(unresearched):
                        # Timer
                        if i > 0:
                            elapsed = time.time() - start_time
                            avg_time = elapsed / i
                            remaining = avg_time * (len(unresearched) - i)
                            mins = int(remaining // 60)
                            secs = int(remaining % 60)
                            timer = f"⏱️ ~{mins}m {secs}s remaining"
                        else:
                            timer = "⏱️ Starting..."
                        
                        # Get query text
                        if qid.startswith('custom_'):
                            custom_h = next((h for h in st.session_state.custom_headings 
                                           if h['id'] == qid), None)
                            if custom_h:
                                q_text = custom_h['query']
                                if custom_h.get('table_instruction'):
                                    q_text = f"{q_text}. Specifically: {custom_h['table_instruction']}"
                            else:
                                continue
                        else:
                            q_idx = int(qid.split('_')[1])
                            q_text = queries[q_idx]['query']
                        
                        status.text(f"{timer} | Researching: {q_text[:60]}...")
                        
                        res = call_perplexity(q_text)
                        
                        if res and 'choices' in res and len(res['choices']) > 0:
                            st.session_state.research_results[qid] = {
                                'query': q_text,
                                'result': res['choices'][0]['message']['content']
                            }
                        
                        progress.progress((i + 1) / len(unresearched))
                        time.sleep(1.5)
                    
                    elapsed_total = time.time() - start_time
                    mins_total = int(elapsed_total // 60)
                    secs_total = int(elapsed_total % 60)
                    
                    st.success(f"✅ Research Complete in {mins_total}m {secs_total}s!")
                    st.rerun()
            else:
                st.info("All selected queries already researched!")
        elif not perplexity_key:
            st.error("⚠️ Add Perplexity API key in sidebar")
        elif not all_selected:
            st.info("Select queries above to research")

with tab3:
    st.header("Step 3: Outline Structure")
    
    if not st.session_state.research_results:
        st.warning("⚠️ Complete research in Tab 2 first")
        st.stop()
    
    st.success(f"✅ {len(st.session_state.research_results)} queries researched and ready")
    
    # Generate semantic H1
    st.markdown("---")
    st.subheader("📝 Article Title (H1)")
    
    if not st.session_state.content_outline:
        if st.button("Generate Semantic H1", type="primary"):
            with st.spinner("Generating H1..."):
                h1 = generate_semantic_h1(st.session_state.focus_keyword, st.session_state.research_results)
                st.session_state.content_outline = {'article_title': h1, 'headings': []}
                st.rerun()
    
    if st.session_state.content_outline:
        # Display and allow editing of H1
        h1_edit = st.text_input("Edit H1 Title:", 
                               value=st.session_state.content_outline['article_title'],
                               key="h1_editor")
        if h1_edit != st.session_state.content_outline['article_title']:
            st.session_state.content_outline['article_title'] = h1_edit
        
        # Build H2 structure from researched queries
        st.markdown("---")
        st.subheader("📋 Content Structure - H2 Headings")
        st.info("These are your researched queries. They will become H2 headings in the article.")
        
        col_regen, _ = st.columns([1, 3])
        with col_regen:
            if st.button("🔄 Regenerate Outline", type="secondary", help="Re-optimize headings from research queries"):
                st.session_state.content_outline['headings'] = []
                st.rerun()
        
        # Convert researched queries to outline headings
        if not st.session_state.content_outline.get('headings'):
            
            with st.spinner("Optimizing headings for SEO..."):
                all_queries = [data['query'] for data in st.session_state.research_results.values()]
                optimized_map = optimize_headings_batch(all_queries)
            
            headings = []
            for qid, data in st.session_state.research_results.items():
                # Determine if it needs table or bullets
                needs_table = True  # Default
                custom_instruction = ""
                
                # Check if it's a custom heading with specific instructions
                if qid.startswith('custom_'):
                    custom_h = next((h for h in st.session_state.custom_headings if h['id'] == qid), None)
                    if custom_h:
                        needs_table = custom_h.get('content_type') == 'Table Required'
                        custom_instruction = custom_h.get('table_instruction', '')
                
                original_q = data['query']
                short_h2 = optimized_map.get(original_q, original_q)
                
                # Double check length
                if len(short_h2.split()) > 10:
                     short_h2 = " ".join(short_h2.split()[:8]).title()

                headings.append({
                    'qid': qid,
                    'h2_title': short_h2,
                    'original_query': original_q, # Store original for context
                    'needs_table': needs_table,
                    'needs_bullets': not needs_table,
                    'custom_table_instruction': custom_instruction,
                    'content_focus': f"Write about {short_h2}"
                })
            
            st.session_state.content_outline['headings'] = headings
        
        # Display outline with ability to reorder and edit
        st.markdown(f"**Total Sections:** {len(st.session_state.content_outline['headings'])}")
        
        for idx, heading in enumerate(st.session_state.content_outline['headings']):
            with st.expander(f"**H2 #{idx+1}: {heading['h2_title']}**", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Edit H2 title
                    new_title = st.text_input("H2 Title:", value=heading['h2_title'], 
                                            key=f"h2_edit_{idx}")
                    if new_title != heading['h2_title']:
                        st.session_state.content_outline['headings'][idx]['h2_title'] = new_title
                    
                    # Content focus
                    st.caption(f"**Context:** {heading.get('original_query', '')[:100]}...")
                
                with col2:
                    # Move up/down
                    col_up, col_down, col_del = st.columns(3)
                    with col_up:
                        if idx > 0 and st.button("⬆️", key=f"up_{idx}", help="Move up"):
                            headings = st.session_state.content_outline['headings']
                            headings[idx], headings[idx-1] = headings[idx-1], headings[idx]
                            st.rerun()
                    with col_down:
                        if idx < len(st.session_state.content_outline['headings'])-1 and \
                           st.button("⬇️", key=f"down_{idx}", help="Move down"):
                            headings = st.session_state.content_outline['headings']
                            headings[idx], headings[idx+1] = headings[idx+1], headings[idx]
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"remove_{idx}", help="Remove"):
                            st.session_state.content_outline['headings'].pop(idx)
                            st.rerun()
                
                # Structure display
                structure_tags = []
                if heading.get('needs_table'):
                    structure_tags.append("📊 Table")
                if heading.get('needs_bullets'):
                    structure_tags.append("📝 Bullets")
                structure_tags.append("📄 Paragraph")
                
                st.info(f"Structure: {' + '.join(structure_tags)}")
                
                if heading.get('custom_table_instruction'):
                    st.success(f"**Custom Table:** {heading['custom_table_instruction']}")
                
                # Toggle table/bullets
                col_t, col_b = st.columns(2)
                with col_t:
                    if st.checkbox("Needs Table", value=heading.get('needs_table', True), 
                                 key=f"table_{idx}"):
                        st.session_state.content_outline['headings'][idx]['needs_table'] = True
                        st.session_state.content_outline['headings'][idx]['needs_bullets'] = False
                with col_b:
                    if st.checkbox("Needs Bullets", value=heading.get('needs_bullets', False),
                                 key=f"bullets_{idx}"):
                        st.session_state.content_outline['headings'][idx]['needs_bullets'] = True
                        st.session_state.content_outline['headings'][idx]['needs_table'] = False
        
        # Preview full outline
        st.markdown("---")
        st.markdown("### 📄 Full Article Outline Preview")
        st.markdown(f"# {st.session_state.content_outline['article_title']}")
        for idx, h in enumerate(st.session_state.content_outline['headings'], 1):
            structure = "Table" if h.get('needs_table') else "Bullets"
            st.markdown(f"{idx}. **{h['h2_title']}** [{structure}]")
        
        st.markdown("---")
        st.info("✅ Outline ready! Go to Tab 4 to generate content.")

with tab4:
    st.header("Step 4: Generate Content")
    
    if not st.session_state.content_outline or not st.session_state.content_outline.get('headings'):
        st.warning("⚠️ Create outline structure in Tab 3 first")
        st.stop()
    
    st.success(f"✅ Outline ready with {len(st.session_state.content_outline['headings'])} sections")
    
    # Show outline summary
    with st.expander("📋 Article Outline", expanded=False):
        st.markdown(f"**H1:** {st.session_state.content_outline['article_title']}")
        for idx, h in enumerate(st.session_state.content_outline['headings'], 1):
            st.markdown(f"{idx}. {h['h2_title']}")
    
    if st.button("🚀 Generate Publication-Ready Article", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        start_time = time.time()
        
        # Step 1: Get latest updates
        status.text("⏱️ Checking for latest updates...")
        latest_updates = get_latest_news_updates(st.session_state.focus_keyword, st.session_state.target_country)
        st.session_state.latest_updates = latest_updates
        
        # Step 2: Prepare research context
        research_context = "\n\n".join([f"Q: {d['query']}\nA: {d['result']}" 
                                      for d in st.session_state.research_results.values()])
        
        # Step 3: Generate sections based on outline
        st.session_state.generated_sections = []
        total = len(st.session_state.content_outline['headings'])
        
        for idx, heading_data in enumerate(st.session_state.content_outline['headings']):
            # Timer
            if idx > 0:
                elapsed = time.time() - start_time
                avg = elapsed / idx
                remaining = avg * (total - idx)
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                timer = f"⏱️ ~{mins}m {secs}s remaining"
            else:
                timer = "⏱️ Starting..."
            
            status.text(f"{timer} | Writing: {heading_data['h2_title'][:50]}...")
            
            is_first = (idx == 0)
            content, _ = generate_section_content(
                heading_data, 
                research_context, 
                is_first_section=is_first,
                latest_updates=latest_updates if is_first else None
            )
            
            # Generate table if needed
            table = None
            if heading_data.get('needs_table'):
                status.text(f"{timer} | Creating table...")
                table, error = generate_intelligent_table(heading_data, research_context)
                if error and "incomplete" in error.lower():
                    status.warning(f"⚠️ Table skipped - incomplete data")
            
            st.session_state.generated_sections.append({
                'heading': heading_data,
                'content': content,
                'table': table
            })
            
            progress.progress((idx + 1) / total)
        
        # Step 4: Generate FAQs
        status.text("⏱️ Generating FAQs...")
        faqs, _ = generate_faqs(st.session_state.focus_keyword, 
                               st.session_state.paa_keywords, 
                               research_context)
        st.session_state.generated_faqs = faqs.get('faqs', []) if faqs else []
        
        # Step 5: Final coherence check
        status.text("⏱️ Final quality check...")
        h1 = st.session_state.content_outline['article_title']
        passed, message = final_coherence_check(h1, st.session_state.generated_sections, 
                                                st.session_state.generated_faqs)
        
        elapsed_total = time.time() - start_time
        mins_total = int(elapsed_total // 60)
        secs_total = int(elapsed_total % 60)
        
        if passed:
            st.success(f"✅ Article Ready in {mins_total}m {secs_total}s! {message}")
        else:
            st.warning(f"⚠️ Quality check: {message}")
        
        st.rerun()
    
    # Display generated content
    if st.session_state.generated_sections:
            st.markdown("---")
            st.markdown("## 📄 Article Preview")
            
            h1 = st.session_state.content_outline['article_title']
            st.markdown(f"# {h1}")
            
            if st.session_state.latest_updates:
                st.markdown("**Latest Updates**")
                for update in st.session_state.latest_updates:
                    st.markdown(f"• {update}")
                st.markdown("---")
            
            for section in st.session_state.generated_sections:
                st.markdown(f"## {section['heading']['h2_title']}")
                
                if section.get('content'):
                    st.markdown(section['content'])
                
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
            html = export_to_html(h1, st.session_state.generated_sections, 
                                st.session_state.generated_faqs, st.session_state.latest_updates)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📄 Download HTML", html,
                                 file_name=f"{st.session_state.focus_keyword.replace(' ', '_')}.html",
                                 mime="text/html", use_container_width=True)
            with col2:
                # Plain text
                text = f"{h1}\n\n"
                for sec in st.session_state.generated_sections:
                    text += f"{sec['heading']['h2_title']}\n\n{sec.get('content', '')}\n\n"
                st.download_button("📝 Download Text", text,
                                 file_name=f"{st.session_state.focus_keyword.replace(' ', '_')}.txt",
                                 mime="text/plain", use_container_width=True)
