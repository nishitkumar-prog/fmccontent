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
if 'h2_headings' not in st.session_state: st.session_state.h2_headings = []

# --- API CONFIGURATION SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
with st.sidebar.expander("🔑 API Keys", expanded=True):
    gemini_key = st.text_input("Gemini API Key", type="password")
    perplexity_key = st.text_input("Perplexity API Key", type="password")
    grok_key = st.text_input("Grok API Key", type="password")

gemini_model = st.sidebar.selectbox("Gemini Model", ["gemini-2.0-flash-exp", "gemini-2.0-flash"], index=0)

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

TASK: Write content for: "{heading['h2_title']}"
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
    
    prompt = f"""Create focused data table for: "{heading['h2_title']}"

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
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
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
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
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
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
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
tab1, tab2, tab3 = st.tabs(["1. Setup & Keywords", "2. Research & Outline", "3. Generate Content"])

with tab1:
    st.header("Step 1: Topic & Keyword Setup")
    st.info(f"📅 Content Date: **{formatted_date}**")
    
    col1, col2 = st.columns(2)
    with col1:
        focus_keyword = st.text_input(
            "Main Topic *",
            value=st.session_state.main_topic,
            placeholder="e.g., MBA Colleges India, Cloud Computing, Investment Banking"
        )
        if focus_keyword:
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
    st.header("Step 2: Research & Outline")
    
    if not st.session_state.focus_keyword:
        st.error("⚠️ Set topic in Tab 1 first")
        st.stop()
    
    st.success(f"📌 **{st.session_state.focus_keyword}** | {st.session_state.target_country}")
    
    # Convert keywords to H2 headings
    if st.session_state.keyword_planner_data and not st.session_state.h2_headings:
        if st.button("🔄 Convert Keywords to H2 Headings", type="primary"):
            st.session_state.h2_headings = []
            for kw in st.session_state.keyword_planner_data[:15]:  # Top 15
                st.session_state.h2_headings.append({
                    'h2': kw['keyword'],
                    'research_instruction': f"Find detailed data about {kw['keyword']} for {st.session_state.focus_keyword}",
                    'needs_table': True,
                    'content_type': 'Table Required'
                })
            st.success(f"✓ Created {len(st.session_state.h2_headings)} H2 headings")
            st.rerun()
    
    # Display and edit H2 headings
    if st.session_state.h2_headings:
        st.markdown("---")
        st.subheader(f"📝 {len(st.session_state.h2_headings)} H2 Headings")
        
        # Add manual heading
        with st.expander("➕ Add Custom H2"):
            new_h2 = st.text_input("H2 Heading:", placeholder="e.g., Top MBA Colleges")
            research_inst = st.text_area("Research Instruction:", 
                                        placeholder="What specific data should Perplexity collect?")
            needs_table = st.checkbox("Needs Table", value=True)
            
            if st.button("Add H2") and new_h2:
                st.session_state.h2_headings.append({
                    'h2': new_h2,
                    'research_instruction': research_inst,
                    'needs_table': needs_table,
                    'content_type': 'Table Required' if needs_table else 'Bullets'
                })
                st.success(f"✓ Added: {new_h2}")
                st.rerun()
        
        # Display headings
        for idx, h in enumerate(st.session_state.h2_headings):
            col1, col2, col3 = st.columns([3, 1, 0.5])
            with col1:
                h2_id = f"h2_{idx}"
                is_researched = h2_id in st.session_state.research_results
                status = "✅" if is_researched else "⏳"
                st.markdown(f"{status} **{h['h2']}** {'[Table]' if h.get('needs_table') else '[Bullets]'}")
                if h.get('research_instruction'):
                    st.caption(f"📋 {h['research_instruction'][:60]}...")
            with col2:
                pass
            with col3:
                if st.button("🗑️", key=f"del_{idx}"):
                    st.session_state.h2_headings.pop(idx)
                    st.rerun()
        
        # Research button
        st.markdown("---")
        unresearched = [f"h2_{i}" for i in range(len(st.session_state.h2_headings)) 
                       if f"h2_{i}" not in st.session_state.research_results]
        
        if unresearched and perplexity_key:
            if st.button(f"🔍 Research All {len(unresearched)} Headings", type="primary", use_container_width=True):
                progress = st.progress(0)
                status_text = st.empty()
                start_time = time.time()
                
                for i, h2_id in enumerate(unresearched):
                    idx = int(h2_id.split('_')[1])
                    heading = st.session_state.h2_headings[idx]
                    
                    # Estimate time remaining
                    if i > 0:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / i
                        remaining = avg_time * (len(unresearched) - i)
                        mins = int(remaining // 60)
                        secs = int(remaining % 60)
                        timer = f"⏱️ ~{mins}m {secs}s remaining"
                    else:
                        timer = "⏱️ Starting..."
                    
                    status_text.text(f"{timer} | Researching: {heading['h2'][:50]}...")
                    
                    # Build query with instruction
                    query = heading.get('research_instruction', '') or heading['h2']
                    query = f"{query}. Topic context: {st.session_state.focus_keyword}"
                    
                    res = call_perplexity(query)
                    
                    if res and 'choices' in res:
                        st.session_state.research_results[h2_id] = {
                            'query': heading['h2'],
                            'result': res['choices'][0]['message']['content']
                        }
                    
                    progress.progress((i + 1) / len(unresearched))
                    time.sleep(1.5)
                
                elapsed_total = time.time() - start_time
                mins_total = int(elapsed_total // 60)
                secs_total = int(elapsed_total % 60)
                
                st.success(f"✅ Research Complete in {mins_total}m {secs_total}s!")
                st.rerun()
        elif not perplexity_key:
            st.error("⚠️ Add Perplexity API key")

with tab3:
    st.header("Step 3: Generate Content")
    
    if not st.session_state.research_results:
        st.warning("Complete research in Tab 2 first")
    else:
        st.success(f"✅ {len(st.session_state.research_results)} headings researched")
        
        if st.button("🚀 Generate Publication-Ready Article", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            start_time = time.time()
            
            # Step 1: Generate semantic H1
            status.text("⏱️ Generating semantic H1...")
            h1 = generate_semantic_h1(st.session_state.focus_keyword, st.session_state.research_results)
            
            # Step 2: Get latest updates
            status.text("⏱️ Checking for latest updates...")
            latest_updates = get_latest_news_updates(st.session_state.focus_keyword, st.session_state.target_country)
            st.session_state.latest_updates = latest_updates
            
            # Step 3: Prepare research context
            research_context = "\n\n".join([f"Q: {d['query']}\nA: {d['result']}" 
                                          for d in st.session_state.research_results.values()])
            
            # Step 4: Generate sections
            st.session_state.generated_sections = []
            total = len(st.session_state.h2_headings)
            
            for idx, heading_data in enumerate(st.session_state.h2_headings):
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
                
                status.text(f"{timer} | Writing: {heading_data['h2']}...")
                
                # Generate content
                heading_obj = {
                    'h2_title': heading_data['h2'],
                    'needs_table': heading_data.get('needs_table', True),
                    'needs_bullets': not heading_data.get('needs_table', True)
                }
                
                is_first = (idx == 0)
                content, _ = generate_section_content(
                    heading_obj, 
                    research_context, 
                    is_first_section=is_first,
                    latest_updates=latest_updates if is_first else None
                )
                
                # Generate table if needed
                table = None
                if heading_data.get('needs_table'):
                    status.text(f"{timer} | Creating table...")
                    table, error = generate_intelligent_table(heading_obj, research_context)
                    if error and "incomplete" in error.lower():
                        status.warning(f"⚠️ Table skipped for {heading_data['h2']} - incomplete data")
                
                st.session_state.generated_sections.append({
                    'heading': heading_obj,
                    'content': content,
                    'table': table
                })
                
                progress.progress((idx + 1) / total)
            
            # Step 5: Generate FAQs
            status.text("⏱️ Generating FAQs...")
            faqs, _ = generate_faqs(st.session_state.focus_keyword, 
                                   st.session_state.paa_keywords, 
                                   research_context)
            st.session_state.generated_faqs = faqs.get('faqs', []) if faqs else []
            
            # Step 6: Final coherence check
            status.text("⏱️ Final quality check...")
            passed, message = final_coherence_check(h1, st.session_state.generated_sections, 
                                                    st.session_state.generated_faqs)
            
            elapsed_total = time.time() - start_time
            mins_total = int(elapsed_total // 60)
            secs_total = int(elapsed_total % 60)
            
            if passed:
                st.session_state.content_outline = {'article_title': h1}
                st.success(f"✅ Article Ready in {mins_total}m {secs_total}s! {message}")
            else:
                st.warning(f"⚠️ Quality check: {message}")
                st.session_state.content_outline = {'article_title': h1}
            
            st.rerun()
        
        # Display content
        if st.session_state.generated_sections and st.session_state.content_outline:
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
