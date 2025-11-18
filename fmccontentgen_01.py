import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
import time
from datetime import datetime
import io

# App config
st.set_page_config(page_title="SEO Content Generator", layout="wide")
st.title("SEO Content Generator Pro")

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
if 'semantic_outline' not in st.session_state:
    st.session_state.semantic_outline = None
if 'generated_sections' not in st.session_state:
    st.session_state.generated_sections = []
if 'generated_faqs' not in st.session_state:
    st.session_state.generated_faqs = []
if 'main_topic' not in st.session_state:
    st.session_state.main_topic = ""

# Sidebar Configuration
st.sidebar.header("Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password")
grok_key = st.sidebar.text_input("Grok API Key", type="password")

# Gemini model selector
st.sidebar.subheader("Gemini Model")
gemini_model = st.sidebar.selectbox(
    "Select Model",
    ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-pro"],
    index=0
)

# Configure Gemini
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(gemini_model)
        st.sidebar.success(f"✓ {gemini_model}")
    except Exception as e:
        st.sidebar.error(f"Gemini error: {e}")
        model = None
else:
    model = None

# Utility Functions
def call_perplexity(query, system_prompt=None):
    """Enhanced Perplexity call with content-optimized prompts"""
    if not perplexity_key:
        return {"error": "Missing Perplexity API key"}
    
    # Default content-focused system prompt
    if not system_prompt:
        system_prompt = """You are a content research expert. Provide comprehensive, factual information optimized for content creation:
- Include specific data points, statistics, and numbers
- Provide factual examples and case studies
- Include current trends and market insights
- Add comparison data when relevant
- Structure information clearly for easy content extraction
- Focus on actionable, verifiable information"""
    
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
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Perplexity error: {e}"}

def call_grok(messages, max_tokens=4000, temperature=0.7):
    if not grok_key:
        return None, "Missing Grok API key"
    headers = {
        "Authorization": f"Bearer {grok_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": messages,
        "model": "grok-3",
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
        return None, f"Unexpected response: {result}"
    except Exception as e:
        return None, f"Grok error: {str(e)}"

def generate_research_queries(topic, mode="AI Overview (simple)"):
    if not model:
        return None, "Gemini not configured"
    min_queries = 12 if mode == "AI Overview (simple)" else 25
    prompt = f"""Generate {min_queries}+ research queries for: "{topic}"

Create diverse queries covering:
1. Core definitions and fundamentals
2. Key features and specifications
3. Benefits and advantages
4. Comparisons and alternatives
5. Statistics and market data
6. Best practices and implementation
7. Common challenges and solutions
8. Expert opinions and case studies
9. Future trends and predictions
10. User experience and testimonials

Return ONLY valid JSON:
{{
  "queries": [
    {{
      "query": "specific research question",
      "category": "category name",
      "priority": "high/medium/low",
      "purpose": "what this will reveal"
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
        return None, f"Error: {e}"

def generate_outline_from_research(topic, research_data, all_keywords=None):
    """Generate outline with keyword awareness"""
    if not model:
        return None, "Gemini not configured"
    research_summary = "\n".join([
        f"- {data['query']}: {data['result'][:200]}..."
        for data in list(research_data.values())[:15]
    ])
    
    keyword_context = ""
    if all_keywords:
        keyword_context = f"\nAVAILABLE KEYWORDS: {', '.join(all_keywords[:50])}"
    
    prompt = f"""Based on this research, create a comprehensive article outline for: "{topic}"

RESEARCH DATA:
{research_summary}
{keyword_context}

Create 8-12 H2 headings that:
- Cover the topic comprehensively
- Are SEO-optimized (use available keywords naturally)
- Follow logical flow
- Each heading should have factual data to support it
- Consider user search intent

Return ONLY valid JSON:
{{
  "article_title": "SEO-optimized H1 title",
  "meta_description": "150-160 char description",
  "headings": [
    {{
      "h2_title": "Heading with keyword",
      "purpose": "what this covers",
      "data_available": true/false,
      "table_topic": "what data to show in table"
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
        return None, f"Error: {e}"

def integrate_keywords_into_outline(outline, keyword_combinations, google_ads_keywords):
    """Integrate both keyword types using semantic SEO"""
    if not model:
        return outline, "Gemini not configured"
    
    all_keywords = list(set(keyword_combinations + google_ads_keywords))
    if not all_keywords:
        return outline, "No keywords to integrate"
    
    keywords_text = "\n".join([f"- {kw}" for kw in all_keywords[:80]])
    prompt = f"""Integrate these keywords into article headings using semantic SEO principles:

CURRENT HEADINGS:
{json.dumps([h['h2_title'] for h in outline['headings']], indent=2)}

AVAILABLE KEYWORDS:
{keywords_text}

Rules:
- Match keywords naturally to relevant headings
- Use semantic variations and natural language
- Prioritize high-volume keywords from Google Ads data
- Keep headings readable and engaging
- If no keyword matches, leave heading as-is
- Use long-tail keywords where appropriate

Return ONLY valid JSON:
{{
  "headings": [
    {{
      "original": "original heading",
      "optimized": "heading with keyword",
      "keyword_used": "keyword or null",
      "search_volume": "from google ads data if available"
    }}
  ]
}}"""
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        result = json.loads(json_text.strip())
        # Update outline with optimized headings
        for i, heading_update in enumerate(result['headings']):
            if i < len(outline['headings']):
                outline['headings'][i]['h2_title'] = heading_update['optimized']
                outline['headings'][i]['keyword_used'] = heading_update.get('keyword_used')
                outline['headings'][i]['search_volume'] = heading_update.get('search_volume')
        return outline, None
    except Exception as e:
        return outline, f"Error: {e}"

def generate_section_content(heading, research_context, all_keywords):
    """Generate factual content with keyword optimization"""
    if not grok_key:
        return None, "Grok API required"
    
    keyword_context = ""
    if all_keywords:
        keyword_context = f"\nSEO KEYWORDS TO INTEGRATE: {', '.join(all_keywords[:20])}"
    
    prompt = f"""Write highly factual, SEO-optimized content for: "{heading['h2_title']}"

CONTEXT: {heading.get('purpose', '')}
RESEARCH DATA: {research_context[:2500]}
{keyword_context}

Requirements:
- Write 200-300 words of factual content
- Every statement must be backed by data or research
- Include specific numbers, statistics, percentages
- Use simple, direct language (8th grade reading level)
- Naturally integrate relevant SEO keywords
- Focus on user intent and value
- No fluff, filler, or generic statements
- Start with a strong, keyword-rich opening sentence

Write the complete paragraph content now (plain text, no markdown)."""
    
    messages = [{"role": "user", "content": prompt}]
    content, error = call_grok(messages, max_tokens=1000, temperature=0.6)
    return content, error

def generate_data_table(heading, research_context, all_keywords):
    """Generate comprehensive data table covering 100% of topic"""
    if not grok_key:
        return None, "Grok API required"
    
    prompt = f"""Create a COMPREHENSIVE data table for: "{heading['h2_title']}"

TOPIC: {heading.get('table_topic', heading['h2_title'])}
RESEARCH: {research_context[:2000]}

CRITICAL Requirements:
- Table MUST cover 100% of the topic thoroughly
- Minimum 7-10 rows of data (more if needed for complete coverage)
- 3-6 columns with relevant metrics
- ALL cells must contain accurate, specific data
- Include numbers, percentages, comparisons
- Use factual data from research
- Make it comprehensive enough to stand alone

Return ONLY valid JSON:
{{
  "table_title": "Descriptive, keyword-rich title",
  "headers": ["Column 1", "Column 2", "Column 3", "Column 4"],
  "rows": [
    ["Data", "Data", "Data", "Data"],
    ["Data", "Data", "Data", "Data"],
    ["Data", "Data", "Data", "Data"]
  ]
}}

Create complete table now:"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=1500, temperature=0.5)
    if error:
        return None, error
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except Exception as e:
        return None, f"Parse error: {e}"

def generate_faqs(topic, paa_keywords, research_context, all_keywords):
    """Generate comprehensive FAQs from PAA and research"""
    if not grok_key:
        return None, "Grok API required"
    
    paa_text = "\n".join([f"- {kw}" for kw in paa_keywords[:40]])
    keyword_text = "\n".join([f"- {kw}" for kw in all_keywords[:30]])
    
    prompt = f"""Generate comprehensive FAQs for: "{topic}"

PAA QUESTIONS (Must answer these):
{paa_text}

RELEVANT KEYWORDS:
{keyword_text}

RESEARCH DATA: {research_context[:2500]}

Create 12-20 FAQs that:
- Directly answer ALL PAA questions provided
- Add related important questions users ask
- Provide factual, specific answers (60-120 words each)
- Include statistics and data where relevant
- Use natural language with keywords
- Focus on user value and clarity

Return ONLY valid JSON:
{{
  "faqs": [
    {{
      "question": "Question?",
      "answer": "Detailed factual answer with specific data",
      "source": "paa or research"
    }}
  ]
}}

Generate complete FAQ set now:"""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=4000, temperature=0.6)
    if error:
        return None, error
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except Exception as e:
        return None, f"Parse error: {e}"

def export_to_html(article_title, meta_description, sections, faqs):
    """Export clean HTML with semantic tags only"""
    html = []
    html.append(f'<h1>{article_title}</h1>')
    html.append(f'<p><em>{meta_description}</em></p>')
    html.append('')
    
    for section in sections:
        html.append(f'<h2>{section["heading"]["h2_title"]}</h2>')
        
        # Paragraph content
        if section.get('content'):
            paragraphs = section['content'].split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if para and not para.startswith('#'):
                    html.append(f'<p>{para}</p>')
        
        # Data table
        if section.get('table'):
            table = section['table']
            html.append(f'<h3>{table.get("table_title", "Data Table")}</h3>')
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
    
    # FAQs section
    if faqs:
        html.append('<h2>Frequently Asked Questions</h2>')
        for faq in faqs:
            html.append(f'<h3>{faq["question"]}</h3>')
            html.append(f'<p>{faq["answer"]}</p>')
        html.append('')
    
    return '\n'.join(html)

# Main Interface
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Research", 
    "2. Upload Keywords", 
    "3. Research Results",
    "4. Generate Outline", 
    "5. Generate Content"
])

with tab1:
    st.header("Step 1: Research Your Topic")
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "Enter your topic:", 
            placeholder="e.g., Certified Management Accounting",
            help="Main topic for your article"
        )
        if topic:
            st.session_state.main_topic = topic
    with col2:
        mode = st.selectbox("Depth", ["AI Overview (simple)", "AI Mode (complex)"])
    
    if st.button("Generate Research Queries", type="primary", use_container_width=True):
        if not topic.strip():
            st.error("Enter a topic")
        elif not gemini_key:
            st.error("Enter Gemini API key")
        else:
            with st.spinner("Generating queries..."):
                result, error = generate_research_queries(topic, mode)
                if result:
                    st.session_state.fanout_results = result
                    st.session_state.selected_queries = set()
                    st.success(f"✓ Generated {len(result['queries'])} queries")
                    st.rerun()
                else:
                    st.error(error)
    
    if st.session_state.fanout_results and 'queries' in st.session_state.fanout_results:
        st.markdown("---")
        queries = st.session_state.fanout_results['queries']
        st.subheader(f"Research Queries ({len(queries)})")
        
        # Category filter
        categories = sorted(list(set(q.get('category', 'Unknown') for q in queries)))
        selected_cats = st.multiselect("Filter by Category:", categories, default=categories)
        
        # Filter queries
        filtered = [q for q in queries if q.get('category', 'Unknown') in selected_cats]
        
        # Select all
        filtered_ids = {f"q_{queries.index(q)}" for q in filtered}
        all_selected = all(qid in st.session_state.selected_queries for qid in filtered_ids) if filtered_ids else False
        
        select_all = st.checkbox("Select All Visible", value=all_selected, key="select_all_checkbox")
        if select_all and not all_selected:
            st.session_state.selected_queries.update(filtered_ids)
            st.rerun()
        elif not select_all and all_selected:
            st.session_state.selected_queries.difference_update(filtered_ids)
            st.rerun()
        
        st.write(f"Showing {len(filtered)} of {len(queries)} queries")
        
        for q in filtered:
            qid = f"q_{queries.index(q)}"
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    is_selected = qid in st.session_state.selected_queries
                    selected = st.checkbox(
                        f"**{q['query']}**",
                        value=is_selected,
                        key=f"cb_{qid}"
                    )
                    if selected != is_selected:
                        if selected:
                            st.session_state.selected_queries.add(qid)
                        else:
                            st.session_state.selected_queries.discard(qid)
                        st.rerun()
                    st.caption(f"{q['category']} | Priority: {q.get('priority', 'medium')} | {q.get('purpose', '')}")
                with col2:
                    if qid in st.session_state.research_results:
                        st.success("✓ Done")
                    elif perplexity_key:
                        if st.button("Research", key=f"btn_{qid}"):
                            with st.spinner("Researching..."):
                                res = call_perplexity(q['query'])
                                if 'choices' in res:
                                    st.session_state.research_results[qid] = {
                                        'query': q['query'],
                                        'category': q.get('category', 'Unknown'),
                                        'result': res['choices'][0]['message']['content']
                                    }
                                    st.rerun()
                                else:
                                    st.error(f"Error: {res.get('error')}")
                    else:
                        st.caption("Need Key")
        
        if st.session_state.selected_queries and perplexity_key:
            st.markdown("---")
            selected_count = len(st.session_state.selected_queries)
            unreserached = [qid for qid in st.session_state.selected_queries if qid not in st.session_state.research_results]
            
            if st.button(f"Research {len(unreserached)} Selected Queries", type="secondary", disabled=len(unreserached)==0):
                progress = st.progress(0)
                status = st.empty()
                
                for idx, qid in enumerate(unreserached):
                    q_idx = int(qid.split('_')[1])
                    q = queries[q_idx]
                    status.text(f"Researching ({idx+1}/{len(unreserached)}): {q['query'][:60]}...")
                    res = call_perplexity(q['query'])
                    if 'choices' in res:
                        st.session_state.research_results[qid] = {
                            'query': q['query'],
                            'category': q.get('category', 'Unknown'),
                            'result': res['choices'][0]['message']['content']
                        }
                    time.sleep(1)
                    progress.progress((idx + 1) / len(unreserached))
                status.success("✓ Research complete!")
                time.sleep(1)
                st.rerun()

with tab2:
    st.header("Step 2: Upload Keywords (Optional but Recommended)")
    st.info("Upload keyword data to optimize your content for SEO and user intent")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 AnswerThePublic (PAA)")
        st.caption("For FAQ generation")
        paa_file = st.file_uploader(
            "Upload PAA CSV",
            type=['csv'],
            key="paa_upload",
            help="Questions from AnswerThePublic"
        )
        if paa_file:
            try:
                # Try different encodings
                try:
                    df = pd.read_csv(paa_file, encoding='utf-8')
                except:
                    df = pd.read_csv(paa_file, encoding='utf-16')
                
                st.write(f"Loaded {len(df)} rows")
                
                # Auto-detect question column or let user select
                if len(df.columns) == 1:
                    col_name = df.columns[0]
                else:
                    col_name = st.selectbox("Select question column:", df.columns.tolist(), key="paa_col")
                
                if st.button("Load PAA Keywords", key="load_paa"):
                    # Clean and extract questions
                    paa_list = df[col_name].dropna().astype(str).tolist()
                    # Remove topic line if present
                    paa_list = [q.strip() for q in paa_list if '?' in q or len(q.split()) > 3]
                    st.session_state.paa_keywords = paa_list
                    st.success(f"✓ Loaded {len(st.session_state.paa_keywords)} PAA keywords")
            except Exception as e:
                st.error(f"Error loading PAA: {e}")
    
    with col2:
        st.subheader("🔍 Keyword Combinations")
        st.caption("For heading optimization")
        kw_file = st.file_uploader(
            "Upload Keywords CSV",
            type=['csv'],
            key="kw_upload",
            help="Suggestions or combinations"
        )
        if kw_file:
            try:
                df = pd.read_csv(kw_file, encoding='utf-8')
                st.write(f"Loaded {len(df)} rows")
                st.write("Columns:", df.columns.tolist())
                
                # Look for 'Suggestion' column or let user choose
                if 'Suggestion' in df.columns:
                    col_name = 'Suggestion'
                else:
                    col_name = st.selectbox("Select keyword column:", df.columns.tolist(), key="kw_col")
                
                if st.button("Load Keywords", key="load_kw"):
                    kw_list = df[col_name].dropna().astype(str).tolist()
                    st.session_state.keyword_combinations = kw_list
                    st.success(f"✓ Loaded {len(st.session_state.keyword_combinations)} keywords")
            except Exception as e:
                st.error(f"Error loading keywords: {e}")
    
    with col3:
        st.subheader("📊 Google Ads Keywords")
        st.caption("For search volume data")
        gads_file = st.file_uploader(
            "Upload Google Ads CSV",
            type=['csv'],
            key="gads_upload",
            help="From Google Keyword Planner"
        )
        if gads_file:
            try:
                # Google Ads exports can be UTF-16
                try:
                    df = pd.read_csv(gads_file, encoding='utf-16', sep='\t')
                except:
                    df = pd.read_csv(gads_file, encoding='utf-8')
                
                # Skip header rows if present
                if 'Keyword' not in df.columns and len(df) > 3:
                    df = pd.read_csv(gads_file, encoding='utf-16', sep='\t', skiprows=2)
                
                st.write(f"Loaded {len(df)} keywords")
                
                if 'Keyword' in df.columns:
                    if st.button("Load Google Ads Keywords", key="load_gads"):
                        # Extract keywords with volume data
                        keywords_with_volume = []
                        for _, row in df.iterrows():
                            kw = str(row['Keyword']).strip()
                            if kw and kw != 'nan':
                                keywords_with_volume.append(kw)
                        st.session_state.google_ads_keywords = keywords_with_volume
                        st.success(f"✓ Loaded {len(st.session_state.google_ads_keywords)} Google Ads keywords")
                else:
                    st.error("Could not find 'Keyword' column")
            except Exception as e:
                st.error(f"Error loading Google Ads: {e}")
    
    st.markdown("---")
    st.subheader("Loaded Keywords Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.paa_keywords:
            st.metric("PAA Keywords", len(st.session_state.paa_keywords))
            with st.expander("View PAA"):
                for kw in st.session_state.paa_keywords[:15]:
                    st.caption(f"• {kw}")
    with col2:
        if st.session_state.keyword_combinations:
            st.metric("Keyword Combos", len(st.session_state.keyword_combinations))
            with st.expander("View Keywords"):
                for kw in st.session_state.keyword_combinations[:15]:
                    st.caption(f"• {kw}")
    with col3:
        if st.session_state.google_ads_keywords:
            st.metric("Google Ads KWs", len(st.session_state.google_ads_keywords))
            with st.expander("View Keywords"):
                for kw in st.session_state.google_ads_keywords[:15]:
                    st.caption(f"• {kw}")

with tab3:
    st.header("Step 3: Research Results")
    
    if not st.session_state.research_results:
        st.info("No research completed yet. Go to Step 1 to research your topic.")
    else:
        # Summary metrics
        total = len(st.session_state.research_results)
        categories = set(r.get('category', 'Unknown') for r in st.session_state.research_results.values())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Researched Queries", total)
        with col2:
            st.metric("Categories", len(categories))
        with col3:
            st.metric("Ready for Outline", "✓ Yes" if total >= 5 else "Need more")
        
        # Export research
        st.markdown("---")
        if st.button("📥 Export Research Data (CSV)", use_container_width=True):
            research_df = pd.DataFrame([
                {
                    'Query': data['query'],
                    'Category': data.get('category', 'Unknown'),
                    'Research Findings': data['result']
                }
                for data in st.session_state.research_results.values()
            ])
            csv_data = research_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        st.subheader("Research Findings")
        
        # Filter by category
        cat_filter = st.multiselect(
            "Filter by category:",
            sorted(list(categories)),
            default=sorted(list(categories))
        )
        
        # Display results
        for qid, data in st.session_state.research_results.items():
            if data.get('category', 'Unknown') in cat_filter:
                with st.expander(f"**{data['query']}** ({data.get('category', 'Unknown')})", expanded=False):
                    st.markdown("**Research Findings:**")
                    st.write(data['result'])
                    
                    # Word count
                    words = len(data['result'].split())
                    st.caption(f"📝 {words} words")

with tab4:
    st.header("Step 4: Generate Content Outline")
    
    if not st.session_state.research_results:
        st.warning("⚠️ Complete Step 1 research first")
    else:
        st.success(f"✓ {len(st.session_state.research_results)} research queries completed")
        
        # Compile all keywords
        all_kw = (st.session_state.keyword_combinations + 
                  st.session_state.google_ads_keywords)
        
        if all_kw:
            st.info(f"✓ {len(all_kw)} keywords loaded for SEO optimization")
        
        if st.button("Generate Outline from Research", type="primary", use_container_width=True):
            with st.spinner("Creating outline from research data..."):
                outline, error = generate_outline_from_research(
                    st.session_state.main_topic or "Article",
                    st.session_state.research_results,
                    all_kw if all_kw else None
                )
                if outline:
                    st.session_state.content_outline = outline
                    st.success("✓ Outline generated!")
                    
                    # Integrate keywords if available
                    if all_kw:
                        with st.spinner("Optimizing headings with keywords..."):
                            optimized, error = integrate_keywords_into_outline(
                                outline,
                                st.session_state.keyword_combinations,
                                st.session_state.google_ads_keywords
                            )
                            if not error:
                                st.session_state.content_outline = optimized
                                st.success("✓ Headings optimized with SEO keywords!")
                    st.rerun()
                else:
                    st.error(error)
        
        if st.session_state.content_outline:
            st.markdown("---")
            outline = st.session_state.content_outline
            
            st.subheader("Article Outline Preview")
            
            # Editable title and meta
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text_input("Title (H1):", value=outline['article_title'], key="edit_title", disabled=True)
            with col2:
                st.metric("Sections", len(outline['headings']))
            
            st.text_area("Meta Description:", value=outline['meta_description'], height=80, key="edit_meta", disabled=True)
            
            st.markdown("---")
            st.subheader(f"Content Headings ({len(outline['headings'])})")
            
            for i, heading in enumerate(outline['headings']):
                with st.expander(f"**H2 #{i+1}:** {heading['h2_title']}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Heading:** {heading['h2_title']}")
                        st.caption(f"**Purpose:** {heading.get('purpose', 'N/A')}")
                        st.caption(f"**Table Topic:** {heading.get('table_topic', 'N/A')}")
                    with col2:
                        if heading.get('keyword_used'):
                            st.success(f"✓ Keyword")
                            st.caption(heading['keyword_used'])
                        else:
                            st.info("No keyword")
                        
                        if heading.get('search_volume'):
                            st.caption(f"Vol: {heading['search_volume']}")
            
            st.info("💡 Tip: Outline is ready. Proceed to Step 5 to generate full content!")

with tab5:
    st.header("Step 5: Generate Content")
    
    if not st.session_state.content_outline:
        st.warning("⚠️ Complete Step 4 outline generation first")
    else:
        outline = st.session_state.content_outline
        total_sections = len(outline['headings'])
        completed = len(st.session_state.generated_sections)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sections", f"{completed}/{total_sections}")
        with col2:
            if st.session_state.generated_faqs:
                st.metric("FAQs", len(st.session_state.generated_faqs))
            else:
                st.metric("FAQs", "Not generated")
        with col3:
            progress_pct = int((completed / total_sections) * 100) if total_sections > 0 else 0
            st.metric("Progress", f"{progress_pct}%")
        
        if st.button("🚀 Generate Full Article", type="primary", use_container_width=True):
            # Compile all data
            research_context = "\n\n".join([
                f"**{data['query']}**\n{data['result']}"
                for data in list(st.session_state.research_results.values())[:20]
            ])
            
            all_kw = list(set(
                st.session_state.keyword_combinations + 
                st.session_state.google_ads_keywords
            ))
            
            # Generate sections
            progress = st.progress(0)
            status = st.empty()
            
            st.session_state.generated_sections = []
            
            for idx, heading in enumerate(outline['headings']):
                status.text(f"📝 Generating content: {heading['h2_title']}...")
                
                # Generate paragraph
                content, error = generate_section_content(heading, research_context, all_kw)
                if error:
                    st.error(f"Error on {heading['h2_title']}: {error}")
                    continue
                
                # Generate comprehensive table
                status.text(f"📊 Creating data table: {heading['h2_title']}...")
                table, table_error = generate_data_table(heading, research_context, all_kw)
                
                st.session_state.generated_sections.append({
                    'heading': heading,
                    'content': content,
                    'table': table if not table_error else None
                })
                
                progress.progress((idx + 1) / total_sections)
                time.sleep(1)
            
            # Generate FAQs
            if st.session_state.paa_keywords:
                status.text("❓ Generating comprehensive FAQs...")
                faqs, error = generate_faqs(
                    outline['article_title'],
                    st.session_state.paa_keywords,
                    research_context,
                    all_kw
                )
                if faqs and not error:
                    st.session_state.generated_faqs = faqs['faqs']
            
            status.success("✅ Content generation complete!")
            time.sleep(1)
            st.rerun()
        
        # Display generated content
        if st.session_state.generated_sections:
            st.markdown("---")
            st.subheader("📄 Generated Article Preview")
            
            st.markdown(f"# {outline['article_title']}")
            st.caption(f"*{outline['meta_description']}*")
            st.markdown("---")
            
            total_words = 0
            
            for section in st.session_state.generated_sections:
                st.markdown(f"## {section['heading']['h2_title']}")
                
                if section['content']:
                    st.markdown(section['content'])
                    total_words += len(section['content'].split())
                
                if section.get('table'):
                    table = section['table']
                    st.markdown(f"### {table.get('table_title', 'Data Table')}")
                    if table.get('rows') and table.get('headers'):
                        try:
                            df = pd.DataFrame(table['rows'], columns=table['headers'])
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        except:
                            st.warning("Table format issue")
                
                st.markdown("---")
            
            # FAQs
            if st.session_state.generated_faqs:
                st.markdown("## Frequently Asked Questions")
                for faq in st.session_state.generated_faqs:
                    st.markdown(f"### {faq['question']}")
                    st.markdown(faq['answer'])
                st.markdown("---")
            
            st.success(f"📊 **Total Word Count:** {total_words:,} words")
            
            # Export section
            st.markdown("---")
            st.subheader("📥 Download Content")
            
            html_content = export_to_html(
                outline['article_title'],
                outline['meta_description'],
                st.session_state.generated_sections,
                st.session_state.generated_faqs
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download HTML (Clean)",
                    data=html_content.encode('utf-8'),
                    file_name=f"{outline['article_title'][:50].replace(' ', '_')}.html",
                    mime="text/html",
                    use_container_width=True
                )
            with col2:
                with st.expander("Preview HTML Code"):
                    st.code(html_content[:2000] + "..." if len(html_content) > 2000 else html_content, language='html')
            
            st.info("✅ HTML includes: `<h1>`, `<h2>`, `<h3>`, `<p>`, `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<td>`, `<th>` - Ready for CMS/WordPress!")

# Sidebar tools
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Data", use_container_width=True):
    for key in list(st.session_state.keys()):
        if key not in ['gemini_key', 'perplexity_key', 'grok_key']:
            del st.session_state[key]
    st.success("✅ All data cleared!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("**SEO Content Generator Pro v2.5**")
st.sidebar.caption("Research → Keywords → Outline → Content")
