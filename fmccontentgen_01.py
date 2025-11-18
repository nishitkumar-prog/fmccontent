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
if 'semantic_outline' not in st.session_state:
    st.session_state.semantic_outline = None
if 'generated_sections' not in st.session_state:
    st.session_state.generated_sections = []
if 'generated_faqs' not in st.session_state:
    st.session_state.generated_faqs = []

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
def call_perplexity(query, system_prompt="Provide comprehensive, factual information with specific data points and statistics."):
    if not perplexity_key:
        return {"error": "Missing Perplexity API key"}
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

def generate_outline_from_research(topic, research_data):
    if not model:
        return None, "Gemini not configured"
    research_summary = "\n".join([
        f"- {data['query']}: {data['result'][:200]}..."
        for data in list(research_data.values())[:15]
    ])
    prompt = f"""Based on this research, create a comprehensive article outline for: "{topic}"

RESEARCH DATA:
{research_summary}

Create 8-12 H2 headings that:
- Cover the topic comprehensively
- Are SEO-optimized (include relevant keywords)
- Follow logical flow
- Each heading should have factual data to support it

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

def integrate_keywords_into_outline(outline, keyword_combinations):
    if not model or not keyword_combinations:
        return outline, "No keywords to integrate or Gemini not configured"
    keywords_text = "\n".join([f"- {kw}" for kw in keyword_combinations[:50]])
    prompt = f"""Integrate these keywords into the article headings using semantic SEO principles:

CURRENT HEADINGS:
{json.dumps([h['h2_title'] for h in outline['headings']], indent=2)}

AVAILABLE KEYWORDS:
{keywords_text}

Rules:
- Match keywords naturally to relevant headings
- Use semantic variations
- Keep heading readable and natural
- If no keyword matches, leave heading as-is
- Return headings with integrated keywords

Return ONLY valid JSON:
{{
  "headings": [
    {{
      "original": "original heading",
      "optimized": "heading with keyword",
      "keyword_used": "keyword or null"
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
        return outline, None
    except Exception as e:
        return outline, f"Error: {e}"

def generate_section_content(heading, research_context):
    if not grok_key:
        return None, "Grok API required"
    prompt = f"""Write factual content for this heading: "{heading['h2_title']}"

CONTEXT: {heading.get('purpose', '')}
RESEARCH DATA: {research_context[:2000]}

Requirements:
- Start with 2-3 paragraph explanation (150-200 words)
- Every statement must be factual and specific
- Include statistics, data points, examples
- Write in simple, direct language
- No fluff or filler content

Return the paragraph text only."""
    messages = [{"role": "user", "content": prompt}]
    content, error = call_grok(messages, max_tokens=800, temperature=0.6)
    return content, error

def generate_data_table(heading, research_context):
    if not grok_key:
        return None, "Grok API required"
    prompt = f"""Create a comprehensive data table for: "{heading['h2_title']}"

TOPIC: {heading.get('table_topic', heading['h2_title'])}
RESEARCH: {research_context[:1500]}

Requirements:
- Table must cover 100% of the topic
- Include all relevant data points
- 5-10 rows minimum
- 3-5 columns
- All cells must have accurate data
- Use specific numbers, percentages, facts

Return ONLY valid JSON:
{{
  "table_title": "Descriptive title",
  "headers": ["Column 1", "Column 2", "Column 3"],
  "rows": [
    ["Data", "Data", "Data"],
    ["Data", "Data", "Data"]
  ]
}}"""
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=1200, temperature=0.5)
    if error:
        return None, error
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except Exception as e:
        return None, f"Parse error: {e}"

def generate_faqs(topic, paa_keywords, research_context):
    if not grok_key:
        return None, "Grok API required"
    paa_text = "\n".join([f"- {kw}" for kw in paa_keywords[:30]])
    prompt = f"""Generate comprehensive FAQs for: "{topic}"

PAA QUESTIONS (AnswerThePublic):
{paa_text}

RESEARCH DATA: {research_context[:2000]}

Create 10-15 FAQs that:
- Answer PAA questions directly
- Add related important questions
- Provide factual, specific answers (50-100 words each)
- Include data where relevant

Return ONLY valid JSON:
{{
  "faqs": [
    {{
      "question": "Question?",
      "answer": "Detailed answer with facts",
      "source": "paa or research"
    }}
  ]
}}"""
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=3000, temperature=0.6)
    if error:
        return None, error
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except Exception as e:
        return None, f"Parse error: {e}"

def export_to_html(article_title, meta_description, sections, faqs):
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
                if para:
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
                html.append(' <tr>')
                for header in headers:
                    html.append(f'  <th>{header}</th>')
                html.append(' </tr>')
                html.append(' </thead>')
            if rows:
                html.append(' <tbody>')
                for row in rows:
                    html.append(' <tr>')
                    for cell in row:
                        html.append(f'  <td>{cell}</td>')
                    html.append(' </tr>')
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
tab1, tab2, tab3, tab4 = st.tabs(["1. Research", "2. Upload Keywords", "3. Generate Outline", "4. Generate Content"])

with tab1:
    st.header("Step 1: Research Your Topic")
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "Enter your topic:", 
            placeholder="e.g., Certified Management Accounting",
            help="Main topic for your article"
        )
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
                    st.success(f"Generated {len(result['queries'])} queries")
                    st.rerun()
                else:
                    st.error(error)
    
    if st.session_state.fanout_results:
        st.markdown("---")
        queries = st.session_state.fanout_results['queries']
        st.subheader(f"Research Queries ({len(queries)})")
        
        # Select all
        all_ids = {f"q_{i}" for i in range(len(queries))}
        all_selected = all(qid in st.session_state.selected_queries for qid in all_ids)
        if st.checkbox("Select All", value=all_selected):
            st.session_state.selected_queries = all_ids
        else:
            st.session_state.selected_queries = set()
        
        for i, q in enumerate(queries):
            qid = f"q_{i}"
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    selected = st.checkbox(
                        f"**{q['query']}**",
                        value=qid in st.session_state.selected_queries,
                        key=f"cb_{qid}"
                    )
                    if selected:
                        st.session_state.selected_queries.add(qid)
                    else:
                        st.session_state.selected_queries.discard(qid)
                    st.caption(f"{q['category']} | {q.get('purpose', '')}")
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
                                        'result': res['choices'][0]['message']['content']
                                    }
                                    st.rerun()
        
        if st.session_state.selected_queries and perplexity_key:
            st.markdown("---")
            if st.button(f"Research {len(st.session_state.selected_queries)} Selected", type="secondary"):
                progress = st.progress(0)
                status = st.empty()
                selected_list = list(st.session_state.selected_queries)
                for idx, qid in enumerate(selected_list):
                    if qid not in st.session_state.research_results:
                        q_idx = int(qid.split('_')[1])
                        q = queries[q_idx]
                        status.text(f"Researching: {q['query'][:60]}...")
                        res = call_perplexity(q['query'])
                        if 'choices' in res:
                            st.session_state.research_results[qid] = {
                                'query': q['query'],
                                'result': res['choices'][0]['message']['content']
                            }
                        time.sleep(1)
                    progress.progress((idx + 1) / len(selected_list))
                status.success("Research complete!")
                time.sleep(1)
                st.rerun()

with tab2:
    st.header("Step 2: Upload Keywords (Optional)")
    st.info("Upload CSV files with keywords to optimize your content for SEO")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("AnswerThePublic Keywords (PAA)")
        st.caption("For FAQ section generation")
        paa_file = st.file_uploader(
            "Upload CSV with questions",
            type=['csv'],
            key="paa_upload",
            help="CSV should have a column with questions/queries"
        )
        if paa_file:
            try:
                df = pd.read_csv(paa_file)
                st.write(f"Loaded {len(df)} rows")
                st.write("Columns:", df.columns.tolist())
                col_name = st.selectbox("Select question column:", df.columns.tolist(), key="paa_col")
                if st.button("Load PAA Keywords"):
                    st.session_state.paa_keywords = df[col_name].dropna().tolist()
                    st.success(f"Loaded {len(st.session_state.paa_keywords)} PAA keywords")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.subheader("Keyword Combinations")
        st.caption("For heading optimization")
        kw_file = st.file_uploader(
            "Upload CSV with keywords",
            type=['csv'],
            key="kw_upload",
            help="CSV should have keywords/phrases for semantic SEO"
        )
        if kw_file:
            try:
                df = pd.read_csv(kw_file)
                st.write(f"Loaded {len(df)} rows")
                st.write("Columns:", df.columns.tolist())
                col_name = st.selectbox("Select keyword column:", df.columns.tolist(), key="kw_col")
                if st.button("Load Keywords"):
                    st.session_state.keyword_combinations = df[col_name].dropna().tolist()
                    st.success(f"Loaded {len(st.session_state.keyword_combinations)} keywords")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.paa_keywords:
            st.metric("PAA Keywords Loaded", len(st.session_state.paa_keywords))
            with st.expander("View PAA Keywords"):
                st.write(st.session_state.paa_keywords[:20])
    with col2:
        if st.session_state.keyword_combinations:
            st.metric("Keywords Loaded", len(st.session_state.keyword_combinations))
            with st.expander("View Keywords"):
                st.write(st.session_state.keyword_combinations[:20])

with tab3:
    st.header("Step 3: Generate Content Outline")
    
    if not st.session_state.research_results:
        st.warning("Complete Step 1 research first")
    else:
        st.success(f"✓ {len(st.session_state.research_results)} research queries completed")
        
        if st.button("Generate Outline from Research", type="primary", use_container_width=True):
            with st.spinner("Creating outline..."):
                outline, error = generate_outline_from_research(
                    topic,
                    st.session_state.research_results
                )
                if outline:
                    st.session_state.content_outline = outline
                    st.success("Outline generated!")
                    
                    # Integrate keywords if available
                    if st.session_state.keyword_combinations:
                        with st.spinner("Optimizing with keywords..."):
                            optimized, error = integrate_keywords_into_outline(
                                outline,
                                st.session_state.keyword_combinations
                            )
                            if not error:
                                st.session_state.content_outline = optimized
                                st.success("Outline optimized with keywords!")
                    st.rerun()
                else:
                    st.error(error)
        
        if st.session_state.content_outline:
            st.markdown("---")
            outline = st.session_state.content_outline
            
            st.subheader("Article Outline")
            st.text_input("Title (H1):", value=outline['article_title'], key="edit_title")
            st.text_area("Meta Description:", value=outline['meta_description'], height=80, key="edit_meta")
            
            st.subheader(f"Headings ({len(outline['headings'])})")
            for i, heading in enumerate(outline['headings']):
                with st.expander(f"H2: {heading['h2_title']}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text_input("Heading:", value=heading['h2_title'], key=f"h_{i}")
                        st.caption(f"Purpose: {heading.get('purpose', 'N/A')}")
                        st.caption(f"Table: {heading.get('table_topic', 'N/A')}")
                    with col2:
                        if heading.get('keyword_used'):
                            st.success(f"✓ {heading['keyword_used']}")
                        else:
                            st.caption("No keyword")

with tab4:
    st.header("Step 4: Generate Content")
    
    if not st.session_state.content_outline:
        st.warning("Complete Step 3 outline generation first")
    else:
        outline = st.session_state.content_outline
        total_sections = len(outline['headings'])
        completed = len(st.session_state.generated_sections)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.metric("Sections", f"{completed}/{total_sections}")
        with col2:
            if st.session_state.generated_faqs:
                st.metric("FAQs", len(st.session_state.generated_faqs))
        
        if st.button("Generate Full Article", type="primary", use_container_width=True):
            # Compile research context
            research_context = "\n".join([
                f"{data['query']}: {data['result'][:300]}"
                for data in st.session_state.research_results.values()
            ])
            
            # Generate sections
            progress = st.progress(0)
            status = st.empty()
            
            st.session_state.generated_sections = []
            
            for idx, heading in enumerate(outline['headings']):
                status.text(f"Generating: {heading['h2_title']}...")
                
                # Generate paragraph
                content, error = generate_section_content(heading, research_context)
                if error:
                    st.error(f"Error on {heading['h2_title']}: {error}")
                    continue
                
                # Generate table
                status.text(f"Creating table for: {heading['h2_title']}...")
                table, table_error = generate_data_table(heading, research_context)
                
                st.session_state.generated_sections.append({
                    'heading': heading,
                    'content': content,
                    'table': table if not table_error else None
                })
                
                progress.progress((idx + 1) / total_sections)
                time.sleep(1)
            
            # Generate FAQs
            if st.session_state.paa_keywords:
                status.text("Generating FAQs...")
                faqs, error = generate_faqs(
                    outline['article_title'],
                    st.session_state.paa_keywords,
                    research_context
                )
                if faqs and not error:
                    st.session_state.generated_faqs = faqs['faqs']
            
            status.success("Content generation complete!")
            time.sleep(1)
            st.rerun()
        
        # Display generated content
        if st.session_state.generated_sections:
            st.markdown("---")
            st.subheader("Generated Article")
            
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
                        df = pd.DataFrame(table['rows'], columns=table['headers'])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
            
            # FAQs
            if st.session_state.generated_faqs:
                st.markdown("## Frequently Asked Questions")
                for faq in st.session_state.generated_faqs:
                    st.markdown(f"### {faq['question']}")
                    st.markdown(faq['answer'])
                st.markdown("---")
            
            st.success(f"Total Words: {total_words:,}")
            
            # Export
            st.subheader("Download Content")
            html_content = export_to_html(
                outline['article_title'],
                outline['meta_description'],
                st.session_state.generated_sections,
                st.session_state.generated_faqs
            )
            
            st.download_button(
                "📥 Download HTML",
                data=html_content.encode('utf-8'),
                file_name=f"{outline['article_title'].replace(' ', '_')}.html",
                mime="text/html",
                use_container_width=True
            )
            
            with st.expander("Preview HTML"):
                st.code(html_content, language='html')

# Clear data
if st.sidebar.button("Clear All Data"):
    for key in ['fanout_results', 'research_results', 'selected_queries', 'content_outline',
                'paa_keywords', 'keyword_combinations', 'semantic_outline', 'generated_sections', 'generated_faqs']:
        st.session_state[key] = [] if 'keywords' in key or 'sections' in key or 'faqs' in key else None if 'outline' in key else {} if 'results' in key else set()
    st.success("Cleared!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("SEO Content Generator v2.0")
