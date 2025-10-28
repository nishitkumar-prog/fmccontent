import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
import time
from datetime import datetime
import PyPDF2
import io

# DOCX support for Word export
try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("⚠️ python-docx not installed. Install: pip install python-docx")

# App config
st.set_page_config(page_title="Qforia Research Platform", layout="wide")
st.title("SYU Content Guide")

# Grok API Configuration
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# Initialize session states
if 'fanout_results' not in st.session_state:
    st.session_state.fanout_results = None
if 'generation_details' not in st.session_state:
    st.session_state.generation_details = None
if 'research_results' not in st.session_state:
    st.session_state.research_results = {}
if 'selected_queries' not in st.session_state:
    st.session_state.selected_queries = set()
if 'pdf_analysis' not in st.session_state:
    st.session_state.pdf_analysis = None
if 'enhanced_topics' not in st.session_state:
    st.session_state.enhanced_topics = []
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = {}
if 'content_structure' not in st.session_state:
    st.session_state.content_structure = []
if 'user_query' not in st.session_state:
    st.session_state.user_query = ''

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password")
grok_key = st.sidebar.text_input("Grok API Key (for content generation)", type="password", help="Enter your xAI Grok API key")

# Gemini model selector with WORKING models (no models/ prefix needed)
st.sidebar.subheader("Gemini Model")
gemini_model = st.sidebar.selectbox(
    "Select Model",
    [
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ],
    index=0
)

# Configure Gemini with working model
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(gemini_model)
    st.sidebar.success(f"✅ {gemini_model}")
else:
    model = None

# Utility Functions
def call_perplexity(query, system_prompt="Provide comprehensive, actionable insights with specific data points, statistics, and practical recommendations."):
    if not perplexity_key:
        return {"error": "Missing Perplexity API key"}
    
    headers = {
        "Authorization": f"Bearer {perplexity_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "model": "sonar",  # CHEAPEST model - $0.2 per 1M tokens (vs sonar-pro at $3)
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "max_tokens": 500  # Reduced from 1000 to minimize cost
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
        return {"error": f"Perplexity API error: {e}"}

def call_grok(messages, max_tokens=4000, temperature=0.7):
    """Call Grok API for content generation"""
    if not grok_key:
        return None, "Missing Grok API key"
    
    headers = {
        "Authorization": f"Bearer {grok_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": messages,
        "model": "grok-2-1212",
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
        return None, "Unexpected response format"
    except Exception as e:
        return None, f"Grok API error: {str(e)}"

def generate_content_structure(research_data, topic):
    """Generate SEO-optimized article structure"""
    if not grok_key:
        return None, "Grok API key required"
    
    research_summary = ""
    for query_id, data in list(research_data.items())[:10]:  # Limit to 10 to reduce tokens
        research_summary += f"\n{data['query']}: {data['result'][:200]}..."
    
    prompt = f"""Create SEO-optimized article structure for: "{topic}"

RESEARCH DATA:
{research_summary}

REQUIREMENTS:
1. H1 Title: Include main keyword, make it compelling (50-60 chars)
2. Meta Description: 150-160 chars with main keyword and value proposition
3. 8-10 H2 Sections covering user search intent
4. Each section answers a specific user question
5. Include sections with tables for comparisons/data
6. Mark sections where visuals enhance understanding
7. Logical flow: Problem → Solution → Implementation → Results

USER INTENT:
- What are they trying to learn?
- What problem are they solving?
- What action will they take?

JSON FORMAT:
{{
    "article_title": "SEO-friendly H1 title",
    "meta_description": "150-160 char description",
    "primary_keyword": "main keyword",
    "semantic_keywords": ["keyword1", "keyword2", "keyword3"],
    "sections": [
        {{
            "title": "H2 Section Title with Keyword",
            "description": "What this covers",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "needs_table": true/false,
            "table_description": "What to compare",
            "needs_infographic": true/false,
            "infographic_description": "What to visualize",
            "estimated_words": 400
        }}
    ]
}}

Create structure NOW:
"""

    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=3000, temperature=0.6)
    
    if error:
        return None, error
    
    try:
        # Extract JSON from response
        json_match = response
        if "```json" in response:
            json_match = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_match = response.split("```")[1].split("```")[0]
        
        return json.loads(json_match.strip()), None
    except Exception as e:
        return None, f"Failed to parse structure: {str(e)}"

def generate_section_content(section, research_context, tone="professional"):
    """Generate detailed content for a specific section with SEO optimization"""
    if not grok_key:
        return None, "Grok API key required"
    
    prompt = f"""Write SEO-optimized content for this section.

SECTION: {section['title']}
DESCRIPTION: {section['description']}
KEY POINTS: {', '.join(section['key_points'])}
TARGET LENGTH: {section.get('estimated_words', 400)} words
TONE: {tone}

RESEARCH DATA:
{research_context[:2000]}

SEO REQUIREMENTS:
1. Use primary keyword "{section['title']}" in first paragraph
2. Include semantic keywords naturally throughout
3. Use header hierarchy properly (H2 for section, H3 for subsections)
4. Write for user intent - answer what users actually want to know
5. Include specific data, statistics, examples from research
6. Short paragraphs (2-3 sentences max)
7. Use transition words between paragraphs

STRUCTURE:
- Opening paragraph with main keyword
- 2-3 body paragraphs with semantic keywords
- Each paragraph should answer a specific user question
- Use bullet points or numbered lists where helpful
- End with actionable insight or transition

{f"CREATE TABLE: {section['table_description']}" if section.get('needs_table') else ''}

Write the complete section content now:"""

    messages = [{"role": "user", "content": prompt}]
    return call_grok(messages, max_tokens=1500, temperature=0.7)

def generate_table_content(section):
    """Generate table data for a section"""
    if not grok_key or not section.get('needs_table'):
        return None, "No table needed or Grok API not configured"
    
    prompt = f"""Create a useful comparison table for: {section['title']}

Table should show: {section.get('table_description', 'relevant comparisons')}

Return ONLY valid JSON in this format:
{{
    "table_title": "Title for the table",
    "headers": ["Column 1", "Column 2", "Column 3"],
    "rows": [
        ["Data 1", "Data 2", "Data 3"],
        ["Data 1", "Data 2", "Data 3"]
    ]
}}

Create 3-5 rows with accurate, relevant data."""

    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=800, temperature=0.5)
    
    if error:
        return None, error
    
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        return json.loads(response.strip()), None
    except Exception as e:
        return None, f"Failed to parse table: {str(e)}"

def export_to_docx(article_title, meta_description, sections_content, keywords):
    """Export article to formatted DOCX file"""
    if not DOCX_AVAILABLE:
        return None
    
    doc = Document()
    
    # Set normal style
    style = doc.styles['Normal']
    style.font.name = 'Arial'
    style.font.size = Pt(11)
    
    # Add H1 Title (ONLY ONE)
    title = doc.add_heading(article_title, level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    
    # Add meta description
    meta = doc.add_paragraph(meta_description)
    meta.italic = True
    doc.add_paragraph()
    
    # Add keywords section
    doc.add_heading('Target Keywords', level=2)
    kw_text = ', '.join([k.get('keyword', str(k)) if isinstance(k, dict) else str(k) for k in keywords[:10]])
    doc.add_paragraph(kw_text)
    doc.add_paragraph()
    
    # Add all sections (H2 level)
    for section_data in sections_content:
        section = section_data['section']
        content = section_data['content']
        
        # Add H2 section title
        doc.add_heading(section['title'], level=2)
        
        # Add content paragraphs
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Check if it's a subheading (starts with ###)
                if para.strip().startswith('###'):
                    doc.add_heading(para.replace('###', '').strip(), level=3)
                else:
                    doc.add_paragraph(para.strip())
        
        # Add table if exists
        if 'table' in section_data:
            table_info = section_data['table']
            doc.add_paragraph()
            doc.add_heading(table_info.get('table_title', 'Comparison Table'), level=3)
            
            headers = table_info.get('headers', [])
            rows = table_info.get('rows', [])
            
            if headers and rows:
                table = doc.add_table(rows=len(rows)+1, cols=len(headers))
                table.style = 'Light Grid Accent 1'
                
                # Header row
                for idx, header in enumerate(headers):
                    cell = table.rows[0].cells[idx]
                    cell.text = str(header)
                    cell.paragraphs[0].runs[0].font.bold = True
                
                # Data rows
                for row_idx, row in enumerate(rows):
                    for col_idx, cell_value in enumerate(row):
                        table.rows[row_idx+1].cells[col_idx].text = str(cell_value)
        
        # Add infographic note
        if section.get('needs_infographic'):
            doc.add_paragraph()
            note = doc.add_paragraph()
            note.add_run('💡 Infographic Suggestion: ').bold = True
            note.add_run(section.get('infographic_description', 'Visual content recommended'))
        
        doc.add_paragraph()
    
    # Save to BytesIO
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def export_to_html(article_title, meta_description, sections_content, keywords):
    """Export article to clean HTML (no <html>, <head>, <meta> tags)"""
    
    html = []
    
    # H1 Title (ONLY ONE)
    html.append(f'<h1>{article_title}</h1>')
    html.append(f'<p class="meta-description"><em>{meta_description}</em></p>')
    html.append('')
    
    # Keywords
    html.append('<h2>Target Keywords</h2>')
    kw_list = ', '.join([k.get('keyword', str(k)) if isinstance(k, dict) else str(k) for k in keywords[:10]])
    html.append(f'<p class="keywords">{kw_list}</p>')
    html.append('')
    
    # All sections (H2 level)
    for section_data in sections_content:
        section = section_data['section']
        content = section_data['content']
        
        # H2 section title
        html.append(f'<h2>{section["title"]}</h2>')
        
        # Content paragraphs
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Check for H3 subheadings
                if para.strip().startswith('###'):
                    h3_text = para.replace('###', '').strip()
                    html.append(f'<h3>{h3_text}</h3>')
                # Check for bullet points
                elif para.strip().startswith('- ') or para.strip().startswith('• '):
                    items = para.strip().split('\n')
                    html.append('<ul>')
                    for item in items:
                        clean_item = item.strip().lstrip('- ').lstrip('• ')
                        if clean_item:
                            html.append(f'  <li>{clean_item}</li>')
                    html.append('</ul>')
                # Regular paragraph
                else:
                    html.append(f'<p>{para.strip()}</p>')
        
        # Add table if exists
        if 'table' in section_data:
            table_info = section_data['table']
            html.append(f'<h3>{table_info.get("table_title", "Comparison Table")}</h3>')
            html.append('<table>')
            
            headers = table_info.get('headers', [])
            rows = table_info.get('rows', [])
            
            if headers:
                html.append('  <thead>')
                html.append('    <tr>')
                for header in headers:
                    html.append(f'      <th>{header}</th>')
                html.append('    </tr>')
                html.append('  </thead>')
            
            if rows:
                html.append('  <tbody>')
                for row in rows:
                    html.append('    <tr>')
                    for cell in row:
                        html.append(f'      <td>{cell}</td>')
                    html.append('    </tr>')
                html.append('  </tbody>')
            
            html.append('</table>')
        
        # Infographic note
        if section.get('needs_infographic'):
            html.append(f'<div class="infographic-note">')
            html.append(f'  <strong>💡 Infographic Suggestion:</strong> {section.get("infographic_description", "Visual content recommended")}')
            html.append('</div>')
        
        html.append('')
    
    return '\n'.join(html)

def extract_pdf_text(uploaded_file):
    """Extract text content from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        if len(text.strip()) < 100:
            return None, "PDF content too short or unreadable"
        
        # Limit text length for analysis
        return text[:10000], None
    except Exception as e:
        return None, f"PDF extraction error: {e}"

def analyze_pdf_content(content, filename):
    """Analyze PDF content and extract keywords and key information"""
    if not model:
        return None, "Gemini not configured"
    
    try:
        prompt = f"""
        Analyze this PDF content and extract key information with focus on specific keywords and topics:
        
        Filename: {filename}
        Content: {content}

        Provide a comprehensive analysis in JSON format with concise, specific keywords and topics:
        {{
            "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
            "main_topics": ["topic1", "topic2", "topic3"],
            "key_concepts": ["concept1", "concept2", "concept3"],
            "content_type": "research|report|article|manual|guide|presentation",
            "domain": "technology|business|science|education|healthcare|finance|other",
            "credibility_indicators": ["indicator1", "indicator2"],
            "missing_context": [
                {{"topic": "specific topic", "missing_info": "what's missing", "research_query": "targeted query"}}
            ],
            "fact_check_items": [
                {{"claim": "specific factual claim", "verification_query": "query to verify"}}
            ],
            "enhancement_opportunities": [
                {{"area": "specific area", "suggested_research": "focused research query"}}
            ],
            "summary": "Brief 2-3 sentence summary of the document"
        }}
        
        Focus on:
        - Extract specific keywords (1-3 words each), not long phrases
        - Identify concrete topics, not abstract concepts
        - Keep claims and opportunities specific and actionable
        """
        
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        # Clean JSON
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()
        
        return json.loads(json_text), None
    except Exception as e:
        return None, f"Analysis error: {e}"

def QUERY_FANOUT_PROMPT(q, mode):
    min_queries_simple = 12
    min_queries_complex = 25

    if mode == "AI Overview (simple)":
        num_queries_instruction = (
            f"Analyze the user's query: \"{q}\". For '{mode}' mode, "
            f"generate **at least {min_queries_simple} diverse queries** that cover: "
            f"basic information, comparisons, alternatives, practical considerations, and user scenarios. "
            f"Focus on queries that would provide comprehensive coverage for someone researching this topic."
        )
    else:  # AI Mode (complex)
        num_queries_instruction = (
            f"Analyze the user's query: \"{q}\". For '{mode}' mode, "
            f"generate **at least {min_queries_complex} comprehensive queries** that include: "
            f"deep analysis, market trends, technical specifications, expert opinions, case studies, "
            f"future predictions, regulatory considerations, and advanced comparisons. "
            f"Create queries suitable for exhaustive research and strategic decision-making."
        )

    return (
        f"You are an expert research strategist creating a comprehensive query fan-out for: \"{q}\"\n"
        f"Mode: {mode}\n\n"
        f"{num_queries_instruction}\n\n"
        f"Create queries across these categories (ensure good distribution):\n"
        f"1. **Core Information**: Direct answers and fundamental concepts\n"
        f"2. **Comparative Analysis**: Comparisons with alternatives and competitors\n"
        f"3. **Market Intelligence**: Trends, statistics, market dynamics\n"
        f"4. **Technical Deep-Dive**: Specifications, features, capabilities\n"
        f"5. **User Experience**: Reviews, testimonials, real-world usage\n"
        f"6. **Strategic Considerations**: Cost analysis, ROI, decision factors\n"
        f"7. **Future Outlook**: Predictions, upcoming developments\n"
        f"8. **Expert Insights**: Professional opinions, industry analysis\n\n"
        f"Return ONLY valid JSON in this exact format:\n"
        f"{{\n"
        f"  \"generation_details\": {{\n"
        f"    \"target_query_count\": <number>,\n"
        f"    \"reasoning_for_count\": \"<explanation>\",\n"
        f"    \"research_strategy\": \"<overall approach>\"\n"
        f"  }},\n"
        f"  \"expanded_queries\": [\n"
        f"    {{\n"
        f"      \"query\": \"<specific research question>\",\n"
        f"      \"category\": \"<one of the 8 categories above>\",\n"
        f"      \"priority\": \"<high/medium/low>\",\n"
        f"      \"expected_insights\": \"<what this query should reveal>\",\n"
        f"      \"research_value\": \"<why this is important for decision-making>\"\n"
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )

def generate_fanout(query, mode):
    if not model:
        st.error("Please configure Gemini API key")
        return None
        
    prompt = QUERY_FANOUT_PROMPT(query, mode)
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        # Clean JSON
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        data = json.loads(json_text)
        return data
    except Exception as e:
        st.error(f"Error generating fanout: {e}")
        return None

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Query Research", "📄 PDF Analyzer", "✅ Fact Checker", "📊 Research Dashboard", "✍️ Content Generator"])

with tab1:
    st.header("🎯 Qforia Query Fan-Out Research")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Research Query")
        user_query = st.text_area(
            "What would you like to research?", 
            value="JEE Main How to Prepare",
            height=100,
            help="Enter any topic you want to research comprehensively"
        )

    with col2:
        st.subheader("Research Settings")
        mode = st.selectbox(
            "Research Depth",
            ["AI Overview (simple)", "AI Mode (complex)"],
            help="Simple: 12+ focused queries | Complex: 25+ comprehensive queries"
        )
        
        if st.button("🚀 Generate Research Queries", type="primary", use_container_width=True):
            if not user_query.strip():
                st.warning("Please enter a research query")
            elif not gemini_key:
                st.warning("Please enter your Gemini API key")
            else:
                with st.spinner("🤖 Generating comprehensive research queries..."):
                    # Store user query for later use
                    st.session_state.user_query = user_query
                    results = generate_fanout(user_query, mode)
                    
                if results:
                    st.session_state.fanout_results = results
                    st.session_state.generation_details = results.get("generation_details", {})
                    st.session_state.selected_queries = set()
                    st.success("✅ Research queries generated successfully!")
                    st.rerun()

    # Display fanout results
    if st.session_state.fanout_results:
        st.markdown("---")
        
        # Show generation details
        details = st.session_state.generation_details
        if details:
            st.subheader("🧠 Research Strategy")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Queries", details.get('target_query_count', 'N/A'))
            with col2:
                st.metric("Generated", len(st.session_state.fanout_results.get('expanded_queries', [])))
            with col3:
                if perplexity_key:
                    st.metric("Research Ready", "✅")
                else:
                    st.metric("Research Ready", "❌ Need Perplexity Key")
            
            st.info(f"**Strategy:** {details.get('research_strategy', 'Not provided')}")

        # Interactive query selection
        st.subheader("📋 Research Queries - Select for Deep Research")
        
        queries = st.session_state.fanout_results.get('expanded_queries', [])
        if queries:
            # Category filter
            categories = list(set(q.get('category', 'Unknown') for q in queries))
            selected_categories = st.multiselect(
                "Filter by Category:", 
                categories, 
                default=categories
            )
            
            # Priority filter
            priorities = list(set(q.get('priority', 'medium') for q in queries))
            selected_priorities = st.multiselect(
                "Filter by Priority:", 
                priorities, 
                default=priorities
            )
            
            # Filter queries
            filtered_queries = [
                q for q in queries 
                if q.get('category', 'Unknown') in selected_categories 
                and q.get('priority', 'medium') in selected_priorities
            ]
            
            st.write(f"Showing {len(filtered_queries)} of {len(queries)} queries")
            
            # Display queries with selection
            for i, query_data in enumerate(filtered_queries):
                query_id = f"query_{hash(query_data['query'])}"
                
                with st.container():
                    col1, col2, col3 = st.columns([1, 6, 2])
                    
                    with col1:
                        selected = st.checkbox("Select", key=f"checkbox_{query_id}")
                        if selected:
                            st.session_state.selected_queries.add(query_id)
                        else:
                            st.session_state.selected_queries.discard(query_id)
                    
                    with col2:
                        priority_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(query_data.get('priority', 'medium'), '🟡')
                        st.markdown(f"**{priority_color} {query_data['query']}**")
                        st.caption(f"📁 {query_data.get('category', 'Unknown')} | 💡 {query_data.get('expected_insights', 'N/A')}")
                    
                    with col3:
                        if query_id in st.session_state.research_results:
                            st.success("✅ Researched")
                        elif perplexity_key:
                            if st.button("🔍 Research", key=f"research_{query_id}"):
                                with st.spinner("Researching..."):
                                    result = call_perplexity(query_data['query'])
                                    if 'choices' in result:
                                        st.session_state.research_results[query_id] = {
                                            'query': query_data['query'],
                                            'result': result['choices'][0]['message']['content'],
                                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                            'category': query_data.get('category', 'Unknown'),
                                            'priority': query_data.get('priority', 'medium')
                                        }
                                        st.rerun()
                        else:
                            st.caption("Need Perplexity Key")
                            
                    st.markdown("---")
            
            # Bulk research
            if st.session_state.selected_queries and perplexity_key:
                if st.button("🚀 Research Selected Queries", type="secondary"):
                    selected_query_data = [
                        q for q in filtered_queries 
                        if f"query_{hash(q['query'])}" in st.session_state.selected_queries
                    ]
                    
                    progress_bar = st.progress(0)
                    for i, query_data in enumerate(selected_query_data):
                        query_id = f"query_{hash(query_data['query'])}"
                        if query_id not in st.session_state.research_results:
                            result = call_perplexity(query_data['query'])
                            if 'choices' in result:
                                st.session_state.research_results[query_id] = {
                                    'query': query_data['query'],
                                    'result': result['choices'][0]['message']['content'],
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                    'category': query_data.get('category', 'Unknown'),
                                    'priority': query_data.get('priority', 'medium')
                                }
                            time.sleep(1)
                        progress_bar.progress((i + 1) / len(selected_query_data))
                    st.success("✅ Bulk research completed!")
                    st.rerun()

with tab2:
    st.header("📄 PDF Document Analyzer")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload PDF document for analysis or url pdf:", 
            type=['pdf'],
            help="Upload a PDF file to extract keywords, topics, and analyze content"
        )
        
    with col2:
        if st.button("🔍 Analyze PDF", type="primary", use_container_width=True):
            if not uploaded_file:
                st.warning("Please upload a PDF file")
            elif not gemini_key:
                st.warning("Please enter your Gemini API key")
            else:
                with st.spinner("📄 Extracting PDF content..."):
                    pdf_text, error = extract_pdf_text(uploaded_file)
                    
                if pdf_text:
                    st.success("✅ PDF content extracted successfully!")
                    
                    with st.spinner("🤖 Analyzing content and extracting keywords..."):
                        analysis, error = analyze_pdf_content(pdf_text, uploaded_file.name)
                    
                    if analysis:
                        st.session_state.pdf_analysis = analysis
                        st.success("✅ Analysis completed!")
                        st.rerun()
                    else:
                        st.error(f"Analysis failed: {error}")
                else:
                    st.error(f"PDF extraction failed: {error}")

    # Display PDF analysis results
    if st.session_state.pdf_analysis:
        st.markdown("---")
        analysis = st.session_state.pdf_analysis
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Content Type", analysis.get('content_type', 'Unknown').title())
        with col2:
            st.metric("Domain", analysis.get('domain', 'Unknown').title())
        with col3:
            st.metric("Keywords Found", len(analysis.get('keywords', [])))
        with col4:
            st.metric("Enhancement Ops", len(analysis.get('enhancement_opportunities', [])))
        
        # Document Summary
        if analysis.get('summary'):
            st.subheader("📋 Document Summary")
            st.info(analysis['summary'])
        
        # Keywords and Topics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            if analysis.get('keywords'):
                st.subheader("🔑 Keywords")
                # Display keywords as tags
                keywords_html = " ".join([f'<span style="background-color: #e1f5fe; padding: 4px 8px; margin: 2px; border-radius: 12px; font-size: 0.9em;">{keyword}</span>' for keyword in analysis['keywords']])
                st.markdown(keywords_html, unsafe_allow_html=True)
        
        with col2:
            if analysis.get('main_topics'):
                st.subheader("📝 Main Topics")
                for topic in analysis['main_topics']:
                    st.write(f"• {topic}")
        
        # Key concepts
        if analysis.get('key_concepts'):
            st.subheader("🎯 Key Concepts")
            concepts_html = " ".join([f'<span style="background-color: #f3e5f5; padding: 4px 8px; margin: 2px; border-radius: 12px; font-size: 0.9em;">{concept}</span>' for concept in analysis['key_concepts']])
            st.markdown(concepts_html, unsafe_allow_html=True)
        
        # Credibility indicators
        if analysis.get('credibility_indicators'):
            st.subheader("✅ Credibility Indicators")
            for indicator in analysis['credibility_indicators']:
                st.write(f"• {indicator}")
        
        # Missing context & enhancement opportunities
        col1, col2 = st.columns(2)
        
        with col1:
            if analysis.get('missing_context'):
                st.subheader("❓ Missing Context")
                for item in analysis['missing_context']:
                    with st.expander(f"📌 {item['topic']}"):
                        st.write(f"**Missing:** {item['missing_info']}")
                        if perplexity_key and st.button(f"Research: {item['topic']}", key=f"missing_{hash(item['topic'])}"):
                            result = call_perplexity(item['research_query'])
                            if 'choices' in result:
                                st.write("**Research Result:**")
                                st.write(result['choices'][0]['message']['content'])
        
        with col2:
            if analysis.get('enhancement_opportunities'):
                st.subheader("🚀 Enhancement Opportunities")
                for item in analysis['enhancement_opportunities']:
                    with st.expander(f"💡 {item['area']}"):
                        st.write(f"**Suggested Research:** {item['suggested_research']}")
                        if perplexity_key and st.button(f"Enhance: {item['area']}", key=f"enhance_{hash(item['area'])}"):
                            result = call_perplexity(item['suggested_research'])
                            if 'choices' in result:
                                st.write("**Enhancement Data:**")
                                st.write(result['choices'][0]['message']['content'])

with tab3:
    st.header("✅ Fact Checker & Claim Verification")
    
    # Manual fact checking
    st.subheader("🔍 Manual Fact Check")
    fact_query = st.text_input("Enter claim to verify:", placeholder="e.g., Tesla Model Y is the best-selling EV in 2024")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 Verify Fact", type="primary"):
            if fact_query and perplexity_key:
                with st.spinner("Verifying claim..."):
                    verification_prompt = f"Fact-check this claim with current data and sources: {fact_query}. Provide verification status, supporting evidence, and source citations."
                    result = call_perplexity(fact_query, verification_prompt)
                    if 'choices' in result:
                        st.write("**Verification Result:**")
                        st.write(result['choices'][0]['message']['content'])
            elif not perplexity_key:
                st.warning("Please enter Perplexity API key")
            else:
                st.warning("Please enter a claim to verify")
    
    # Auto fact-checking from PDF analysis
    if st.session_state.pdf_analysis and st.session_state.pdf_analysis.get('fact_check_items'):
        st.markdown("---")
        st.subheader("🤖 Auto-Detected Claims for Verification")
        
        for item in st.session_state.pdf_analysis['fact_check_items']:
            with st.expander(f"📋 {item['claim']}", expanded=False):
                if perplexity_key:
                    if st.button(f"Verify Claim", key=f"verify_{hash(item['claim'])}"):
                        with st.spinner("Verifying..."):
                            result = call_perplexity(item['verification_query'], "Fact-check this claim with current data, sources, and verification status.")
                            if 'choices' in result:
                                st.write("**Verification Result:**")
                                st.write(result['choices'][0]['message']['content'])
                else:
                    st.caption("Perplexity API key required for verification")

with tab4:
    st.header("📊 Research Dashboard")
    
    if st.session_state.research_results:
        # Summary metrics
        total_researched = len(st.session_state.research_results)
        categories_researched = len(set(r['category'] for r in st.session_state.research_results.values()))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Researched", total_researched)
        with col2:
            st.metric("Categories Covered", categories_researched)
        with col3:
            st.metric("Research Depth", "Comprehensive" if total_researched > 10 else "Basic")
        
        # Export research results
        research_df = pd.DataFrame([
            {
                'Query': data['query'],
                'Category': data['category'],
                'Priority': data['priority'],
                'Research Findings': data['result'],
                'Timestamp': data['timestamp']
            }
            for data in st.session_state.research_results.values()
        ])
        
        csv_data = research_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Complete Research Results",
            data=csv_data,
            file_name=f"qforia_complete_research_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        # Detailed results
        st.subheader("📋 Detailed Research Results")
        for query_id, data in st.session_state.research_results.items():
            with st.expander(f"🔍 {data['query']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("**Research Findings:**")
                    st.write(data['result'])
                with col2:
                    st.caption(f"**Category:** {data['category']}")
                    st.caption(f"**Priority:** {data['priority']}")
                    st.caption(f"**Researched:** {data['timestamp']}")
    else:
        st.info("No research results yet. Start by using the Query Research or PDF Analyzer tabs.")

with tab5:
    st.header("✍️ AI Content Generator")
    st.markdown("Generate comprehensive articles based on your research data using Grok AI")
    
    if not grok_key:
        st.warning("⚠️ Please enter your Grok API key in the sidebar to use content generation")
    else:
        # Check if there's research data available
        if not st.session_state.research_results:
            st.info("💡 **Tip:** First conduct research in the Query Research tab to gather data for content generation")
        
        # Content Generation Configuration
        st.subheader("📝 Content Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get default topic from session state if available
            default_topic = ""
            if 'user_query' in st.session_state:
                default_topic = st.session_state.user_query
            
            content_topic = st.text_input(
                "Article Topic/Title",
                value=default_topic,
                placeholder="e.g., Complete Guide to JEE Main Preparation",
                help="Main topic for your article"
            )
            
            content_description = st.text_area(
                "Article Description (optional)",
                placeholder="Describe what the article should cover...",
                height=100
            )
        
        with col2:
            content_tone = st.selectbox(
                "Content Tone",
                ["Professional", "Conversational", "Educational", "Technical", "Persuasive"],
                help="Writing style for the article"
            )
            
            target_words = st.selectbox(
                "Target Word Count",
                [1500, 2000, 2500, 3000, 4000, 5000],
                index=2,
                help="Approximate total words for the article"
            )
        
        st.divider()
        
        # Step 1: Generate Structure
        if 'content_structure' not in st.session_state or not st.session_state.content_structure:
            st.subheader("Step 1: Generate Article Structure")
            
            if st.button("🏗️ Generate Article Structure", type="primary", use_container_width=True):
                if not content_topic:
                    st.error("Please enter an article topic")
                elif not st.session_state.research_results:
                    st.warning("⚠️ No research data available. The article will be based on AI knowledge only.")
                    
                with st.spinner("🤖 Generating article structure..."):
                    structure, error = generate_content_structure(
                        st.session_state.research_results,
                        content_topic
                    )
                    
                    if error:
                        st.error(f"❌ {error}")
                    elif structure:
                        st.session_state.content_structure = structure
                        st.success("✅ Article structure generated!")
                        st.rerun()
        
        # Step 2: Review and Edit Structure
        if st.session_state.content_structure:
            st.success("✅ Article structure generated")
            
            structure = st.session_state.content_structure
            
            st.subheader("📋 Article Structure")
            
            # Display article metadata
            with st.expander("📄 Article Metadata", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    article_title = st.text_input(
                        "Article Title",
                        value=structure.get('article_title', content_topic),
                        key="edit_title"
                    )
                with col2:
                    st.metric("Total Sections", len(structure.get('sections', [])))
                
                meta_desc = st.text_area(
                    "Meta Description",
                    value=structure.get('meta_description', ''),
                    height=80,
                    key="edit_meta"
                )
            
            # Display and edit sections
            st.subheader("📑 Article Sections")
            
            edited_sections = []
            for idx, section in enumerate(structure.get('sections', [])):
                with st.expander(f"**Section {idx+1}: {section['title']}**", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        section_title = st.text_input(
                            "Section Title",
                            value=section['title'],
                            key=f"section_title_{idx}"
                        )
                        
                        section_desc = st.text_area(
                            "Description",
                            value=section.get('description', ''),
                            height=80,
                            key=f"section_desc_{idx}"
                        )
                        
                        key_points = st.text_area(
                            "Key Points",
                            value="\n".join(section.get('key_points', [])),
                            height=100,
                            key=f"section_points_{idx}"
                        )
                    
                    with col2:
                        include_section = st.checkbox(
                            "Include",
                            value=True,
                            key=f"include_{idx}"
                        )
                        
                        word_count = st.number_input(
                            "Words",
                            min_value=200,
                            max_value=1000,
                            value=section.get('estimated_words', 400),
                            step=50,
                            key=f"words_{idx}"
                        )
                        
                        needs_table = st.checkbox(
                            "📊 Table",
                            value=section.get('needs_table', False),
                            key=f"table_{idx}"
                        )
                        
                        needs_infographic = st.checkbox(
                            "📈 Infographic",
                            value=section.get('needs_infographic', False),
                            key=f"infographic_{idx}"
                        )
                    
                    if needs_table:
                        table_desc = st.text_input(
                            "Table Description",
                            value=section.get('table_description', ''),
                            key=f"table_desc_{idx}"
                        )
                    else:
                        table_desc = ""
                    
                    if needs_infographic:
                        infographic_desc = st.text_input(
                            "Infographic Description",
                            value=section.get('infographic_description', ''),
                            key=f"infographic_desc_{idx}"
                        )
                    else:
                        infographic_desc = ""
                    
                    if include_section:
                        edited_sections.append({
                            'title': section_title,
                            'description': section_desc,
                            'key_points': [p.strip() for p in key_points.split('\n') if p.strip()],
                            'estimated_words': word_count,
                            'needs_table': needs_table,
                            'table_description': table_desc,
                            'needs_infographic': needs_infographic,
                            'infographic_description': infographic_desc
                        })
            
            # Update structure with edits
            st.session_state.content_structure['sections'] = edited_sections
            st.session_state.content_structure['article_title'] = article_title
            st.session_state.content_structure['meta_description'] = meta_desc
            
            st.divider()
            
            # Step 3: Generate Content
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("🔄 Regenerate Structure", use_container_width=True):
                    st.session_state.content_structure = []
                    st.session_state.generated_content = {}
                    st.rerun()
            
            with col2:
                # Show current progress
                generated_count = len(st.session_state.generated_content)
                total_count = len(edited_sections)
                st.metric("Progress", f"{generated_count}/{total_count} sections")
            
            with col3:
                if st.button("🚀 Generate Full Article", type="primary", use_container_width=True):
                    if not edited_sections:
                        st.error("No sections to generate!")
                    else:
                        # Compile research context
                        research_context = ""
                        for data in st.session_state.research_results.values():
                            research_context += f"\n{data['query']}: {data['result'][:300]}..."
                        
                        # Generate each section
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, section in enumerate(edited_sections):
                            section_key = f"section_{idx}"
                            
                            if section_key in st.session_state.generated_content:
                                continue
                            
                            status_text.text(f"✍️ Generating: {section['title']}...")
                            
                            # Generate section content
                            content, error = generate_section_content(
                                section,
                                research_context,
                                content_tone.lower()
                            )
                            
                            if error:
                                st.error(f"❌ Error generating {section['title']}: {error}")
                                continue
                            
                            section_data = {
                                'content': content,
                                'section': section
                            }
                            
                            # Generate table if needed
                            if section.get('needs_table'):
                                table, table_error = generate_table_content(section)
                                if table and not table_error:
                                    section_data['table'] = table
                            
                            st.session_state.generated_content[section_key] = section_data
                            
                            # Update progress
                            progress = (idx + 1) / len(edited_sections)
                            progress_bar.progress(progress)
                            
                            time.sleep(1)  # Rate limiting
                        
                        status_text.text("✅ Article generation complete!")
                        progress_bar.progress(1.0)
                        st.success("🎉 Article generated successfully!")
                        time.sleep(2)
                        st.rerun()
            
            # Display Generated Content
            if st.session_state.generated_content:
                st.divider()
                st.subheader("📄 Generated Article")
                
                # Article preview
                st.markdown(f"# {st.session_state.content_structure['article_title']}")
                st.caption(f"*{st.session_state.content_structure.get('meta_description', '')}*")
                st.divider()
                
                total_words = 0
                
                for idx, section_key in enumerate(sorted(st.session_state.generated_content.keys())):
                    section_data = st.session_state.generated_content[section_key]
                    section = section_data['section']
                    content = section_data['content']
                    
                    st.markdown(f"## {section['title']}")
                    st.markdown(content)
                    
                    # Show table if generated
                    if 'table' in section_data:
                        table = section_data['table']
                        st.markdown(f"### {table.get('table_title', 'Comparison Table')}")
                        
                        # Create dataframe for display
                        df = pd.DataFrame(table['rows'], columns=table['headers'])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Show infographic suggestion
                    if section.get('needs_infographic'):
                        st.info(f"💡 **Infographic Suggestion:** {section.get('infographic_description', 'Visual representation recommended')}")
                    
                    st.divider()
                    
                    total_words += len(content.split())
                
                # Article stats
                st.success(f"📊 **Total Word Count:** {total_words:,} words")
                
                # Export options
                st.subheader("📥 Export Article")
                
                # Prepare data for export
                sections_list = []
                for section_key in sorted(st.session_state.generated_content.keys()):
                    sections_list.append(st.session_state.generated_content[section_key])
                
                # Get keywords
                article_keywords = st.session_state.get('keywords', [])
                
                # Export buttons in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # DOCX Export
                    if DOCX_AVAILABLE:
                        docx_buffer = export_to_docx(
                            st.session_state.content_structure['article_title'],
                            st.session_state.content_structure.get('meta_description', ''),
                            sections_list,
                            article_keywords
                        )
                        
                        if docx_buffer:
                            st.download_button(
                                "📄 Download DOCX",
                                data=docx_buffer,
                                file_name=f"{content_topic.replace(' ', '_')}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True
                            )
                    else:
                        st.warning("Install python-docx for DOCX export")
                
                with col2:
                    # Clean HTML Export
                    html_content = export_to_html(
                        st.session_state.content_structure['article_title'],
                        st.session_state.content_structure.get('meta_description', ''),
                        sections_list,
                        article_keywords
                    )
                    
                    st.download_button(
                        "🌐 Download HTML",
                        data=html_content.encode('utf-8'),
                        file_name=f"{content_topic.replace(' ', '_')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                
                with col3:
                    # Markdown Export (original)
                    full_article = f"# {st.session_state.content_structure['article_title']}\n\n"
                    full_article += f"*{st.session_state.content_structure.get('meta_description', '')}*\n\n"
                    full_article += "---\n\n"
                    
                    for section_key in sorted(st.session_state.generated_content.keys()):
                        section_data = st.session_state.generated_content[section_key]
                        section = section_data['section']
                        
                        full_article += f"## {section['title']}\n\n"
                        full_article += f"{section_data['content']}\n\n"
                        
                        if 'table' in section_data:
                            table = section_data['table']
                            full_article += f"### {table.get('table_title', 'Table')}\n\n"
                            headers = table['headers']
                            full_article += "| " + " | ".join(headers) + " |\n"
                            full_article += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                            for row in table['rows']:
                                full_article += "| " + " | ".join(str(cell) for cell in row) + " |\n"
                            full_article += "\n"
                        
                        if section.get('needs_infographic'):
                            full_article += f"*💡 Infographic: {section.get('infographic_description', 'Visual recommended')}*\n\n"
                        
                        full_article += "---\n\n"
                    
                    st.download_button(
                        "📝 Download Markdown",
                        data=full_article.encode('utf-8'),
                        file_name=f"{content_topic.replace(' ', '_')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                # Show format info
                with st.expander("ℹ️ Export Format Details"):
                    st.markdown("""
                    **DOCX (Word):**
                    - Full formatting (bold, italic, tables)
                    - One H1 title only
                    - H2 for sections, H3 for subsections
                    - Editable in Microsoft Word/Google Docs
                    - Tables properly formatted
                    
                    **HTML (Clean):**
                    - No html, head, or meta tags
                    - One H1 title only
                    - Ready to paste into CMS/website
                    - Semantic HTML structure
                    - Tables with proper markup
                    
                    **Markdown:**
                    - Plain text with formatting
                    - Works with any markdown editor
                    - Easy to convert to other formats
                    """)

# Clear all data button
if st.sidebar.button("🗑️ Clear All Data"):
    st.session_state.fanout_results = None
    st.session_state.generation_details = None
    st.session_state.research_results = {}
    st.session_state.selected_queries = set()
    st.session_state.pdf_analysis = None
    st.session_state.enhanced_topics = []
    st.session_state.generated_content = {}
    st.session_state.content_structure = []
    st.session_state.user_query = ''
    st.success("All data cleared!")
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Qforia Complete Research Platform** - Query Fan-Out, PDF Analysis, Fact Checking, Research Dashboard & AI Content Generation | *Powered by Gemini AI, Perplexity & Grok*")
