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
st.title("fmc Content Guide")

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
if 'user_intent' not in st.session_state:
    st.session_state.user_intent = ''

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
        "max_tokens": 800  # Increased from 500 for more comprehensive answers
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

def generate_content_structure_from_queries(research_data, topic, selected_queries):
    """Generate article structure directly from selected query fan-out queries"""
    if not grok_key:
        return None, "Grok API key required"
    
    # Get the actual queries from selected items
    sections_list = []
    for query_id in selected_queries:
        if query_id in research_data:
            query_text = research_data[query_id]['query']
            sections_list.append({
                "title": query_text,
                "query_id": query_id,
                "description": f"Content based on: {query_text}",
                "key_points": [],
                "needs_table": True,  # Enable tables by default
                "table_description": f"Data and comparisons for {query_text}",
                "needs_infographic": False,
                "infographic_description": "",
                "estimated_words": 600
            })
    
    if not sections_list:
        return None, "No queries selected"
    
    # Create structure
    structure = {
        "article_title": topic,
        "meta_description": f"Comprehensive guide about {topic} with detailed analysis and insights.",
        "primary_keyword": topic,
        "semantic_keywords": [topic],
        "sections": sections_list
    }
    
    return structure, None

def generate_section_content(section, research_context, topic, user_intent):
    """Generate data-driven content with emphasis on tables, bullet points, and structured data"""
    if not grok_key:
        return None, "Grok API key required"
    
    # Get research data for this section
    section_research = ""
    if 'query_id' in section and section['query_id'] in research_context:
        research_data = research_context[section['query_id']]
        if 'result' in research_data:
            result = research_data['result']
            if isinstance(result, dict) and 'choices' in result:
                section_research = result['choices'][0]['message']['content']
            elif isinstance(result, str):
                section_research = result
    
    # If no research data available, skip this section
    if not section_research or len(section_research.strip()) < 50:
        return None, "Insufficient research data for this section"
    
    prompt = f"""Create highly structured, data-driven content for this section.

ARTICLE TOPIC: {topic}
SECTION TITLE: {section['title']}

USER INTENT/PURPOSE:
{user_intent}

RESEARCH DATA:
{section_research}

CRITICAL REQUIREMENTS:
1. PRIORITIZE DATA OVER PROSE - Use tables, bullet points, and structured lists
2. Extract ALL specific data points, numbers, statistics, prices, dates, percentages from research
3. Use ONLY bullet points and numbered lists - minimize paragraph text to 1-2 sentences max per concept
4. If research data is insufficient or unclear, return "INSUFFICIENT_DATA"
5. Focus on comparisons, specifications, metrics, and actionable data
6. Use H3 subheadings (###) to organize different aspects
7. Every claim must be backed by specific data from research
8. Format for scannability - readers should get value by skimming
9. DO NOT add conclusions or summaries
10. DO NOT repeat the section title

CONTENT FORMAT PREFERENCE (in order):
- Bullet points with specific data (most important)
- Numbered steps/lists for processes
- Short 1-2 sentence explanations only when necessary
- Reserve paragraphs only for critical context (2-3 sentences max)

OUTPUT STRUCTURE:
### First Aspect
- Data point 1: [specific number/fact]
- Data point 2: [specific number/fact]
- Data point 3: [specific number/fact]

### Second Aspect
1. Step/item with specific data
2. Step/item with specific data
3. Step/item with specific data

Write content NOW (400-600 words, markdown format, prioritize bullet points):
"""

    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=1500, temperature=0.6)
    
    if error:
        return None, error
    
    # Check if content indicates insufficient data
    if response and "INSUFFICIENT_DATA" in response:
        return None, "Insufficient data to generate accurate content"
    
    return response, None

def generate_table_content(section, research_context, user_intent):
    """Generate comprehensive comparison tables with maximum data extraction"""
    if not grok_key or not section.get('needs_table'):
        return None, None
    
    # Get research data for this section
    section_research = ""
    if 'query_id' in section and section['query_id'] in research_context:
        research_data = research_context[section['query_id']]
        if 'result' in research_data:
            result = research_data['result']
            if isinstance(result, dict) and 'choices' in result:
                section_research = result['choices'][0]['message']['content']
            elif isinstance(result, str):
                section_research = result
    
    # If no research data, skip table generation
    if not section_research or len(section_research.strip()) < 50:
        return None, "Insufficient data for table generation"
    
    prompt = f"""Create a COMPREHENSIVE comparison table extracting ALL available data points from research.

SECTION: {section['title']}
USER INTENT: {user_intent}

RESEARCH DATA:
{section_research}

CRITICAL REQUIREMENTS:
1. Extract EVERY data point, statistic, price, feature, metric from the research
2. Create 4-8 columns (more columns = better)
3. Include 8-15 rows of ACTUAL data (more rows = better)
4. Use specific, accurate values - NO placeholders like "Varies", "TBD", "Contact for pricing"
5. If research lacks enough data for a proper comparison table, return "INSUFFICIENT_DATA"
6. Make it actionable - readers should be able to compare and decide
7. Include units (%, $, GB, days, etc.) with numbers
8. Sort rows logically (by price, popularity, alphabetically, etc.)

TABLE TYPES TO CONSIDER:
- Feature comparison (Product A vs B vs C)
- Pricing tiers (Plan name, Price, Features, Limits)
- Statistical data (Year, Metric 1, Metric 2, Growth %)
- Specifications (Model, Specs, Performance, Price)
- Timeline (Date, Event, Impact, Key figures)
- Process steps (Step, Action, Time, Cost)

EXAMPLE QUALITY TABLE:
{{
    "table_title": "Cloud Storage Pricing Comparison 2024",
    "headers": ["Provider", "Free Tier", "Pro Price/Month", "Storage Limit", "File Size Limit", "Collaboration Features"],
    "rows": [
        ["Google Drive", "15 GB", "$9.99", "2 TB", "5 TB", "Real-time editing, Comments"],
        ["Dropbox", "2 GB", "$11.99", "2 TB", "50 GB", "Smart Sync, Paper docs"],
        ["OneDrive", "5 GB", "$6.99", "1 TB", "250 GB", "Office integration, Vault"]
    ]
}}

JSON FORMAT:
{{
    "table_title": "Descriptive, Specific Table Title",
    "headers": ["Column 1", "Column 2", "Column 3", "Column 4", "Column 5"],
    "rows": [
        ["Specific value 1", "Specific value 2", "Specific value 3", "Specific value 4", "Specific value 5"],
        (8-15 rows with actual data)
    ]
}}

Create comprehensive table NOW (return ONLY valid JSON, extract ALL available data):
"""

    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=3000, temperature=0.4)  # Increased for bigger tables
    
    if error:
        return None, error
    
    # Check for insufficient data
    if response and "INSUFFICIENT_DATA" in response:
        return None, "Insufficient data for table"
    
    try:
        # Extract JSON
        json_match = response
        if "```json" in response:
            json_match = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_match = response.split("```")[1].split("```")[0].strip()
        
        table_data = json.loads(json_match)
        
        # Validate table has substantial data
        if not table_data.get('rows') or len(table_data['rows']) < 3:
            return None, "Table has insufficient data rows"
        
        if not table_data.get('headers') or len(table_data['headers']) < 3:
            return None, "Table has insufficient columns"
        
        return table_data, None
    except Exception as e:
        return None, f"Table generation error: {str(e)}"

def export_to_docx(title, meta_desc, sections, keywords):
    """Export article to properly formatted DOCX (no markdown symbols)"""
    if not DOCX_AVAILABLE:
        return None
    
    doc = Document()
    
    # Set document properties
    core_properties = doc.core_properties
    core_properties.title = title
    core_properties.keywords = ", ".join(keywords) if keywords else ""
    
    # Title (H1) - Only one H1 in the document
    h1 = doc.add_heading(title, level=1)
    h1.alignment = WD_ALIGN_PARAGRAPH.LEFT
    
    # Meta description
    if meta_desc:
        meta = doc.add_paragraph(meta_desc)
        meta.italic = True
        meta_format = meta.paragraph_format
        meta_format.space_after = Pt(12)
    
    doc.add_paragraph()  # Spacing
    
    # Add sections
    for section_data in sections:
        section = section_data['section']
        content = section_data['content']
        
        # Section heading (H2) - clean any markdown
        section_title_clean = section['title'].replace('**', '').replace('*', '').replace('#', '')
        h2 = doc.add_heading(section_title_clean, level=2)
        h2.alignment = WD_ALIGN_PARAGRAPH.LEFT
        
        # Section content - process and clean markdown
        content_lines = content.split('\n')
        for line in content_lines:
            line = line.strip()
            if not line:
                continue
            
            # H3 subheadings - remove ### and **
            if line.startswith('### '):
                subheading_text = line.replace('### ', '').replace('**', '').replace('*', '')
                h3 = doc.add_heading(subheading_text, level=3)
                h3.alignment = WD_ALIGN_PARAGRAPH.LEFT
            # Bullet points - remove - or *
            elif line.startswith('- ') or line.startswith('* '):
                text = line[2:].strip().replace('**', '').replace('*', '').replace('__', '').replace('_', '')
                doc.add_paragraph(text, style='List Bullet')
            # Numbered lists - remove number prefix
            elif line[0].isdigit() and line[1:3] == '. ':
                text = line.split('. ', 1)[1] if '. ' in line else line
                text = text.replace('**', '').replace('*', '').replace('__', '').replace('_', '')
                doc.add_paragraph(text, style='List Number')
            # Regular paragraph - clean all markdown
            else:
                text = line.replace('**', '').replace('*', '').replace('__', '').replace('_', '').replace('###', '').replace('##', '').replace('#', '')
                if text:  # Only add non-empty paragraphs
                    doc.add_paragraph(text)
        
        # Add table if available
        if 'table' in section_data:
            table_data = section_data['table']
            doc.add_paragraph()  # Spacing
            
            # Table title (H3) - clean markdown
            table_title_clean = table_data.get('table_title', 'Data Table').replace('**', '').replace('*', '')
            table_title = doc.add_heading(table_title_clean, level=3)
            table_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
            
            # Create table
            headers = table_data['headers']
            rows = table_data['rows']
            
            table = doc.add_table(rows=1 + len(rows), cols=len(headers))
            table.style = 'Light Grid Accent 1'
            
            # Header row - clean markdown
            header_cells = table.rows[0].cells
            for i, header in enumerate(headers):
                clean_header = str(header).replace('**', '').replace('*', '')
                header_cells[i].text = clean_header
                # Bold header
                for paragraph in header_cells[i].paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
            
            # Data rows - clean markdown
            for row_idx, row_data in enumerate(rows):
                row_cells = table.rows[row_idx + 1].cells
                for col_idx, cell_value in enumerate(row_data):
                    clean_value = str(cell_value).replace('**', '').replace('*', '')
                    row_cells[col_idx].text = clean_value
        
        # Infographic note - clean markdown
        if section.get('needs_infographic'):
            doc.add_paragraph()
            infographic_desc = section.get('infographic_description', 'Visual recommended').replace('**', '').replace('*', '')
            info_note = doc.add_paragraph(f"💡 Infographic: {infographic_desc}")
            info_note.italic = True
        
        doc.add_paragraph()  # Section spacing
    
    # Save to buffer
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def export_to_html(title, meta_desc, sections, keywords):
    """Export article to clean HTML (no html/head tags, one H1, no markdown symbols)"""
    html = f"<article>\n"
    
    # Title (H1) - Only one H1
    html += f"  <h1>{title}</h1>\n"
    
    # Meta description
    if meta_desc:
        html += f"  <p class=\"meta-description\"><em>{meta_desc}</em></p>\n"
    
    html += "\n"
    
    # Add sections
    for section_data in sections:
        section = section_data['section']
        content = section_data['content']
        
        # Section heading (H2)
        html += f"  <h2>{section['title']}</h2>\n"
        
        # Process content line by line and remove markdown formatting
        content_lines = content.split('\n')
        in_list = False
        list_type = None
        
        for line in content_lines:
            line = line.strip()
            if not line:
                if in_list:
                    html += f"  </{list_type}>\n"
                    in_list = False
                continue
            
            # H3 subheadings - remove ###
            if line.startswith('### '):
                if in_list:
                    html += f"  </{list_type}>\n"
                    in_list = False
                subheading_text = line.replace('### ', '').replace('**', '').replace('*', '')
                html += f"  <h3>{subheading_text}</h3>\n"
            # Bullet points - remove - or *
            elif line.startswith('- ') or line.startswith('* '):
                text = line[2:].strip()
                if not in_list or list_type != 'ul':
                    if in_list:
                        html += f"  </{list_type}>\n"
                    html += "  <ul>\n"
                    in_list = True
                    list_type = 'ul'
                # Clean markdown formatting from text
                text = text.replace('**', '').replace('*', '').replace('__', '').replace('_', '')
                # Convert back bold for HTML
                if '**' in line or '__' in line:
                    # Extract bold parts (between ** or __)
                    parts = []
                    temp = text
                    while '**' in temp:
                        before, rest = temp.split('**', 1)
                        if '**' in rest:
                            bold_text, after = rest.split('**', 1)
                            parts.append(before)
                            parts.append(f"<strong>{bold_text}</strong>")
                            temp = after
                        else:
                            parts.append(before)
                            parts.append(rest)
                            break
                    text = ''.join(parts) if parts else text
                html += f"    <li>{text}</li>\n"
            # Numbered lists - remove numbers and periods
            elif line[0].isdigit() and '. ' in line:
                text = line.split('. ', 1)[1] if '. ' in line else line
                if not in_list or list_type != 'ol':
                    if in_list:
                        html += f"  </{list_type}>\n"
                    html += "  <ol>\n"
                    in_list = True
                    list_type = 'ol'
                # Clean markdown formatting
                text = text.replace('**', '').replace('*', '').replace('__', '').replace('_', '')
                html += f"    <li>{text}</li>\n"
            # Regular paragraph
            else:
                if in_list:
                    html += f"  </{list_type}>\n"
                    in_list = False
                # Clean all markdown formatting
                text = line.replace('**', '').replace('*', '').replace('__', '').replace('_', '').replace('###', '').replace('##', '').replace('#', '')
                html += f"  <p>{text}</p>\n"
        
        if in_list:
            html += f"  </{list_type}>\n"
        
        # Add table if available
        if 'table' in section_data:
            table_data = section_data['table']
            html += "\n"
            
            # Table title (H3) - clean formatting
            table_title = table_data.get('table_title', 'Data Table').replace('**', '').replace('*', '')
            html += f"  <h3>{table_title}</h3>\n"
            
            # Create table
            html += "  <table>\n"
            html += "    <thead>\n      <tr>\n"
            for header in table_data['headers']:
                clean_header = str(header).replace('**', '').replace('*', '')
                html += f"        <th>{clean_header}</th>\n"
            html += "      </tr>\n    </thead>\n"
            
            html += "    <tbody>\n"
            for row in table_data['rows']:
                html += "      <tr>\n"
                for cell in row:
                    clean_cell = str(cell).replace('**', '').replace('*', '')
                    html += f"        <td>{clean_cell}</td>\n"
                html += "      </tr>\n"
            html += "    </tbody>\n"
            html += "  </table>\n"
        
        # Infographic note
        if section.get('needs_infographic'):
            infographic_desc = section.get('infographic_description', 'Visual recommended').replace('**', '').replace('*', '')
            html += f"  <p class=\"infographic-note\"><em>💡 Infographic: {infographic_desc}</em></p>\n"
        
        html += "\n"
    
    html += "</article>"
    return html

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def generate_query_fanout(topic):
    """Generate diverse research queries"""
    if not model:
        return None, "Gemini API key required"
    
    prompt = f"""Generate 15-20 diverse, specific research queries for comprehensive coverage of: "{topic}"

REQUIREMENTS:
1. Mix of query types:
   - Definitional ("What is...", "How does...")
   - Comparative ("X vs Y", "Compare...")
   - Practical ("How to...", "Steps to...")
   - Statistical ("Statistics on...", "Data about...")
   - Historical ("History of...", "Evolution of...")
   - Future-looking ("Trends in...", "Future of...")
   - Problem-solving ("Challenges in...", "Solutions for...")

2. Vary specificity (broad to narrow)
3. Target different user intents
4. Include industry-specific queries
5. Add location/region-specific queries if relevant

Return ONLY a JSON array of query strings:
["query1", "query2", "query3", ...]

Generate queries NOW:
"""

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Extract JSON
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        queries = json.loads(result)
        return queries, None
    except Exception as e:
        return None, f"Query generation error: {str(e)}"

def analyze_pdf_content(pdf_text):
    """Analyze PDF and extract key topics"""
    if not model:
        return None, "Gemini API key required"
    
    prompt = f"""Analyze this PDF content and extract 5-10 key topics/themes for research:

PDF CONTENT:
{pdf_text[:4000]}  # Limit context

Return ONLY a JSON array of topic strings:
["topic1", "topic2", "topic3", ...]

Topics should be:
- Specific and focused
- Researchable
- Actionable for content creation

Extract topics NOW:
"""

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Extract JSON
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        topics = json.loads(result)
        return topics, None
    except Exception as e:
        return None, f"PDF analysis error: {str(e)}"

# Main App Layout
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query Fan-Out", "📄 PDF Analysis", "✅ Fact Checker", "📊 Research Dashboard"])

# TAB 1: Query Fan-Out
with tab1:
    st.header("Query Fan-Out Generator")
    
    query_input = st.text_area(
        "Enter your research topic or query:",
        height=100,
        placeholder="E.g., 'AI in healthcare', 'Sustainable energy solutions', 'Remote work trends 2024'"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Generate Queries", use_container_width=True):
            if not query_input:
                st.error("Please enter a topic")
            elif not model:
                st.error("Please provide Gemini API key")
            else:
                with st.spinner("Generating diverse research queries..."):
                    queries, error = generate_query_fanout(query_input)
                    
                    if error:
                        st.error(f"Error: {error}")
                    else:
                        st.session_state.fanout_results = queries
                        st.session_state.generation_details = {
                            'topic': query_input,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'model': gemini_model
                        }
                        st.session_state.user_query = query_input
                        st.success(f"✅ Generated {len(queries)} queries!")
                        st.rerun()
    
    with col2:
        if st.session_state.fanout_results:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.fanout_results = None
                st.session_state.generation_details = None
                st.session_state.research_results = {}
                st.session_state.selected_queries = set()
                st.session_state.user_query = ''
                st.rerun()
    
    # Display Results
    if st.session_state.fanout_results:
        st.divider()
        
        # Generation info
        if st.session_state.generation_details:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Topic", st.session_state.generation_details['topic'])
            with col2:
                st.metric("Queries Generated", len(st.session_state.fanout_results))
            with col3:
                st.metric("Model Used", st.session_state.generation_details['model'])
        
        st.divider()
        st.subheader("📋 Generated Queries")
        
        # Query selection
        st.info("💡 Select queries below to perform research")
        
        # Select all / Deselect all buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Select All"):
                st.session_state.selected_queries = set(range(len(st.session_state.fanout_results)))
                st.rerun()
        with col2:
            if st.button("❌ Deselect All"):
                st.session_state.selected_queries = set()
                st.rerun()
        
        # Display queries with checkboxes
        for idx, query in enumerate(st.session_state.fanout_results):
            is_selected = idx in st.session_state.selected_queries
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                if st.checkbox("", value=is_selected, key=f"query_{idx}", label_visibility="collapsed"):
                    st.session_state.selected_queries.add(idx)
                else:
                    st.session_state.selected_queries.discard(idx)
            with col2:
                st.markdown(f"**{idx + 1}.** {query}")
        
        # Research button
        if st.session_state.selected_queries:
            st.divider()
            st.info(f"🎯 {len(st.session_state.selected_queries)} queries selected for research")
            
            if st.button("🔬 Start Research on Selected Queries", use_container_width=True):
                if not perplexity_key:
                    st.error("❌ Perplexity API key required for research")
                else:
                    # Clear previous research results
                    st.session_state.research_results = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    selected_list = sorted(list(st.session_state.selected_queries))
                    
                    for i, query_idx in enumerate(selected_list):
                        query = st.session_state.fanout_results[query_idx]
                        status_text.text(f"Researching: {query}")
                        
                        # Call Perplexity
                        result = call_perplexity(query)
                        
                        # Store result
                        st.session_state.research_results[query_idx] = {
                            'query': query,
                            'result': result,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Update progress
                        progress = (i + 1) / len(selected_list)
                        progress_bar.progress(progress)
                        
                        time.sleep(0.5)  # Rate limiting
                    
                    status_text.text("✅ Research complete!")
                    progress_bar.progress(1.0)
                    st.success(f"🎉 Completed research on {len(selected_list)} queries!")
                    time.sleep(1)
                    st.rerun()
        
        # Display Research Results Section (FIXED SECTION ADDED HERE)
        if st.session_state.research_results:
            st.divider()
            st.subheader("🔍 Research Results")
            
            st.info(f"✅ Completed {len(st.session_state.research_results)} research queries")
            
            # Show research results in expandable sections
            for query_id, research_data in st.session_state.research_results.items():
                with st.expander(f"📊 {research_data['query']}", expanded=False):
                    result = research_data['result']
                    
                    # Display result
                    if isinstance(result, dict):
                        if 'error' in result:
                            st.error(f"❌ {result['error']}")
                        elif 'choices' in result:
                            content = result['choices'][0]['message']['content']
                            st.markdown(content)
                            
                            # Show timestamp
                            st.caption(f"📅 Researched: {research_data['timestamp']}")
                        else:
                            st.json(result)
                    else:
                        st.markdown(result)
                    
                    st.caption(f"🔑 Query ID: {query_id}")
        
        # Export options
        if st.session_state.fanout_results:
            st.divider()
            st.subheader("📥 Export Options")
            
            # Prepare export data
            export_data = {
                'topic': st.session_state.generation_details['topic'],
                'timestamp': st.session_state.generation_details['timestamp'],
                'model': st.session_state.generation_details['model'],
                'queries': st.session_state.fanout_results
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    "📄 Download JSON",
                    data=json_data,
                    file_name=f"queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # CSV export
                df = pd.DataFrame({
                    'Query_Number': range(1, len(st.session_state.fanout_results) + 1),
                    'Query': st.session_state.fanout_results
                })
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📊 Download CSV",
                    data=csv_data,
                    file_name=f"queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# TAB 2: PDF Analysis
with tab2:
    st.header("PDF Content Analysis")
    
    uploaded_pdf = st.file_uploader("Upload PDF Document", type=['pdf'])
    
    if uploaded_pdf:
        if st.button("📖 Analyze PDF", use_container_width=True):
            if not model:
                st.error("Please provide Gemini API key")
            else:
                with st.spinner("Extracting and analyzing PDF content..."):
                    # Extract text
                    pdf_text = extract_text_from_pdf(uploaded_pdf)
                    
                    if pdf_text.startswith("Error"):
                        st.error(pdf_text)
                    else:
                        # Analyze content
                        topics, error = analyze_pdf_content(pdf_text)
                        
                        if error:
                            st.error(f"Analysis error: {error}")
                        else:
                            st.session_state.pdf_analysis = {
                                'filename': uploaded_pdf.name,
                                'topics': topics,
                                'text_preview': pdf_text[:1000],
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.success(f"✅ Extracted {len(topics)} key topics!")
                            st.rerun()
    
    # Display PDF Analysis
    if st.session_state.pdf_analysis:
        st.divider()
        
        analysis = st.session_state.pdf_analysis
        
        st.subheader("📄 Analysis Results")
        st.info(f"**File:** {analysis['filename']} | **Analyzed:** {analysis['timestamp']}")
        
        st.subheader("🔑 Extracted Topics")
        
        for idx, topic in enumerate(analysis['topics'], 1):
            st.markdown(f"**{idx}.** {topic}")
        
        # Generate queries button
        st.divider()
        if st.button("🚀 Generate Queries for All Topics"):
            if not model:
                st.error("Gemini API key required")
            else:
                with st.spinner("Generating queries..."):
                    all_queries = []
                    
                    for topic in analysis['topics']:
                        queries, error = generate_query_fanout(topic)
                        if not error:
                            all_queries.extend(queries)
                    
                    st.session_state.fanout_results = all_queries
                    st.session_state.generation_details = {
                        'topic': f"PDF Analysis: {analysis['filename']}",
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'model': gemini_model
                    }
                    st.success(f"✅ Generated {len(all_queries)} queries!")
                    st.info("👉 Go to 'Query Fan-Out' tab to view and research")

# TAB 3: Fact Checker
with tab3:
    st.header("Fact Checking Tool")
    
    claim_input = st.text_area(
        "Enter claim to verify:",
        height=100,
        placeholder="E.g., 'The Great Wall of China is visible from space'"
    )
    
    if st.button("🔍 Check Fact", use_container_width=True):
        if not claim_input:
            st.error("Please enter a claim")
        elif not perplexity_key:
            st.error("Perplexity API key required")
        else:
            with st.spinner("Verifying claim..."):
                system_prompt = "You are a fact-checker. Verify the claim and provide: 1) Verdict (True/False/Partially True/Unverifiable), 2) Evidence with sources, 3) Context"
                
                result = call_perplexity(claim_input, system_prompt)
                
                if 'error' in result:
                    st.error(result['error'])
                elif 'choices' in result:
                    st.divider()
                    st.subheader("🔎 Fact Check Result")
                    content = result['choices'][0]['message']['content']
                    st.markdown(content)
                else:
                    st.error("Unexpected response format")

# TAB 4: Research Dashboard
with tab4:
    st.header("📊 Research Dashboard")
    
    if not st.session_state.research_results:
        st.info("👈 Generate queries and run research in the 'Query Fan-Out' tab first")
    else:
        st.success(f"📈 {len(st.session_state.research_results)} research results available")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", len(st.session_state.fanout_results) if st.session_state.fanout_results else 0)
        with col2:
            st.metric("Researched", len(st.session_state.research_results))
        with col3:
            error_count = sum(1 for r in st.session_state.research_results.values() 
                            if isinstance(r.get('result'), dict) and 'error' in r['result'])
            st.metric("Errors", error_count)
        
        st.divider()
        
        # Research Results Display
        st.subheader("Research Results")
        
        for query_id, data in st.session_state.research_results.items():
            with st.expander(f"Query {query_id + 1}: {data['query']}", expanded=False):
                result = data['result']
                
                if isinstance(result, dict):
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                    elif 'choices' in result:
                        content = result['choices'][0]['message']['content']
                        st.markdown(content)
                        st.caption(f"Timestamp: {data['timestamp']}")
                    else:
                        st.json(result)
                else:
                    st.markdown(result)
                    st.caption(f"Timestamp: {data['timestamp']}")
        
        # Content Generation Section (MODIFIED TO USE QUERY FAN-OUT HEADINGS)
        st.divider()
        st.header("✍️ AI Content Generation")
        
        if not st.session_state.research_results:
            st.warning("⚠️ Complete research first before generating content")
        else:
            st.info("💡 Content will be generated using your selected research queries as section headings")
            
            content_topic = st.text_input(
                "Article Main Title:",
                value=st.session_state.user_query if st.session_state.user_query else "",
                placeholder="Enter the main title for your article"
            )
            
            user_intent = st.text_area(
                "User Intent / Purpose (Important):",
                height=100,
                placeholder="E.g., 'Compare pricing and features of different tools', 'Show step-by-step implementation with specific data', 'Provide statistical comparison of market leaders', etc.",
                help="Describe what the reader wants to accomplish. This will optimize content format (more tables, bullet points, data-driven)"
            )
            
            if st.button("📝 Generate Article Structure", use_container_width=True):
                if not content_topic:
                    st.error("❌ Please provide an article title")
                elif not user_intent:
                    st.error("❌ Please describe the user intent/purpose")
                elif not grok_key:
                    st.error("❌ Grok API key required")
                else:
                    with st.spinner("Creating article structure from your research queries..."):
                        structure, error = generate_content_structure_from_queries(
                            st.session_state.research_results,
                            content_topic,
                            st.session_state.selected_queries
                        )
                        
                        if error:
                            st.error(f"❌ Structure generation failed: {error}")
                        else:
                            st.session_state.content_structure = structure
                            st.session_state.user_intent = user_intent  # Store intent
                            st.success("✅ Article structure created from your queries!")
                            st.rerun()
            
            # Display and Edit Structure
            if st.session_state.content_structure:
                st.divider()
                st.subheader("📋 Article Structure (Based on Your Queries)")
                
                structure = st.session_state.content_structure
                
                # Display title and meta
                st.markdown(f"**Title (H1):** {structure['article_title']}")
                st.markdown(f"**Meta Description:** {structure['meta_description']}")
                
                st.divider()
                st.markdown("**Sections (H2 Headings):**")
                
                # Editable sections
                edited_sections = []
                
                for idx, section in enumerate(structure['sections']):
                    with st.expander(f"Section {idx + 1}: {section['title']}", expanded=True):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            section_title = st.text_input(
                                "Section Title",
                                value=section['title'],
                                key=f"section_title_{idx}",
                                label_visibility="collapsed"
                            )
                        
                        with col2:
                            needs_table = st.checkbox(
                                "Include Table",
                                value=section.get('needs_table', True),
                                key=f"table_{idx}"
                            )
                        
                        section_copy = section.copy()
                        section_copy['title'] = section_title
                        section_copy['needs_table'] = needs_table
                        edited_sections.append(section_copy)
                
                structure['sections'] = edited_sections
                st.session_state.content_structure = structure
                
                # Generate Content Button
                st.divider()
                
                if st.button("🚀 Generate Full Article", use_container_width=True, type="primary"):
                    if not grok_key:
                        st.error("❌ Grok API key required")
                    else:
                        # Clear previous content
                        st.session_state.generated_content = {}
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Get user intent from session state
                        user_intent = st.session_state.get('user_intent', 'Provide comprehensive information')
                        
                        for idx, section in enumerate(edited_sections):
                            section_key = f"section_{idx}"
                            status_text.text(f"Generating: {section['title']}")
                            
                            # Generate section content with user intent
                            content, error = generate_section_content(
                                section,
                                st.session_state.research_results,
                                content_topic,
                                user_intent
                            )
                            
                            # Skip section if insufficient data
                            if error and "Insufficient" in error:
                                status_text.text(f"⚠️ Skipping {section['title']} - insufficient data")
                                time.sleep(1)
                                continue
                            
                            if error:
                                st.error(f"❌ Error generating {section['title']}: {error}")
                                continue
                            
                            section_data = {
                                'content': content,
                                'section': section
                            }
                            
                            # Generate table if needed with user intent
                            if section.get('needs_table'):
                                table, table_error = generate_table_content(
                                    section,
                                    st.session_state.research_results,
                                    user_intent
                                )
                                if table and not table_error:
                                    section_data['table'] = table
                                elif table_error and "Insufficient" not in table_error:
                                    st.warning(f"⚠️ Could not generate table for {section['title']}")
                            
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
                article_keywords = st.session_state.content_structure.get('semantic_keywords', [])
                
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
                    # Markdown Export - clean version without markdown symbols
                    full_article = f"{st.session_state.content_structure['article_title']}\n\n"
                    full_article += f"{st.session_state.content_structure.get('meta_description', '')}\n\n"
                    full_article += "---\n\n"
                    
                    for section_key in sorted(st.session_state.generated_content.keys()):
                        section_data = st.session_state.generated_content[section_key]
                        section = section_data['section']
                        
                        # Clean section title
                        section_title = section['title'].replace('**', '').replace('*', '').replace('#', '')
                        full_article += f"{section_title}\n\n"
                        
                        # Clean content - remove all markdown symbols
                        content_clean = section_data['content']
                        content_clean = content_clean.replace('###', '').replace('##', '').replace('#', '')
                        content_clean = content_clean.replace('**', '').replace('__', '')
                        content_clean = content_clean.replace('*', '').replace('_', '')
                        full_article += f"{content_clean}\n\n"
                        
                        if 'table' in section_data:
                            table = section_data['table']
                            table_title = table.get('table_title', 'Table').replace('**', '').replace('*', '')
                            full_article += f"{table_title}\n\n"
                            headers = [str(h).replace('**', '').replace('*', '') for h in table['headers']]
                            full_article += " | ".join(headers) + "\n"
                            full_article += " | ".join(["---"] * len(headers)) + "\n"
                            for row in table['rows']:
                                clean_row = [str(cell).replace('**', '').replace('*', '') for cell in row]
                                full_article += " | ".join(clean_row) + "\n"
                            full_article += "\n"
                        
                        if section.get('needs_infographic'):
                            infographic = section.get('infographic_description', 'Visual recommended').replace('**', '').replace('*', '')
                            full_article += f"💡 Infographic: {infographic}\n\n"
                        
                        full_article += "---\n\n"
                    
                    st.download_button(
                        "📝 Download Text",
                        data=full_article.encode('utf-8'),
                        file_name=f"{content_topic.replace(' ', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                # Show format info
                with st.expander("ℹ️ Export Format Details"):
                    st.markdown("""
                    **DOCX (Word):**
                    - Full formatting with proper headings
                    - One H1 title, H2 for sections, H3 for subsections
                    - Tables properly formatted with bold headers
                    - Clean output with no markdown symbols
                    - Editable in Microsoft Word/Google Docs
                    
                    **HTML (Clean):**
                    - No html, head, or meta tags (ready for CMS)
                    - One H1 title, H2 for sections, H3 for subsections
                    - Semantic HTML structure with proper lists
                    - Tables with thead/tbody markup
                    - Clean output with no markdown symbols
                    
                    **Text:**
                    - Plain text format
                    - No formatting symbols or markdown
                    - Clean, readable content
                    - Tables in simple text format
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
    st.session_state.user_intent = ''
    st.success("All data cleared!")
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Qforia Complete Research Platform** - Query Fan-Out, PDF Analysis, Fact Checking, Research Dashboard & AI Content Generation | *Powered by Gemini AI, Perplexity & Grok*")
