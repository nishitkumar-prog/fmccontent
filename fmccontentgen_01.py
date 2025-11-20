import streamlit as st
import json

# Placeholder for Grok API key
grok_key = "your_grok_api_key_here"

def call_grok(messages, max_tokens=3000, temperature=0.2):
    # Placeholder for Grok API call
    return "{}", None

def generate_semantic_h1(focus_keyword, research_results):
    """Generate a semantic H1 title based on the focus keyword and research results using Grok"""
    if not grok_key:
        return None, "Grok API Key required"
    
    # Prepare prompt for H1 generation
    queries_text = "\n".join([data['query'] for qid, data in research_results.items()])
    prompt = f"""Generate a crisp, SEO-optimized H1 title for an article on: "{focus_keyword}"

RESEARCH QUERIES:
{queries_text}

REQUIREMENTS:
1. The title should be SHORT and ATTRACTIVE (5-12 words)
2. It should encapsulate the main theme of the research queries
3. Use the focus keyword prominently
4. Ensure the title is semantic and user-friendly

Return ONLY the title as a string."""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=3000, temperature=0.2)
    
    if error:
        return None, error
    
    try:
        if "\`\`\`" in response:
            response = response.split("\`\`\`")[1].split("\`\`\`")[0]
        return response.strip(), None
    except Exception as e:
        return None, f"Parse error: {str(e)}"

def generate_semantic_h2_headings(research_results, focus_keyword):
    """Convert long research queries into crisp, semantic H2 headings using Grok"""
    if not grok_key: 
        return None, "Grok API Key required"
    
    # Prepare all queries for transformation
    queries_list = [{'qid': qid, 'query': data['query']} 
                   for qid, data in research_results.items()]
    
    queries_text = "\n".join([f"{i+1}. {q['query']}" for i, q in enumerate(queries_list)])
    
    prompt = f"""Transform these research queries into crisp, SEO-optimized H2 headings for: "{focus_keyword}"

RESEARCH QUERIES:
{queries_text}

REQUIREMENTS:
1. Convert each long query into a SHORT, CLEAR H2 heading (3-8 words)
2. Remove question format - make it a statement/topic heading
3. Extract main keywords for SEO optimization
4. Remove duplicates - merge similar concepts into one heading
5. Make headings semantic and user-friendly

EXAMPLES:
❌ "Analyze the historical trends and current enrollment data for MBA programs in India with detailed breakdowns"
✅ "MBA Enrollment Trends and Statistics"
✅ Keywords: MBA enrollment, admission statistics, student data

❌ "What are all the specific fee components including tuition, hostel, and other charges for B.Tech programs?"
✅ "B.Tech Fee Structure and Breakdown"
✅ Keywords: B.Tech fees, tuition cost, college charges

Return ONLY valid JSON:
{{
  "headings": [
    {{
      "original_query": "long research query",
      "semantic_h2": "Short SEO Heading",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }}
  ]
}}

IMPORTANT: Keep the array length same as input. Transform ALL {len(queries_list)} queries."""
    
    messages = [{"role": "user", "content": prompt}]
    response, error = call_grok(messages, max_tokens=3000, temperature=0.2)
    
    if error:
        return None, error
    
    try:
        if "\`\`\`json" in response:
            response = response.split("\`\`\`json")[1].split("\`\`\`")[0]
        result = json.loads(response.strip())
        
        # Map back to original qids
        semantic_mapping = {}
        for i, heading_data in enumerate(result.get('headings', [])):
            if i < len(queries_list):
                qid = queries_list[i]['qid']
                semantic_mapping[qid] = {
                    'original_query': heading_data.get('original_query', queries_list[i]['query']),
                    'semantic_h2': heading_data.get('semantic_h2', queries_list[i]['query']),
                    'keywords': heading_data.get('keywords', [])
                }
        
        return semantic_mapping, None
    except Exception as e:
        return None, f"Parse error: {str(e)}"

# Streamlit app
st.title("Research Query Generator")

# Tab setup
tab1, tab2, tab3, tab4 = st.tabs(["Step 1: Input", "Step 2: Research", "Step 3: Outline Structure", "Step 4: Generate Content"])

with tab1:
    st.header("Step 1: Input")
    focus_keyword = st.text_input("Enter Focus Keyword:", key="focus_keyword")
    if focus_keyword:
        st.session_state.focus_keyword = focus_keyword

with tab2:
    st.header("Step 2: Research")
    if not st.session_state.get('focus_keyword'):
        st.warning("⚠️ Enter a focus keyword in Tab 1 first")
        st.stop()
    
    research_results = {}
    # Placeholder for research logic
    st.session_state.research_results = research_results

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
        
        st.markdown("---")
        st.subheader("🎯 Convert to Semantic H2 Headings")
        st.info("Transform long research queries into crisp, SEO-optimized H2 headings. The original queries will be kept as context for Grok to generate content.")
        
        if not st.session_state.content_outline.get('headings'):
            if st.button("✨ Generate Semantic H2 Headings", type="primary", use_container_width=True):
                with st.spinner("Converting queries to semantic headings..."):
                    semantic_mapping, error = generate_semantic_h2_headings(
                        st.session_state.research_results,
                        st.session_state.focus_keyword
                    )
                    
                    if error:
                        st.error(f"Error: {error}")
                    elif semantic_mapping:
                        # Build headings with semantic titles
                        headings = []
                        for qid, data in st.session_state.research_results.items():
                            # Get semantic version
                            semantic_data = semantic_mapping.get(qid, {})
                            semantic_h2 = semantic_data.get('semantic_h2', data['query'])
                            keywords = semantic_data.get('keywords', [])
                            original_query = data['query']
                            
                            # Determine if it needs table or bullets
                            needs_table = True  # Default
                            custom_instruction = ""
                            
                            # Check if it's a custom heading with specific instructions
                            if qid.startswith('custom_'):
                                custom_h = next((h for h in st.session_state.custom_headings if h['id'] == qid), None)
                                if custom_h:
                                    needs_table = custom_h.get('content_type') == 'Table Required'
                                    custom_instruction = custom_h.get('table_instruction', '')
                            
                            headings.append({
                                'qid': qid,
                                'h2_title': semantic_h2,  # Display semantic title
                                'original_query': original_query,  # Keep original for Grok context
                                'keywords': keywords,
                                'needs_table': needs_table,
                                'needs_bullets': not needs_table,
                                'custom_table_instruction': custom_instruction,
                                'content_focus': f"Write about {original_query}"  # Use original query for content generation
                            })
                        
                        st.session_state.content_outline['headings'] = headings
                        st.success(f"✅ Generated {len(headings)} semantic H2 headings!")
                        st.rerun()
        else:
            st.success(f"✅ {len(st.session_state.content_outline['headings'])} semantic headings ready")
            
            # Option to regenerate
            if st.button("🔄 Regenerate Semantic H2 Headings", type="secondary"):
                st.session_state.content_outline['headings'] = []
                st.rerun()
        
        # Build H2 structure from researched queries
        st.markdown("---")
        st.subheader("📋 Content Structure - H2 Headings")
        
        if st.session_state.content_outline.get('headings'):
            st.markdown(f"**Total Sections:** {len(st.session_state.content_outline['headings'])}")
            
            for idx, heading in enumerate(st.session_state.content_outline['headings']):
                with st.expander(f"**H2 #{idx+1}: {heading['h2_title']}**", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Edit H2 title (semantic version)
                        new_title = st.text_input("H2 Title (displayed):", value=heading['h2_title'], 
                                                key=f"h2_edit_{idx}")
                        if new_title != heading['h2_title']:
                            st.session_state.content_outline['headings'][idx]['h2_title'] = new_title
                        
                        # Show original query context
                        if 'original_query' in heading:
                            st.caption(f"**Original Context (for Grok):** {heading['original_query']}")
                        
                        # Show keywords
                        if 'keywords' in heading and heading['keywords']:
                            st.caption(f"🏷️ **Keywords:** {', '.join(heading['keywords'])}")
                        
                        # Content focus
                        st.caption(f"📝 Focus: {heading['content_focus']}")
                    
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
                keywords_display = f" [{', '.join(h['keywords'][:3])}]" if h.get('keywords') else ""
                st.markdown(f"{idx}. **{h['h2_title']}**{keywords_display} [{structure}]")
            
            st.markdown("---")
            st.info("✅ Outline ready! Go to Tab 4 to generate content.")

with tab4:
    st.header("Step 4: Generate Content")
    if not st.session_state.content_outline:
        st.warning("⚠️ Generate content outline in Tab 3 first")
        st.stop()
    
    st.success(f"✅ Content outline ready with {len(st.session_state.content_outline['headings'])} sections")
    # Placeholder for content generation logic

# Session state initialization
if 'research_results' not in st.session_state:
    st.session_state.research_results = {}

if 'content_outline' not in st.session_state:
    st.session_state.content_outline = {}

if 'custom_headings' not in st.session_state:
    st.session_state.custom_headings = []
