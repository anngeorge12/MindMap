import os
import sys
import streamlit as st
import nltk
import streamlit.components.v1 as components
from collections import Counter
import random
import wikipedia
import re

# Download NLTK punkt tokenizer if not already present
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Setup for local module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Custom modules
from utils.preprocess import extract_text_from_pdf, chunk_text, get_document_stats, estimate_document_size
from models.summarizer import summarize_chunks, create_document_summary
from models.relations_extract import extract_relations, extract_relations_batch
from pipeline.concept_graph import parse_triplets, build_graph, visualize_graph, get_layout_options, get_learning_path_mermaid

# Streamlit UI config
st.set_page_config(page_title="MindSketch", layout="wide")
st.markdown(
    """
    <style>
    body, .main, .block-container {
        background: linear-gradient(120deg, #f9fafb 0%, #e0f2fe 100%) !important;
    }
    .main-title {
        font-size:2.5em; font-weight:700; color:#2563eb; margin-bottom:0.2em; letter-spacing:1px;
    }
    .section-title {
        font-size:1.5em; font-weight:600; color:#34d399; margin-top:2em;
    }
    .card {
        background: #fff;
        border-radius: 18px;
        padding: 32px 36px 28px 36px;
        margin-bottom: 28px;
        box-shadow: 0 6px 32px #2563eb18, 0 2px 8px #34d39910;
        border: 1.5px solid #e0f2fe;
        transition: box-shadow 0.2s, transform 0.1s;
    }
    .card:hover {
        box-shadow: 0 12px 48px #38bdf833, 0 4px 16px #fbbf2440;
        transform: scale(1.01);
        border: 1.5px solid #38bdf8;
    }
    .divider {
        height:3px;
        background:linear-gradient(90deg,#2563eb,#34d399,#fbbf24,#f472b6,#38bdf8);
        border-radius:2px;
        margin:36px 0;
    }
    .metric {
        background:#e0f2fe;
        border-radius:8px;
        padding:12px 0;
        color:#2563eb;
        font-weight:600;
    }
    .stButton>button {
        background: linear-gradient(90deg,#2563eb,#34d399);
        color: #fff;
        border: none;
        border-radius: 12px;
        padding: 0.8em 2em;
        font-size: 1.15em;
        font-weight: 700;
        box-shadow: 0 4px 16px #2563eb22, 0 1.5px 6px #34d39920;
        transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
        margin: 0.5em 0.5em 0.5em 0;
        letter-spacing: 0.5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg,#fbbf24,#2563eb,#34d399);
        box-shadow: 0 8px 32px #fbbf2440, 0 2px 8px #38bdf830;
        transform: translateY(-2px) scale(1.04);
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<div class="main-title">🧠 MindSketch – AI Concept Map Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">📄 Document Upload & Stats</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

# Check for Groq API key
groq_available = os.getenv("GROQ_API_KEY") is not None
if groq_available:
    st.success("✅ Groq API detected - Using enhanced AI models!")
else:
    st.warning("⚠️ GROQ_API_KEY not found. Using fallback models (BART + REBEL).")

# Layout selection
st.sidebar.header("🎨 Visualization Options")
layout_options = get_layout_options()
selected_layout = st.sidebar.selectbox(
    "Choose Graph Layout:",
    options=list(layout_options.keys()),
    format_func=lambda x: layout_options[x],
    index=0
)

# Search functionality
st.sidebar.header("🔍 Search & Filter")

# Add state to track view mode
if 'show_filtered' not in st.session_state:
    st.session_state.show_filtered = False
if 'regenerate_graph' not in st.session_state:
    st.session_state.regenerate_graph = False

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    search_term = st.text_input(
        "Search for concepts:",
        placeholder="Type to search...",
        help="Search for specific concepts in the graph"
    )
with col2:
    if st.button("Clear", help="Clear search"):
        search_term = ""
        st.session_state.show_filtered = False
        st.session_state.regenerate_graph = False
        st.rerun()

# Add toggle button for view mode
if search_term:
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("🔍 Filter View", help="Show only matching concepts"):
            st.session_state.show_filtered = True
            st.session_state.regenerate_graph = True
            st.rerun()
    with col2:
        if st.button("🌐 Show All", help="Show complete concept map"):
            st.session_state.show_filtered = False
            st.session_state.regenerate_graph = True
            st.rerun()
    
    # Show regenerate button if view changed
    if st.session_state.regenerate_graph:
        st.sidebar.warning("⚠️ View mode changed!")
        if st.sidebar.button("🔄 Regenerate Graph", help="Regenerate graph with new view settings"):
            st.session_state.regenerate_graph = False
            st.rerun()

# Show search results
if search_term and 'final_triplets' in locals():
    matching_nodes = []
    search_words = search_term.lower().split()  # Split into individual words
    
    for triplet in final_triplets:
        subject, relation, obj = triplet
        subject_words = subject.lower().split()
        obj_words = obj.lower().split()
        
        # Check if any search word exactly matches any word in subject or object
        for search_word in search_words:
            if (search_word in subject_words or search_word in obj_words):
                matching_nodes.extend([subject, obj])
                break  # Found a match, no need to check other search words
    
    matching_nodes = list(set(matching_nodes))  # Remove duplicates
    
    if matching_nodes:
        st.sidebar.success(f"Found {len(matching_nodes)} matching concepts!")
        
        if st.session_state.show_filtered:
            st.sidebar.info("💡 Graph view is filtered to show only matching concepts and their connections")
        else:
            st.sidebar.info("🌐 Showing complete concept map (all concepts)")
        
        st.sidebar.write("**Matching concepts:**")
        for node in matching_nodes[:5]:  # Show first 5
            st.sidebar.write(f"• {node}")
        if len(matching_nodes) > 5:
            st.sidebar.write(f"... and {len(matching_nodes) - 5} more")
    else:
        st.sidebar.info("No matching concepts found")
        st.sidebar.warning("Graph will show all concepts")
elif search_term:
    st.sidebar.info("Upload a PDF and generate a concept map to search")

# File uploader
uploaded = st.file_uploader("📄 Upload your notes or textbook (PDF)", type=["pdf"])

if uploaded:
    os.makedirs("data", exist_ok=True)
    pdf_path = "data/input.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded.read())

    # Check file size and warn for very large files
    file_size_mb = uploaded.size / (1024 * 1024)
    if file_size_mb > 50:
        st.warning(f"⚠️ Large PDF detected ({file_size_mb:.1f} MB). Processing may take longer.")
        st.info("💡 For very large documents (>100 MB), consider splitting into smaller files for better performance.")
    elif file_size_mb > 10:
        st.info(f"📄 PDF size: {file_size_mb:.1f} MB - Processing should be smooth.")

    # Step 1: Extract text from PDF
    with st.spinner("🔍 Extracting text from PDF..."):
        try:
            progress_placeholder = st.empty()
            def progress_callback(message):
                progress_placeholder.info(message)
            raw_text = extract_text_from_pdf(pdf_path, progress_callback)
            doc_stats = get_document_stats(raw_text)
            # --- Professional Info Card for Document Stats ---
            st.markdown("""
            <div style='display:flex;gap:24px;justify-content:center;margin-bottom:18px;'>
                <div style='background:linear-gradient(120deg,#e0f2fe 60%,#38bdf8 100%);border-radius:12px;padding:18px 28px;min-width:120px;text-align:center;box-shadow:0 2px 8px #2563eb18;'>
                    <div style='font-size:2em;'>📄</div>
                    <div style='font-size:1.2em;font-weight:600;color:#2563eb;'>Pages</div>
                    <div style='font-size:1.3em;font-weight:700;'>{pages}</div>
                </div>
                <div style='background:linear-gradient(120deg,#fbbf24 60%,#ffe082 100%);border-radius:12px;padding:18px 28px;min-width:120px;text-align:center;box-shadow:0 2px 8px #fbbf2418;'>
                    <div style='font-size:2em;'>📝</div>
                    <div style='font-size:1.2em;font-weight:600;color:#b45309;'>Words</div>
                    <div style='font-size:1.3em;font-weight:700;'>{words}</div>
                </div>
                <div style='background:linear-gradient(120deg,#d1fae5 60%,#34d399 100%);border-radius:12px;padding:18px 28px;min-width:120px;text-align:center;box-shadow:0 2px 8px #34d39918;'>
                    <div style='font-size:2em;'>🔤</div>
                    <div style='font-size:1.2em;font-weight:600;color:#059669;'>Sentences</div>
                    <div style='font-size:1.3em;font-weight:700;'>{sentences}</div>
                </div>
                <div style='background:linear-gradient(120deg,#fce7f3 60%,#f472b6 100%);border-radius:12px;padding:18px 28px;min-width:120px;text-align:center;box-shadow:0 2px 8px #f472b618;'>
                    <div style='font-size:2em;'>📊</div>
                    <div style='font-size:1.2em;font-weight:600;color:#be185d;'>Size</div>
                    <div style='font-size:1.3em;font-weight:700;'>{size}</div>
                </div>
            </div>
            """.format(
                pages=f"{doc_stats.get('estimated_pages', 0):.0f}",
                words=f"{doc_stats.get('words', 0):,}",
                sentences=f"{doc_stats.get('sentences', 0):,}",
                size=doc_stats.get('size_category', 'unknown').title()
            ), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to extract text: {e}")
            st.stop()

    # Step 2: Create document overview
    if groq_available:
        with st.spinner("📋 Creating document overview..."):
            try:
                doc_summary = create_document_summary(raw_text)
                st.markdown("""
                <div style='background:linear-gradient(120deg,#f9fafb 60%,#e0f2fe 100%);border-radius:12px;padding:20px 28px;margin:18px 0 0 0;box-shadow:0 2px 8px #2563eb12;'>
                    <div style='font-size:1.15em;font-weight:600;color:#2563eb;margin-bottom:6px;'>📝 Document Overview</div>
                    <div style='font-size:1.08em;color:#222;'>{summary}</div>
                </div>
                """.format(summary=doc_summary), unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Document overview failed: {e}")

    # Step 3: Sentence tokenization
    with st.spinner("🔤 Processing sentences..."):
        try:
            sentences = sent_tokenize(raw_text)
        except Exception as e:
            st.error(f"Sentence tokenization failed: {e}")
            st.stop()

    # Step 4: Chunking text
    with st.spinner("Processing document..."):
        try:
            chunk_progress_placeholder = st.empty()
            def chunk_progress_callback(message):
                pass  # No UI update
            doc_size, recommended_chunk_size, recommended_overlap = estimate_document_size(raw_text)
            chunks = chunk_text(raw_text, max_tokens=recommended_chunk_size, progress_callback=chunk_progress_callback)
        except Exception as e:
            st.error(f"Chunking failed: {e}")
            st.stop()

    # Step 5: Overlap chunks
    with st.spinner():
        try:
            def overlap_progress_callback(message):
                pass  # No UI update
            from utils.preprocess import overlap_chunks
            chunks = overlap_chunks(chunks, overlap=recommended_overlap, progress_callback=overlap_progress_callback)
        except Exception as e:
            st.error(f"Overlap creation failed: {e}")
            st.stop()

    # Step 6: Summarize chunks
    try:
        summaries = summarize_chunks(chunks)
    except Exception as e:
        st.error(f"Summarization failed: {e}")
        st.stop()

    # Step 7: Extract relations
    st.info("🔗 Extracting relations...")
    all_triplets = []
    
    # Use batch extraction if Groq is available
    if groq_available:
        try:
            with st.spinner("Extracting relations using Groq..."):
                all_triplets = extract_relations_batch(chunks)
        except Exception as e:
            st.warning(f"Batch extraction failed: {e}. Trying individual extraction...")
            all_triplets = []
    
    # Fallback to individual extraction
    if not all_triplets:
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Extracting relations from chunk {i+1}/{len(chunks)}..."):
                try:
                    rel_text = extract_relations(chunk)
                    triplets = parse_triplets(rel_text)
                    all_triplets.extend(triplets)
                except Exception as e:
                    st.warning(f"Relation extraction failed for chunk {i+1}: {e}")

    # Deduplicate and filter triplets
    def is_valid_triplet(triplet):
        s, r, o = triplet
        if not s or not r or not o:
            return False
        if len(s) < 2 or len(r) < 2 or len(o) < 2:
            return False
        if s == o:
            return False
        generic_relations = {"has", "is", "part", "of", "in", "on", "with"}
        if r.lower() in generic_relations:
            return False
        return True

    filtered_triplets = [t for t in all_triplets if is_valid_triplet(t)]
    triplet_counts = Counter(filtered_triplets)
    final_triplets = [t for t, count in triplet_counts.items() if count > 1]
    if len(final_triplets) < 5:
        final_triplets = filtered_triplets

    if not final_triplets:
        st.warning("⚠️ No relations could be extracted. Try uploading clearer text or check your model output above.")
        st.stop()

    # Step 8: Build and visualize graph
    st.info("🌐 Building concept map...")
    try:
        G = build_graph(final_triplets)
        os.makedirs("outputs", exist_ok=True)
        out_file = "outputs/concept_map.html"
        
        # Pass search term only if filtering is enabled
        search_term_for_graph = search_term if st.session_state.show_filtered else None
        visualize_graph(G, out_file, selected_layout, search_term_for_graph)
        
        # Reset regenerate flag after successful generation
        st.session_state.regenerate_graph = False

        st.success("🎉 Concept map created!")
        
        # Only display the graph if not regenerating
        if not st.session_state.regenerate_graph:
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=800, scrolling=True)
                st.download_button("Download Concept Map (HTML)", data=html_content, file_name="concept_map.html", mime="text/html")
            except FileNotFoundError:
                st.error("Graph file not found. Please try regenerating the concept map.")
            except Exception as e:
                st.error(f"Error displaying graph: {e}")
        else:
            st.info("🔄 Click 'Regenerate Graph' in the sidebar to apply view changes")
        
        # --- Learning Path (Visually Enhanced, Themed Colors, Max 6 Steps) ---
        st.subheader("🛤️ Learning Path (Recommended Order)")
        try:
            import networkx as nx
            order = list(nx.topological_sort(G))
        except Exception:
            order = list(G.nodes)
        # Remove duplicates, preserve order
        seen = set()
        unique_order = []
        for concept in order:
            c_lower = concept.strip().lower()
            if c_lower not in seen:
                unique_order.append(concept)
                seen.add(c_lower)
        # Prioritize by degree centrality for importance
        degree_centrality = nx.degree_centrality(G)
        unique_order = sorted(unique_order, key=lambda x: -degree_centrality.get(x, 0))
        max_steps = 6
        if unique_order:
            st.markdown("""
            <style>
            .learning-path-step {
                color: #fff;
                margin-bottom: 8px;
                padding: 10px 16px 10px 16px;
                border-radius: 7px;
                box-shadow: 0 1px 4px #2563eb18;
                font-size: 1em;
                display: flex;
                align-items: center;
            }
            .learning-path-number {
                background: rgba(255,255,255,0.18);
                color: #fff;
                font-weight: bold;
                border-radius: 50%;
                width: 26px;
                height: 26px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 12px;
                font-size: 1em;
                box-shadow: 0 1px 2px #2563eb10;
            }
            </style>
            """, unsafe_allow_html=True)
            # Themed gradient color palette
            gradients = [
                "linear-gradient(90deg,#2563eb 0%,#38bdf8 100%)",   # Blue to Sky Blue
                "linear-gradient(90deg,#34d399 0%,#a7f3d0 100%)",  # Mint to Light Mint
                "linear-gradient(90deg,#fbbf24 0%,#ffe082 100%)",  # Orange to Gold
                "linear-gradient(90deg,#a259f7 0%,#c4b5fd 100%)",  # Purple to Lavender
                "linear-gradient(90deg,#38bdf8 0%,#f472b6 100%)",  # Sky Blue to Pink
                "linear-gradient(90deg,#f472b6 0%,#fbbf24 100%)"   # Pink to Orange
            ]
            steps_html = ""
            for idx, concept in enumerate(unique_order[:max_steps], 1):
                gradient = gradients[(idx-1) % len(gradients)]
                steps_html += f'<div class="learning-path-step" style="background:{gradient};"><div class="learning-path-number">{idx}</div> <b>{concept}</b></div>'
            st.markdown(steps_html, unsafe_allow_html=True)
        else:
            st.info("No learning path could be determined (graph may be empty or cyclic).")
            
        # --- Glossary Section ---
        st.markdown('<div class="section-title">📚 Glossary of Key Terms</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Use node names from the concept map as glossary terms
        if 'G' in locals() and G is not None:
            terms = sorted(set(G.nodes))
        else:
            terms = []
        def get_definition(term, summaries):
            for s in summaries:
                if term in s and len(s) < 120:
                    return s
            try:
                return wikipedia.summary(term, sentences=1, auto_suggest=True, redirect=True)
            except Exception:
                return "No definition found."
        search = st.text_input("Search glossary:", "", key="glossary_search")
        filtered_terms = [t for t in terms if search.lower() in t.lower()]
        if filtered_terms:
            for term in filtered_terms:
                definition = get_definition(term, summaries)
                if definition and definition != "No definition found.":
                    st.markdown(f"""
                    <div style='background:linear-gradient(120deg,#e0f2fe 60%,#38bdf8 100%);border-radius:10px;padding:14px 20px;margin-bottom:12px;box-shadow:0 1px 4px #2563eb10;'>
                        <b style='color:#2563eb;font-size:1.1em;'>{term}</b><br>
                        <span style='color:#222;font-size:1em;'>{definition}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No terms found.")
        st.markdown('</div>', unsafe_allow_html=True)
        # --- Further Reading Section ---
        st.markdown('<div class="section-title">🔗 Suggested Further Reading</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for term in terms[:15]:
            wiki_url = f"https://en.wikipedia.org/wiki/{term.replace(' ', '_')}"
            st.markdown(f"""
            <div style='background:linear-gradient(120deg,#38bdf8 0%,#a259f7 100%);border-radius:12px;padding:16px 22px;margin-bottom:12px;box-shadow:0 2px 8px #2563eb18;'>
                <b style='color:#fff;font-size:1.13em;text-shadow:0 1px 4px #2563eb30;'>{term}</b><br>
                <a href='{wiki_url}' target='_blank' style='color:#ffe082;font-weight:700;text-decoration:none;font-size:1.07em;display:inline-block;margin-top:6px;'>
                    <span style="vertical-align:middle;">🌐 Read more on Wikipedia</span>
                </a>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Graph building or visualization failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌐 Concept Map Visualization</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
