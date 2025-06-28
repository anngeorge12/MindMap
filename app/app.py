import os
import sys
import streamlit as st
import nltk
import streamlit.components.v1 as components
from collections import Counter

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
from pipeline.concept_graph import parse_triplets, build_graph, visualize_graph, get_layout_options

# Streamlit UI config
st.set_page_config(page_title="MindSketch", layout="wide")
st.title("🧠 MindSketch – AI Concept Map Generator")

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
st.sidebar.header("�� Search & Filter")

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
            # Create a progress placeholder
            progress_placeholder = st.empty()
            
            def progress_callback(message):
                progress_placeholder.info(message)
            
            raw_text = extract_text_from_pdf(pdf_path, progress_callback)
            
            # Get document statistics
            doc_stats = get_document_stats(raw_text)
            
            # Display document statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Pages", f"{doc_stats.get('estimated_pages', 0):.0f}")
            with col2:
                st.metric("📝 Words", f"{doc_stats.get('words', 0):,}")
            with col3:
                st.metric("🔤 Sentences", f"{doc_stats.get('sentences', 0):,}")
            with col4:
                st.metric("📊 Size", doc_stats.get('size_category', 'unknown').title())
        except Exception as e:
            st.error(f"Failed to extract text: {e}")
            st.stop()

    # Step 2: Create document overview
    if groq_available:
        with st.spinner("📋 Creating document overview..."):
            try:
                doc_summary = create_document_summary(raw_text)
                st.subheader("📋 Document Overview")
                st.write(doc_summary)
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
    with st.spinner("📚 Processing document..."):
        try:
            # Create a progress placeholder for chunking
            chunk_progress_placeholder = st.empty()
            
            def chunk_progress_callback(message):
                chunk_progress_placeholder.info(message)
            
            # Get recommended chunk size based on document size
            doc_size, recommended_chunk_size, recommended_overlap = estimate_document_size(raw_text)
            
            # Use adaptive chunking
            chunks = chunk_text(raw_text, max_tokens=recommended_chunk_size, progress_callback=chunk_progress_callback)
        except Exception as e:
            st.error(f"Chunking failed: {e}")
            st.stop()

    # Step 5: Overlap chunks
    with st.spinner("🔄 Creating overlapping chunks..."):
        try:
            overlap_progress_placeholder = st.empty()
            
            def overlap_progress_callback(message):
                overlap_progress_placeholder.info(message)
            
            from utils.preprocess import overlap_chunks
            chunks = overlap_chunks(chunks, overlap=recommended_overlap, progress_callback=overlap_progress_callback)
        except Exception as e:
            st.error(f"Overlap creation failed: {e}")
            st.stop()

    # Step 6: Summarize chunks
    st.info("📝 Summarizing chunks...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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
            
    except Exception as e:
        st.error(f"Graph building or visualization failed: {e}")
