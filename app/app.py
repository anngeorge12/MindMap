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
from utils.preprocess import extract_text_from_pdf, chunk_text
from models.summarizer import summarize_chunks, create_document_summary
from models.relations_extract import extract_relations, extract_relations_batch
from pipeline.concept_graph import parse_triplets, build_graph, visualize_graph

# Streamlit UI config
st.set_page_config(page_title="MindSketch", layout="wide")
st.title("🧠 MindSketch – AI Concept Map Generator")

# Check for Groq API key
groq_available = os.getenv("GROQ_API_KEY") is not None
if groq_available:
    st.success("✅ Groq API detected - Using enhanced AI models!")
else:
    st.warning("⚠️ GROQ_API_KEY not found. Using fallback models (BART + REBEL).")

# File uploader
uploaded = st.file_uploader("📄 Upload your notes or textbook (PDF)", type=["pdf"])

if uploaded:
    os.makedirs("data", exist_ok=True)
    pdf_path = "data/input.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded.read())

    # Step 1: Extract text from PDF
    with st.spinner("🔍 Extracting text from PDF..."):
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            st.write(f"First 500 characters of extracted text:\n{raw_text[:500]}")
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
            st.write(f"Total sentences detected: {len(sentences)}")
        except Exception as e:
            st.error(f"Sentence tokenization failed: {e}")
            st.stop()

    # Step 4: Chunking text
    chunk_size = 512
    with st.spinner("📚 Chunking text..."):
        try:
            chunks = chunk_text(raw_text, max_tokens=chunk_size)
            st.write(f"📚 Total chunks: {len(chunks)} (chunk size: {chunk_size} tokens)")
        except Exception as e:
            st.error(f"Chunking failed: {e}")
            st.stop()

    # Step 5: Overlap chunks
    def overlap_chunks(chunks, overlap=1):
        overlapped = []
        for i in range(len(chunks)):
            chunk = chunks[i]
            if i > 0:
                chunk = chunks[i-1] + " " + chunk
            overlapped.append(chunk)
        return overlapped

    chunks = overlap_chunks(chunks, overlap=1)

    # Step 6: Summarize chunks
    st.info("📝 Summarizing chunks...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        summaries = summarize_chunks(chunks)
        
        # Display summaries
        st.subheader("📌 Chunk Summaries")
        for i, summary in enumerate(summaries):
            with st.expander(f"Chunk {i+1} Summary"):
                st.write(summary)
        
        # Combine summaries for relation extraction
        combined_summary = " ".join(summaries)
        st.subheader("📌 Combined Summary")
        st.write(combined_summary)
        
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
                st.success(f"✅ Extracted {len(all_triplets)} triplets using Groq!")
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
    else:
        st.success(f"✅ Extracted {len(final_triplets)} triplets!")
        st.subheader("📎 Extracted Triplets")
        for triplet in final_triplets:
            st.write(f"• {triplet[0]} → {triplet[1]} → {triplet[2]}")

    # Step 8: Build and visualize graph
    st.info("🌐 Building concept map...")
    try:
        G = build_graph(final_triplets)
        st.write("🧠 Nodes:", list(G.nodes))
        st.write("🔗 Edges:", list(G.edges))

        os.makedirs("outputs", exist_ok=True)
        out_file = "outputs/concept_map.html"
        visualize_graph(G, out_file)

        st.success("🎉 Concept map created!")
        with open(out_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
        st.download_button("Download Concept Map (HTML)", data=html_content, file_name="concept_map.html", mime="text/html")
    except Exception as e:
        st.error(f"Graph building or visualization failed: {e}")
