import fitz  # PyMuPDF
import nltk
import os
import time
from typing import List, Tuple

def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)

def extract_text_from_pdf(file_path, progress_callback=None):
    """Extract text from PDF file with progress tracking."""
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        if progress_callback:
            progress_callback(f"📄 Extracting text from {total_pages} pages...")
        
        text_chunks = []
        for page_num in range(total_pages):
            if progress_callback and page_num % 10 == 0:  # Update every 10 pages
                progress_callback(f"📄 Processing page {page_num + 1}/{total_pages}")
            
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                text_chunks.append(page_text)
        
        doc.close()
        
        if progress_callback:
            progress_callback("📄 Text extraction completed!")
        
        return "\n".join(text_chunks)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def estimate_document_size(text):
    """Estimate document size and recommend chunking strategy."""
    char_count = len(text)
    word_count = len(text.split())
    page_estimate = char_count / 2000  # Rough estimate: 2000 chars per page
    
    if char_count < 50000:  # Small document
        return "small", 512, 1
    elif char_count < 200000:  # Medium document
        return "medium", 768, 2
    elif char_count < 500000:  # Large document
        return "large", 1024, 3
    else:  # Very large document
        return "very_large", 1536, 4

def chunk_text(text, max_tokens=512, progress_callback=None):
    """Split text into chunks for processing with adaptive sizing."""
    ensure_nltk_data()
    
    if not text:
        return []
    
    # Estimate document size and adjust chunking strategy
    doc_size, recommended_chunk_size, overlap_size = estimate_document_size(text)
    
    # Use recommended chunk size if not specified
    if max_tokens == 512:  # Default value
        max_tokens = recommended_chunk_size
    
    if progress_callback:
        progress_callback(f"📚 Document size: {doc_size} ({len(text):,} characters)")
        progress_callback(f"📚 Using chunk size: {max_tokens} tokens, overlap: {overlap_size}")
    
    # Clean up text
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Remove extra whitespace
    
    # Split by sentences first
    sentences = nltk.sent_tokenize(text)
    
    if progress_callback:
        progress_callback(f"📚 Processing {len(sentences)} sentences...")
    
    chunks = []
    current_chunk = ""
    chunk_count = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = len(sentence.split())
        
        # Progress update for large documents
        if progress_callback and i % 100 == 0:
            progress_callback(f"📚 Processing sentence {i + 1}/{len(sentences)}")
        
        # If adding this sentence would exceed max_tokens, save current chunk
        if len(current_chunk.split()) + sentence_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            chunk_count += 1
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        chunk_count += 1
    
    # Ensure we have at least one chunk
    if not chunks and text:
        chunks = [text[:max_tokens * 4]]  # Rough character limit
    
    if progress_callback:
        progress_callback(f"📚 Created {len(chunks)} chunks")
    
    return chunks

def overlap_chunks(chunks, overlap=1, progress_callback=None):
    """Create overlapping chunks for better context preservation."""
    if not chunks:
        return chunks
    
    if progress_callback:
        progress_callback(f"🔄 Creating overlapping chunks (overlap: {overlap})...")
    
    overlapped = []
    for i, chunk in enumerate(chunks):
        if progress_callback and i % 10 == 0:
            progress_callback(f"🔄 Processing chunk {i + 1}/{len(chunks)}")
        
        if i > 0 and overlap > 0:
            # Add previous chunk content for overlap
            prev_chunk = chunks[i-1]
            # Take last few sentences from previous chunk
            prev_sentences = nltk.sent_tokenize(prev_chunk)
            overlap_text = " ".join(prev_sentences[-overlap:]) if len(prev_sentences) >= overlap else prev_chunk
            chunk = overlap_text + " " + chunk
        
        overlapped.append(chunk)
    
    if progress_callback:
        progress_callback(f"🔄 Created {len(overlapped)} overlapping chunks")
    
    return overlapped

def get_document_stats(text):
    """Get comprehensive document statistics."""
    if not text:
        return {}
    
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len(nltk.sent_tokenize(text))
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    return {
        'characters': char_count,
        'words': word_count,
        'sentences': sentence_count,
        'paragraphs': paragraph_count,
        'estimated_pages': char_count / 2000,
        'size_category': estimate_document_size(text)[0]
    }
