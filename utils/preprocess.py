import fitz  # PyMuPDF
import nltk
import os

def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)

def extract_text_from_pdf(file_path):
    """Extract text from PDF file."""
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text, max_tokens=512):
    """Split text into chunks for processing."""
    ensure_nltk_data()
    
    if not text:
        return []
    
    # Clean up text
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Remove extra whitespace
    
    # Split by sentences first
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        
        # If adding this sentence would exceed max_tokens, save current chunk
        if len(current_chunk.split()) + sentence_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Ensure we have at least one chunk
    if not chunks and text:
        chunks = [text[:max_tokens * 4]]  # Rough character limit
    
    return chunks
