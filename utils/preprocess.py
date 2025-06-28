import fitz  # PyMuPDF
import nltk

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, max_tokens=512):
    nltk.download('punkt')
    # Clean up text
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Split by paragraphs if possible
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for para in paragraphs:
        sentences = nltk.sent_tokenize(para)
        chunk = ""
        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) < max_tokens:
                chunk += " " + sentence
            else:
                chunks.append(chunk.strip())
                chunk = sentence
        if chunk:
            chunks.append(chunk.strip())
    return chunks
