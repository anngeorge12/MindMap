import sys
import os

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.groq_utils import summarize_chunks as groq_summarize_chunks, summarize_text, create_concept_summary

def summarize_chunks(chunks):
    """
    Summarize chunks using Groq's Llama3-70b model for better quality.
    Falls back to BART if Groq is not available.
    """
    try:
        # Check if GROQ_API_KEY is set
        if not os.getenv("GROQ_API_KEY"):
            print("Warning: GROQ_API_KEY not found. Using BART fallback.")
            return bart_summarize_chunks(chunks)
        
        print("Using Groq for summarization...")
        return groq_summarize_chunks(chunks)
    except Exception as e:
        print(f"Groq summarization failed: {e}. Falling back to BART...")
        return bart_summarize_chunks(chunks)

def bart_summarize_chunks(chunks):
    """
    Fallback to BART summarization if Groq is not available.
    """
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        summarized = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
            summarized.append(summary)
        return summarized
    except Exception as e:
        print(f"BART summarization also failed: {e}")
        return chunks  # Return original chunks if all summarization fails

def create_document_summary(text):
    """
    Create a high-level summary of the entire document using Groq.
    """
    try:
        if not os.getenv("GROQ_API_KEY"):
            return "Document summary not available without Groq API key."
        
        return create_concept_summary(text)
    except Exception as e:
        print(f"Document summary failed: {e}")
        return f"Summary error: {str(e)}"
