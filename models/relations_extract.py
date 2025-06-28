# models/relations_extract.py
import sys
import os

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.groq_utils import extract_relations_enhanced, extract_triplets

def extract_relations(text):
    """
    Extract relations using Groq's Llama3-70b model for better quality.
    Falls back to REBEL if Groq is not available.
    """
    try:
        # Check if GROQ_API_KEY is set
        if not os.getenv("GROQ_API_KEY"):
            print("Warning: GROQ_API_KEY not found. Using REBEL fallback.")
            return rebel_extract_relations(text)
        
        print("Using Groq for relation extraction...")
        # Use the enhanced version that returns parsed triplets
        triplets = extract_relations_enhanced(text)
        if triplets:
            # Convert triplets back to text format for compatibility
            return format_triplets_as_text(triplets)
        else:
            # Fallback to text-based extraction
            return extract_triplets(text)
    except Exception as e:
        print(f"Groq relation extraction failed: {e}. Falling back to REBEL...")
        return rebel_extract_relations(text)

def format_triplets_as_text(triplets):
    """
    Format triplets as text for compatibility with existing pipeline.
    """
    formatted = []
    for subject, relation, obj in triplets:
        formatted.append(f"({subject}, {relation}, {obj})")
    return "\n".join(formatted)

def rebel_extract_relations(text):
    """
    Fallback to REBEL model for relation extraction.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Load REBEL model and tokenizer
        model_name = "Babelscape/rebel-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Add task-specific prompt
        prompt = f"extract all factual (subject, relation, object) triplets from the following text: {text}"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate prediction
        output_ids = model.generate(**inputs, max_length=256)
        
        # Decode output
        decoded_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return decoded_text
    except Exception as e:
        print(f"REBEL extraction also failed: {e}")
        return f"Extraction error: {str(e)}"

def extract_relations_batch(texts):
    """
    Extract relations from multiple texts using Groq.
    """
    all_triplets = []
    
    for i, text in enumerate(texts):
        print(f"Extracting relations from text {i+1}/{len(texts)}")
        try:
            triplets = extract_relations_enhanced(text)
            all_triplets.extend(triplets)
        except Exception as e:
            print(f"Failed to extract relations from text {i+1}: {e}")
    
    return all_triplets
