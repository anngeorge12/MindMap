# models/relations_extract.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load REBEL model and tokenizer
model_name = "Babelscape/rebel-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def extract_relations(text):
    # Add task-specific prompt (this is crucial)
    prompt = f"extract all factual (subject, relation, object) triplets from the following text: {text}"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Generate prediction
    output_ids = model.generate(**inputs, max_length=256)
    
    # Decode output
    decoded_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    return decoded_text
