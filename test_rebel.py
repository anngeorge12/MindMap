from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# Load REBEL model and tokenizer
model_name = "Babelscape/rebel-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def extract_triplets_from_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            early_stopping=True,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    return decoded

def parse_triplets(text):
    triplets = []
    pattern = r"<triplet>\s*(.*?)\s*<subj>\s*(.*?)\s*<obj>\s*(.*?)(?=<triplet>|</s>|$)"
    matches = re.findall(pattern, text)

    for pred, subj, obj in matches:
        clean_pred = pred.strip().replace("</s>", "")
        clean_subj = subj.strip().replace("</s>", "")
        clean_obj = obj.strip().replace("</s>", "")
        triplets.append((clean_subj, clean_pred, clean_obj))

    return triplets

# Example test input
text = "The heart pumps blood. The brain controls muscles."
raw_output = extract_triplets_from_text(text)
print("🔎 Raw output:\n", raw_output)

triplets = parse_triplets(raw_output)
print("\n✅ Triplets extracted:")
for t in triplets:
    print(t)
