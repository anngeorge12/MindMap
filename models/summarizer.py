from transformers import pipeline

# Load pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_chunks(chunks):
    summarized = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        summarized.append(summary)
    return summarized
