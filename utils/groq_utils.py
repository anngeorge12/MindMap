# utils/groq_utils.py

import os
import openai
import time
from typing import List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

def get_groq_client():
    """Get a configured Groq client."""
    return openai.OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

def summarize_text(text: str, max_length: int = 150) -> str:
    """
    Summarize text using Groq's Llama3-70b model with better prompting.
    """
    try:
        client = get_groq_client()
        prompt = f"""Please provide a clear, concise summary of the following educational content in 3-5 sentences. 
Focus on the main concepts, key relationships, and important facts. Make it suitable for creating a concept map.

Text to summarize:
{text}

Summary:"""
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_length
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in summarization: {e}")
        return f"Summary error: {str(e)}"

def summarize_chunks(chunks: List[str]) -> List[str]:
    """
    Summarize multiple text chunks using Groq.
    """
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}")
        summary = summarize_text(chunk)
        summaries.append(summary)
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    return summaries

def extract_triplets(text: str) -> str:
    """
    Extract subject-relation-object triplets using Groq with improved prompting.
    """
    try:
        client = get_groq_client()
        prompt = f"""Extract all factual (subject, relation, object) triplets from the following text. 
Format each triplet as: (subject, relation, object)
Focus on educational concepts, relationships, and factual information.
Only include meaningful relationships that would be useful for a concept map.

Text:
{text}

Triplets:"""
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in triplet extraction: {e}")
        return f"Extraction error: {str(e)}"

def extract_relations_enhanced(text: str) -> List[Tuple[str, str, str]]:
    """
    Enhanced relation extraction that returns parsed triplets directly.
    """
    try:
        client = get_groq_client()
        prompt = f"""Extract all factual (subject, relation, object) triplets from the following text. 
Return ONLY the triplets in this exact format:
(subject, relation, object)
(subject, relation, object)
...

Focus on educational concepts and meaningful relationships. Do not include generic or obvious relationships.

Text:
{text}"""
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )
        
        result = response.choices[0].message.content.strip()
        return parse_triplets_from_text(result)
    except Exception as e:
        print(f"Error in enhanced relation extraction: {e}")
        return []

def parse_triplets_from_text(text: str) -> List[Tuple[str, str, str]]:
    """
    Parse triplets from text output.
    """
    triplets = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            # Remove parentheses and split by comma
            content = line[1:-1]
            parts = [part.strip() for part in content.split(',')]
            if len(parts) >= 3:
                subject = parts[0]
                relation = parts[1]
                object_part = ','.join(parts[2:])  # Handle objects with commas
                triplets.append((subject, relation, object_part))
    
    return triplets

def create_concept_summary(text: str) -> str:
    """
    Create a high-level concept summary for the entire document.
    """
    try:
        client = get_groq_client()
        prompt = f"""Create a comprehensive concept summary of the following educational content. 
Identify the main themes, key concepts, and their relationships. This will be used as an overview for a concept map.

Text:
{text}

Concept Summary:"""
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in concept summary: {e}")
        return f"Summary error: {str(e)}"
