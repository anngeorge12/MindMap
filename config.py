# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"  # Default model

# Text Processing Configuration
CHUNK_SIZE = 512  # Maximum tokens per chunk
OVERLAP_SIZE = 1  # Number of overlapping chunks

# Summarization Configuration
SUMMARY_MAX_LENGTH = 150
SUMMARY_TEMPERATURE = 0.3

# Relation Extraction Configuration
EXTRACTION_TEMPERATURE = 0.2
EXTRACTION_MAX_TOKENS = 500

# Graph Configuration
MIN_TRIPLET_COUNT = 5  # Minimum triplets to create a meaningful graph

def check_groq_setup():
    """
    Check if Groq is properly configured.
    """
    if not GROQ_API_KEY:
        print("⚠️  GROQ_API_KEY not found!")
        print("To use enhanced AI features:")
        print("1. Get your API key from https://console.groq.com/")
        print("2. Set the environment variable: export GROQ_API_KEY='your-key-here'")
        print("3. Or create a .env file with: GROQ_API_KEY=your-key-here")
        return False
    return True

def get_available_models():
    """
    Return available Groq models.
    """
    return {
        "llama3-70b-8192": "Llama3-70B (Recommended for best quality)",
        "llama3-8b-8192": "Llama3-8B (Faster, good quality)",
        "mixtral-8x7b-32768": "Mixtral-8x7B (Good balance)",
        "gemma2-9b-it": "Gemma2-9B (Fast, efficient)"
    } 