#!/usr/bin/env python3
"""
Test script to verify MindSketch setup before running main.py
"""

import os
import sys
import importlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    required_modules = [
        'streamlit',
        'nltk',
        'networkx',
        'pyvis',
        'fitz',  # PyMuPDF
        'openai'
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_local_modules():
    """Test if local modules can be imported."""
    print("\n🔍 Testing local modules...")
    
    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    local_modules = [
        'utils.preprocess',
        'utils.groq_utils',
        'models.summarizer',
        'models.relations_extract',
        'pipeline.concept_graph'
    ]
    
    failed_imports = []
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_groq_setup():
    """Test Groq API setup."""
    print("\n🔍 Testing Groq setup...")
    
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"  ✅ GROQ_API_KEY found (length: {len(groq_key)})")
        
        # Test basic Groq functionality
        try:
            from utils.groq_utils import summarize_text
            test_result = summarize_text("This is a test sentence.")
            if "error" not in test_result.lower():
                print("  ✅ Groq API test successful")
                return True
            else:
                print("  ⚠️  Groq API test failed - check your API key")
                return False
        except Exception as e:
            print(f"  ❌ Groq API test failed: {e}")
            return False
    else:
        print("  ⚠️  GROQ_API_KEY not found - will use fallback models")
        return True

def test_directories():
    """Test if required directories exist."""
    print("\n🔍 Testing directories...")
    
    required_dirs = ['data', 'outputs', 'app', 'models', 'utils', 'pipeline']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
            return False
    
    return True

def test_nltk_data():
    """Test if NLTK data is available."""
    print("\n🔍 Testing NLTK data...")
    
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print("  ✅ NLTK punkt tokenizer available")
        return True
    except LookupError:
        print("  ⚠️  NLTK punkt tokenizer not found - will download automatically")
        return True

def main():
    """Run all tests."""
    print("🧠 MindSketch Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Local Module Tests", test_local_modules),
        ("Directory Tests", test_directories),
        ("NLTK Data Tests", test_nltk_data),
        ("Groq Setup Tests", test_groq_setup)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! You can now run:")
        print("   python main.py")
    else:
        print("❌ Some tests failed. Please fix the issues above before running main.py")
        print("\n💡 Try running: python setup.py")

if __name__ == "__main__":
    main() 