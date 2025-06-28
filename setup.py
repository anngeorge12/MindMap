#!/usr/bin/env python3
"""
Setup script for MindSketch - AI Concept Map Generator
"""

import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def check_groq_setup():
    """Check if Groq API key is configured."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print("✅ Groq API key found!")
        return True
    else:
        print("⚠️  Groq API key not found.")
        print("\nTo get the best experience:")
        print("1. Visit https://console.groq.com/")
        print("2. Create an account and get your API key")
        print("3. Set the environment variable:")
        print("   export GROQ_API_KEY='your-key-here'")
        print("   Or create a .env file with: GROQ_API_KEY=your-key-here")
        print("\nThe app will work without Groq, but with reduced quality.")
        return False

def create_env_template():
    """Create a template .env file."""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env template...")
        with open(env_file, "w") as f:
            f.write("# MindSketch Configuration\n")
            f.write("# Get your API key from https://console.groq.com/\n")
            f.write("GROQ_API_KEY=your-api-key-here\n")
        print("✅ Created .env template file")
        print("📝 Edit .env and add your Groq API key")

def main():
    """Main setup function."""
    print("🧠 MindSketch Setup")
    print("=" * 50)
    
    # Install dependencies
    if not install_requirements():
        print("❌ Setup failed. Please check the error messages above.")
        return
    
    # Check Groq setup
    groq_configured = check_groq_setup()
    
    # Create .env template if needed
    if not groq_configured:
        create_env_template()
    
    print("\n🎉 Setup complete!")
    print("\nTo run the application:")
    print("  python main.py")
    print("  or")
    print("  streamlit run app/app.py")
    
    if groq_configured:
        print("\n🚀 You're all set! The app will use enhanced AI models.")
    else:
        print("\n⚠️  The app will work with fallback models.")
        print("   Consider adding your Groq API key for better results.")

if __name__ == "__main__":
    main() 