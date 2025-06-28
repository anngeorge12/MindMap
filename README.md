# 🧠 MindSketch - AI Concept Map Generator

An intelligent tool that automatically generates concept maps from educational PDFs using advanced AI models. Enhanced with Groq's powerful language models for superior summarization and relation extraction.

## ✨ Features

- **📄 PDF Processing**: Extract text from educational PDFs
- **🤖 AI-Powered Summarization**: Uses Groq's Llama3-70B for high-quality summaries
- **🔗 Intelligent Relation Extraction**: Automatically identifies concept relationships
- **🌐 Interactive Concept Maps**: Beautiful, interactive visualizations
- **⚡ Fast Processing**: Optimized for quick results
- **🔄 Fallback Support**: Works with or without Groq API

## 🎯 How It Works

1. **📄 Upload PDF**: Upload your educational document
2. **🔍 Text Extraction**: Extract and process text content
3. **📝 AI Summarization**: Generate concise summaries using Groq
4. **🔗 Relation Extraction**: Identify concept relationships
5. **🌐 Concept Map**: Create interactive visualization
6. **💾 Download**: Save your concept map as HTML
7. **Glossary**: definitions of words used in the pdf
8. **Resources**:provides external links to read more about the resources

## 🔧 Configuration

### Models Used

- **With Groq API**: Llama3-70B for best quality
- **Without Groq**: BART + REBEL models as fallback


## 🛠️ Technical Details

### Architecture

```
MindMap/
├── app/                 # Streamlit web interface
├── models/             # AI model integrations
├── pipeline/           # Processing pipeline
├── utils/              # Utility functions
├── data/               # Input/output data
└── outputs/            # Generated concept maps
```

### Key Components

- **`utils/groq_utils.py`**: Groq API integration
- **`models/summarizer.py`**: Summarization logic
- **`models/relations_extract.py`**: Relation extraction
- **`pipeline/concept_graph.py`**: Graph generation
- **`app/app.py`**: Web interface

## 🎨 Example Output

The tool generates interactive concept maps showing:
- **Nodes**: Key concepts from your document
- **Edges**: Relationships between concepts
- **Interactive Features**: Zoom, pan, and explore

