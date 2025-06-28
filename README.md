# 🧠 MindSketch - AI Concept Map Generator

An intelligent tool that automatically generates concept maps from educational PDFs using advanced AI models. Enhanced with Groq's powerful language models for superior summarization and relation extraction.

## ✨ Features

- **📄 PDF Processing**: Extract text from educational PDFs
- **🤖 AI-Powered Summarization**: Uses Groq's Llama3-70B for high-quality summaries
- **🔗 Intelligent Relation Extraction**: Automatically identifies concept relationships
- **🌐 Interactive Concept Maps**: Beautiful, interactive visualizations
- **⚡ Fast Processing**: Optimized for quick results
- **🔄 Fallback Support**: Works with or without Groq API

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Groq API (Optional but Recommended)

1. Get your API key from [Groq Console](https://console.groq.com/)
2. Set the environment variable:
   ```bash
   export GROQ_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your-api-key-here
   ```

### 3. Run the Application

```bash
python main.py
```

Or directly with Streamlit:
```bash
streamlit run app/app.py
```

## 🎯 How It Works

1. **📄 Upload PDF**: Upload your educational document
2. **🔍 Text Extraction**: Extract and process text content
3. **📝 AI Summarization**: Generate concise summaries using Groq
4. **🔗 Relation Extraction**: Identify concept relationships
5. **🌐 Concept Map**: Create interactive visualization
6. **💾 Download**: Save your concept map as HTML

## 🔧 Configuration

### Models Used

- **With Groq API**: Llama3-70B for best quality
- **Without Groq**: BART + REBEL models as fallback

### Customization

Edit `config.py` to adjust:
- Chunk sizes for text processing
- Summarization parameters
- Model selection
- Graph generation settings

## 📊 Performance Comparison

| Feature | With Groq | Without Groq |
|---------|-----------|--------------|
| Summarization Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Relation Extraction | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Processing Speed | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Cost | 💰 | 🆓 |

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

## 🔍 Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not found"**
   - Set your API key as described above
   - The app will work with fallback models

2. **"No relations extracted"**
   - Try uploading clearer, more structured text
   - Check that the PDF contains educational content

3. **"Summarization failed"**
   - Ensure your Groq API key is valid
   - Check your internet connection

### Performance Tips

- Use shorter documents for faster processing
- Ensure PDFs have good text quality
- Close other applications to free up memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Groq for providing fast, high-quality AI models
- Streamlit for the web framework
- PyVis for graph visualization
- The open-source community for various dependencies

---

**Happy Concept Mapping! 🧠✨** 