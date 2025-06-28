import os

# Change working directory to project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.system("streamlit run app/app.py")
