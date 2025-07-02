import streamlit as st
import PyPDF2
import docx
import csv
import spacy
import re
import matplotlib.pyplot as plt
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Text Extractors
# -----------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return ' '.join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join([p.text for p in doc.paragraphs])

def extract_text_from_csv(file):
    text = []
    decoded = file.read().decode('latin-1').splitlines()
    reader = csv.reader(decoded)
    for row in reader:
        text.append(' '.join(row))
    return ' '.join(text)

# -----------------------------
# Text Processing
# -----------------------------
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def extract_keywords(text):
    doc = nlp(text)
    return {token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop}

def score_resume(resume_text, keywords):
    words = set(resume_text.split())
    matched_keywords = words & keywords
    return len(matched_keywords), matched_keywords

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Resume Ranker", layout="wide")
st.title("Resume Ranking Dashboard")
st.markdown("An NLP-powered tool that scores resumes against a job description.")

# Sidebar
st.sidebar.header("Upload Inputs")
job_description = st.sidebar.text_area("Paste Job Description Here", height=200)
uploaded_files = st.sidebar.file_uploader("Upload Resumes (PDF, DOCX, CSV)", type=["pdf", "docx", "csv"], accept_multiple_files=True)

# Processing resumes
if st.sidebar.button("Rank Resumes"):
    if not job_description or not uploaded_files:
        st.warning("Please provide both a job description and resume files.")
    else:
        st.success("Processing resumes...")
        keywords = extract_keywords(preprocess_text(job_description))
        results = []

        for file in uploaded_files:
            file_name = file.name
            if file_name.endswith(".pdf"):
                text = extract_text_from_pdf(file)
            elif file_name.endswith(".docx"):
                text = extract_text_from_docx(file)
            elif file_name.endswith(".csv"):
                text = extract_text_from_csv(file)
            else:
                st.warning(f"Unsupported file format: {file_name}")
                continue

            processed_text = preprocess_text(text)
            matched_count, matched = score_resume(processed_text, keywords)
            score_percent = (matched_count / len(keywords)) * 100 if keywords else 0

            results.append({
                "File Name": file_name,
                "Score (%)": round(score_percent, 2),
                "Matched Keywords": ', '.join(sorted(matched))
            })

        # Display results
        df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False).reset_index(drop=True)
        st.subheader("Resume Match Results")
        st.dataframe(df)

        # Bar chart
        if not df.empty:
            st.subheader("Similarity Score Bar Chart")
            fig, ax = plt.subplots()
            ax.bar(df["File Name"], df["Score (%)"], color="#6C63FF")
            plt.xticks(rotation=45, ha='right')
            plt.xlabel("Resumes")
            plt.ylabel("Similarity Score (%)")
            plt.title("Resume Screening Overview")
            st.pyplot(fig)
        else:
            st.info("No data to visualize.")
