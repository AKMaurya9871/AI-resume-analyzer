import streamlit as st
import matplotlib.pyplot as pt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download necessary NLTK resources

nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")

                # Page Setup

st.set_page_config(page_title="Resume Scorer",page_icon="📄",layout="wide")

st.markdown("""
Upload your resume(PDF) and paste a job description to see how well they 
match!\n
This tool uses **TF-IDF + Cosine Similarity** to analyze your resume against job
requirements.
""")

with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps you:\n
    - Measures hoy your resume matches a job description.\n
    - Identify important keywords in your resume.\n
    - Improve your resume by adding relevant keywords.
    """)
    st.header("How It works")
    st.write("""
    1. Upload your resume in PDF format.\n
    2. Paste the job description in the text box.\n
    3. Click the 'Analyze' button to see the match score.
    4. Review score & suggestions
    """)


                # helper functions

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Convert to lowercase
    text = text.lower()
    return text


def stopword_removal(text):

    # Collect common English stopwords

    stop_words = set(stopwords.words("english"))

    # Tokenize the text into words

    word_tok = word_tokenize(text)

    # Remove stopwords from the tokenized words

    return " ".join([word for word in word_tok if word not in stop_words])


    # Function to calculate cosine similarity between resume and job description

def calculate_cosine_similarity(resume_text, job_description):

    # Clean the text by removing special characters, digits, and converting to lowercase

    resume_text = stopword_removal(clean_text(resume_text))
    job_description = stopword_removal(clean_text(job_description))

    # Remove stopwords from both resume and job description

    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the resume and job description text to create TF-IDF vectors

    tfidf_matrix = tfidf_vectorizer.fit_transform([resume_text, job_description])

    # Calculate cosine similarity between the two TF-IDF vectors

    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]*100
    return round(cosine_sim, 2),resume_text,job_description

        # Main function to run the Streamlit app

def main():
    # Streamlit app title and description

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    # Text area for job description input

    job_description = st.text_area("Paste the job descrioption here",height=200)

    # Analyze button to trigger the similarity calculation

    if st.button("Analze Match"):
        if not uploaded_file:
            st.error("Please upload a resume in PDF format.")
            return
        if not job_description:
            st.error("Please paste the job description.")
            return

        resume_text = extract_text_from_pdf(uploaded_file)

        # Calculate cosine similarity and display the results

        if resume_text:
            similarity_score, cleaned_resume, cleaned_job_des = calculate_cosine_similarity(resume_text, job_description)
            st.subheader("Results")
            st.metric("Match Score", f"{similarity_score:.2f}%")

            # Display a horizontal bar chart to visualize the match score

            fig, ax = pt.subplots(figsize=(6,0.5))
            colors = ['#ff4b4b','#ffa726','#0f9d58']
            color_index = min(int(similarity_score / 33.33), 2)
            ax.barh([0], [similarity_score], color=colors[color_index])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Match Score (%)")
            ax.set_yticks([])
            ax.set_title("Resume job Match")
            st.pyplot(fig)

            if(similarity_score < 40):
                st.warning("Your resume has a low match score. Consider adding relevant keywords from the job description to improve your chances.")
            elif(similarity_score < 70):
                st.info("Your resume has a moderate match score. You may want to review the job description and add more relevant keywords.")
            else:
                st.success("Great job! Your resume has a high match score. You are well-aligned with the job requirements.")

        else: st.error("Failed to extract text from the file. Please ensure the file is a valid PDF and try again.")

    
main()
