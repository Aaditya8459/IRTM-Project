import nltk
import pdfplumber
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except:
        return ""
    return text

def preprocess(text):
    if not text:
        return ""

    tokens = word_tokenize(text.lower())

    words = []
    for w in tokens:
        if w.isalnum() and w not in stop_words and len(w) > 2:
            lemma = lemmatizer.lemmatize(w)
            words.append(lemma)

    return " ".join(words)

def skill_score(resume_text, job_description):
    resume_words = set(preprocess(resume_text).split())
    jd_words = set(preprocess(job_description).split())

    matched = resume_words.intersection(jd_words)

    if len(jd_words) == 0:
        return 0

    return (len(matched) / len(jd_words)) * 100

def semantic_score(resume_text, job_description):
    resume_clean = preprocess(resume_text)
    jd_clean = preprocess(job_description)

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    vectors = vectorizer.fit_transform([resume_clean, jd_clean])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return similarity * 100

def calculate_score(resume_text, job_description):
    sem_score = semantic_score(resume_text, job_description)
    skill_match = skill_score(resume_text, job_description)

    final_score = (0.7 * sem_score) + (0.3 * skill_match)

    return round(min(final_score, 100), 2)

def suggest_role(resume_text):
    text = resume_text.lower()

    if any(x in text for x in ["machine learning", "deep learning", "nlp", "pandas"]):
        return "Data Scientist / ML Engineer"

    elif any(x in text for x in ["react", "javascript", "frontend", "css"]):
        return "Frontend Developer"

    elif any(x in text for x in ["java", "spring", "api", "backend"]):
        return "Backend Developer"

    elif any(x in text for x in ["sql", "database", "analysis"]):
        return "Data Analyst"

    else:
        return "Software Developer"
