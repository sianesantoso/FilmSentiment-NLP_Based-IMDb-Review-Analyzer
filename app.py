import streamlit as st
import re
import nltk
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download NLP resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')


# NLP Tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Text Prepocessing
def preprocess_text(text):
    cleaned_text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text.lower()
    )
    tokens = word_tokenize(cleaned_text)
    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return ' '.join(filtered_tokens)

# Load Model
with open('vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit Config
st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)



# UI Design 
st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }
    
    .title {
        text-align:center;
        font-size:45px;
        font-weight:800;
        color:#111827;
    }

    .subtitle {
        text-align:center;
        color:#6b7280;
        font-size:18px;
        margin-bottom:35px;
    }

    .stTextArea textarea {
    background-color: white !important;
    border-radius: 15px !important;
    border: 1px solid #e5e7eb !important;
    padding: 15px !important;
    font-size: 16px !important;
    }


    .stTextArea textarea:focus {
    border: 2px solid #2563eb !important;
    }

    .positive {
        color:#16a34a;
        font-size:35px;
        font-weight:bold;
    }

    .negative {
        color:#dc2626;
        font-size:35px;
        font-weight:bold;
    }

    .warning {
        color:#dc2626;
        font-weight:bold;
        text-align:center;
    }

    div.stButton > button {
        width:100%;
        height:50px;
        border-radius:12px;
        background:#2563eb;
        color:white;
        font-size:18px;
        font-weight:bold;
        border:none;
    }

    div.stButton > button:hover {
        background:#1d4ed8;
        color:white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <div class="title">
    🎬 IMDb Sentiment Analysis
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    A ML application that predicts movie review sentiment
    using NLP.

    </div>
    """,
    unsafe_allow_html=True
)

# Input Review
st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

user_input = st.text_area(
    "Write your movie review in English",
    placeholder=
    "Example: The movie was very good !",
    height=200
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# Prediction
if st.button("🔍 Analyze Sentiment"):
    if user_input:
        # preprocessing
        cleaned_input = preprocess_text(user_input)

        # TF-IDF
        transformed_input = tfidf.transform(
            [cleaned_input]
        )

        # prediction
        prediction = model.predict(
            transformed_input
        )[0]

        sentiment = (
            "Positive"
            if prediction == 1
            else
            "Negative"
        )

        # result display
        if sentiment == "Positive":
            st.markdown(
                f"""
                <div class="result-card">
                    <h3>Prediction Result</h3>
                    
                    <div class="positive">
                    😊 {sentiment}
                    </div>
    
                    <p>
                    The review indicates a positive sentiment.
                    </p>
                </div>
                
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"""
                <div class="result-card">
                <h3>Prediction Result</h3>

                <div class="negative">
                😞 {sentiment}
                </div>

                <p>
                The review indicates a negative sentiment.
                </p>
                </div>
                
                """,
                unsafe_allow_html=True
            )

    else:
        st.markdown(
            """
            <p class="warning">
            ⚠️ Please enter a review before analyzing.
            </p>
            """,
            unsafe_allow_html=True

        )
