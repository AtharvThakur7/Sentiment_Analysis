# from textblob import TextBlob
# import pandas as pd
# import streamlit as st
# import cleantext
# import logging
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import plotly.express as px
# import numpy as np

# # Configure logging
# logging.basicConfig(
#     filename="sentiment_analysis.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# # Initialize VADER Sentiment Analyzer
# analyzer = SentimentIntensityAnalyzer()

# # Set Streamlit Page Configuration
# st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# st.title("📝 Sentiment Analysis App")

# # ---------------- TEXT INPUT SENTIMENT ANALYSIS (REAL-TIME) ----------------
# with st.expander("🔍 Analyze Text"):
#     text = st.text_area("Enter text for sentiment analysis:", key="real_time_text")
    

#     if st.button("Analyze Sentiment"):
#         if text:
#             # Real-time sentiment prediction with VADER
#             sentiment_scores = analyzer.polarity_scores(text)
#             compound_score = sentiment_scores["compound"]

#             # Assign sentiment label based on compound score
#             if compound_score >= 0.05:
#                 sentiment = "Positive 😊"
#             elif compound_score <= -0.05:
#                 sentiment = "Negative 😡"
#             else:
#                 sentiment = "Neutral 😐"

#             st.write("**Sentiment Score:**", round(compound_score, 2))
#             st.write("**Sentiment:**", sentiment)

#             logging.info(f"Text analyzed: {text} | Sentiment: {sentiment}")

#         # ------------------- Word-Level Sentiment Analysis -------------------
#         # Split the input text into words
#         words = text.split()
        
#         word_sentiments = []
#         for word in words:
#             word_score = analyzer.polarity_scores(word)["compound"]
#             word_sentiments.append((word, word_score))
        
#         # Convert word sentiments to a DataFrame for display
#         word_df = pd.DataFrame(word_sentiments, columns=["Word", "Sentiment Score"])
#         word_df["Sentiment"] = word_df["Sentiment Score"].apply(lambda x: 'Positive 😊' if x > 0.05 else ('Negative 😡' if x < -0.05 else 'Neutral 😐'))

#         st.write("**Word-Level Sentiment Analysis:**")
#         st.dataframe(word_df)

#     # Text Cleaning
#     clean_text_input = st.text_area("Enter text to clean:")
    
#     if st.button("Clean Text"):
#         try:
#             if clean_text_input:
#                 cleaned_text = cleantext.clean(clean_text_input, 
#                                                clean_all=False, 
#                                                extra_spaces=True,
#                                                stopwords=True, 
#                                                lowercase=True, 
#                                                numbers=True, 
#                                                punct=True)
#                 st.write("**Cleaned Text:**", cleaned_text)
#                 logging.info(f"Text cleaned: {clean_text_input}")

#         except Exception as e:
#             st.error("⚠️ An error occurred while cleaning text.")
#             logging.error(f"Error in text cleaning: {e}")

# # ---------------- CSV UPLOAD & SENTIMENT ANALYSIS ----------------
# with st.expander("📂 Analyze CSV"):
#     upl = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

#     if upl:
#         try:
#             # Detect file type and load data
#             file_extension = upl.name.split('.')[-1]
#             df = pd.read_csv(upl) if file_extension == 'csv' else pd.read_excel(upl)

#             st.write("**Uploaded Data Preview:**", df.head())

#             # Detect text (categorical) columns only
#             text_columns = df.select_dtypes(include=['object']).columns.tolist()

#             if not text_columns:
#                 st.error("❌ No text columns found! Please upload a file with textual data.")
#                 logging.warning("CSV uploaded with no text columns detected.")
#             else:
#                 selected_col = st.selectbox("📝 Select a column for sentiment analysis:", text_columns)

#                 if st.button("Analyze CSV"):
#                     def vader_score(text):
#                         scores = analyzer.polarity_scores(str(text))
#                         return scores['compound']

#                     def analyze(score):
#                         if score >= 0.05:
#                             return 'Positive 😊'
#                         elif score <= -0.05:
#                             return 'Negative 😡'
#                         else:
#                             return 'Neutral 😐'

#                     # Apply sentiment functions
#                     df_result = df[[selected_col]].copy()
#                     df_result['Sentiment Score'] = df_result[selected_col].astype(str).apply(vader_score)
#                     df_result['Sentiment Analysis'] = df_result['Sentiment Score'].apply(analyze)

#                     st.write("**Processed Data Preview:**", df_result.head(10))
#                     logging.info(f"CSV analyzed successfully. Column: {selected_col}")

#                     # Visualization: Sentiment Distribution
#                     sentiment_counts = df_result['Sentiment Analysis'].value_counts()
#                     sentiment_fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values, 
#                                            labels={'x': 'Sentiment', 'y': 'Count'}, 
#                                            title="Sentiment Distribution")
#                     st.plotly_chart(sentiment_fig)

#                     # Download button
#                     @st.cache_data
#                     def convert_df(df):
#                         return df.to_csv(index=False).encode('utf-8')

#                     csv = convert_df(df_result)

#                     st.download_button(
#                         label="📥 Download Processed Data as CSV",
#                         data=csv,
#                         file_name='sentiment_analysis.csv',
#                         mime='text/csv',
#                     )
#         except Exception as e:
#             st.error("⚠️ An error occurred while processing the CSV file.")
#             logging.error(f"Error in CSV processing: {e}")









from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import re
from transformers import pipeline

# Configure logging
logging.basicConfig(
    filename="sentiment_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Load Pretrained Sarcasm Detection Model (Optional)
try:
    sarcasm_detector = pipeline("text-classification", model="mvanvo/sarcasm-detector")
    use_sarcasm_model = True
except Exception as e:
    logging.error(f"Failed to load sarcasm model: {e}")
    use_sarcasm_model = False

# Set Streamlit Page Configuration
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.title("📝 Sentiment Analysis App with Sarcasm Detection")

# ---------------- SARCASTIC TEXT DETECTION FUNCTION ----------------
def detect_sarcasm(text):
    """
    Detects sarcasm using a combination of rule-based methods and a deep-learning model.
    """
    sarcastic_patterns = [
        r"oh\s*(wow|great|sure)",  # "Oh wow", "Oh great"
        r"(yeah|sure|right),?\s*because",  # "Yeah, because that totally makes sense"
        r"just\s*(love|hate)",  # "I just love waiting in line"
        r"as\s*if",  # "As if that would ever work"
        r"not\s+(like|that)",  # "Not like I was busy or anything"
    ]
    
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in sarcastic_patterns):
        return True

    # Use deep learning model if available
    if use_sarcasm_model:
        prediction = sarcasm_detector(text)[0]
        return prediction["label"] == "sarcasm"
    
    return False

# ---------------- TEXT INPUT SENTIMENT ANALYSIS (REAL-TIME) ----------------
with st.expander("🔍 Analyze Text"):
    text = st.text_area("Enter text for sentiment analysis:", key="real_time_text")

    if st.button("Analyze Sentiment"):
        if text:
            sarcasm_detected = detect_sarcasm(text)

            # Real-time sentiment prediction with VADER
            sentiment_scores = analyzer.polarity_scores(text)
            compound_score = sentiment_scores["compound"]

            # Adjust sentiment if sarcasm is detected
            if sarcasm_detected:
                sentiment = "Sarcastic 🧐"
                compound_score = 0.0  # Neutralizing sarcastic sentiment as it often contradicts words
            elif compound_score >= 0.05:
                sentiment = "Positive 😊"
            elif compound_score <= -0.05:
                sentiment = "Negative 😡"
            else:
                sentiment = "Neutral 😐"

            st.write("**Sentiment Score:**", round(compound_score, 2))
            st.write("**Sentiment:**", sentiment)

            logging.info(f"Text analyzed: {text} | Sentiment: {sentiment}")

        # ------------------- Word-Level Sentiment Analysis -------------------
        words = text.split()
        word_sentiments = [(word, analyzer.polarity_scores(word)["compound"]) for word in words]

        # Convert to DataFrame
        word_df = pd.DataFrame(word_sentiments, columns=["Word", "Sentiment Score"])
        word_df["Sentiment"] = word_df["Sentiment Score"].apply(
            lambda x: 'Positive 😊' if x > 0.05 else ('Negative 😡' if x < -0.05 else 'Neutral 😐')
        )

        st.write("**Word-Level Sentiment Analysis:**")
        st.dataframe(word_df)

    # Text Cleaning
    clean_text_input = st.text_area("Enter text to clean:")
    
    if st.button("Clean Text"):
        try:
            if clean_text_input:
                cleaned_text = cleantext.clean(clean_text_input, extra_spaces=True,
                                               stopwords=True, lowercase=True,
                                               numbers=True, punct=True)
                st.write("**Cleaned Text:**", cleaned_text)
                logging.info(f"Text cleaned: {clean_text_input}")
        except Exception as e:
            st.error("⚠️ An error occurred while cleaning text.")
            logging.error(f"Error in text cleaning: {e}")

# ---------------- CSV UPLOAD & SENTIMENT ANALYSIS ----------------
with st.expander("📂 Analyze CSV"):
    upl = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

    if upl:
        try:
            file_extension = upl.name.split('.')[-1]
            df = pd.read_csv(upl) if file_extension == 'csv' else pd.read_excel(upl)
            st.write("**Uploaded Data Preview:**", df.head())

            text_columns = df.select_dtypes(include=['object']).columns.tolist()

            if not text_columns:
                st.error("❌ No text columns found! Please upload a file with textual data.")
                logging.warning("CSV uploaded with no text columns detected.")
            else:
                selected_col = st.selectbox("📝 Select a column for sentiment analysis:", text_columns)

                if st.button("Analyze CSV"):
                    def vader_score(text):
                        if detect_sarcasm(str(text)):
                            return 0.0  # Neutralize sarcasm sentiment
                        scores = analyzer.polarity_scores(str(text))
                        return scores['compound']

                    def analyze(score):
                        if score == 0.0:
                            return 'Sarcastic 🧐'
                        elif score >= 0.05:
                            return 'Positive 😊'
                        elif score <= -0.05:
                            return 'Negative 😡'
                        else:
                            return 'Neutral 😐'

                    df_result = df[[selected_col]].copy()
                    df_result['Sentiment Score'] = df_result[selected_col].astype(str).apply(vader_score)
                    df_result['Sentiment Analysis'] = df_result['Sentiment Score'].apply(analyze)

                    st.write("**Processed Data Preview:**", df_result.head(10))
                    logging.info(f"CSV analyzed successfully. Column: {selected_col}")

                    sentiment_counts = df_result['Sentiment Analysis'].value_counts()
                    sentiment_fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values, title="Sentiment Distribution")
                    st.plotly_chart(sentiment_fig)

        except Exception as e:
            st.error("⚠️ An error occurred while processing the CSV file.")
            logging.error(f"Error in CSV processing: {e}")

