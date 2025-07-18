import re
import nltk
import random
import string
import warnings
import json
import pdfplumber
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import std_questions

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load configurations and knowledge base from JSON files
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

with open('knowledge_base.json', 'r') as kb_file:
    kb = json.load(kb_file)

# Extract commonly used values from config
exit_commands = config['exit_commands']
greeting_inputs = config['greeting_inputs']
greeting_responses = config['greeting_responses']
meet_questions = config['meet_questions']
ignorant_responses = config['ignorant_responses']

# Gemini section (Google Generative AI API)
import os
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key="AIzaSyBt8o3P6KJoaC8nixCFzWxROrHnavQoTz4")

# Set up generation configuration for Gemini
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 40,
    "response_mime_type": "text/plain",
}

# Initialize Gemini model and chat session
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

chat_session = model.start_chat(history=[])

# Function to call Gemini and get response
def call_gemini(query):
    query += ", answer in 1 sentence exactly within 20 tokens if related to farming else say i don't understand"
    response = chat_session.send_message(query)
    print("AgroGuru: " + response.text[:-2])  # Trim unnecessary end of response

# Function to match a user response with predefined questions
def present(user_input, question_dict):
    for key, value in question_dict.items():
        if re.match(value, user_input):
            return True, key
    return False, ""

# Data extraction functions for PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Failed to extract text from {pdf_path}: {e}")
    return text

# Data extraction function for websites
def extract_text_from_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            return '\n'.join([para.get_text() for para in paragraphs])
        else:
            print(f"Failed to retrieve content from {url}")
            return ""
    except Exception as e:
        print(f"An error occurred while fetching {url}: {e}")
        return ""

# Function to load data from both files and websites in the knowledge base
def load_data_from_sources():
    combined_text = ""
    for file_path in kb.get('files', []):
        if file_path.endswith('.pdf'):
            combined_text += extract_text_from_pdf(file_path) + "\n"
        elif file_path.endswith('.txt'):
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    combined_text += f.read() + "\n"
            except Exception as e:
                print(f"Failed to read file {file_path}: {e}")
    
    for url in kb.get('websites', []):
        combined_text += extract_text_from_website(url) + "\n"
    
    return combined_text

# Text preprocessing functions for NLP
lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting function to handle user greetings
def greeting(user_response):
    for word in user_response.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)

# Main response function using TF-IDF and cosine similarity
def response(user_response, sent_tokens):
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]  # Index of the second most similar sentence
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        chatbot_response = "I am sorry! I am unable to understand you."
    else:
        chatbot_response = sent_tokens[idx]

    sent_tokens.remove(user_response)
    return chatbot_response

# Main chatbot loop
def chatbot():
    raw_text = load_data_from_sources().lower()
    sent_tokens = nltk.sent_tokenize(raw_text)  # List of sentences
    word_tokens = nltk.word_tokenize(raw_text)  # List of words
    print("Hello, there! My name is AgroGuru Assistant. How can I help you?")
    flag = True
    while flag:
        user_response = input("You: ").lower()
        
        if user_response in exit_commands:
            flag = False
            print("AgroGuru: Bye! Have a great time!")
        elif user_response in ['thanks', 'thank you']:
            flag = False
            print("AgroGuru: You're welcome!")
        elif user_response in meet_questions:
            print("AgroGuru: I am doing well..")
        elif user_response in greeting_inputs:
            print("AgroGuru: Hi, How can I help you..")
        elif present(user_response, std_questions.std_questions)[0]:
            matched_response = present(user_response, std_questions.std_questions)[1]
            print("AgroGuru: " + response(matched_response, sent_tokens))
        else:
            call_gemini(user_response)

# Run the chatbot
if __name__ == "__main__":

    chatbot()
