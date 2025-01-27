import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import re

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")

# Set page configuration
st.set_page_config(page_title="AI-Powered Symptom Checker", page_icon="🩺", layout="wide")

# Add logo image
logo_path = r"C:\Users\LENOVO\Documents\AI\GenAI\LangChain\1-Langchain\Projects\Hackathon_Health_Bot\Designer.jpeg"  

# Display logo above the title
st.image(logo_path, width=100, use_column_width=False)

# Header section with styling
st.markdown("""  
    <style>
        .main-header {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #FFFFFF;  /* White for header text */
            margin-bottom: 20px;
        }
        .sub-header {
            text-align: center;
            font-size: 18px;
            color: #D1E8E4;  /* Softer, light color for sub-header */
            margin-bottom: 40px;
        }
        .input-box {
            font-size: 16px;
            border-radius: 5px;
        }
        .button {
            background-color: #58D68D;  /* Green background for the button */
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .button:hover {
            background-color: #45B26B;  /* Darker green on hover */
        }
        .response-card {
            background-color: #F4F6F7;  /* Light pastel background for response cards */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
            border-left: 5px solid #58D68D;  /* Green left border to match the button */
        }
        .footer {
            text-align: center;
            font-size: 12px;
            color: #BDC3C7;
            margin-top: 50px;
        }
        body {
            background-color: #F0F8FF;  /* Light blue background for a soothing feel */
            color: #333;  /* Dark text for readability */
        }
    </style>
""", unsafe_allow_html=True)

# Display title and description
st.markdown('<div class="main-header">AI-Powered Symptom Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Get potential diagnoses based on your symptoms. Remember, this is not a substitute for professional healthcare advice.</div>', unsafe_allow_html=True)

# Step 1: Load the dataset and process it
dataset = pd.read_csv(r"C:\Users\LENOVO\Documents\AI\GenAI\LangChain\1-Langchain\Projects\Hackathon_Health_Bot\data\Symptom2Disease.csv")

# Step 2: Cache the model and vectorstore
@st.cache_resource
def load_model_and_vectorstore():
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = dataset['text'].tolist()
    metadata = [{"label": label} for label in dataset['label'].tolist()]
    vectorstore = FAISS.from_texts(documents, embedder, metadatas=metadata)
    
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    retriever = vectorstore.as_retriever()

    return llm, retriever

with st.spinner('Building the FAISS vectorstore. Please wait...'):
    llm, retriever = load_model_and_vectorstore()

st.success("Vectorstore built successfully!")

# Step 3: Initialize LLM and retriever
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Step 4: Set up LangChain pipeline
contextualize_q_system_prompt = (
    "You are a multilingual assistant for diagnosing symptoms. "
    "Your task is to help users understand health issues by formulating clear and concise questions based on their input. "
    "Always consider possible diseases related to the symptoms provided and include them in your response unless the user is engaging in normal conversation. "
    "If the user's question references previous conversations, reframe it as a standalone question. "
    "Always ensure the question can be understood without additional context."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_chain = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are an AI assistant specialized in healthcare, normal conversation along with language translation capabilities. "
    "You should help users by diagnosing symptoms and providing possible diseases related to their health conditions. "
    "When a user describes symptoms like chest pain, fatigue, or other medical issues, identify relevant diseases and include their names in your response, "
    "and explain them in simple terms, along with suggested solutions.. "
    "However, if the user inputs a casual greeting or normal conversation (e.g., 'hello', 'how are you', 'what's up' etc), "
    "respond in a friendly, conversational manner without attempting a medical diagnosis."
    "Only provide a diagnosis if the user mentions symptoms related to health conditions. "
    "Always ensure that when the user provides symptoms, your response focuses on possible health conditions, "
    "but for normal conversations, respond casually."
    "If the user asks their question in a different language, translate your response into that language. "
    "If you don't know the answer, say 'I don't know' and suggest consulting a healthcare professional.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_chain, question_answer_chain)

# Step 5: Enable chat history
def Get_Session_History(session:str)->BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

conversational_rag_chain = RunnableWithMessageHistory(rag_chain, 
                                                      Get_Session_History, 
                                                      input_messages_key="input", 
                                                      history_messages_key="chat_history", 
                                                      output_messages_key="answer")

# Function to extract disease names from the response
def extract_disease(response: str):
    diseases = ['Psoriasis', 'Varicose Veins', 'Typhoid', 'Chicken pox', 'Impetigo', 'Dengue', 'Fungal infection', 'Common Cold', 
                'Pneumonia', 'Dimorphic Hemorrhoids', 'Arthritis', 'Acne', 'Bronchial Asthma', 'Hypertension', 'Migraine', 
                'Cervical spondylosis', 'Jaundice', 'Malaria', 'Urinary tract infection', 'Allergy', 'Gastroesophageal reflux disease', 
                'Drug reaction', 'Peptic ulcer disease', 'Diabetes']
    
    found_diseases = [disease for disease in diseases if re.search(r'\b' + re.escape(disease) + r'\b', response, re.IGNORECASE)]
    return found_diseases

# Step 6: User interface
session_id = st.text_input("Session_ID", value="default_session", key="session_id")
if "store" not in st.session_state:
    st.session_state.store = {}

# Use st.text_area for multi-line input
user_input = st.text_area("Describe your symptoms:", height=100, key="input_area", placeholder="Type your symptoms here...")

# Create a submit button with enhanced styling
submit_button = st.button("Submit", key="submit_button", help="Click to submit your symptoms")

if submit_button and user_input:
    with st.spinner('Processing your symptoms...'):
        session_history = Get_Session_History(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

    # Extract possible diseases
    possible_diseases = extract_disease(response["answer"])
    
    # Display results in styled response card
    st.markdown('<div class="response-card">', unsafe_allow_html=True)
    if possible_diseases:
        st.write(f"Possible condition(s): **{', '.join(possible_diseases)}**")
    else:
        st.write("Assistant: ", response["answer"])
    st.markdown('</div>', unsafe_allow_html=True)

# Footer section
st.markdown("""
    <div class="footer">
        This tool is for educational purposes only. Always consult a healthcare professional for medical advice.
    </div>
""", unsafe_allow_html=True)
