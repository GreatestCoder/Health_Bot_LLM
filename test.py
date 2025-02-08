import os
import io
import re
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

# Additional imports for voice functionality
import speech_recognition as sr
from gtts import gTTS
from streamlit_audiorecorder import audiorecorder

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")

# Set page configuration
st.set_page_config(page_title="AI-Powered Symptom Checker", page_icon="🩺", layout="wide")

# Add logo image
logo_path = r"C:\Users\LENOVO\Documents\AI\GenAI\LangChain\1-Langchain\Projects\Hackathon_Health_Bot\Designer.jpeg"
st.image(logo_path, width=100, use_column_width=False)

# Display title and description
st.markdown("""
    <h1 style='text-align: center; color: #2C3E50;'>AI-Powered Symptom Checker</h1>
    <p style='text-align: center; color: #7F8C8D;'>
        Get potential diagnoses based on your symptoms. This tool is for educational purposes only and is not a substitute for professional healthcare advice.
    </p>
""", unsafe_allow_html=True)

# Load dataset
dataset = pd.read_csv(r"C:\Users\LENOVO\Documents\AI\GenAI\LangChain\1-Langchain\Projects\Hackathon_Health_Bot\data\Symptom2Disease.csv")

# Cache model and vectorstore
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

# LangChain pipeline setup
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

# Ensure chat history initialization
def get_session_history(session: str) -> BaseChatMessageHistory:
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# Function to extract disease names from response
def extract_disease(response: str):
    diseases = ['Psoriasis', 'Varicose Veins', 'Typhoid', 'Chicken pox', 'Impetigo', 'Dengue', 'Fungal infection',
                'Common Cold', 'Pneumonia', 'Dimorphic Hemorrhoids', 'Arthritis', 'Acne', 'Bronchial Asthma', 'Hypertension',
                'Migraine', 'Cervical spondylosis', 'Jaundice', 'Malaria', 'Urinary tract infection', 'Allergy',
                'Gastroesophageal reflux disease', 'Drug reaction', 'Peptic ulcer disease', 'Diabetes']
    return [disease for disease in diseases if re.search(r'\b' + re.escape(disease) + r'\b', response, re.IGNORECASE)]

# User interface: Choose between Text and Voice Input
session_id = st.text_input("Session ID", value="default_session", key="session_id")
input_mode = st.radio("Choose input mode:", ["Text", "Voice"])

user_input = ""
if input_mode == "Text":
    user_input = st.text_area("Describe your symptoms:", height=100, placeholder="Type your symptoms here...")
else:
    st.markdown("Click the button below to record your symptoms.")
    audio_bytes = audiorecorder("Record", text="Recording...")
    if audio_bytes is not None:
        recognizer = sr.Recognizer()
        try:
            # Use SpeechRecognition to convert the audio bytes to text
            audio_file = sr.AudioFile(io.BytesIO(audio_bytes))
            with audio_file as source:
                audio_data = recognizer.record(source)
            user_input = recognizer.recognize_google(audio_data)
            st.write("You said:", user_input)
        except Exception as e:
            st.error("Error processing audio: " + str(e))

# Process input when user submits
if st.button("Submit", key="submit_button") and user_input:
    with st.spinner('Processing your symptoms...'):
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

    # Extract possible diseases
    possible_diseases = extract_disease(response["answer"])
    
    # Display text response
    st.markdown('<div style="background-color: #F4F6F7; padding: 15px; border-radius: 10px;">', unsafe_allow_html=True)
    if possible_diseases:
        st.write(f"Possible condition(s): **{', '.join(possible_diseases)}**")
    else:
        st.write("Assistant: ", response["answer"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate voice response using gTTS and play the audio
    try:
        tts = gTTS(response["answer"])
        tts.save("response.mp3")
        st.audio("response.mp3", format="audio/mp3")
    except Exception as e:
        st.error("Error generating voice response: " + str(e))

# Footer
st.markdown("""
    <p style="text-align: center; color: #7F8C8D; font-size: 12px;">
        This tool is for educational purposes only. Always consult a healthcare professional.
    </p>
""", unsafe_allow_html=True)
