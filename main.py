import os
import io
import re
import math
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
import speech_recognition as sr
from gtts import gTTS
from st_audiorec import st_audiorec
from PIL import Image

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")

st.set_page_config(page_title="AI-Powered Symptom Checker", page_icon="🩺", layout="wide")
logo_path = r"C:\Users\LENOVO\Documents\AI\GenAI\LangChain\1-Langchain\Projects\Hackathon_Health_Bot\Designer.jpeg"
st.image(logo_path, width=100, use_column_width=False)

st.markdown("""
    <h1 style='text-align: center; color: #2C3E50;'>AI-Powered Symptom Checker</h1>
    <p style='text-align: center; color: #7F8C8D;'>
        Get potential diagnoses based on your symptoms. This tool is for educational purposes only and is not a substitute for professional healthcare advice.
    </p>
""", unsafe_allow_html=True)

dataset = pd.read_csv(r"C:\Users\LENOVO\Documents\AI\GenAI\LangChain\1-Langchain\Projects\Hackathon_Health_Bot\data\Symptom2Disease.csv")

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
    "You are an AI assistant specialized in healthcare, with normal conversation and language translation capabilities. "
    "You should help users by diagnosing symptoms and providing possible diseases related to their health conditions. "
    "When a user describes symptoms like chest pain, fatigue, or other medical issues, identify relevant diseases and include their names in your response, "
    "and explain them in simple terms, along with suggested solutions. "
    "However, if the user inputs a casual greeting or normal conversation (e.g., 'hello', 'how are you', 'what's up' etc), "
    "respond in a friendly, conversational manner without attempting a medical diagnosis. "
    "Only provide a diagnosis if the user mentions symptoms related to health conditions. "
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

def extract_disease(response: str):
    diseases = ['Psoriasis', 'Varicose Veins', 'Typhoid', 'Chicken pox', 'Impetigo', 'Dengue', 'Fungal infection',
                'Common Cold', 'Pneumonia', 'Dimorphic Hemorrhoids', 'Arthritis', 'Acne', 'Bronchial Asthma', 'Hypertension',
                'Migraine', 'Cervical spondylosis', 'Jaundice', 'Malaria', 'Urinary tract infection', 'Allergy',
                'Gastroesophageal reflux disease', 'Drug reaction', 'Peptic ulcer disease', 'Diabetes']
    return [disease for disease in diseases if re.search(r'\b' + re.escape(disease) + r'\b', response, re.IGNORECASE)]

session_id = st.text_input("Session ID", value="default_session", key="session_id")
input_mode = st.radio("Choose input mode:", ["Text", "Voice", "Image"])

user_input = ""
image_input = None
if input_mode == "Text":
    user_input = st.text_area("Describe your symptoms:", height=100, placeholder="Type your symptoms here...")
elif input_mode == "Voice":
    st.markdown("Click the button below to record your symptoms.")
    audio_bytes = st_audiorec()
    if audio_bytes is not None:
        recognizer = sr.Recognizer()
        try:
            audio_file = sr.AudioFile(io.BytesIO(audio_bytes))
            with audio_file as source:
                audio_data = recognizer.record(source)
            user_input = recognizer.recognize_google(audio_data)
            st.write("You said:", user_input)
        except Exception as e:
            st.error("Error processing audio: " + str(e))
elif input_mode == "Image":
    image_input = st.file_uploader("Upload an image of your symptom (e.g., a rash or skin condition)", type=["jpg", "jpeg", "png"])
    if image_input:
        image = Image.open(image_input)
        st.image(image, caption="Uploaded image", use_column_width=False)

vision_output = ""
if image_input:
    try:
        import torch
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model_vision = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor_vision = AutoProcessor.from_pretrained(model_id)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Read the symptom shown in this image and give the possible diseases, and remedies if asked for as well."}
            ]}
        ]
        input_text_vision = processor_vision.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor_vision(
            image,
            input_text_vision,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model_vision.device)
        output = model_vision.generate(**inputs, max_new_tokens=30)
        vision_output = processor_vision.decode(output[0])
        st.write("Vision model output:", vision_output)
    except Exception as e:
        st.error("Error processing image with vision model: " + str(e))

# Combine inputs from different modes
combined_input = ""
if image_input and (user_input or vision_output):
    combined_input = f"{user_input}\nImage description: {vision_output}"
elif image_input:
    combined_input = f"Image description: {vision_output}"
else:
    combined_input = user_input

# Process input when user submits
if st.button("Submit", key="submit_button") and combined_input:
    with st.spinner('Processing your input...'):
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": combined_input},
            config={"configurable": {"session_id": session_id}}
        )

    possible_diseases = extract_disease(response["answer"])
    st.markdown('<div style="background-color: #F4F6F7; padding: 15px; border-radius: 10px;">', unsafe_allow_html=True)
    if possible_diseases:
        st.write(f"Possible condition(s): **{', '.join(possible_diseases)}**")
    else:
        st.write("Assistant: ", response["answer"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        tts = gTTS(response["answer"])
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer, format="audio/mp3")
    except Exception as e:
        st.error("Error generating voice response: " + str(e))

st.markdown("""
    <p style="text-align: center; color: #7F8C8D; font-size: 12px;">
        This tool is for educational purposes only. Always consult a healthcare professional.
    </p>
""", unsafe_allow_html=True)

def top_k_accuracy(predictions, ground_truth, k):
    correct = 0
    total = len(predictions)
    for pred, gt in zip(predictions, ground_truth):
        if any(item in pred[:k] for item in gt):
            correct += 1
    return correct / total if total > 0 else 0

def mean_reciprocal_rank(predictions, ground_truth):
    mrr = 0.0
    total = len(predictions)
    for pred, gt in zip(predictions, ground_truth):
        for rank, item in enumerate(pred, start=1):
            if item in gt:
                mrr += 1.0 / rank
                break
    return mrr / total if total > 0 else 0

def average_precision(pred, gt):
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(pred, start=1):
        if item in gt:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / len(gt) if gt else 0

def mean_average_precision(predictions, ground_truth):
    total_ap = 0.0
    total = len(predictions)
    for pred, gt in zip(predictions, ground_truth):
        total_ap += average_precision(pred, gt)
    return total_ap / total if total > 0 else 0

def dcg_at_k(pred, gt, k):
    dcg = 0.0
    for i, item in enumerate(pred[:k], start=1):
        if item in gt:
            dcg += 1.0 / math.log2(i + 1)
    return dcg

def ndcg_at_k(predictions, ground_truth, k):
    ndcg_total = 0.0
    total = len(predictions)
    for pred, gt in zip(predictions, ground_truth):
        dcg = dcg_at_k(pred, gt, k)
        ideal_hits = min(len(gt), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_total += ndcg
    return ndcg_total / total if total > 0 else 0

if __name__ == "__main__":
    predictions = [
        ["Hypertension", "Migraine", "Diabetes"],
        ["Common Cold", "Pneumonia", "Asthma"],
        ["Diabetes", "Hypertension", "Arthritis"]
    ]
    ground_truth = [
        {"Migraine", "Hypertension"},
        {"Pneumonia"},
        {"Hypertension"}
    ]
    
    k = 3
    print("Top-{} Accuracy: {:.2f}".format(k, top_k_accuracy(predictions, ground_truth, k)))
    print("Mean Reciprocal Rank (MRR): {:.2f}".format(mean_reciprocal_rank(predictions, ground_truth)))
    print("Mean Average Precision (MAP): {:.2f}".format(mean_average_precision(predictions, ground_truth)))
    print("NDCG@{}: {:.2f}".format(k, ndcg_at_k(predictions, ground_truth, k)))