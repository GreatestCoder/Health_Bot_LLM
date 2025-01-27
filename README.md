Hackathon Health Bot 🩺
Hackathon Health Bot is an AI-powered symptom checker that uses natural language understanding to assist users in diagnosing potential health conditions based on their input symptoms. The bot is built using the LangChain framework, with vector-based retrieval and LLM (Large Language Models) to enhance user interactions.

Features:
Provides possible health conditions based on the user's symptoms.
Interactive and friendly conversation flow.
Supports multiple languages and healthcare-related queries.
Utilizes FAISS for efficient symptom similarity search.
Powered by Groq for LLM inference and HuggingFace embeddings.

Folder Structure:
Hackathon_Health_Bot/
│
├── data/
│   └── Symptom2Disease.csv          # Dataset containing symptoms and related diseases
│
├── .env                             # Environment variables (excluded from GitHub)
│
├── app.py                           # Main Python script for running the bot
│
└── README.md                        # This file

Setup Instructions:
1. Clone the Repository
First, clone this repository to your local machine:
git clone https://github.com/GreatestCoder/AI_Projects/Hackathon_Health_Bot.git
cd Hackathon_Health_Bot
2. Set Up Environment Variables
Create a .env file in the root directory and add your API keys:
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
3. Prepare the Dataset
Ensure the data/Symptom2Disease.csv file is present.
Download the dataset [here](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease).
4. Run the Bot
To run the AI-powered symptom checker, execute:
streamlit run app.py
This will launch the app in your default browser. You can start interacting with the bot to diagnose symptoms.

Usage:
Start the Application: Once the app is running, input symptoms like fever, cough, etc.
Receive Diagnosis: The bot will suggest possible conditions based on your input.
User-Friendly Responses: The bot offers explanations in simple language and also supports casual conversation.

Example Output:
User: I have a sore throat and fever.
Bot: Based on your symptoms, you may be experiencing **Common Cold** or **Flu**. Please consult a healthcare professional for further guidance.

Technologies Used:
Python: Core programming language.
Streamlit: For creating the web interface.
LangChain: Manages conversation flow and interaction with large language models.
Groq: Utilized for LLM-based responses.
FAISS: For efficient similarity search in symptom data.
HuggingFace: Provides the model embeddings.

Contributing:
Contributions are welcome! If you'd like to improve this project, please open an issue or submit a pull request.

License
This project is not under any license.


