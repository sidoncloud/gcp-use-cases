import streamlit as st
from google.cloud import storage
from langchain.vectorstores.chroma import Chroma
from langchain_google_vertexai import VertexAI,VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os

# GCS bucket details
BUCKET_NAME = "your-bucket-name"
GCS_PERSIST_PATH = "chroma/"
LOCAL_PERSIST_PATH = "./local_chromadb/"

# Initialize GCS client
storage_client = storage.Client()

def download_directory_from_gcs(gcs_directory, local_directory, bucket_name):
    """Download all files from a GCS directory to a local directory."""
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_directory)

    for blob in blobs:
        if not blob.name.endswith("/"):  # Avoid directory blobs
            relative_path = os.path.relpath(blob.name, gcs_directory)
            local_file_path = os.path.join(local_directory, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)
            print(f"Downloaded {blob.name} to {local_file_path}")

# Download Chroma persisted data from GCS to local directory
download_directory_from_gcs(GCS_PERSIST_PATH, LOCAL_PERSIST_PATH, BUCKET_NAME)

# Step to use the data locally in retrieval
EMBEDDING_MODEL = "textembedding-gecko@003"
EMBEDDING_NUM_BATCH = 5

# Load embeddings and persisted data
embeddings = VertexAIEmbeddings(
    model_name=EMBEDDING_MODEL, batch_size=EMBEDDING_NUM_BATCH
)

# Load Chroma data from local persisted directory
db = Chroma(persist_directory=LOCAL_PERSIST_PATH, embedding_function=embeddings)

# Now use db for retrieval
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

template = """
    You are a helpful AI assistant. You're tasked to answer the question given below, but only based on the context provided.
    context:

    {context}


    question:

    {input}


    If you cannot find an answer ask the user to rephrase the question.
    answer:
"""
prompt = PromptTemplate.from_template(template)

# OpenAI model configuration
api_key = "YOUR_API_KEY"
llm_openai = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0)

llm_gemini = VertexAI(
    model="gemini-1.5-pro",
    max_output_tokens=2048,
    temperature=0.2,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

conversational_retrieval = ConversationalRetrievalChain.from_llm(
    llm=llm_openai, retriever=retriever, memory=memory, verbose=False
)

# Streamlit app
st.set_page_config(page_title="Conversational AI Chatbot", layout="centered")

st.title("AI Assistant Chatbot")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user's query
user_input = st.chat_input("Your message")

if user_input:
    # Display user's message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Store user's query in the chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get the AI assistant's response
    response = conversational_retrieval({"question": user_input})["answer"]

    # Store AI's response in the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display assistant's message
    with st.chat_message("assistant"):
        st.markdown(response)

# Option to clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    memory.clear()
    st.experimental_rerun()
