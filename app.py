import streamlit as st
import os
import torch
import chromadb
from pypdf import PdfReader
from PIL import Image
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from google.genai import errors as genai_errors
import time

# -----------------------------------------
# 1. INITIALIZATION & CACHING
# -----------------------------------------
st.set_page_config(page_title="Multimodal AI Assistant", layout="wide")

@st.cache_resource
def load_models():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in .env")
        st.stop()
        
    client = genai.Client(api_key=api_key)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    embed_model = SentenceTransformer(
        "google/embeddinggemma-2",
        model_kwargs={"torch_dtype": dtype},
        device=device
    )
    
    db_client = chromadb.PersistentClient(path="./my_ai_memory_local")
    
    return client, embed_model, db_client

client, embed_model, db_client = load_models()
COLLECTION_NAME = "dynamic_workspace"

# -----------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------
def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    words = text.split()
    
    # Chunking: 500 words, 50 overlap
    chunks = [" ".join(words[i:i + 500]) for i in range(0, len(words), 450) if words[i:i + 500]]
    formatted_chunks = [f"title: none | text: {c}" for c in chunks]
    
    collection = db_client.get_or_create_collection(name=COLLECTION_NAME)
    
    try:
        db_client.delete_collection(name=COLLECTION_NAME)
        collection = db_client.create_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    
    # Embed on RTX 3060
    embeddings = embed_model.encode(formatted_chunks, batch_size=16, normalize_embeddings=True)
    
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks
    )
    return len(chunks)

def retrieve_context(query):
    try:
        collection = db_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return "No document uploaded yet."
        
    formatted_query = f"task: search result | query: {query}"
    query_vec = embed_model.encode([formatted_query], normalize_embeddings=True).tolist()[0]
    results = collection.query(query_embeddings=[query_vec], n_results=2)
    
    if results["documents"] and results["documents"][0]:
        return "\n".join(results["documents"][0])
    return "No relevant context found."

# -----------------------------------------
# 3. STREAMLIT UI LAYOUT
# -----------------------------------------
st.title("Multimodal AI Assistant")

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# SIDEBAR: Document Ingestion
with st.sidebar:
    st.header("1. Upload Knowledge (PDF)")
    uploaded_file = st.file_uploader("Drag & Drop a manual or datasheet", type=["pdf"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Extracting and calculating GPU embeddings..."):
            num_chunks = process_pdf(uploaded_file)
            st.success(f"Successfully indexed {num_chunks} text chunks into local ChromaDB!")

# MAIN AREA: Camera & Chat
col1, col2 = st.columns([1, 1])

with col1:
    st.header("2. Visual Input")
    camera_photo = st.camera_input("Point at a component and click capture")

with col2:
    st.header("3. Ask Question")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_query = st.chat_input("Ask about the document or the captured image...")
    
    if user_query:
        # Display user message
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # 1. Retrieve Context
        context = retrieve_context(user_query)
        
        # 2. Extract Recent Conversation History (Exclude the current query just added)
        recent_history = st.session_state.messages[-5:-1]
        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_history])
        if not history_str:
            history_str = "No prior conversation."

        # 3. Build Prompt with History + Context
        prompt = (
            f"You are a technical assistant. Answer concisely and accurately using the document context and recent dialogue.\n\n"
            f"Document Context:\n{context}\n\n"
            f"Prior Discussion:\n{history_str}\n\n"
            f"Current Question: {user_query}"
        )
        
        contents = [prompt]
        
        # Add image if captured
        if camera_photo:
            pil_img = Image.open(camera_photo)
            contents.insert(1, pil_img)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking... (connecting to Gemini API)"):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.5-flash", # Configured to fallback model to prevent 503 errors
                            contents=contents
                        )
                        
                        # Display the AI answer
                        st.markdown(response.text)
                        
                        # EXPANDER: Display Ground Truth Source
                        with st.expander("🔍 View Retrieved Document Sources"):
                            st.info(context)
                            
                        # Save assistant response to memory
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                        break 
                        
                    except genai_errors.ServerError as e:
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt) 
                            continue
                        else:
                            st.warning("⚠️ Google's Gemini servers are currently overloaded. Please wait a few seconds and try clicking 'Submit' again.")
                            
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                        break