import os
import time
import torch
import chromadb
import pygame
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# 1. Environment & API Setup
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
client = genai.Client(api_key=api_key)

# 2. Local GPU Embedding Model
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
print(f"Loading EmbeddingGemma 2 on: {device}...")

embed_model = SentenceTransformer(
    "google/embeddinggemma-2",
    model_kwargs={"torch_dtype": dtype},
    device=device
)

# 3. Connect to ChromaDB
DB_PATH = "./my_ai_memory_local"
COLLECTION_NAME = "rpi5_hardware_specs"
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_collection(name=COLLECTION_NAME)

def speak(text):
    if not text:
        return
    print(f"\nAI: {text}\n")
    tts = gTTS(text=text, lang="en")
    tts.save("temp_speech.mp3")
    
    pygame.mixer.init()
    pygame.mixer.music.load("temp_speech.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.quit()
    time.sleep(0.1)
    if os.path.exists("temp_speech.mp3"):
        os.remove("temp_speech.mp3")

def listen():
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = False
    
    with sr.Microphone() as source:
        print("\n[Listening... Speak your question now]")
        try:
            audio = r.listen(source, timeout=8, phrase_time_limit=10)
            text = r.recognize_google(audio, language="en-in")
            print(f"User: {text}")
            return text
        except sr.WaitTimeoutError:
            print("Listening timed out. No speech detected.")
            return None
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return None
        except Exception as e:
            print(f"Microphone error: {e}")
            return None

def retrieve_context(query):
    formatted_query = f"task: search result | query: {query}"
    query_vec = embed_model.encode([formatted_query], normalize_embeddings=True).tolist()[0]
    results = collection.query(query_embeddings=[query_vec], n_results=2)
    return "\n".join(results["documents"][0])

def main():
    print("\n--- VOICE-ONLY RAG ASSISTANT ---")
    print("Type 'ask' or hit Enter to speak. Type 'q' to exit.\n")
    
    while True:
        cmd = input("Command [Enter/ask to speak, q to quit]: ").strip().lower()
        if cmd == "q":
            print("Exiting...")
            break
        
        query = listen()
        if not query:
            speak("I didn't hear anything. Please try again.")
            continue
        
        # 1. Retrieve context via local GPU
        context = retrieve_context(query)
        
        # 2. Query Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                f"You are a technical assistant. Answer the user's question concisely using ONLY the provided document context. If the answer is not in the text, say so.\n\nContext:\n{context}\n\nQuestion: {query}"
            ]
        )
        
        # 3. Speak the answer
        speak(response.text)

if __name__ == "__main__":
    main()