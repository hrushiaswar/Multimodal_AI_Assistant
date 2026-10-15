import os
import time
import torch
import chromadb
import pygame
import pyaudio
import keyboard
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

# 3. Vector Database
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
        try:
            os.remove("temp_speech.mp3")
        except PermissionError:
            pass

def listen():
    print("\n[Press SPACEBAR to START recording]")
    keyboard.wait('space')
    while keyboard.is_pressed('space'):
        time.sleep(0.01)

    CHUNK, FORMAT, CHANNELS, RATE = 1024, pyaudio.paInt16, 1, 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("🔴 RECORDING... (Press SPACEBAR again to STOP)")
    frames = []
    while not keyboard.is_pressed('space'):
        frames.append(stream.read(CHUNK))

    while keyboard.is_pressed('space'):
        time.sleep(0.01)
    print("🟢 Recording Stopped. Processing...")
    
    stream.stop_stream()
    stream.close()
    sample_size = p.get_sample_size(FORMAT)
    p.terminate()

    # Process audio directly in memory (avoids Windows file lock issues)
    raw_audio_data = b''.join(frames)
    audio_data = sr.AudioData(raw_audio_data, RATE, sample_size)
    r = sr.Recognizer()

    try:
        text = r.recognize_google(audio_data, language="en-in")
        print(f"User: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not decipher audio. Please try again.")
        return None
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def retrieve_context(query):
    formatted_query = f"task: search result | query: {query}"
    query_vec = embed_model.encode([formatted_query], normalize_embeddings=True).tolist()[0]
    results = collection.query(query_embeddings=[query_vec], n_results=2)
    return "\n".join(results["documents"][0])

def run_continuous_session():
    print("\n=======================================================")
    print("   MANUAL TOGGLE VOICE ASSISTANT ACTIVE")
    print("   Say 'quit', 'exit', or 'stop' to end the session.")
    print("   Press Ctrl + C in terminal to force stop.")
    print("=======================================================\n")

    speak("Toggle mode active. Press the spacebar when you are ready to speak.")
    conversation_history = []

    while True:
        query = listen()
        if not query:
            continue

        if query.strip().lower() in ["quit", "exit", "stop", "bye", "goodbye"]:
            speak("Shutting down.")
            break

        doc_context = retrieve_context(query)
        history_context = "\n".join([f"Q: {turn['q']}\nA: {turn['a']}" for turn in conversation_history[-3:]])

        prompt = (f"You are a technical mechatronics assistant. Answer concisely and accurately "
                  f"using the provided document context and prior conversation context.\n\n"
                  f"Document Ground Truth:\n{doc_context}\n\n"
                  f"Recent Conversation:\n{history_context}\n\n"
                  f"User Question: {query}")

        response = client.models.generate_content(model="gemini-3.8-flash", contents=[prompt])
        speak(response.text)
        conversation_history.append({"q": query, "a": response.text})

if __name__ == "__main__":
    try:
        run_continuous_session()
    except KeyboardInterrupt:
        print("\nSession interrupted manually. Exiting cleanly.")