import cv2
import os
import time
import torch
import chromadb
import pygame
import speech_recognition as sr
from gtts import gTTS
from PIL import Image
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

# 3. Vector Database Connection
DB_PATH = "./my_ai_memory_local"
COLLECTION_NAME = "rpi5_hardware_specs"  # Match the collection name you built
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_collection(name=COLLECTION_NAME)

# 4. Camera Init
cap = cv2.VideoCapture(0)

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
            print(f"Recognized: {text}")
            return text
        except sr.WaitTimeoutError:
            print("Listening timed out. No speech detected.")
            return None
        except sr.UnknownValueError:
            print("Speech was unintelligible.")
            return None
        except Exception as e:
            print(f"Audio capture error: {e}")
            return None

def retrieve_context(query):
    formatted_query = f"task: search result | query: {query}"
    query_vec = embed_model.encode([formatted_query], normalize_embeddings=True).tolist()[0]
    results = collection.query(query_embeddings=[query_vec], n_results=2)
    return "\n".join(results["documents"][0])

def run_assistant():
    print("\n==========================================")
    print("   MULTIMODAL AI ASSISTANT READY         ")
    print("   Press 'a' : Ask (Vision + Audio + PDF) ")
    print("   Press 'q' : Quit                       ")
    print("==========================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera feed unavailable.")
            break

        cv2.imshow("Multimodal AI Assistant", frame)
        key = cv2.waitKey(1)

        # Emergency Stop / Clean Exit
        if key & 0xFF == ord("q"):
            print("Exiting assistant...")
            break

        # Trigger Question Capture
        elif key & 0xFF == ord("a"):
            user_query = listen()
            if not user_query:
                speak("I did not hear a question. Press 'a' to try again.")
                continue

            # 1. Retrieve PDF Ground Truth from Local GPU Vector DB
            context = retrieve_context(user_query)

            # 2. Extract current visual frame
            cv2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_rgb)

            # 3. Multimodal Reasoning Call
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    f"You are a technical mechatronics assistant. Answer the user's question using the provided document context and the attached camera frame. If the question cannot be answered from the document, state that clearly.\n\nDocument Context:\n{context}",
                    pil_img,
                    user_query
                ]
            )

            # 4. Synthesize voice response
            speak(response.text)

    # Clean release of hardware handles