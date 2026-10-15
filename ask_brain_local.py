import os
import torch
import chromadb
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. Load API Key and Initialize Client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
client = genai.Client(api_key=api_key)

# 2. Hardware and Model Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

print(f"Loading EmbeddingGemma 2 on: {device}...")
embed_model = SentenceTransformer(
    "google/embeddinggemma-2",
    model_kwargs={"torch_dtype": dtype},
    device=device
)

# 3. Connect to ChromaDB (Match DB_PATH and COLLECTION_NAME used in build_brain_local.py)
DB_PATH = "./my_ai_memory_local"
COLLECTION_NAME = "rpi5_hardware_specs"  # Adjust if you used a different name

db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_collection(name=COLLECTION_NAME)

def ask_question(query: str):
    print(f"\nUser Query: {query}")

    # Format query with task prefix
    formatted_query = f"task: search result | query: {query}"
    query_vec = embed_model.encode([formatted_query], normalize_embeddings=True).tolist()[0]

    # Search local database for top-2 matching chunks
    results = collection.query(query_embeddings=[query_vec], n_results=2)
    context = "\n".join(results["documents"][0])

    print("\n--- Retrieved Context from Vector DB ---")
    print(context[:400] + "...\n" if len(context) > 400 else context + "\n")

    # Generate grounded response via Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            f"You are a technical assistant. Answer the question using ONLY the provided document context.\n\nContext:\n{context}\n\nQuestion: {query}"
        ]
    )

    print("--- Model Response ---")
    print(response.text)

if __name__ == "__main__":
    # Test with a question from the PDF
    ask_question("What processor and power supply specifications are required?")