import os
import torch
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_PATH = "rpi5_product_brief.pdf"
DB_PATH = "./my_ai_memory_local"
COLLECTION_NAME = "rpi5_hardware_specs"

# Hardware Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
print(f"Embedding on {device} ({torch.cuda.get_device_name(0)}) using {dtype}")

# Load EmbeddingGemma 2 into VRAM
embed_model = SentenceTransformer(
    "google/embeddinggemma-2", 
    model_kwargs={"torch_dtype": dtype}, 
    device=device
)

def extract_and_chunk(pdf_path, chunk_size=500, overlap=50):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Missing {pdf_path}. Place it in this folder.")
    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step) if words[i:i + chunk_size]]

def build():
    chunks = extract_and_chunk(PDF_PATH)
    print(f"Extracted {len(chunks)} chunks.")

    # Format with EmbeddingGemma 2 document instruction prefix
    formatted_chunks = [f"title: none | text: {c}" for c in chunks]

    # Initialize ChromaDB vault
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    print("Encoding on RTX 3060...")
    embeddings = embed_model.encode(formatted_chunks, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks  # store the raw text without prefix for cleaner prompt context
    )
    print(f"Local ChromaDB vault saved at '{DB_PATH}'.")

if __name__ == "__main__":
    build()