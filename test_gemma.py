import torch
from sentence_transformers import SentenceTransformer

# 1. Target the RTX 3060
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
print(f"Loading google/embeddinggemma-2 in {dtype}...")

# 2. Download weights from Hugging Face & allocate to VRAM
embed_model = SentenceTransformer(
    "google/embeddinggemma-2", 
    model_kwargs={"torch_dtype": dtype}, 
    device=device
)

# 3. Test a quick inference with task prefix
sample_text = ["title: none | text: Testing local multimodal embedding on RTX 3060."]
vector = embed_model.encode(sample_text, normalize_embeddings=True)

print("Embedding successful!")
print(f"Vector shape: {vector.shape} (Dimension: {vector.shape[1]})")
print(f"Allocated VRAM: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")