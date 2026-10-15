# Multimodal AI Assistant

Multimodal AI Assistant: A hybrid Edge-AI RAG pipeline running EmbeddingGemma2 on local GPU with ChromaDB. Features live camera perception, push-to-talk voice I/O, and Gemini reasoning wrapped in a dynamic Streamlit dashboard for grounded engineering doc QA.

## 🚀 Key Features

* **Local Data Privacy:** Proprietary engineering PDFs are extracted, chunked, and embedded 100% locally. Your documents never leave your machine.
* **Hardware-Accelerated Embeddings:** Uses Google's `EmbeddingGemma 2` open-weights model, running in `bfloat16` precision directly on an NVIDIA RTX 3060 (12GB VRAM).
* **Multimodal Sensor Fusion:** Combines text queries, push-to-talk voice recordings, and live webcam capture into a single reasoning pipeline.
* **Deterministic RAG:** Eliminates LLM hallucinations by forcing the reasoning engine (Gemini API) to answer *strictly* using the retrieved context from your vector vault.
* **Streamlit Web Dashboard:** Clean UI with drag-and-drop document upload, live camera widget, conversational memory, and source citation auditing.

## 🧠 System Architecture (The Pipeline)

This project utilizes a **Hybrid Edge-Cloud Architecture**:

1. **Ingestion (Local):** `pypdf` extracts text from uploaded manuals. The text is chunked and passed to `google/embeddinggemma-2` (running via `sentence-transformers` on the local GPU) to generate high-dimensional semantic vectors.
2. **Storage (Local):** Vectors and text chunks are indexed persistently in a local SQLite-based `ChromaDB` vault (`./my_ai_memory_local`).
3. **Query & Retrieval (Local):** User queries (typed or spoken) are vectorized on the GPU. ChromaDB performs a semantic distance search to retrieve the most relevant ground-truth document chunks.
4. **Multimodal Reasoning (Cloud):** The retrieved text context, the user's prompt, and the captured webcam frame (image array) are packaged and sent to the Gemini API (`gemini-3.5-flash` or `3.8-flash`) for final reasoning and answer synthesis.

## ⚙️ Prerequisites

* **OS:** Windows 10/11
* **Hardware:** NVIDIA GPU with at least 8GB VRAM (Tested on RTX 3060 12GB)
* **Python:** Python 3.12 (64-bit)
* **API Key:** Google AI Studio API Key (for Gemini reasoning)

## 🛠️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/mechatronics-multimodal-assistant.git
cd mechatronics-multimodal-assistant
```

**2. Create and activate a Python 3.12 virtual environment**
```cmd
py -3.12 -m venv venv
venv\Scripts\activate
```

**3. Install PyTorch with CUDA acceleration**
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**4. Install remaining dependencies**
```cmd
pip install sentence-transformers chromadb pypdf python-dotenv opencv-python pillow google-genai SpeechRecognition gTTS pyaudio keyboard streamlit
```

**5. Configure your Environment Variables**
Create a file named `.env` in the root directory and add your Google AI Studio API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

## 💻 Usage

### 1. The Streamlit Web Dashboard (Primary Interface)
This is the fully integrated UI featuring document drag-and-drop, chat history, and live camera capture.
```cmd
streamlit run app.py
```
* **Upload:** Drag a PDF into the sidebar and click "Process Document" to index it into your RTX 3060.
* **Vision:** Click "Capture" on the camera feed to take a photo of a hardware component.
* **Ask:** Type your question. Click the "View Retrieved Document Sources" expander in the response to audit the exact text pulled from your local database.

### 2. Desktop Push-to-Talk Audio Assistant (Terminal)
If you want to test the low-latency voice pipeline without the web UI, use the terminal audio script.
```cmd
python continuous_voice_rag.py
```
* Note: You must run `python build_brain_local.py` first to manually index a PDF for this script.
* Press and hold the **Spacebar** to speak. Release to submit your query.

## 📂 Project Structure

* `app.py`: The main Streamlit web dashboard.
* `continuous_voice_rag.py`: Terminal-based push-to-talk audio loop with multi-turn memory.
* `build_brain_local.py`: Standalone script for indexing PDFs into the local ChromaDB.
* `test_gemma.py`: Hardware diagnostic script to verify GPU VRAM allocation and embedding generation.
* `.env`: (Ignored by Git) Stores API credentials.
* `my_ai_memory_local/`: (Ignored by Git) The local ChromaDB vector storage directory.
