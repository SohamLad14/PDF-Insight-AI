# PDF Insight AI

**PDF Insight AI** is a Retrieval-Augmented Generation (RAG) application that allows users to chat with their PDF documents in real-time. It uses a **FastAPI** backend for processing and vector storage, and a **Streamlit** frontend for an interactive user experience. The application leverages **Google Gemini** for generation and **HuggingFace** models for embeddings.

## 🚀 Features

- **Upload & Process**: Upload multiple PDF files simultaneously.
- **Vector Database**: Automatically extracts text, chunks it, and creates a FAISS vector store.
- **Interactive Chat**: Chat with your documents using a conversational interface.
- **Context-Aware Answers**: Uses RAG to provide accurate answers based *only* on the document context.
- **Session Management**: Maintains chat history per session.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **LLM**: [Google Gemini (gemini-2.0-flash)](https://ai.google.dev/)
- **Embeddings**: [HuggingFace (all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss)
- **Orchestration**: [LangChain](https://www.langchain.com/)

## 📋 Prerequisites

- Python 3.8 or higher
- A Google API Key (for Gemini)

## ⚙️ Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 🏃‍♂️ Usage

This application consists of two parts: the backend API and the frontend UI. You need to run both.

### 1. Start the Backend (FastAPI)

Open a terminal and run:
```bash
fastapi dev main.py
# OR
uvicorn main:app --reload
```
The API will start at `http://localhost:8000`.

### 2. Start the Frontend (Streamlit)

Open a **new** terminal window and run:
```bash
streamlit run app.py
```
The application will open in your browser (usually at `http://localhost:8501`).

## 💡 How it Works

1. **Upload**: Use the sidebar to upload your PDF files and click "Process PDF".
2. **Indexing**: The backend extracts text, splits it into chunks, and builds a vector index.
3. **Chat**: Type your question in the chat input. The system retrieves relevant chunks and uses Gemini to generate an answer.