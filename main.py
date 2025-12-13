from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid
from rag import extract_text, create_chunks, create_vectorstore, rag_answer
from config import settings, logger

app = FastAPI(title="PDF Insight RAG API")

# Session Management
# In a production app, this should be a Redis or database.
# For now, in-memory dictionary is fine as per plan, but we structure it better.
class SessionData:
    def __init__(self):
        self.vectorstore = None
        self.history: List[Dict[str, str]] = []

SESSIONS: Dict[str, SessionData] = {}

@app.post("/upload")
async def upload_pdf(
    files: list[UploadFile] = File(...),
    session_id: str = Form(...)
):
    try:
        logger.info(f"Received upload request for session: {session_id}")
        
        # Initialize session if not exists
        if session_id not in SESSIONS:
            SESSIONS[session_id] = SessionData()
        
        text = extract_text(files)
        chunks = create_chunks(text)
        vectorstore = create_vectorstore(chunks)
        
        SESSIONS[session_id].vectorstore = vectorstore
        
        logger.info(f"Vector store updated for session: {session_id}")
        return {"message": "PDF uploaded & vector database created successfully!"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class Query(BaseModel):
    session_id: str
    question: str


@app.post("/ask")
async def ask_question(data: Query):
    try:
        session_id = data.session_id
        if session_id not in SESSIONS or SESSIONS[session_id].vectorstore is None:
            logger.warning(f"Session {session_id} not found or no PDF uploaded.")
            return {"answer": "Please upload a PDF first."}
            # Alternatively raise 400, but returning a message is friendlier for the chat UI
        
        session = SESSIONS[session_id]
        
        # store user message
        session.history.append({"role": "user", "content": data.question})

        # build the history string
        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in session.history]
        )

        # get answer
        answer = rag_answer(session.vectorstore, data.question, history_text)

        # store bot message
        session.history.append({"role": "assistant", "content": answer})

        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/")
def home():
    return {"status": "RAG API running", "config": {"model": settings.MODEL_NAME}}
