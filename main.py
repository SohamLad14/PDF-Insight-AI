from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag import extract_text, create_chunks, create_vectorstore, rag_answer

app = FastAPI(title="PDF Insight RAG API")


VECTORSTORE = None
MEMORY = {}         


@app.post("/upload")
async def upload_pdf(files: list[UploadFile] = File(...)):
    global VECTORSTORE

    text = extract_text(files)
    chunks = create_chunks(text)
    VECTORSTORE = create_vectorstore(chunks)

    return {"message": "PDF uploaded & vector database created successfully!"}


class Query(BaseModel):
    session_id: str
    question: str


@app.post("/ask")
async def ask_question(data: Query):
    global VECTORSTORE, MEMORY

    if VECTORSTORE is None:
        return {"error": "Upload PDF first using /upload"}

    # init memory for session
    if data.session_id not in MEMORY:
        MEMORY[data.session_id] = []

    # store user message
    MEMORY[data.session_id].append({"role": "user", "content": data.question})

    # build the history string
    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in MEMORY[data.session_id]]
    )

    # get answer
    answer = rag_answer(VECTORSTORE, data.question, history_text)

    # store bot message
    MEMORY[data.session_id].append({"role": "assistant", "content": answer})

    return {"answer": answer}


@app.get("/")
def home():
    return {"status": "RAG API running"}
