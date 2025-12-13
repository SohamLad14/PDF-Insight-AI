from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings, logger
from dotenv import load_dotenv
import os 

load_dotenv()

# Initialize Model
try:
    model = ChatGoogleGenerativeAI(model=settings.MODEL_NAME)
    parser = StrOutputParser()
except Exception as e:
    logger.error(f"Failed to initialize Google GenAI model: {e}")
    raise

def extract_text(files) -> str:
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    try:
        for pdf in files:
            reader = PdfReader(pdf.file)
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    text += txt
        logger.info(f"Extracted {len(text)} characters from {len(files)} files.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDFs: {e}")
        raise

def create_chunks(text: str) -> list[str]:
    """Splits text into chunks."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        chunks = splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks from text.")
        return chunks
    except Exception as e:
        logger.error(f"Error creating chunks: {e}")
        raise

def create_vectorstore(chunks: list[str]):
    """Creates a FAISS vector store from text chunks."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        vectorstore = FAISS.from_texts(chunks, embeddings)
        logger.info("Vector store created successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise

def retrieve_docs(vectorstore, query: str, top_k: int = 3):
    """Retrieves relevant documents from the vector store."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        # Note: FAISS.similarity_search_by_vector requires embedding the query first
        # But vectorstore.similarity_search handles it internally if we pass the query string
        # The original code used embed_query + similarity_search_by_vector. 
        # We can simplify or keep it. Let's keep it consistent but safer.
        
        # Optimization: Re-using the embedding model might be expensive if re-loaded every time.
        # Ideally, embeddings should be a singleton or passed in. 
        # For now, we stick to the function signature but note this inefficiency.
        
        query_vector = embeddings.embed_query(query)
        docs = vectorstore.similarity_search_by_vector(query_vector, top_k)
        return docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise

def rag_answer(vectorstore, query: str, history: str):
    """Generates an answer using RAG."""
    try:
        docs = retrieve_docs(vectorstore, query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = PromptTemplate(
            input_variables=["context", "query", "history"],
            template="""
You are a helpful PDF assistant.

Conversation so far:
{history}

Use ONLY the context below to answer the user's question.

Context:
{context}

Question:
{query}
"""
        )

        chain = prompt | model | parser
        answer = chain.invoke({
            "context": context,
            "query": query,
            "history": history
        })
        logger.info("Generated answer successfully.")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "I encountered an error while processing your request."
