import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
parser = StrOutputParser()


#extracting the text
def extract_text(files):
    text = ""
    for pdf in files:
        reader = PdfReader(pdf.file)
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt
    return text


#chunking
def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


#vector db
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return FAISS.from_texts(chunks, embeddings)


#retrieval
def retrieve_docs(vectorstore, query, top_k=3):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    query_vector = embeddings.embed_query(query)
    return vectorstore.similarity_search_by_vector(query_vector, top_k)


# final answer
def rag_answer(vectorstore, query, history):
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
    return chain.invoke({
        "context": context,
        "query": query,
        "history": history
    })
