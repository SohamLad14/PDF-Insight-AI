from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv , find_dotenv
from langchain_community.vectorstores import FAISS
import streamlit as st

load_dotenv(find_dotenv())

#Loading the pdf using pdf parser
def get_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#Splitting the text and creating chunks
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = splitter.split_text(text)

    return chunks

def get_vector_db(text_chunks):
    #embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    vector_store = FAISS.from_texts(text_chunks , embedding = embeddings)
    vector_store.save_local('faiss_index')

Template = """

Here is the context :
<context> {context} </context>

The question that must be answered using context:
<question> {input} </question>

-Please read through the context carefully. Use this context to answer the user questions.
-Provide direct answer using the context
-If question is not related to the context then reply with the question or topic is not related in the doc
-Do not create a fictional answer
-Output response should be in plain english

"""

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash' , temperature=0.3)

    prompt = PromptTemplate(template=Template , input_variables = ['context' , 'question'])

    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain

def user_input(question):
    #embeddings = GoogleGenerativeAIEmbeddings(model='model/gemini-embedding001')
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    new_db = FAISS.load_local("faiss_index" , embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()

    response = chain.invoke(
         {"context":docs, "input": question}, return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response)




def main():
    st.set_page_config("PDF Insight AI")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_db(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()