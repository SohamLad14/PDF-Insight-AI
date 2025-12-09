import streamlit as st
import requests
import uuid

# API_URL = "http://localhost:8000"
API_URL = "https://pdf-insight-ai.onrender.com"

st.set_page_config(page_title="PDF Insight", layout="wide")


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


with st.sidebar:
    st.title("💬 Chat with your PDF")
    st.write("Upload PDFs and chat with them in real time.")

    uploaded_files = st.file_uploader(
        "Upload PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDF"):
        if not uploaded_files:
            st.error("Please upload at least one PDF!")
        else:
            files = [("files", (f.name, f, "application/pdf")) for f in uploaded_files]

            with st.spinner("Creating vector database..."):
                response = requests.post(f"{API_URL}/upload", files=files)

            st.success(response.json()["message"])

    st.markdown("---")
    st.write("Made with ❤️ using FastAPI + Streamlit + LangChain By SOHAM LAD")


st.title("📄 PDF Insight RAG",width="stretch")
if "messages" not in st.session_state:
    st.session_state.messages = []

# show chat bubbles
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# text input
query = st.chat_input("Ask something from your PDF...")

if query:
    # show user msg
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # send to FastAPI
    with st.spinner("Thinking..."):
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "session_id": st.session_state.session_id,
                "question": query
            }
        )
        answer = response.json().get("answer", "Error: No answer returned")

    # show bot message
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
