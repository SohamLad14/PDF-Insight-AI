import streamlit as st
import requests
import uuid
from config import settings

API_URL = settings.API_URL

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
                try:
                    response = requests.post(
                        f"{API_URL}/upload", 
                        files=files,
                        data={"session_id": st.session_state.session_id}
                    )
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.markdown("---")
    st.write("Made with ❤️ using FastAPI + Streamlit + LangChain By SOHAM LAD")


st.title("📄 PDF Insight RAG")
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
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "session_id": st.session_state.session_id,
                    "question": query
                }
            )
            if response.status_code == 200:
                answer = response.json().get("answer", "Error: No answer returned")
            else:
                answer = f"Error: {response.text}"
        except Exception as e:
            answer = f"Connection Error: {e}"

    # show bot message
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
