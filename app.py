
import streamlit as st
from src.rag_pipeline import process_documents, ask_question

st.title("Ask My PDF Bot")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    vectorstore = process_documents(uploaded_files)

    query = st.text_input("Ask a Question")

    if query:
        result = ask_question(vectorstore, query)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        for s in result["sources"]:
            st.write(s)
