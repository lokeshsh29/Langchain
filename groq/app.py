import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

load_dotenv()

## Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is not None:
    os.environ["GROQ_API_KEY"] = groq_api_key

google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is not None:
    os.environ["GOOGLE_API_KEY"] = google_api_key

if "vector" not in st.session_state:
    st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
    st.session_state.loader=WebBaseLoader("https://docs.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("ChatGroq Demo")
llm=ChatGroq(groq_api_key=groq_api_key,model="gemma2-9b-it")

prompt=ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions:{input}
    """
)

document_chain = create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.vectors.as_retriever()

retrieval_chain = create_retrieval_chain(retriever,document_chain)

prompt=st.text_input("Input your prompt here.")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time:",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):

        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------")