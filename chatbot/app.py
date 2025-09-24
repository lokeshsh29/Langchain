from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os


from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Only set environment variables if they exist
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is not None:
    os.environ["GOOGLE_API_KEY"] = google_api_key

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key is not None:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

os.environ["LANGCHAIN_TRACING_V2"] = "true"


prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Question: {question}")
    ]
)


st.title("Langchain Demo with Open source LLM")
input_text = st.text_input("Search the topic of your choice")

## Calling the LLM model

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
