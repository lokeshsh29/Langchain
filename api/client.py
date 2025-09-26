import requests
import streamlit as st

def get_GeminiAI_response(input_text):
    response=requests.post("http://localhost:8000/essay/invoke",
                           json={'input':{'topic':input_text}})
    
    return response.json()['output']['content']

def get_Ollama_response(input_text):
    response=requests.post("http://localhost:8000/poem/invoke",
                           json={'input':{'topic':input_text}})
    
    return response.json()['output']['content']


st.title('Langchain Demo with Llama2 and Gemini API')
essay_text = st.text_input("Write an essay on")
poem_text = st.text_input("Write a poem on")

if essay_text:
    st.write(get_GeminiAI_response(essay_text))

if poem_text:
    st.write(get_Ollama_response(poem_text))
