from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import langserve
import uvicorn
import os
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from fastapi.middleware.cors import CORSMiddleware

# Set all CORS enabled origins

load_dotenv()

google_api_key = os.environ.get("GOOGLE_API_KEY")
if google_api_key is not None:
    os.environ["GOOGLE_API_KEY"] = google_api_key


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

GeminiModel=ChatGoogleGenerativeAI(model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2)

llm=OllamaLLM(model="codellama",streaming=False)

langserve.add_routes(app,GeminiModel,path="/gemini")

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)