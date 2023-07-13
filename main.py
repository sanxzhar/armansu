import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from fastapi.middleware.cors import CORSMiddleware


def openai(text: str):
    load_dotenv()

    os.getenv("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings()

    knowledge_base = FAISS.load_local("./storage/index3", embeddings)
    chain = RetrievalQA.from_chain_type(llm = ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever = knowledge_base.as_retriever())
    response = chain.run(text)
    return(response)

class QueryModel(BaseModel):
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/query')
async def response(query: QueryModel):
    text = query.text
    response = openai(text)
    return{"status": 200, "response" : response}


