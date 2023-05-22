import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def main():
    load_dotenv()
    st.set_page_config(page_title="@armansu")
    st.header("Ask your question from @Armansu")
    os.getenv("OPENAI_API_KEY")
    
    embeddings = OpenAIEmbeddings()

    knowledge_base = FAISS.load_local("./storage/index3", embeddings)
    user_question = st.text_input("Type your question")
    if user_question:
        chain = RetrievalQA.from_chain_type(llm = ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever = knowledge_base.as_retriever())
        response = chain.run(user_question)
        st.write(response)

if __name__ == "__main__":
    main()




