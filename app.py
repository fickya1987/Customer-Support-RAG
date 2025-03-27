import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="Chat Customer Service", layout="wide")
st.title("📄 Chat with Customer Service")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    ai_greeting = "Hi,I'm a Customer Service bot who can help you about your problem. Don't be shy to ask!"
    st.session_state.messages.append({"role": "assistant", "content": ai_greeting})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

loader = PyPDFLoader("src\customer_support_guide.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
splits = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest question, rewrite it as a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_promp = (
                "You are an assistant for question-answering tasks."
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise. If user complain show a empathy."
                "\n\n"
                "{context}"
            )

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_promp),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_session_history() -> BaseChatMessageHistory:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()
    return st.session_state.chat_history

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response = conversational_rag_chain.invoke({"input": user_input, "context": ""})
    bot_response = response['answer']
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    with st.chat_message("assistant"):
        st.markdown(bot_response)
