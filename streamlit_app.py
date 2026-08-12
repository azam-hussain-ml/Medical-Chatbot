import os
import streamlit as st

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt


st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Chatbot")
st.caption("Ask questions based on the medical knowledge base.")


# API keys from Streamlit Secrets
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


@st.cache_resource
def load_rag_chain():

    embeddings = download_hugging_face_embeddings()

    index_name = "medical-chatbot"

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    chat_model = ChatOpenAI(model="gpt-4o")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        chat_model,
        prompt
    )

    return create_retrieval_chain(
        retriever,
        question_answer_chain
    )


rag_chain = load_rag_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask your medical question...")


if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": question})
            answer = response["answer"]

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
