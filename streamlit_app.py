import os 
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv 
import streamlit as st 
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


load_dotenv()

os.environ['GOOGLE_API_KEY'] = 'AIzaSyAd8i43T-iHpgc7OqnFiuf9HgL3zyaK194'
#os.environ['ACTIVELOOP_TOKEN'] = base64.b64decode(os.getenv('eyJhbGciOiJIUzUxMiIsImlhdCI6MTY5NTk0ODgyMiwiZXhwIjoxNzI3NTcxMjE1fQ.eyJpZCI6ImRpa3NoYXNhbmRodTEzMjAwMiJ9.ymCKty32iLmatS4n4wPNYI1EKmStu-7jaKmDhSilItdXCQmvmsFMGfyjnA4hkhSnqh77NthgjNnhshtolG7VLA'))
#os.environ['dikshasandhu132002']= os.getenv('dikshasandhu132002')

def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

def embeddings_store():
    embedding = GooglePalmEmbeddings()
    texts = doc_preprocessing()
    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
    vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
    retriever = vectordb.as_retriever()

    return retriever

def search_db():
    retriever = embeddings_store()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10
    model = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    #qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    
    qa= ConversationalRetrievalChain.from_llm(
    llm=GooglePalm(),
    memory=memory,
    retriever=retriever)

    return qa

qa = search_db()

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

def main():
    # Initialize Streamlit app with a title
    st.title("LLM Powered Chatbot")

    # Get user input from text input
    user_input = st.text_input("", key="input")

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]
        
    # Search the database for a response based on user input and update session state
    if user_input:
        input_dict = {"question": user_input}
        output = qa(input_dict)
        st.session_state.past.append(user_input)
        response = str(output["answer"])
        st.session_state.generated.append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)

if __name__ == "__main__":
    main()








