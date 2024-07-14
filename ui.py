# All the required imports
import os
import streamlit as st
import time
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from utils.content_loader import load_web_content
from utils.text_processing import process, index_from_text, index_from_docs
from utils.retriever_generator import setup_chain, invoke_chain
from utils.display_file import displayPDF
from langchain_chroma import Chroma
from PyPDF2 import PdfReader

# Loading the environment variables
from dotenv import load_dotenv
load_dotenv() 

# Sidebar title and Input URL/PDF
st.title("Healthcare Chatbot Tool")
st.sidebar.title("Any Healthcare Website URL")

url = st.sidebar.text_input("URL ")
process_url = st.sidebar.button("Process URL", key="url")

pdf = st.sidebar.file_uploader("Upload the Medical Document in a PDF file")
process_pdf = st.sidebar.button("Process PDF", key="pdf")

st.sidebar.text("Must click on 'Process URL' or 'Process PDF'!",)

file_path = "./chroma_db"

# Main section 
main_placeholder = st.empty()
ask_question = st.button("Ask Question")

path = "Temp_Files/saved_pdf.pdf"

text = ''

# If PDF is uploaded then display it
if pdf:
    pdfreader = PdfReader(pdf)

    # Extracting the text from the pdf
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            text += content
    
    with open(path, "wb") as f:
        f.write(pdf.getbuffer())

    st.success("File saved successfully!")
    displayPDF(path)


### Code for OCR based PDF's. Yet to be implemented ###

# from pdf2image import convert_from_path
# import pytesseract

# pages = convert_from_path('Sample_PDFs/H-Star_ Health Insurance Plan.pdf')
# for page in pages:
#     text = pytesseract.image_to_string(page)
#     print(text)

# Define the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0)

# Processing the web page contents
if process_url:   
    
    # load data
    main_placeholder.text("Data Loading...Started...✅✅✅")
    docs = load_web_content(url)

    main_placeholder.text("Text Splitter...Started...✅✅✅")

    # Create vector store
    vectorstore = index_from_docs(docs,file_path)
    main_placeholder.text("Embedding Vector Ready...✅✅✅")

    time.sleep(1.2)


# Processing the PDF contents
if process_pdf:   
    
    # load data
    main_placeholder.text("Data Loading...Started...✅✅✅")
    # docs = load_web_content(url)

    main_placeholder.text("Text Splitter...Started...✅✅✅")

    # Create vector store
    clean_text = process(text)
    vectorstore = index_from_text(clean_text,file_path)
    main_placeholder.text("Embedding Vector Ready...✅✅✅")

    time.sleep(1.2)


query = main_placeholder.text_input("Question: (Please process URL/PDF before asking the query)")


# Loading the existing Vector Database
if os.path.exists(file_path):
    vectorstore = Chroma(persist_directory=file_path, embedding_function=OpenAIEmbeddings())

# Prompt, LLM Chain and Memory setup
chain, history = setup_chain(vectorstore, llm)

# Taking Question Input and updating the chat history to answer the query.
if ask_question and query:

        answer, store = invoke_chain(chain,history,query)
        history = store

        st.sidebar.header("Answer")
        st.sidebar.write(answer)