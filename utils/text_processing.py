from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
import re


### Clean the text with NON-ASCII characters ###

# Note: Here we do not use stemming, tokenization and other techniques because we are using LLM's for RAG purpose. Just basic cleaning is sufficient.

def process(docs):

    # Remove all non-ASCII characters
    clean_text = re.sub(r'[^\x00-\x7F]+', '', docs)

    # Remove extra spaces and new lines
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text


### Alternatively we can use Semantic Chunker depending upon the text document ###
# text_splitter = SemanticChunker(embeddings=embeddings)

    # text_splitter = SemanticChunker(
    #     embeddings=embeddings,
    #     breakpoint_threshold_type="percentile",
    #     breakpoint_threshold_amount=85
    # )

# For Website based text. (As Webbasedloader directly provides a str instead of List of docs)
def index_from_text(text,directory=None):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    
    # USE OF CHROMA DB VECTOR DATABASE
    if directory is not None:
        vectorstore = Chroma.from_texts(texts=splits, embedding=OpenAIEmbeddings(),persist_directory=directory)
    vectorstore = Chroma.from_texts(texts=splits, embedding=OpenAIEmbeddings())
    return vectorstore

# For the PDFs based documents.
def index_from_docs(docs,directory=None):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    if directory is not None:
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(),persist_directory=directory)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore
