### Main File is only for ```Command Line Interface``` and currently defaults to only answering queries based on URL provided. ###
### Run the file using ``python main.py`` ###

from utils.content_loader import load_web_content
from utils.text_processing import index_from_docs
from utils.retriever_generator import retriever_generator
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# These should be initialized with the necessary API keys or configurations
llm = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0.0)  # We can use ANYSCALE for OPEN-SOURCE models

def answer_question():
        
    # Take URL input from user
    url = input("Enter the URL of the website: ")
    query = input("> Enter your query: ")

    # Load the contents of the webpage
    docs = load_web_content(url)

    # Process and index the contents
    vectorstore = index_from_docs(docs)

    # Setup retriever and generator
    answer = retriever_generator(vectorstore, query, llm)

    return answer

answer = answer_question()

# Answer is displayed in the Command Line Interface #
print(answer)