from datasets import Dataset
from text_processing import index_from_text, process
from PyPDF2 import PdfReader
from retriever_generator import setup_chain, invoke_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv() 


llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0)

# Sample questions to Evaluate the RAG system.
questions = ["What is covered under the hospitalization benefit in the H-Star Health Insurance Plan?", 
             "Does the H-Star Health Insurance Plan cover pre-hospitalization expenses?",
             "Are ambulance charges covered under the H-Star Health Insurance Plan?",
             "What conditions are excluded from coverage in the H-Star Health Insurance Plan?"
            ]

# Ground Truths are basically ideal responses to which we compare the results.
ground_truths = [["The H-Star Health Insurance Plan covers room charges, doctor's fees, and surgery costs associated with hospital stays. It ensures that all necessary medical expenses incurred during hospitalization are covered, providing financial protection against significant medical bills."],
                ["Yes, the H-Star Health Insurance Plan covers pre-hospitalization expenses incurred 30 days before hospitalization. This includes diagnostic tests, consultations, and treatments necessary for the illness leading to hospitalization, ensuring preparatory costs are taken care of."],
                ["Yes, the H-Star Health Insurance Plan includes coverage for ambulance expenses up to $500 per claim. This ensures that transportation costs to the hospital are covered in case of emergencies."],
                ["The plan excludes coverage for pre-existing conditions, cosmetic surgery, alternative treatments like Ayurveda and Homeopathy, maternity expenses, and routine dental treatments except for those related to accidental injuries. This helps manage risk and ensures fair premium pricing."]
                ]
answers = []
contexts = []

text =""

# Replace the pdf location with your relative path
pdfreader = PdfReader("C:/Users/LENOVO/Desktop/Healthcare_Git/Healthcare_Chatbot/Sample_PDFs/H-Star_ Health Insurance Plan.pdf")

# Extracting the text from the pdf
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        text += content

# Clean the pdf's text and store the embeddings into vector database
clean_text = process(text)
vectorstore = index_from_text(clean_text,"chroma_db")

retriever = vectorstore.as_retriever()

# Setup the chain
rag_chain, store = setup_chain(vectorstore, llm)

# Inference
for query in questions:
  
  answer, store = invoke_chain(rag_chain, store, query)
  answers.append(answer)
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

# Compute the RAGAS evaluation score
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()

df.to_csv('Evaluation.csv')