from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retriever_generator(vectorstore,query, llm):

    # Defining the vectorstore retriever
    retriever = vectorstore.as_retriever()

    # Defining a Custom Prompt
    prompt = """You are an assistant for answering questions related to healthcare documents. Use the following pieces of retrieved context from the documents to answer the question. If the answer is not in the context, just say that you don't know. Use 5-6 sentences maximum and keep the answer concise.  \nQuestion: {question} \nContext: {context} \nAnswer:"""

    # Using prompt template from langchain
    prompt_template = PromptTemplate(
        template=prompt, input_variables=["context", "question"]
    )

    # Create a memory store with 3 conversations as limit
    memory = ConversationBufferMemory(k=3)

    # Defining the chain
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
    )

    # Getting the result from the LLM chain
    result = rag_chain.invoke(query)
    
    # Updating the memory
    memory.save_context({"input": query}, {"output": result})

    return result
