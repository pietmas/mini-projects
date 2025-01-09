import os
from typing import List

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# Load text files
# return them as a list of LangChain Document objects
def load_documents(data_dir: str) -> List[Document]:

    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                docs.append(Document(page_content=text, metadata={"source": filename}))
    return docs

# Chuncking the documents
def chunk_documents(documents: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:

            new_doc = Document(page_content=chunk, metadata=doc.metadata)
            all_chunks.append(new_doc)
    return all_chunks

# Create a FAISS vector store from documents
def create_faiss_vector_store(docs: List[Document], 
                              embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    vector_store = FAISS.from_documents(docs, embedding_model)
    return vector_store

# Load a pretrained model for causal language modeling using the Transformers pipeline
def load_hf_model(model_name: str = "tiiuae/falcon-7b-instruct", device: str = "cuda"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"  # Use GPU if available
    )
    
    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=4096,
        truncation=True,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    return generation_pipeline

# Build a RetrievalQA chain that uses FAISS for retrieval and a HuggingFace pipeline for generation
def build_rag_chain(vector_store: FAISS, generation_pipeline) -> RetrievalQA:

    llm = HuggingFacePipeline(pipeline=generation_pipeline)
    
    retriever = vector_store.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain

# Main function
def main():
    data_dir = "./data"
    
    # Load documents
    print("Loading documents...")
    docs = load_documents(data_dir)
    
    # Chunk documents
    print("Chunking documents...")
    chunked_docs = chunk_documents(docs)
    
    # Create FAISS vector store
    print("Creating FAISS vector store...")
    vector_store = create_faiss_vector_store(chunked_docs)
    
    # Load Language Model
    print("Loading HF model (this may take a while)...")
    generation_pipeline = load_hf_model(
        model_name="tiiuae/falcon-7b-instruct", 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Build the RAG chain
    print("Building RetrievalQA chain...")
    rag_chain = build_rag_chain(vector_store, generation_pipeline)
    
    print("\n==== RAG Chatbot ====")
    print("Type 'exit' to quit the chatbot.")
    
    # Start the chatbot
    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting chatbot...")
            break
        
        # Get the answer from the RAG pipeline
        result = rag_chain.invoke({"query": query})
        answer = result["result"]
        
        # Print the final answer
        print(f"\nChatbot: {answer}")
        

if __name__ == "__main__":
    main()
