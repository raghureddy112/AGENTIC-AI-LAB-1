import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

class DataAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None

    def process_pdf(self, file_path):
        """Processes unstructured PDF data into vector store."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Stores embeddings in memory using ChromaDB
        self.vector_store = Chroma.from_documents(chunks, self.embeddings)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        return "PDF processed and indexed into Vector Database!"

    def query_pdf(self, query):
        """Queries the vector database for unstructured context."""
        if not self.qa_chain:
            return "Please index the PDF document first."
        return self.qa_chain.run(query)

    def analyze_csv(self, file_path, query):
        """Analyzes structured CSV data using Pandas + LLM context."""
        df = pd.read_csv(file_path)
        prompt = f"""
        Dataset Preview:
        {df.head(5).to_string()}
        
        Dataset Summary Metrics:
        {df.describe(include='all').to_string()}
        
        User Query: {query}
        Provide clear insights, trends, or direct answers based on this dataset.
        """
        response = self.llm.invoke(prompt)
        return response.content