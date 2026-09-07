from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ----------------------------
# Load Document
# ----------------------------

loader = TextLoader("sample.txt")
documents = loader.load()

print("Documents Loaded :", len(documents))

# ----------------------------
# Split Document
# ----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

print("Chunks Created   :", len(docs))

# ----------------------------
# Embeddings
# ----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# Vector Store
# ----------------------------

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# ----------------------------
# Gemini Model
# ----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3
)

print("\n===== Basic RAG Application =====")

while True:

    question = input("\nAsk a Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # ----------------------------
    # Check if context is relevant
    # ----------------------------

    if len(context.strip()) > 50:

        prompt = f"""
You are a helpful AI assistant.

Use the following document context to answer the question.

If the answer is present in the document,
answer using the document.

If the answer is NOT present,
use your own general knowledge to answer naturally.

Document:
{context}

Question:
{question}

Answer:
"""

    else:

        prompt = f"""
Answer the following question using your own knowledge.

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)