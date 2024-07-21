import sys
import warnings
from string import Template
import os
from dotenv import load_dotenv

import gradio as gr

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Load and Split the Corpus
corpus = PyMuPDFLoader("sample.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)

# Split Documents into Chunks
chunks = text_splitter.split_documents(corpus)
print(f"Number of chunks: {len(chunks)}")

# Initialize Embeddings
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create a Vector Store
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

# Initialize the Language Model
llm = ChatGroq(
    model_name='llama3-70b-8192',
    groq_api_key=groq_api_key,
)

# Define template for question-answering
template = Template("""Answer the question based only on the following context:

Context:

```
$context
```

$history

User: $question

Response: \
""")

def get_chat_res(message, history):
    docs = vectorstore.similarity_search_by_vector(embeddings.embed_query(message), k=10, fetch_k=50)
    context = "\n\n".join([doc.page_content for doc in docs])
    history_string = "\n\n".join([f"User: {user}\nResponse: {ai}" for user, ai in history])
    prompt = template.substitute(context=context, history=history_string, question=message)
    
    try:
        response = llm.invoke(prompt).content
    except Exception:
        response = "Please contact the business directly for more information."
        
    return response
    
gr.ChatInterface(get_chat_res).launch()

