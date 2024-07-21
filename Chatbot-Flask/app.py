import sys
import warnings
from string import Template
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session

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

# Initialize Flask app and session
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

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

$context

Question: $question

Response: \
""")

def get_chatbot_response(user_message, history):
    query = user_message
    docs = vectorstore.similarity_search_by_vector(embeddings.embed_query(query), k=10, fetch_k=50)
    context = "\n\n".join([doc.page_content for doc in docs])
    context += "\n\n" + "\n".join(history)
    try:
        import time
        start_time = time.time()
        response = llm.invoke(template.substitute(context=context, question=query)).content
        end_time = time.time()
        response_time = end_time - start_time
    except Exception as e:
        response = "Please contact the business directly for more information."
        response_time = 0
    
    return response, response_time

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.form.get("question")
    if 'history' not in session:
        session['history'] = []
    
    session['history'].append(user_message)
    response, response_time = get_chatbot_response(user_message, session['history'])
    session['history'].append(response)
    return jsonify({"answer": response, "response_time": response_time})

if __name__ == "__main__":
    app.run(debug=True)
