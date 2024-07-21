# Jessup Cellars Chatbot

This is chatbot that answers questions about Jessup Cellars using information extracted from a PDF document. The chatbot uses Groq's LLM for generating responses.

This application is implemented using python.One is a flask application and another is an application hosted using gradio.

## Requirements

- Python 3.8+

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/ahmad-meda/Chatbot-Jessup-Cellars/tree/main
    cd jessup-cellars-chatbot
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up your environment variables**:
    Create a `.env` file in the project directory with the following content:
    ```sh
    GROQ_API_KEY=your_groq_api_key
    ```

5. **Ensure your `sample.pdf` file is in the project directory**.

## Running the Application

1. **Start the Flask application**:
    ```sh
    python app.py
    ```
2. **Start the Gradio application**:
    ```sh
    python app.py
    ```

2. **Open your web browser and go to**: `http://127.0.0.1:5000/`

## Usage

- Type a question in the input box and click "Submit".
- The chatbot will respond with information extracted from the PDF and generated using Groq's LLM.
- The response time for the Groq API call will be displayed below the chatbox.
