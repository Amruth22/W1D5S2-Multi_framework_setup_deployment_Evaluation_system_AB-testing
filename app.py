import os
from flask import Flask, request, jsonify
from flasgger import Swagger
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Import configurations
import config

# Initialize Flask App
app = Flask(__name__)
Swagger(app) # Initialize Flasgger

# --- LLM Manager (Multi-model Setup) ---
def get_gemini_llm(model_name: str):
    """
    Initializes and returns a LangChain ChatGoogleGenerativeAI instance for the specified model.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.7, top_k=20, google_api_key=api_key)

# Pre-initialize LLM instances for easy access
llm_pro = get_gemini_llm(config.GEMINI_PRO_MODEL)
llm_flash = get_gemini_llm(config.GEMINI_FLASH_MODEL)
llm_pro_text = get_gemini_llm(config.GEMINI_PRO_TEXT_MODEL) # New: For general text tasks

# --- Framework Implementations (LangChain) ---

# Framework 1: Text Summarization (using gemini-1.5-flash for speed)
def summarize_text_framework(text: str) -> str:
    """
    Summarizes text using LangChain with gemini-1.5-flash.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert summarizer. Summarize the following text concisely."),
        ("user", "{text}")
    ])
    chain = prompt | llm_flash | StrOutputParser()
    return chain.invoke({"text": text})

# Framework 2: Complex Q&A/Reasoning (using gemini-1.5-pro for capability)
def complex_qa_framework(question: str, context: str) -> str:
    """
    Answers complex questions using LangChain with gemini-1.5-pro and provided context.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based on the provided context. If the answer is not in the context, state that you don't know."),
        ("user", "Context: {context}\nQuestion: {question}")
    ])
    chain = prompt | llm_pro | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

# Framework 3: Simple Chat/Conversation (using gemini-pro for general interaction)
def simple_chat_framework(message: str) -> str:
    """
    Engages in a simple chat conversation using LangChain with gemini-pro.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly and helpful AI assistant."),
        ("user", "{message}")
    ])
    chain = prompt | llm_pro_text | StrOutputParser()
    return chain.invoke({"message": message})

# --- API Endpoints (Model Deployment) ---

@app.route("/summarize", methods=["POST"])
def summarize_api():
    """
    Summarize text using the gemini-1.5-flash framework.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - text
          properties:
            text:
              type: string
              description: The text to be summarized.
              example: "The quick brown fox jumps over the lazy dog. This is a classic pangram."
    responses:
      200:
        description: Summarized text.
        schema:
          type: object
          properties:
            summary:
              type: string
      400:
        description: Invalid request.
    tags:
      - Text Processing
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400
    
    summary = summarize_text_framework(data["text"])
    return jsonify({"summary": summary})

@app.route("/ask_pro", methods=["POST"])
def ask_pro_api():
    """
    Ask a complex question with context using the gemini-1.5-pro framework.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - question
            - context
          properties:
            question:
              type: string
              description: The question to ask.
              example: "What is the main benefit of using LangChain?"
            context:
              type: string
              description: Relevant context for the question.
              example: "LangChain is a framework for developing applications powered by large language models. It enables applications that are data-aware and agentic."
    responses:
      200:
        description: Answer to the question.
        schema:
          type: object
          properties:
            answer:
              type: string
      400:
        description: Invalid request.
    tags:
      - Q&A
    """
    data = request.get_json()
    if not data or "question" not in data or "context" not in data:
        return jsonify({"error": "Missing 'question' or 'context' in request body"}), 400
    
    answer = complex_qa_framework(data["question"], data["context"])
    return jsonify({"answer": answer})

@app.route("/chat", methods=["POST"])
def chat_api():
    """
    Engage in a simple chat conversation using the gemini-pro framework.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - message
          properties:
            message:
              type: string
              description: The message for the chat bot.
              example: "Hello, how are you today?"
    responses:
      200:
        description: Chat bot's response.
        schema:
          type: object
          properties:
            response:
              type: string
      400:
        description: Invalid request.
    tags:
      - Chat
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request body"}), 400
    
    response = simple_chat_framework(data["message"])
    return jsonify({"response": response})

@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    ---
    responses:
      200:
        description: API is healthy.
        schema:
          type: object
          properties:
            status:
              type: string
              example: "healthy"
            gemini_models_loaded:
              type: object
              properties:
                pro:
                  type: boolean
                flash:
                  type: boolean
                pro_text:
                  type: boolean
    tags:
      - Monitoring
    """
    return jsonify({
        "status": "healthy",
        "gemini_models_loaded": {
            "pro": llm_pro is not None,
            "flash": llm_flash is not None,
            "pro_text": llm_pro_text is not None
        }
    })

# --- Main execution ---
if __name__ == "__main__":
    app.run(debug=True, host=config.API_HOST, port=config.API_PORT)