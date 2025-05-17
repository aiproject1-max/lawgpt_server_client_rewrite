# lawgpt_server.py

from flask import Flask, request, jsonify
from typing import Optional
import logging

from llm import get_llm_response, LLMError
from RAG.retrieval import retrieve_relevant_documents, get_document_stats
from data_processing.preprocess_docs import preprocess_document
from config import TOP_K_RESULTS

app = Flask(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Simulated token for demonstration
API_TOKEN = "secret-token-123"

# Silence third-party logging
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('ollama').setLevel(logging.WARNING)


def check_auth(req):
    """Check if request has valid bearer token."""
    auth_header = req.headers.get("Authorization")
    return auth_header == f"Bearer {API_TOKEN}"


def format_response(response: str, sources: list) -> str:
    """Format response with optional source listing."""
    formatted = response.strip()
    if sources:
        formatted += "\n\n" + "─" * 50 + "\nSources:"
        for source in sources:
            formatted += f"\n• {source}"
    return formatted


def process_query(user_query: str) -> Optional[str]:
    try:
        stats = get_document_stats()
        if stats['total_documents'] == 0:
            return "I don't have any legal documents loaded yet. Please add some documents first."

        processed_query = preprocess_document(user_query)

        relevant_docs = retrieve_relevant_documents(processed_query, top_k=TOP_K_RESULTS)

        doc_contents = [doc['content'] for doc in relevant_docs] if relevant_docs else []
        doc_metadata = [doc['metadata'] for doc in relevant_docs] if relevant_docs else []

        response = get_llm_response(user_query, doc_contents)

        if doc_metadata:
            sources = [
                f"{meta['filename']} (Last modified: {meta['last_modified']})"
                for meta in doc_metadata
            ]
            return format_response(response, sources)

        return response

    except LLMError:
        return "I'm having trouble processing your request right now. Please try again later."
    except Exception:
        return "An unexpected error occurred. Please try again later."


@app.route("/query", methods=["POST"])
def query_endpoint():
    if not check_auth(request):
        logging.warning(f"Unauthorized access attempt from {request.remote_addr}")
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    user_query = data.get("query", "").strip()

    # Log the incoming query and source IP
    logging.info(f"Received query from {request.remote_addr}: {user_query}")

    if not user_query:
        return jsonify({"error": "Empty query provided."}), 400

    response = process_query(user_query)
    return jsonify({"response": response})



if __name__ == "__main__":
    app.run(debug=True, port=5000)
