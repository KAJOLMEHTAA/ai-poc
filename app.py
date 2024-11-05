from flask import Flask, request, jsonify, render_template_string
import os
import threading
import gradio as gr
import logging
from logging.handlers import RotatingFileHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from embedding import load_pdf_files


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a file handler
handler = RotatingFileHandler("app.log", maxBytes=10000000, backupCount=5)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

app = Flask(__name__)

# Initialize components
PROPOSALS_DIR = "proposals"
CASE_STUDIES_DIR = "case_studies"
PERSIST_DIRECTORY = "chroma_db"

if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY not found in environment variables")


def init_vectorstore():
    """Initialize or load the vector store"""
    logger.info("Initializing vector store")
    try:
        embeddings = OpenAIEmbeddings()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        proposals_path = os.path.join(
            current_dir,
            PROPOSALS_DIR,
            "City of Oakland On-Call, 2024 (Transportation Eng.)_WOOD RODGERS.pdf",
        )
        texts = load_pdf_files(file_path=proposals_path)

        logger.info("Creating new vector store")
        vectorstore = Chroma.from_documents(
            documents=texts, embedding=embeddings, persist_directory=PERSIST_DIRECTORY
        )

        # Verify the vectorstore
        collection = vectorstore._collection
        count = collection.count()
        logger.info(f"Vectorstore created with {count} embeddings")

        # Test retrieval
        test_query = "What is the typical project timeline?"
        results = vectorstore.similarity_search(test_query, k=1)
        logger.info(
            f"Test retrieval for '{test_query}' returned {len(results)} results"
        )
        logger.info(
            f"Sample result content: {results[0].page_content[:100] if results else 'No results'}"
        )

        vectorstore.persist()
        logger.info("Vectorstore persisted to disk")

        return vectorstore
    except Exception as e:
        logger.error("Error in init_vectorstore: %s", str(e))
        raise


def generate_proposal(client_name: str, industry: str, requirements: str) -> str:
    """Generate a business proposal"""
    logger.info(
        "Generating proposal for client: %s, industry: %s", client_name, industry
    )
    try:
        vectorstore = init_vectorstore()
        llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo-preview")

        # Test the retrieval before generating the proposal
        test_results = vectorstore.similarity_search(
            f"proposals related to {industry}", k=10
        )
        logger.info(
            f"Found {len(test_results)} relevant documents for {industry} {test_results}"
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        )

        prompt = f"""Generate a detailed business proposal for {client_name} in the {industry} industry.
        
        Requirements:
        {requirements}
        Extract the contents from {test_results} and use them to generate the proposal.
        """

        logger.info("Sending prompt to LLM")
        response = qa.run(prompt)
        logger.info("Successfully generated proposal for %s", client_name)
        return response
    except Exception as e:
        logger.error("Error generating proposal: %s", str(e))
        return f"Error generating proposal: {str(e)}"


@app.route("/api/generate-proposal", methods=["POST"])
def api_generate_proposal():
    logger.info("Received proposal generation request")
    try:
        data = request.json
        if not data:
            logger.error("No data provided in request")
            return jsonify({"error": "No data provided"}), 400

        required_fields = ["client_name", "industry", "requirements"]
        for field in required_fields:
            if field not in data:
                logger.error("Missing required field: %s", field)
                return jsonify({"error": f"Missing required field: {field}"}), 400

        logger.info("Generating proposal for client: %s", data["client_name"])
        proposal = generate_proposal(
            data["client_name"], data["industry"], data["requirements"]
        )

        return jsonify({"success": True, "proposal": proposal})

    except Exception as e:
        logger.error("API error: %s", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


# Gradio Interface
def gradio_generate_proposal(client_name, industry, requirements):
    """Wrapper for Gradio to generate proposal"""
    logger.info("Gradio: Generating proposal for client: %s", client_name)
    return generate_proposal(client_name, industry, requirements)


gr_interface = gr.Interface(
    fn=gradio_generate_proposal,
    inputs=[
        gr.Textbox(label="Client Name"),
        gr.Textbox(label="Industry"),
        gr.Textbox(
            label="Requirements",
            lines=4,
            placeholder="Describe the client's requirements here...",
        ),
    ],
    outputs="text",
    title="Generate Business Proposal",
)


def run_gradio():
    logger.info("Starting Gradio interface on port 7860")
    # Launch Gradio on a separate port
    gr_interface.launch(server_name="0.0.0.0", server_port=7860, share=True)


# Start Gradio in a separate thread
threading.Thread(target=run_gradio).start()


@app.route("/")
def home():
    logger.info("Home page accessed")
    return render_template_string(
        """
        <h1>Welcome to the Proposal Generator</h1>
        <p>Use the Gradio interface to generate proposals: <a href="http://127.0.0.1:7860" target="_blank">Open Gradio Interface</a></p>
    """
    )


if __name__ == "__main__":
    logger.info("Starting Flask application on port 5000")
    app.run(debug=True, port=5000)
