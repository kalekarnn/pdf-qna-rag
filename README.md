# PDF Question Answering with RAG

A web application that allows users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG). The application supports both OpenAI's GPT-3.5 and local Ollama models for question answering.

## Demo url

[https://huggingface.co/spaces/kalekarnn/PDF-QnA-RAG](https://huggingface.co/spaces/kalekarnn/PDF-QnA-RAG)


## Features

- PDF document upload and processing
- Question answering using RAG (Retrieval-Augmented Generation)
- Choice between OpenAI GPT-3.5 and local Ollama model
- User-friendly web interface built with Gradio
- Persistent document storage using ChromaDB
- Document chunking and semantic search capabilities

## Prerequisites

- Python 3.9 or higher
- Ollama (for local model support)
- OpenAI API key (optional, for GPT-3.5)

## Project Structure

- `app.py`: Main application file containing the Gradio interface and RAG implementation
- `requirements.txt`: List of Python dependencies
- `chroma_db/`: Directory for storing the vector database (created automatically)

## Dependencies

- langchain & langchain-community: For RAG implementation
- langchain-openai: OpenAI integration
- chromadb: Vector store database
- gradio: Web interface
- sentence-transformers: Document embeddings
- openai: OpenAI API client

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kalekarnn/pdf-qna-rag.git
cd pdf-qna-rag
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Install and run Ollama if you plan to use the local model:
   - Follow the installation instructions at [Ollama's official website](https://ollama.ai/)
   - Pull the required model:
     ```bash
     ollama pull llama3.1
     ```

## Usage

1. Start the application:

```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:7860`)

3. Using the application:
   - Upload a PDF document using the file upload button
   - Click "Process PDF" to analyze and store the document
   - (Optional) Enter your OpenAI API key to use GPT-3.5
   - Type your question in the input box
   - Click "Get Answer" to receive a response

## How it Works

1. **Document Processing**:
   - The PDF is loaded and split into smaller chunks
   - Text chunks are converted into embeddings using the all-MiniLM-L6-v2 model
   - Embeddings are stored in a local ChromaDB vector database

2. **Question Answering**:
   - User questions are processed using RAG
   - Relevant document chunks are retrieved using semantic search
   - The LLM (either GPT-3.5 or Ollama) generates answers based on the retrieved context
