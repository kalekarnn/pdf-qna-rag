import re
import tempfile

import chromadb
import gradio as gr
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_llm(api_key=None):
    """Get LLM based on API key availability"""
    if api_key and api_key.strip():
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=api_key,
            temperature=0.7,
            max_tokens=1024,
        )
    #return Ollama(model="llama3.1")
    return Ollama(model="deepseek-r1:8b")


def process_pdf(pdf_file):
    """Process PDF and store in ChromaDB"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        if hasattr(pdf_file, "name"):
            # If it's a file path
            with open(pdf_file.name, "rb") as f:
                tmp_file.write(f.read())
        else:
            # If it's the file content directly
            tmp_file.write(pdf_file)
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory="./chroma_db"
    )

    return "PDF processed successfully!"


def query_document(question, api_key=""):
    """Query the document using RAG"""
    # Load the persisted vector store
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # Get appropriate LLM
    llm = get_llm(api_key)

    # Create retrieval chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    # Get response
    response = qa_chain({"query": question})

    result = response["result"]
    cleaned_result = clean_output(result)
    print(cleaned_result)

    return cleaned_result


def clean_output(result):
    # clean for deepseek-r1:8b
    result = re.sub(r"\s+", " ", result)
    cleaned_result = re.sub(r"<think>.*?</think>", "", result)
    return cleaned_result


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# PDF Question Answering with RAG")

    with gr.Row():
        # Left column for PDF upload and API key
        with gr.Column(scale=1):
            gr.Markdown("### Upload PDF")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="binary")
            api_key = gr.Textbox(
                label="OpenAI API Key",
                placeholder="Enter to use GPT-3.5, leave empty for local Ollama",
                type="password",
            )
            upload_button = gr.Button("Process PDF")
            pdf_output = gr.Textbox(label="Status")
            upload_button.click(process_pdf, inputs=[pdf_input], outputs=[pdf_output])

        # Right column for Q&A
        with gr.Column(scale=1):
            gr.Markdown("### Ask Questions")
            question_input = gr.Textbox(label="Ask a question about the document")
            question_button = gr.Button("Get Answer")
            answer_output = gr.Textbox(label="Answer", lines=5)
            question_button.click(
                query_document,
                inputs=[question_input, api_key],
                outputs=[answer_output],
            )

if __name__ == "__main__":
    demo.launch()
