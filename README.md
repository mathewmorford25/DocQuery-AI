# DocQuery AI

DocQuery AI is a **Retrieval-Augmented Generation (RAG)** application that allows users to upload business documents and ask grounded questions about them.  
The app processes documents into searchable chunks, embeds them into a vector database, and uses an LLM to generate answers based only on retrieved context.

This project demonstrates a practical **enterprise AI pattern** for document analysis and question answering.

---

## Features

- Upload multiple document types  
  - PDF  
  - DOCX  
  - TXT  

- Adjustable retrieval settings
  - Chunk size
  - Chunk overlap
  - Retrieved chunks

- Retrieval-Augmented Generation (RAG)

- Source-backed answers with document citations

- Retrieved context inspection

- Conversation history tracking

- Downloadable Q&A conversation

- Clean Streamlit user interface

---

## Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **OpenAI API**
- **FAISS vector database**
- **PyPDF / python-docx for document parsing**

---

## How It Works

1. Documents are uploaded through the Streamlit interface.
2. Documents are split into smaller chunks for better retrieval.
3. Chunks are converted into embeddings using OpenAI.
4. Embeddings are stored in a FAISS vector database.
5. When a user asks a question:
   - Relevant document chunks are retrieved
   - The LLM generates an answer grounded in that context.

---

## Running the App Locally

### 1. Clone the repository

### 2. Install dependencies

### 3. Create a `.env` file

### 4. Run the app

---

## Deployment

This application is designed to be deployed using **Streamlit Community Cloud**.

Add the following secret in the Streamlit deployment settings:

---

## Project Purpose

This project was built to demonstrate the implementation of **Retrieval-Augmented Generation systems for document intelligence applications**, including:

- document ingestion
- chunking strategies
- vector search
- grounded LLM responses
- conversational context

---

## Author

**Mat Morford**

AI portfolio project demonstrating document-based LLM applications.
