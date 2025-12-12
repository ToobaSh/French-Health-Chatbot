# Medical RAG Chatbot (French)

This project is a French medical chatbot using an extractive RAG (Retrieval-Augmented Generation) approach.  
It analyzes local PDF brochures and generates clear, structured answers for patients using Streamlit.

**Note:** This chatbot does not replace professional medical advice.

---

## Features

- Chatbot responding in French
- Uses PDF brochures stored locally
- Extractive RAG pipeline (no external LLM)
- Advanced text cleaning for noisy PDFs
- Structured answers: definition, symptoms, treatment, when to consult
- Streamlit user interface
- Automatic embedding generation

---

## How the RAG Pipeline Works

-User asks a question
-The system retrieves the most relevant PDF chunks
-Text is cleaned (removal of symbols, menus, references)
-Relevant sentences are summarized
-A single, readable answer is generated
-Sources are displayed in the Streamlit interface
