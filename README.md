# YouTube Video Learning Assistant 📚

Chat with any YouTube video in Hindi or English. This Streamlit application allows you to quickly summarize long educational videos and ask specific questions to clarify concepts, making learning more efficient and interactive.


## Overview

YouTube is a vast repository of knowledge, but sifting through long videos to find specific information can be time-consuming. This tool solves that problem by leveraging the power of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to create an interactive learning experience.

Simply provide a YouTube video URL, and the app will:
1.  Generate a concise, bullet-point summary of the video's content.
2.  Provide a chat interface where you can ask questions and get answers based *only* on the video's transcript.

## ✨ Features

-   **Automatic Summarization**: Get the key takeaways from any video without watching the whole thing.
-   **Interactive Q&A Chat**: Ask specific questions and get answers grounded in the video's content.
-   **Multilingual Support**: Seamlessly works with videos that have either **Hindi** or **English** transcripts.
-   **Cross-Lingual Q&A**: Ask questions in English about a Hindi video, and the system will translate your query in the background to find the correct information.
-   **Simple Web Interface**: Built with Streamlit for a clean and user-friendly experience.

## ⚙️ How It Works

This application is built on a Retrieval-Augmented Generation (RAG) pipeline:

1.  **Transcript Fetching**: When a user provides a YouTube URL, the app extracts the video ID and uses the `youtube-transcript-api` to fetch the full transcript, prioritizing Hindi then English.
2.  **Text Chunking**: The transcript is split into smaller, overlapping text chunks using LangChain's `RecursiveCharacterTextSplitter`.
3.  **Embedding & Indexing**: Each text chunk is converted into a numerical vector (embedding) using the Euriai `text-embedding-3-small` model. These embeddings are stored in a highly efficient `FAISS` vector store for fast similarity searches.
4.  **Query Translation**: If the video transcript is in Hindi and the user asks a question in English, a dedicated LLM chain translates the question into Hindi. This ensures that the search query is in the same language as the document chunks, dramatically improving retrieval accuracy.
5.  **Retrieval**: The user's question (or its translation) is embedded, and `FAISS` searches the vector store to find the most relevant text chunks from the transcript.
6.  **Generation**: The original question and the retrieved chunks (the "context") are passed to a Euriai `gpt-4.1-nano` model with a carefully crafted prompt. The model is instructed to formulate an answer based only on the provided context, ensuring the responses are factual and relevant to the video.

## 📁 Project Structure


├── .env # Stores the secret API key

├── app.py # Main Streamlit application file (UI and flow control)

├── langchain_helper.py # Core LangChain logic (chains, vector store, prompts)

├── requirements.txt # List of Python dependencies

└── youtube_utils.py # Utility functions for handling YouTube URLs and transcripts


## 🚀 Getting Started

Follow these steps to get the application running locally.

### Prerequisites

-   Python 3.8+
-   `pip` package manager

### 1. Set Up Your Project

Clone or download the project files into a local directory.

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies. Open your terminal in the project directory and run:

```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
Install all the required Python packages from the requirements.txt file.
pip install -r requirements.txt


### 4. Set Up Your Environment File
Create a file named .env in the root of the project directory. This file will store your Euriai API key securely. Add the following line to it:

```
api_key="YOUR_EURIAI_API_KEY_HERE"
```

▶️ Usage
1. Run the Streamlit application from your terminal:
    streamlit run app.py
2. Your web browser will automatically open with the application running.
3. Paste a YouTube video URL into the input box.
4. Click the "Process Video" button. Wait for the app to fetch the transcript, generate the summary, and build the vector store.
5. Once processed, you can read the summary and start asking questions in the chat box at the bottom of the page.

🛠️ Technologies Used

Backend: Python
Web Framework: Streamlit
LLM & Embeddings: Euriai API (gpt-4.1-nano, text-embedding-3-small)
Core AI Logic: LangChain
Vector Store: FAISS (Facebook AI Similarity Search)
YouTube Integration: youtube-transcript-api
