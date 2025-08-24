import os
import requests
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain, RetrievalQA
from langchain_core.prompts import PromptTemplate

from euriai.langchain import EuriaiChatModel

# --- Custom Embeddings Class for Euriai API ---
class EuraiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.euron.one/api/v1/euri/embeddings"

    def _call_eurai_api(self, text_input: str) -> List[float]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {"input": text_input, "model": self.model}
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['data'][0]['embedding']

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._call_eurai_api(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._call_eurai_api(text)

# --- Vector Store and Chain Creation Functions ---
def create_vector_store_from_transcript(transcript: str, api_key: str) -> FAISS:
    """Splits transcript, generates embeddings, and creates a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([transcript])
    
    embedding_model = EuraiEmbeddings(api_key=api_key)
    
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store

def get_summary_chain(api_key: str) -> LLMChain:
    """Creates a LangChain chain specifically for summarization."""
    prompt_template = """
    You are an expert in summarizing YouTube video transcripts.
    The provided transcript may be in Hindi or English.
    Based on the transcript, create a concise, easy-to-read summary.
    Focus on the main points and key takeaways. Present the summary in 3 to 5 bullet points.

    Transcript:
    "{transcript}"

    Summary:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["transcript"])
    chat_model = EuriaiChatModel(api_key=api_key, model="gpt-4.1-nano")
    return LLMChain(llm=chat_model, prompt=prompt)

def get_qa_chain(vector_store: FAISS, api_key: str) -> RetrievalQA:
    """Creates a Retrieval-Augmented Generation (RAG) chain for Q&A."""
    prompt_template = """
    You are a helpful assistant for a YouTube video. Your task is to answer the user's question based ONLY on the provided context.

    **CRITICAL INSTRUCTIONS:**
    1. The context below is from the video's transcript and may be in a different language than the question (e.g., Hindi context, English question).
    2. You MUST understand both languages and synthesize an answer from the context provided.
    3. Your final answer should be in the SAME LANGUAGE as the user's original question.
    4. If the information required to answer the question is not found in the context, you must say "I couldn't find an answer to that in the video transcript."

    Context:
    {context}
    
    Question: {question}
    
    Helpful Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chat_model = EuriaiChatModel(api_key=api_key, model="gpt-4.1-nano")
    chain_type_kwargs = {"prompt": prompt}
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}), # Increased k for more context
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=False 
    )
    return qa_chain

def get_translation_chain(api_key: str) -> LLMChain:
    """Creates a LangChain chain for translating text."""
    
    # A simple, direct prompt for translation.
    prompt_template = """
    Translate the following text into {target_language}.
    Output ONLY the translated text and nothing else. Do not add any extra explanations or phrases like "Here is the translation:".

    Text to translate:
    "{text}"

    Translation:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text", "target_language"])
    
    chat_model = EuriaiChatModel(api_key=api_key, model="gpt-4.1-nano", temperature=0) # Temp 0 for deterministic translation
    
    return LLMChain(llm=chat_model, prompt=prompt)