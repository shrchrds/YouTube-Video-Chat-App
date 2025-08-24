import streamlit as st
from dotenv import load_dotenv
import os

import youtube_utils
import langchain_helper

load_dotenv()
api_key = os.getenv("api_key")

st.set_page_config(page_title="YouTube Video Learning Assistant", layout="wide")
st.title("📚 YouTube Video Learning Assistant")
st.markdown("Summarize and chat with any YouTube video that has a Hindi or English transcript.")

if not api_key:
    st.error("API key not found. Please create a .env file with your Euriai API key.")
    st.stop()

# --- Initialize Session State ---
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
# NEW: Add lang_code to session state
if 'lang_code' not in st.session_state:
    st.session_state.lang_code = None

# --- Main UI and Logic ---
youtube_url = st.text_input("Enter YouTube Video URL:", key="url_input")

if st.button("Process Video", key="process_button"):
    if youtube_url:
        with st.spinner("Processing video... This may take a moment."):
            st.session_state.vector_store = None
            st.session_state.summary = None
            st.session_state.chat_history = []
            st.session_state.video_processed = False
            st.session_state.lang_code = None # Reset language code

            video_id = youtube_utils.get_video_id(youtube_url)
            
            if video_id:
                # CHANGED: Unpack three values now
                transcript, lang_code, error_message = youtube_utils.get_transcript(video_id)
                
                if transcript:
                    st.session_state.vector_store = langchain_helper.create_vector_store_from_transcript(transcript, api_key)
                    summary_chain = langchain_helper.get_summary_chain(api_key)
                    st.session_state.summary = summary_chain.run(transcript)
                    st.session_state.video_processed = True
                    st.session_state.lang_code = lang_code # Store the language code
                    st.success(f"Video processed! (Transcript language: {lang_code})")
                else:
                    st.error(error_message)
            else:
                st.error("Invalid YouTube URL. Please enter a valid URL.")
    else:
        st.warning("Please enter a YouTube URL.")

# --- Display Summary and Chat Interface ---
if st.session_state.video_processed:
    if st.session_state.summary:
        st.subheader("📝 Video Summary")
        st.write(st.session_state.summary)

    if st.session_state.vector_store:
        st.subheader("💬 Chat with the Video")
        for author, message in st.session_state.chat_history:
            with st.chat_message(author):
                st.markdown(message)
        
        user_question = st.chat_input("Ask a question about the video's content...")
        
        if user_question:
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.markdown(user_question)
            
            with st.spinner("Thinking..."):
                query_for_retrieval = user_question
                
                # ==========================================================
                # THE CORE FIX: Translate query if transcript is not English
                # ==========================================================
                if st.session_state.lang_code != 'en':
                    st.write(f"Translating question to '{st.session_state.lang_code}' for better search...")
                    translation_chain = langchain_helper.get_translation_chain(api_key)
                    query_for_retrieval = translation_chain.run(
                        text=user_question, 
                        target_language=st.session_state.lang_code
                    )
                    st.write(f"Translated query: {query_for_retrieval}")

                qa_chain = langchain_helper.get_qa_chain(st.session_state.vector_store, api_key)
                
                # Use the (potentially translated) query for retrieval
                response = qa_chain({"query": query_for_retrieval})
                answer = response.get("result", "Sorry, I encountered an error.")
            
            st.session_state.chat_history.append(("assistant", answer))
            with st.chat_message("assistant"):
                st.markdown(answer)