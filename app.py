import streamlit as st
from dotenv import load_dotenv
import os

import youtube_utils
import langchain_helper

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="YouTube Video Learning Assistant", layout="wide")
st.title("📚 YouTube Video Learning Assistant")
st.markdown("Summarize and chat with any YouTube video that has a Hindi or English transcript.")

# --- NEW: API Key Handling in Sidebar ---
st.sidebar.title("Configuration")
st.sidebar.markdown("Enter your Euriai API key to begin.")

# We use a password input to keep the key confidential
api_key_input = st.sidebar.text_input(
    "Euriai API Key", 
    type="password",
    placeholder="Enter your key here...",
    help="You can get your API key from the Euriai platform."
)

# When the user enters a key, save it to the session state
if api_key_input:
    st.session_state.api_key = api_key_input
    st.sidebar.success("API Key saved!")

# --- Initialize Session State (if not already done) ---
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'lang_code' not in st.session_state:
    st.session_state.lang_code = None

# --- Main Application Logic ---
# NEW: Check if the API key exists in the session state before proceeding
if st.session_state.get("api_key"):
    youtube_url = st.text_input("Enter YouTube Video URL:", key="url_input")

    if st.button("Process Video", key="process_button"):
        if youtube_url:
            with st.spinner("Processing video... This may take a moment."):
                st.session_state.vector_store = None
                st.session_state.summary = None
                st.session_state.chat_history = []
                st.session_state.video_processed = False
                st.session_state.lang_code = None

                video_id = youtube_utils.get_video_id(youtube_url)
                
                if video_id:
                    transcript, lang_code, error_message = youtube_utils.get_transcript(video_id)
                    
                    if transcript:
                        # CHANGED: Pass the user's API key from session state
                        user_api_key = st.session_state.api_key
                        st.session_state.vector_store = langchain_helper.create_vector_store_from_transcript(transcript, api_key=user_api_key)
                        summary_chain = langchain_helper.get_summary_chain(api_key=user_api_key)
                        
                        st.session_state.summary = summary_chain.run(transcript)
                        st.session_state.video_processed = True
                        st.session_state.lang_code = lang_code
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
                    # CHANGED: Pass the user's API key from session state
                    user_api_key = st.session_state.api_key
                    query_for_retrieval = user_question
                    
                    if st.session_state.lang_code and st.session_state.lang_code != 'en':
                        translation_chain = langchain_helper.get_translation_chain(api_key=user_api_key)
                        query_for_retrieval = translation_chain.run(
                            text=user_question, 
                            target_language=st.session_state.lang_code
                        )

                    qa_chain = langchain_helper.get_qa_chain(st.session_state.vector_store, api_key=user_api_key)
                    response = qa_chain({"query": query_for_retrieval})
                    answer = response.get("result", "Sorry, I encountered an error.")
                
                st.session_state.chat_history.append(("assistant", answer))
                with st.chat_message("assistant"):
                    st.markdown(answer)
else:
    # NEW: Show a prominent message asking for the API key if it's not provided
    st.warning("Please enter your Euriai API key in the sidebar to use the app.")
    st.info("The main application will appear here once the API key is provided.")
