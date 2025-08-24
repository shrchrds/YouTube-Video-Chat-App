import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

def get_video_id(url: str) -> str | None:
    """Extracts the YouTube video ID from a URL."""
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_transcript(video_id: str) -> (str | None, str | None, str | None):
    """
    Fetches the transcript and its language code for a given YouTube video ID.
    
    Returns:
        A tuple containing (full_transcript, language_code, error_message).
    """
    try:
        api_client = YouTubeTranscriptApi()
        transcript_list = api_client.list(video_id)
        transcript_object = transcript_list.find_transcript(['hi', 'en'])
        
        # NEW: Get the language code from the transcript object
        language_code = transcript_object.language_code
        
        transcript_data = transcript_object.fetch()
        full_transcript = " ".join([chunk['text'] for chunk in transcript_data.to_raw_data()])
        
        cleaned_transcript = re.sub(r'\[.*?\]', '', full_transcript)
        cleaned_transcript = re.sub(r'\(\s*\w*\s*\)', '', cleaned_transcript)
        
        # CHANGED: Return the language code as well
        return cleaned_transcript.replace('\n', ' '), language_code, None

    except TranscriptsDisabled:
        error_message = "Transcripts are disabled for this video by the uploader."
        return None, None, error_message
    except NoTranscriptFound:
        error_message = "Could not find a transcript in Hindi or English for this video."
        return None, None, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        return None, None, error_message