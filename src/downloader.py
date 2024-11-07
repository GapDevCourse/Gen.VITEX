# src/downloader.py

from pytube import YouTube
from moviepy.editor import AudioFileClip
import os

def download_youtube_audio(video_url, output_path="output.mp3"):
    """
    Downloads audio from a YouTube video and saves it as an MP3 file.
    
    Parameters:
    - video_url (str): The URL of the YouTube video.
    - output_path (str): The path to save the audio file (default is "output.mp3").
    
    Returns:
    - str: The path to the saved audio file.
    """
    try:
        yt = YouTube(video_url)
        video_stream = yt.streams.filter(only_audio=True).first()
        audio_file = video_stream.download(filename="temp_audio.mp4")
        
        # Convert to MP3 format
        audio = AudioFileClip(audio_file)
        audio.write_audiofile(output_path)
        
        # Clean up temporary file
        audio.close()
        os.remove(audio_file)
        
        return output_path
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
