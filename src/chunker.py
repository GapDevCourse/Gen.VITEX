import pytube
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Download YouTube Video
def download_youtube_video(url, output_path='video.mp4'):
    yt = pytube.YouTube(url)
    video = yt.streams.filter(only_audio=True).first()  # Download audio-only for faster processing
    video.download(filename=output_path)
    print("Video downloaded successfully!")
    return output_path

# 2. Extract Audio and Transcribe
def transcribe_audio(video_path, model="base"):
    # Convert video to audio
    audio_path = "audio.mp3"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    
    # Load Whisper model and transcribe
    transcriber = whisper.load_model(model)
    result = transcriber.transcribe(audio_path)
    print("Transcription completed!")
    return result['text']

# 3. Chunk Text
def chunk_text(text, max_length=512):
    """Splits text into chunks of max_length."""
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])

# 4. Summarize and Vectorize Text with FAISS
def summarize_and_vectorize(text, summarizer_model="facebook/bart-large-cnn", embedding_model="all-MiniLM-L6-v2"):
    # Summarization pipeline
    summarizer = pipeline("summarization", model=summarizer_model)
    chunks = list(chunk_text(text))
    
    # Summarize each chunk
    summarized_chunks = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        summarized_chunks.append(summary[0]['summary_text'])
    
    # Initialize sentence embedding model and convert summaries to embeddings
    embedder = SentenceTransformer(embedding_model)
    embeddings = embedder.encode(summarized_chunks, convert_to_tensor=False)
    
    # Convert embeddings to a FAISS index
    d = embeddings[0].shape[0]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)  # L2 distance for similarity
    index.add(np.array(embeddings))  # Add all embeddings to the index
    print("Vectorization and FAISS index creation completed!")
    
    return summarized_chunks, index

# 5. Main function to run all steps
def process_youtube_video(url):
    # Download and transcribe
    video_path = download_youtube_video(url)
    text = transcribe_audio(video_path)
    
    # Summarize and vectorize transcribed text
    summaries, faiss_index = summarize_and_vectorize(text)
    return summaries, faiss_index

# Usage
youtube_url = "YOUR_YOUTUBE_VIDEO_URL"
summaries, faiss_index = process_youtube_video(youtube_url)

# Example: Querying the FAISS index
def query_faiss_index(faiss_index, query, embedder, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    return indices, distances

# Testing the query
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Same embedding model used in vectorization
query = "your search query text"
indices, distances = query_faiss_index(faiss_index, query, embedder)
print("Indices of most similar summaries:", indices)
print("Distances:", distances)
