from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import os

app = Flask(__name__)

# Set up OpenAI API (Optional for summarization)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to get YouTube video transcript
def get_video_transcript(video_url):
    try:
        video_id = video_url.split('v=')[1]  # Extract the video ID from URL
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        return str(e)

# Function to summarize the text using OpenAI (optional)
def summarize_text(text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize the following content:\n\n{text}",
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

# Function to vectorize text using TF-IDF
def vectorize_text_tfidf(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    return tfidf_matrix.toarray()

# Function to vectorize text using BERT (Optional)
def vectorize_text_bert(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Take the mean of the token embeddings (this is one way to vectorize the text)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

@app.route('/')
def home():
    return "YouTube Video to Text Converter is running!"

@app.route('/convert', methods=['POST'])
def convert_video():
    youtube_url = request.form['youtube_url']
    
    # Step 1: Get Transcript
    transcript = get_video_transcript(youtube_url)
    
    if isinstance(transcript, str):
        return jsonify({"error": transcript}), 400  # Return error if transcript fetch fails
    
    # Step 2: Join all transcript text into a single string
    full_text = " ".join([entry['text'] for entry in transcript])
    
    # Step 3: Summarize the transcript using OpenAI (optional)
    summary = summarize_text(full_text)
    
    # Step 4: Vectorize the full text (TF-IDF or BERT)
    tfidf_vector = vectorize_text_tfidf(full_text)  # OR use `vectorize_text_bert(full_text)`
    
    return jsonify({
        "summary": summary,
        "vectorized_text": tfidf_vector.tolist()  # Convert NumPy array to list for JSON serialization
    })

if __name__ == '__main__':
    app.run(debug=True)
