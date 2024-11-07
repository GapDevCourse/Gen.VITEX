# main.py

from src.downloader import download_youtube_audio

def main():
    video_url = input("Enter the YouTube video URL: ")
    output_path = "output.mp3"
    
    print("Downloading and extracting audio...")
    audio_file_path = download_youtube_audio(video_url, output_path)
    
    if audio_file_path:
        print(f"Audio successfully saved to {audio_file_path}")
    else:
        print("Failed to download audio.")

if __name__ == "__main__":
    main()
