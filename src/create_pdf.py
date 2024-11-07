# app.py

from flask import Flask, render_template, request

app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle form submission
@app.route('/process', methods=['POST'])
def process():
    # Get data from form
    youtube_url = request.form['youtubeUrl']
    # Process the URL (placeholder)
    result = f"Received URL: {youtube_url}"
    return result

if __name__ == '__main__':
    app.run(debug=True)
