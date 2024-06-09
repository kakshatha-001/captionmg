import os
from flask import Flask, render_template, request
import requests

app = Flask(__name__)

CAPTION_SERVER_URL = "http://127.0.0.1:5001"  # Update with your server's URL

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded image to a temporary file
            file_path = 'uploaded_image.jpg'
            uploaded_file.save(file_path)

            # Send the image to the caption server and receive the caption
            files = {'file': open(file_path, 'rb')}
            response = requests.post(f"{CAPTION_SERVER_URL}/predict", files=files)
            if response.status_code == 200:
                predictions = response.json().get('caption')

            # Delete the temporary file
            os.remove(file_path)

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
