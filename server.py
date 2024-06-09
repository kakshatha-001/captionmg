import os
from flask import Flask, request, jsonify, send_from_directory
import cv2
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
from gtts import gTTS
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Load pre-trained model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set generation parameters with updated num_beams
max_length = 30
num_beams = 10
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.equalizeHist(gray)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l)
    enhanced_lab = cv2.merge((clahe_l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    pil_image = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    return pil_image

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image:
            images.append(preprocessed_image)
        else:
            continue
    if not images:
        return []
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        captions = predict_step([filepath])
        if captions:
            caption = captions[0]
            tts = gTTS(caption, lang='en')
            audio_path = os.path.join(AUDIO_FOLDER, f"{filename}.mp3")
            tts.save(audio_path)
            return jsonify({'caption': caption, 'audio_url': f'/audio/{filename}.mp3'})
        else:
            return jsonify({'error': 'Could not generate caption'})
    return jsonify({'error': 'File upload failed'})

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
