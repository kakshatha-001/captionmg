import os
import cv2
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import numpy as np
from gtts import gTTS
from flask import Flask, render_template, request
from IPython.display import Audio, display

app = Flask(__name__)

# Load pre-trained model, image processor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set generation parameters with updated num_beams
max_length = 30
num_beams = 10  # Change this value to control the beam search width
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve contrast
    enhanced_gray = cv2.equalizeHist(gray)

    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into channels
    l, a, b = cv2.split(lab_image)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l)

    # Merge the CLAHE-enhanced L channel with the original A and B channels
    enhanced_lab = cv2.merge((clahe_l, a, b))

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Convert from BGR format to RGB format
    pil_image = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    return pil_image

def predict_step(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Extract features and generate captions
    inputs = image_processor(images=preprocessed_image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    with torch.no_grad():
        output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            predictions = predict_step(image)

            # Convert caption to speech
            tts = gTTS(predictions[0], lang='en')
            tts.save('caption.mp3')

            # Play audio
            display(Audio('caption.mp3', autoplay=True))

            return render_template('index.html', predictions=predictions[0])
    return render_template('index.html', predictions=None)

if __name__ == '__main__':
    app.run(debug=True)
