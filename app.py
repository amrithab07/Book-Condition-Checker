from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import logging
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# MongoDB connection
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['book_condition_db']
    analysis_collection = db['analysis_results']
    logging.info("✅ Connected to MongoDB successfully!")
except Exception as e:
    logging.error(f"❌ Failed to connect to MongoDB: {e}")
    raise RuntimeError(f"❌ Failed to connect to MongoDB: {e}")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the model
MODEL_PATH = 'book_condition_model.h5'
try:
    model = load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    raise RuntimeError(f"❌ Failed to load model: {e}")

IMAGE_SIZE = (224, 224)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    book_id = request.form.get('book_id')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Load and preprocess the image
            image = Image.open(filepath)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize(IMAGE_SIZE, Image.LANCZOS)
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Predict
            probability = model.predict(image_array)[0][0]
            logging.info(f"🔍 Prediction probability: {probability}")

            condition = 'Good' if probability >= 0.5 else 'Damaged'
            confidence = float(probability if probability >= 0.5 else 1 - probability)

            # Store results in MongoDB
            analysis_result = {
                'book_id': book_id,
                'filename': filename,
                'condition': condition,
                'confidence': confidence,
                'timestamp': datetime.utcnow()
            }
            
            # Check for previous analysis of the same book
            alert = None
            if book_id:
                previous_analysis = analysis_collection.find_one({'book_id': book_id})
                if previous_analysis:
                    if previous_analysis['confidence'] == confidence:
                        # For same confidence, keep the existing entry
                        alert = f"⚠️ Entry with same confidence exists: {(confidence * 100):.1f}%"
                        logging.info(f"Skipped duplicate entry with same confidence for book {book_id}")
                        return jsonify({
                            'condition': previous_analysis['condition'],
                            'confidence': previous_analysis['confidence'],
                            'alert': alert
                        })
                    elif previous_analysis['confidence'] != confidence:
                        # Keep the entry with lower confidence
                        if confidence < previous_analysis['confidence']:
                            # Remove the higher confidence entry and insert the new one
                            analysis_collection.delete_one({'_id': previous_analysis['_id']})
                            analysis_collection.insert_one(analysis_result)
                            alert = f"⚠️ Updated to lower confidence analysis: {(confidence * 100):.1f}%"
                            logging.info(f"Replaced higher confidence entry with lower confidence for book {book_id}")
                        else:
                            # Skip inserting the new entry as it has higher confidence
                            alert = f"⚠️ Keeping existing lower confidence analysis: {(previous_analysis['confidence'] * 100):.1f}%"
                            logging.info(f"Skipped higher confidence entry for book {book_id}")
                            return jsonify({
                                'condition': previous_analysis['condition'],
                                'confidence': previous_analysis['confidence'],
                                'alert': alert
                            })
                else:
                    # No previous analysis exists, insert the new one
                    analysis_collection.insert_one(analysis_result)
            logging.info(f"✅ Analysis result stored in MongoDB for {filename}")

            result = {
                'condition': condition,
                'confidence': confidence,
                'alert': alert
            }

            return jsonify(result)

        except Exception as e:
            logging.error(f"❌ Error during prediction or database operation: {str(e)}")
            return jsonify({'error': 'Failed to process image or store results'}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)