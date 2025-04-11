from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from PIL import Image
import io
import mediapipe as mp
import tensorflow as tf
from backend.sign_language_processor import SignLanguageProcessor
from backend.sign_language_transformer import SignLanguageTransformer
from backend.hand_detector import HandDetector
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize models
try:
    logger.info("Initializing AI models...")
    sign_processor = SignLanguageProcessor(
        hand_detector=HandDetector(),
        sign_transformer=SignLanguageTransformer()
    )
    logger.info("AI models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing models: {e}")
    sign_processor = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server status"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': sign_processor is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint for processing sign language from image data"""
    if not sign_processor:
        return jsonify({
            'error': 'Models not initialized properly',
            'status': 'error'
        }), 500

    try:
        # Get image data from request
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({
                'error': 'No image provided',
                'status': 'error'
            }), 400

        # Handle both file uploads and base64 image data
        if 'image' in request.files:
            image_file = request.files['image']
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
        else:
            image_data = request.json['image_data']
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))

        # Convert to numpy array
        image_np = np.array(image)
        
        # Process the image
        translated_text, confidence, _ = sign_processor.process_frame(image_np)

        return jsonify({
            'prediction': translated_text,
            'confidence': float(confidence),
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/process-stream', methods=['POST'])
def process_stream():
    """Endpoint for processing video stream frames"""
    if not sign_processor:
        return jsonify({
            'error': 'Models not initialized properly',
            'status': 'error'
        }), 500

    try:
        # Get frame data from request
        frame_data = request.json.get('frame_data')
        if not frame_data:
            return jsonify({
                'error': 'No frame data provided',
                'status': 'error'
            }), 400

        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame = Image.open(io.BytesIO(frame_bytes))
        frame_np = np.array(frame)

        # Process the frame
        translated_text, confidence, annotated_frame = sign_processor.process_frame(frame_np)

        # Encode annotated frame
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'prediction': translated_text,
            'confidence': float(confidence),
            'annotated_frame': f'data:image/jpeg;base64,{annotated_frame_base64}',
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Error processing video frame: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 