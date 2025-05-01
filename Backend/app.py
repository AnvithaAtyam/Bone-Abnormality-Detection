import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_DIR = 'models'  # Directory where model .pth files are stored
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mapping of body parts to model filenames
MODEL_MAPPING = {
    'shoulder': 'best_mura_model_shoulder.pth',
    'hand': 'best_mura_model_hand.pth',
    'forearm': 'best_mura_model_forearm.pth',
    'wrist': 'best_mura_model_wrist.pth',
    'humerus': 'best_mura_model_humerus.pth',
    'elbow': 'best_mura_model_elbow.pth',
    'finger': 'best_mura_model_finger.pth'
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Image preprocessing - updated to match your training transform
def preprocess_image(image):
    """Preprocess the image for inference"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Matches your validation transform
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Test-time augmentation function - from your training code
def tta_predict(model, images, n_aug=3):
    """Test time augmentation for inference"""
    model.eval()

    # Original prediction
    with torch.no_grad():
        orig_outputs = model(images)
        orig_preds = torch.sigmoid(orig_outputs)

    if n_aug == 0:
        return orig_preds

    # Various augmentations
    aug_preds = [orig_preds]
    aug_list = [
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomRotation(degrees=(5, 5)),
        transforms.RandomRotation(degrees=(-5, -5)),
        transforms.ColorJitter(brightness=0.1),
        transforms.ColorJitter(contrast=0.1)
    ]

    # Apply n_aug augmentations (up to len(aug_list))
    n_aug = min(n_aug, len(aug_list))

    for i in range(n_aug):
        aug_images = images.clone().cpu()
        for j in range(len(images)):
            aug_images[j] = aug_list[i](aug_images[j])
        aug_images = aug_images.to(device)

        with torch.no_grad():
            aug_outputs = model(aug_images)
            preds = torch.sigmoid(aug_outputs)
            aug_preds.append(preds)

    # Average all predictions
    final_preds = torch.mean(torch.stack(aug_preds), dim=0)
    return final_preds

# Model loading function - updated to use torchvision.models instead of efficientnet_pytorch
def load_model(body_part):
    """Load the model for the specified body part"""
    if body_part not in MODEL_MAPPING:
        raise ValueError(f"No model available for body part: {body_part}")
    
    model_path = os.path.join(MODEL_DIR, MODEL_MAPPING[body_part])
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Initialize a new EfficientNet-B3 model
        model = models.efficientnet_b3(pretrained=False)
        
        # Modify final layer for binary classification
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # If the model was saved as a complete model with 'state_dict' key
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove "module." prefix if model was saved with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]  # Remove 'module.' prefix
            new_state_dict[name] = v
        
        # Load state dict
        model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Successfully loaded model for {body_part}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise ValueError(f"Failed to load model: {str(e)}")
    
    model.to(device)
    model.eval()
    return model

@app.route('/api/predict', methods=['POST'])
def predict():
    # Check if both image and body part are provided
    if 'image' not in request.files or 'bodyPart' not in request.form:
        return jsonify({'error': 'Please provide both an image and select a body part'}), 400
    
    body_part = request.form['bodyPart']
    image_file = request.files['image']
    
    try:
        # Open and preprocess the image
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        preprocessed_img = preprocess_image(img)
        
        # Load the appropriate model
        model = load_model(body_part)
        
        # Move tensor to the same device as model
        input_tensor = preprocessed_img.to(device)
        
        # Use test-time augmentation for better prediction
        with torch.no_grad():
            # Get predictions with test-time augmentation
            pred_probs = tta_predict(model, input_tensor, n_aug=3)
            
            # Convert to Python values
            abnormal_prob = pred_probs.item() * 100  # Convert to percentage
            pred_class = 1 if abnormal_prob > 50 else 0
        
        # Determine result
        result = "Abnormal" if pred_class == 1 else "Normal"
        
        # Calculate fracture probability (assuming abnormal = fracture for this use case)
        fracture_probability = abnormal_prob if pred_class == 1 else (100 - abnormal_prob)
        
        return jsonify({
            'prediction': result,
            'confidence': f"{abnormal_prob:.2f}%",
            'body_part': body_part,
            'fracture_probability': f"{fracture_probability:.2f}%"
        })
    
    except FileNotFoundError as e:
        logger.error(str(e))
        return jsonify({'error': str(e)}), 404
    except ValueError as e:
        logger.error(str(e))
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)