# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020
@author: Ayodele Ayodeji
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_resnet50.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x / 255.0  # Scaling between 0 and 1
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The Car IS Audi"
    elif preds == 1:
        preds = "The Car is Lamborghini"
    else:
        preds = "The Car Is Mercedes"

    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        uploads_dir = os.path.join(basepath, 'uploads')

        # Create the 'uploads' directory if it doesn't exist
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        file_path = os.path.join(uploads_dir, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result

if __name__ == '__main__':
    app.run(debug=True)
