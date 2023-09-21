# Car Brand Prediction Model using ResNet50

![Model Demo](![Alt text](<car pred.png>))

## Overview

This project implements a deep learning model for predicting car brands based on input images. It leverages the powerful ResNet50 architecture to achieve high accuracy in classifying various car brands. This README provides an overview of the project, how to set it up, and how to use the model for predictions.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- TensorFlow 2.x
- Keras
- Numpy
- Matplotlib (for visualization)
- A dataset of car images labeled with their respective brands

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Fortunatetech/Vehicle-Brand-Prediction-Deep-Learning-Project-with-Deployment.git
   cd car-brand-prediction
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset:

   - Organize your car images into labeled folders (e.g., "Toyota," "Ford," "Honda," etc.).

2. Train the model:

   - Modify the `Transfer Learning Resnet 50.ipynb` file to set the appropriate parameters such as batch size, number of epochs, and dataset paths.

   - Run the training script:

     ```bash
     python train.py
     ```

3. Make predictions:

   - Use the trained model to predict car brands from input images:

     ```python
     from predict import predict_car_brand

     image_path = "path/to/your/input/image.jpg"
     predicted_brand = predict_car_brand(image_path)
     print(f"The predicted car brand is: {predicted_brand}")
     ```

## Model Training

- Dataset: I used a labeled car image dataset containing classes for Audi, Lamborghini, and Mercedes, split into training, validation, and test sets.

- Data Prep: I resized images to 224x224 pixels and scaled pixel values to 0-1 for uniformity.

- Augmentation: I improved generalization using techniques like rotation, horizontal flip, and brightness adjustments during training.

- Architecture: I employed ResNet50, a pretrained CNN on ImageNet, fine-tuning it for car brand classification.

- Hyperparameters: I adjusted batch size, learning rate, and epochs through experimentation to optimize training.

- Challenges: I encountered limited data and overfitting. To combat these, I applied data augmentation, dropout layers, and fine-tuned hyperparameters.

## Evaluation

- Metrics: I assessed the model with accuracy, precision, recall, F1-score, and confusion matrix.

- Insights: Achieving high test accuracy showed the model's effectiveness. However, class imbalances led to varying precision and recall scores.

- Improvement: Potential enhancements include addressing class imbalance using oversampling or weighted loss functions, fine-tuning the architecture, and exploring advanced techniques like transfer learning.

## Demo

[- Include a GIF or video demonstrating how to use the model for predictions, showcasing its accuracy and speed.](https://github.com/Fortunatetech/Vehicle-Brand-Prediction-Deep-Learning-Project-with-Deployment/assets/104451288/6ba4bc0f-e19e-4b95-a88b-7110d976627b)

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
