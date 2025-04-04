# Traffic-Classifier


This project builds a Convolutional Neural Network (CNN) model using TensorFlow to classify **Traffic signs**.

## 🚀 Project Overview

Traffic signs are crucial for road safety. This project aims to automatically classify traffic sign images using deep learning, specifically Convolutional Neural Networks. 


## 📁 Dataset

The dataset is automatically downloaded and extracted using Python. It contains two folders:

- `train/` – training data for model learning
- `validation/` – validation data for performance evaluation

Each folder contains subfolders representing each class.

## 🧠 Model Architecture

The model is a Sequential CNN with the following layers:

- 3 Convolutional layers with increasing filters (32, 64, 128)
- MaxPooling after each Conv layer
- Flatten layer
- Dense layer with 128 neurons
- Dropout for regularization
- Output Dense layer with 43 neurons (softmax)

```python
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(30,30,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(43, activation='softmax')
])
```

## ⚙️ How to Run

You can run this project on **Google Colab** by copying the code into separate cells per session. The training process uses the downloaded dataset and will save the trained model as `Traffic Classifier.h5`.

### Step-by-step:

1. Install TensorFlow (optional)
2. Import necessary libraries
3. Download & extract dataset
4. Preprocess the images
5. Load dataset into TensorFlow
6. Build and compile the CNN
7. Train the model
8. Save and (optionally) download the model



## 💾 Output

- Trained model saved as: `Traffic_Classifier.h5`

