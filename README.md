# Traffic-Classifier

# 🛣️ German Traffic Sign Classification with CNN

This project builds a Convolutional Neural Network (CNN) model using TensorFlow to classify **German traffic signs** into 43 unique classes based on image input.

## 🚀 Project Overview

Traffic signs are crucial for road safety. This project aims to automatically classify traffic sign images using deep learning, specifically Convolutional Neural Networks. The dataset is sourced from the [German Traffic Sign Recognition Benchmark (GTSRB)].

The images are:
- Size: **30x30 pixels**
- Format: RGB
- Classes: **43 different types of signs**

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

You can run this project on **Google Colab** by copying the code into separate cells per session. The training process uses the downloaded dataset and will save the trained model as `DCML3.h5`.

### Step-by-step:

1. Install TensorFlow (optional)
2. Import necessary libraries
3. Download & extract dataset
4. Preprocess the images
5. Load dataset into TensorFlow
6. Build and compile the CNN
7. Train the model
8. Save and (optionally) download the model

## 📈 Performance

The model is expected to achieve **≥ 95% accuracy** on the validation set after training for about 15 epochs. Make sure to normalize input data and apply dropout to prevent overfitting.


## 💾 Output

- Trained model saved as: `Traffic_Classifier.h5`

## 📚 Reference

- [TensorFlow German Traffic Sign Dataset](https://storage.googleapis.com/download.tensorflow.org/data/certificate/germantrafficsigns.zip)
- [GTSRB: German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html)
