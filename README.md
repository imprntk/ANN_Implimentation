# Simple Artificial Neural Network (ANN) Implementation

## 📌 Overview
This project demonstrates a **basic implementation** of an **Artificial Neural Network (ANN)** using **TensorFlow/Keras**. The model is trained on a simple dataset to perform classification or regression tasks.

## 🔧 Basic Steps in ANN Implementation
1. **Import Libraries**
   ```python
   import tensorflow as tf
   from tensorflow import keras
   import numpy as np
   import pandas as pd
   ```

2. **Load and Preprocess Data**
   - Load dataset using Pandas.
   - Normalize or scale the data if necessary.
   ```python
   df = pd.read_csv("data.csv")
   X = df.drop("target", axis=1).values
   y = df["target"].values
   ```

3. **Define ANN Model**
   ```python
   model = keras.Sequential([
       keras.layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
       keras.layers.Dense(8, activation='relu'),
       keras.layers.Dense(1, activation='sigmoid') # Use 'softmax' for multi-class classification
   ])
   ```

4. **Compile the Model**
   ```python
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   ```

5. **Train the Model**
   ```python
   model.fit(X, y, epochs=50, batch_size=10, validation_split=0.2)
   ```

6. **Evaluate and Make Predictions**
   ```python
   loss, accuracy = model.evaluate(X, y)
   predictions = model.predict(X)
   ```

## 🛠️ Requirements
- Python
- TensorFlow/Keras
- NumPy & Pandas

## 🎯 Usage
Run the script to train and evaluate the model:
```bash
python train.py
```

## 📜 License
This project is licensed under the **MIT License**.

---
🚀 **Happy Coding!** 🎯


