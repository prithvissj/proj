#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionV3, MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Define class labels
class_labels = ["Potato_healthy", "Potato_late_blight", "Potato_early_blight", "Tomato_early_blight", "Tomato_healthy", "Tomato_late_blight"]

# Specify training and testing directories
train_dir = "D:/CVIP_2/Project/Leaf-Disease-Detection-System/train"
test_dir = "D:/CVIP_2/Project/Leaf-Disease-Detection-System/test"

img_height, img_width = 224, 224
batch_size = 32

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Models to Train
models = {
    "ResNet50": ResNet50(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3)),
    "EfficientNetB0": EfficientNetB0(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3)),
    "InceptionV3": InceptionV3(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3)),
    "MobileNetV2": MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
}

# Training each model
best_model = None
best_accuracy = 0
for model_name, base_model in models.items():
    print(f"Training {model_name}...")

    # Add custom layers
    if model_name == "InceptionV3":
        x = GlobalAveragePooling2D()(base_model.output)
    else:
        x = Flatten()(base_model.output)
    
    x = Dense(128, activation="relu")(x)
    output = Dense(len(class_labels), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5,  #Adjust epochs
        verbose=1
    )

    #Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_data, verbose=0)
    print(f"{model_name} Validation Accuracy: {val_accuracy:.4f}")

    #Save best model
    if val_accuracy > best_accuracy:
        best_model = model
        best_accuracy = val_accuracy
        best_model_name = model_name

print(f"Best Model: {best_model_name} with Validation Accuracy: {best_accuracy:.4f}")

#Save the best model
best_model.save("best_leaf_disease_model.h5")

#Tkinter for the Best Model
import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk

#Load best model
model = tf.keras.models.load_model("best_leaf_disease_model.h5")

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resized_image = cv2.resize(image, (224, 224))  # Resize to model input size
    normalized_image = resized_image / 255.0       # Normalize pixel values
    return np.expand_dims(normalized_image, axis=0), image  # Add batch dimension and return original image

#Handle image upload and prediction
def upload_and_predict():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    #Preprocess the image
    input_image, original_image = preprocess_image(file_path)

    #prediction
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100

    #results
    display_image(original_image, f"{predicted_class_label} ({confidence:.2f}%)")


def display_image(image, prediction_text):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    #Update the label
    image_label.config(image=image)
    image_label.image = image

    #label with the prediction text
    result_label.config(text=prediction_text)

# Create the Tkinter application window
app = tk.Tk()
app.title("Leaf Disease Detection")
app.geometry("600x600")

# Add a button to upload an image
upload_button = Button(app, text="Upload Image", command=upload_and_predict, font=("Arial", 14))
upload_button.pack(pady=20)

# Add a label to display the uploaded image
image_label = Label(app)
image_label.pack(pady=20)

# Add a label to display prediction results
result_label = Label(app, text="Prediction will appear here", font=("Arial", 16), fg="green")
result_label.pack(pady=20)

# Run the Tkinter event loop
app.mainloop()

