#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

#training and testing directories
train_dir = "D:/CVIP_2/Project/Leaf-Disease-Detection-System/train"
test_dir = "D:/CVIP_2/Project/Leaf-Disease-Detection-System/test"

#Image preprocessing parameters
img_size = (224, 224)  # Resize images to 224x224
batch_size = 32

#Data augmentation training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True
)

#rescale for testing 
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

#preprocess training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    shuffle=True
)

#preprocess testing data
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

#save data as numpy arrays
def save_dataset_as_numpy(data_generator, output_dir, dataset_type):
    os.makedirs(output_dir, exist_ok=True)
    images = []
    labels = []
    
    #Track progress
    print(f"Processing {dataset_type} data...")
    
    for batch_idx in range(len(data_generator)):
        batch_images, batch_labels = data_generator[batch_idx]
        images.extend(batch_images)
        labels.extend(batch_labels)
        print(f"Processed batch {batch_idx + 1}/{len(data_generator)}")
    
    images = np.array(images)
    labels = np.array(labels)
    
    np.save(os.path.join(output_dir, f"{dataset_type}_images.npy"), images)
    np.save(os.path.join(output_dir, f"{dataset_type}_labels.npy"), labels)
    print(f"Saved {dataset_type} data: {images.shape}, {labels.shape}")

#Saving datasets
save_dataset_as_numpy(train_data, "preprocessed_data", "training")
save_dataset_as_numpy(test_data, "preprocessed_data", "testing")


# In[2]:


#Verify saved datasets
train_images = np.load("preprocessed_data/training_images.npy")
train_labels = np.load("preprocessed_data/training_labels.npy")
test_images = np.load("preprocessed_data/testing_images.npy")
test_labels = np.load("preprocessed_data/testing_labels.npy")

print(f"Training Images Shape: {train_images.shape}")
print(f"Training Labels Shape: {train_labels.shape}")
print(f"Testing Images Shape: {test_images.shape}")
print(f"Testing Labels Shape: {test_labels.shape}")


# In[3]:


def print_dataset_preview(data_generator, dataset_name):
    print(f"\nPreview of {dataset_name} Dataset:")
    batch_images, batch_labels = next(data_generator) 
    print(f"Shape of images: {batch_images.shape}")
    print(f"Shape of labels: {batch_labels.shape}")
    print("First 5 labels:")
    print(batch_labels[:5])#first 5 labels

    #display images
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(batch_images[i])
        plt.title(f"Class: {np.argmax(batch_labels[i])}")
        plt.axis("off")
    plt.show()

#Printing datasets
print_dataset_preview(train_data, "Training")
print_dataset_preview(test_data, "Testing")


# In[4]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import numpy as np
import os
import matplotlib.pyplot as plt

#initialize preprocessed data
data_dir = "preprocessed_data"
train_images_path = os.path.join(data_dir, "training_images.npy")
train_labels_path = os.path.join(data_dir, "training_labels.npy")
test_images_path = os.path.join(data_dir, "testing_images.npy")
test_labels_path = os.path.join(data_dir, "testing_labels.npy")

#Loading preprocessed data
train_images = np.load(train_images_path)
train_labels = np.load(train_labels_path)
test_images = np.load(test_images_path)
test_labels = np.load(test_labels_path)

#Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

#Calculate class weights for imbalanced data
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(np.argmax(train_labels, axis=1)),
    y=np.argmax(train_labels, axis=1)
)
class_weights = dict(enumerate(class_weights))

#AlexNet model
model = Sequential([
    Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D((3, 3), strides=2),
    
    Conv2D(256, (5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((3, 3), strides=2),
    
    Conv2D(384, (3, 3), activation='relu', padding='same'),
    Conv2D(384, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((3, 3), strides=2),
    
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(train_labels.shape[1], activation='softmax')  # Number of classes
])

#Compiling the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    epochs=20,  
    batch_size=16,
    class_weight=class_weights
)

#Save the trained model
model.save("leaf_disease_alexnet_model.h5")
print("Model trained and saved!")

#Test the model
history = model.fit(
    test_images, test_labels,
    validation_data=(test_images, test_labels),
    epochs=20,  
    batch_size=16,
    class_weight=class_weights
)

#Save the tested model
model.save("leaf_disease_alexnet_model_test.h5")
print("Model test and saved!")

#Evaluate the model on training and testing data
train_loss, train_accuracy = model.evaluate(train_images, train_labels, verbose=0)
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

print(f"Training Accuracy: {train_accuracy:.4f}, Training Loss: {train_loss:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}, Testing Loss: {test_loss:.4f}")

#Visualize training history
plt.figure(figsize=(12, 6))

#Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

#Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#Generate predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

#Basic testing result
correct = 0
total = len(test_labels)

#Calculate accuracy
for i in range(total):
    if predicted_classes[i] == true_classes[i]:
        correct += 1

test_accuracy_simple = correct / total
print(f"Simple Testing Accuracy (calculated manually): {test_accuracy_simple:.4f}")

#Visualize few test results
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_images[i])
    plt.title(f"True: {true_classes[i]}, Pred: {predicted_classes[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[20]:


import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

#Load the trained model
model_path = "leaf_disease_alexnet_model.h5"
try:
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

#Load class labels
class_labels = ["Potato_healthy", "Potato_late_blight", "Potato_early_blight",
               "Tomato_early_blight", "Tomato_healthy", "Tomato_late_blight"]

#preprocess webcam frame
def preprocess_frame(frame):
    try:
        resized_frame = cv2.resize(frame, (224, 224))  # Resize to match input size of the model
        normalized_frame = resized_frame / 255.0      # Normalize pixel values
        expanded_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
        return expanded_frame
    except Exception as e:
        print(f"Error in preprocessing frame: {e}")
        return None

#Start webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera, change if necessary

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam accessed successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from the webcam.")
        break

    #Preprocess the frame for prediction
    input_frame = preprocess_frame(frame)
    
    
    #Prediction using the model
    try:
        predictions = model.predict(input_frame)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        confidence = np.max(predictions) * 100
    except Exception as e:
        print(f"Error during prediction: {e}")
        predicted_class_label = "Error"
        confidence = 0.0

    #Display the prediction on the webcam feed
    display_text = f"{predicted_class_label} ({confidence:.2f}%)"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
  
   
    text_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_width, text_height = text_size
    cv2.rectangle(frame, (5, 5), (15 + text_width, 35 + text_height), (0, 255, 0), -1)
    
    # Show the frame
    cv2.imshow("Leaf Disease Detection", frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Model Output Shape: {model.output_shape}")


# In[9]:


print(model.output_shape)


# In[10]:


print(len(class_labels))


# In[11]:


print(train_data.class_indices)


# In[12]:


class_labels = list(train_data.class_indices.keys())


# In[1]:


import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

# Load the trained model
model_path = "leaf_disease_alexnet_model.h5"
model = load_model(model_path)

# Define class labels
class_labels = ["Potato_healthy", "Potato_late_blight", "Potato_early_blight", "Tomato_early_blight", "Tomato_healthy", "Tomato_late_blight"]

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resized_image = cv2.resize(image, (224, 224))  # Resize to model input size
    normalized_image = resized_image / 255.0       # Normalize pixel values
    return np.expand_dims(normalized_image, axis=0), image  # Add batch dimension and return original image

# Function to handle image upload and prediction
def upload_and_predict():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Preprocess the selected image
    input_image, original_image = preprocess_image(file_path)

    # Perform prediction
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100

    # Display the results
    display_image(original_image, f"{predicted_class_label} ({confidence:.2f}%)")

# Function to display the uploaded image and prediction results
def display_image(image, prediction_text):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    # Update the label with the image
    image_label.config(image=image)
    image_label.image = image

    # Update the label with the prediction text
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


# In[2]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionV3, MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Define class labels
class_labels = ["Potato_healthy", "Potato_late_blight", "Potato_early_blight", "Tomato_early_blight", "Tomato_healthy", "Tomato_late_blight"]

# Load and preprocess dataset (replace with your dataset path)
data_dir = "dataset_directory"
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
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    data_dir,
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
        epochs=5,  # Adjust epochs as needed
        verbose=1
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_data, verbose=0)
    print(f"{model_name} Validation Accuracy: {val_accuracy:.4f}")

    # Save the best model
    if val_accuracy > best_accuracy:
        best_model = model
        best_accuracy = val_accuracy
        best_model_name = model_name

print(f"Best Model: {best_model_name} with Validation Accuracy: {best_accuracy:.4f}")

# Save the best model
best_model.save("best_leaf_disease_model.h5")

# Tkinter Application for the Best Model
import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk

# Load the best model
model = tf.keras.models.load_model("best_leaf_disease_model.h5")

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resized_image = cv2.resize(image, (224, 224))  # Resize to model input size
    normalized_image = resized_image / 255.0       # Normalize pixel values
    return np.expand_dims(normalized_image, axis=0), image  # Add batch dimension and return original image

# Function to handle image upload and prediction
def upload_and_predict():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Preprocess the selected image
    input_image, original_image = preprocess_image(file_path)

    # Perform prediction
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100

    # Display the results
    display_image(original_image, f"{predicted_class_label} ({confidence:.2f}%)")

# Function to display the uploaded image and prediction results
def display_image(image, prediction_text):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    # Update the label with the image
    image_label.config(image=image)
    image_label.image = image

    # Update the label with the prediction text
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
        epochs=5,  # Adjust epochs as needed
        verbose=1
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_data, verbose=0)
    print(f"{model_name} Validation Accuracy: {val_accuracy:.4f}")

    # Save the best model
    if val_accuracy > best_accuracy:
        best_model = model
        best_accuracy = val_accuracy
        best_model_name = model_name

print(f"Best Model: {best_model_name} with Validation Accuracy: {best_accuracy:.4f}")

# Save the best model
best_model.save("best_leaf_disease_model.h5")

# Tkinter Application for the Best Model
import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk

# Load the best model
model = tf.keras.models.load_model("best_leaf_disease_model.h5")

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resized_image = cv2.resize(image, (224, 224))  # Resize to model input size
    normalized_image = resized_image / 255.0       # Normalize pixel values
    return np.expand_dims(normalized_image, axis=0), image  # Add batch dimension and return original image

# Function to handle image upload and prediction
def upload_and_predict():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Preprocess the selected image
    input_image, original_image = preprocess_image(file_path)

    # Perform prediction
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100

    # Display the results
    display_image(original_image, f"{predicted_class_label} ({confidence:.2f}%)")

# Function to display the uploaded image and prediction results
def display_image(image, prediction_text):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    # Update the label with the image
    image_label.config(image=image)
    image_label.image = image

    # Update the label with the prediction text
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


# In[ ]:




