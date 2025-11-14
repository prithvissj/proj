Leaf Disease Detection System (Deep Learning + Tkinter GUI)

This project is a Leaf Disease Detection System built using TensorFlow,
Transfer Learning models, and a Tkinter-based GUI for image prediction.

FEATURES: - Multi-class classification for 6 leaf categories:
Potato_healthy, Potato_late_blight, Potato_early_blight,
Tomato_early_blight, Tomato_healthy, Tomato_late_blight - Uses four deep
learning architectures: ResNet50, EfficientNetB0, InceptionV3,
MobileNetV2 - Trains all models and automatically selects the best based
on validation accuracy - Saves the best model as
best_leaf_disease_model.h5 - GUI for uploading leaf images and
displaying prediction + confidence score - Data augmentation for
robustness

DATASET STRUCTURE: train/ class_folders… test/ same class folders…

INSTALLATION:
1. Install required libraries: pip install tensorflow
pillow numpy opencv-python 
2. Update dataset paths inside main.py 
3.Run: python main.py 
4. The GUI will launch automatically after training.

WORKFLOW: - Loads and augments training images - Builds and trains four
transfer-learning CNN models - Compares validation accuracy and saves
the best model - GUI loads the final model and predicts disease from
images

FUTURE IMPROVEMENTS: - Add visual model explanations (Grad-CAM) - Build
a Streamlit web interface - Add camera-based real-time prediction
