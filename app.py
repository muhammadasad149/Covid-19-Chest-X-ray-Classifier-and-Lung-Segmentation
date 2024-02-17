import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt

# Load the segmentation model
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

segmentation_model = load_model('model.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

# Function to preprocess the image for classification
def preprocess_image_classification(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to preprocess the image for segmentation
def preprocess_image_segmentation(image):
    X_shape = 512
    image = Image.open(image).convert('RGB')
    img = np.array(image)  # Convert PIL Image to NumPy array
    img = cv2.resize(img, (X_shape, X_shape))[:, :, 0]
    img = (img.reshape(1, 512, 512, 1) - 127.0) / 127.0
    return img

# Function to make classification predictions
def predict_image_class(image, model):
    processed_img = preprocess_image_classification(image)
    classification_model = tf.keras.models.load_model(model)
    prediction = classification_model.predict(processed_img)
    class_index = np.argmax(prediction)
    return class_index

# Function to perform image segmentation
def segment_image(image):
    processed_img = preprocess_image_segmentation(image)
    op = segmentation_model.predict(processed_img)
    return op.reshape(512, 512)

def main():
    st.title("Covid-19 Chest X-ray Classifier and Lung Segmentation")

    # Choose the classification model
    selected_model = st.selectbox("Select Classification Model", ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'Inception'])

    if selected_model:
        uploaded_image = st.file_uploader("Choose a chest X-ray image...", type=["png"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
            st.write("")

            st.write("Classifying...")

            # Get classification prediction
            classification_model_path = f'{selected_model.lower()}_model.h5'
            classification_prediction = predict_image_class(uploaded_image, classification_model_path)

            # Display the classification result
            if classification_prediction == 0:
                st.write("**Classification Prediction:** COVID-19")
            elif classification_prediction == 1:
                st.write("**Classification Prediction:** Normal")
            else:
                st.write("**Classification Prediction:** Viral")

            st.write("")
            st.write("Segmenting...")

            # Get segmentation result
            segmentation_result = segment_image(uploaded_image)

            # Display the segmentation result
            fig, ax = plt.subplots()
            ax.imshow(segmentation_result, cmap="jet")
            ax.set_title("Lungs Segmentation")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
