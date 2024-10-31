import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load your trained model
model = load_model('Saved Models/real_vs_fake_image_classifier.h5')

# Function to preprocess image and predict
# Image preprocessing
IMG_HEIGHT = 32
IMG_WIDTH = 32
def predict_image(img):
    # img = img.resize((224, 224))  # Resize image to match model input size
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array = img_array / 255.0  # Normalize the image
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return 'AI-generated' if prediction[0][0] > 0.5 else 'Real Image'

# Streamlit UI
st.title("AI-generated Image Detector")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label = predict_image(img)
    st.write(f'This image is: **{label}**')
