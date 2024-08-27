import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Load the trained model
model = torch.load(r"E:\Brain-Tumor-Detection-main (1)\Brain-Tumor-Detection-main\Brain_Tumor_model.pt")
model.eval()  # Set the model to evaluation mode

# Define the transformation to apply to input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class labels
CLA_label = {
    0: 'Brain Tumor',
    1: 'Healthy'
}

def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return CLA_label[predicted.item()]

# Create the Streamlit app
st.title("Brain Tumor Detection and Diagnosis")

st.write("Upload an MRI scan to classify the image as 'Brain Tumor' or 'Healthy'.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label = predict(image)
    st.write(f"Prediction: {label}")
