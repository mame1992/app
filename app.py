import torch
from torchvision.models import vit_b_16
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import streamlit as st
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Load the saved model
num_classes = 5
model = vit_b_16(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
state_dict = torch.load('trained_vit_model.pth', map_location=device)  # Load on the same device
model.load_state_dict(state_dict)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class labels
class_labels = ['10', '100', '200', '5', '50']

# Streamlit app
st.title("Currency Note Classification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        predictions = model(image_tensor)
        probabilities = F.softmax(predictions, dim=1)

    # Check if the maximum probability is above a certain threshold (e.g., 0.5)
    max_probability = torch.max(probabilities).item()
    if max_probability > 0.5:  # Adjust the threshold as needed
        # Display results
        st.subheader("Prediction Results:")
        for label, probability in zip(class_labels, probabilities[0]):
            st.write(f'{label}: {probability.item():.2f}')

        _, predicted_idx = torch.max(predictions, 1)
        predicted_class = class_labels[predicted_idx.item()]
        st.write(f"Predicted Class: {predicted_class}")
    else:
        st.write("Currency not detected. Please upload an image of a valid currency note.")
