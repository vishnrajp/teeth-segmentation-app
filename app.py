import streamlit as st
import torch
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
@st.cache_resource
def load_model(model_path):
    # Load the model and map it to the CPU
    model = torch.load(model_path, map_location=torch.device('cpu'),weights_only=False).eval()
    return model

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((384, 768)),   # Resize to match training size
        transforms.Grayscale(),          # Convert to grayscale
        transforms.ToTensor(),           # Convert to tensor
    ])
    image = transform(image).to(device)  # Add batch dimension
    return image

def toPIL(tensor):
    return transforms.ToPILImage()(tensor.cpu())

# Overlay mask on the original image
def overlay_mask(image, probabilities):
    # Convert image tensor to PIL and then to NumPy array
    input_image = toPIL(image).convert("RGB")
    input_image = np.array(input_image)  # Shape: (384, 768, 3)

    # Convert probabilities tensor to PIL and then to NumPy array
    prediction_mask = toPIL(probabilities).convert("L")
    prediction_mask = np.array(prediction_mask)  # Shape: (384, 768)

    # Create a red mask with the same shape as the input image
    red_mask = np.zeros_like(input_image)  # Shape: (384, 768, 3)
    red_mask[..., 0] = prediction_mask  # Set the red channel

    # Blend the original image with the red mask
    overlayed_image = input_image * 0.7 + red_mask * 0.3  # Blend with original image
    overlayed_image = np.clip(overlayed_image, 0, 255).astype(np.uint8)  # Ensure valid pixel values

    return Image.fromarray(overlayed_image)

# Streamlit app
def main():
    st.title("Dental Caries Segmentation App")
    st.write("Upload an X-ray image or select from existing images to detect dental caries.")

    # Load the model
    model_path = "UNetEfficientnetB0-best.pth"
    model = load_model(model_path)
    image = None

    # Option to upload or select from existing images
    option = st.radio("Choose an option:", ("Upload an image", "Select from existing images"))
    if option == "Upload an image":
        # File uploader
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            overlayed_image = None 
            image = Image.open(uploaded_file)
    else:
        existing_images = [
            "10.jpg",
            "77.jpg",
            "80.jpg",
        ]

        # Dropdown to select an image
        uploaded_file = st.selectbox("Select an image:", existing_images)

        if uploaded_file:
            image_path = os.path.join("Demo_Images", uploaded_file)  # Folder containing existing images
            image = Image.open(image_path)

    if image:
        col1, col2 = st.columns(2)

        # Display the uploaded image in the first column
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Display the overlayed image in the second column
        with col2:
            col2_placeholder = st.empty()  # Create a placeholder for col2

            with st.spinner("Processing image..."):
                # Preprocess the image
                input_tensor = preprocess_image(image)
                
                # Perform inference
                with torch.no_grad():
                    probabilities = F.sigmoid(model(input_tensor.unsqueeze(0))).squeeze(0)
                
                # Overlay the mask on the original image
                overlayed_image = overlay_mask(input_tensor, probabilities)
                col2_placeholder.image(overlayed_image, caption="Affected Areas", use_container_width=True)
if __name__ == "__main__":
    main()