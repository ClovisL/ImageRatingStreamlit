from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import streamlit as st

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 37 * 37, 128)  # Corrected input size calculation
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the saved model (Do this only once at the beginning)
model_path = 'ImagePolarity.pth'
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier()  # Instantiate the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: {model_path} not found. Please ensure the model file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def upload_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True) #Display image
            return image #Return the image object
        except Exception as e:
            st.error(f"Could not open image: {e}")
            return None
    return None

def predict_image(image, model, device): #Take the model and device as an argument
    if image is None:
        st.warning("Please upload an image first.")
        return

    try:
        transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)
            prediction_value = prediction.item()

        st.write(f"Predicted Sentiment: {prediction_value:.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.title("Image Sentiment Analysis")

uploaded_image = upload_image()

if uploaded_image:
    if st.button("Predict Sentiment"):
        predict_image(uploaded_image, model, device)
