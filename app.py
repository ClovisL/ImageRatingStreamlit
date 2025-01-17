import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk  # Import ttk for themed widgets (Scrollbar)
from PIL import Image, ImageTk
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
    global img_path, img_tk
    filename = filedialog.askopenfilename(initialdir=".", title="Select an Image",
                                           filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
    if filename:
        img_path = filename
        try:
            pil_img = Image.open(img_path)
            pil_img = pil_img.resize((300, 300), Image.LANCZOS)  # Resize for display
            img_tk = ImageTk.PhotoImage(pil_img)
            image_label.config(image=img_tk)
            prediction_label.config(text="") # Clear previous predictions
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image: {e}")

def predict_image():
    global img_path
    if not img_path:
        messagebox.showwarning("Warning", "Please upload an image first.")
        return

    try:
        # Preprocess image for PyTorch (assuming input size of 150x150)
        transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Use model to predict
        with torch.no_grad():
            prediction = model(img_tensor)
            prediction_value = prediction.item()  # Extract single value

        prediction_label.config(text=f"Predicted Sentiment: {prediction_value:.2f}")  # Display with 2 decimal places

    except Exception as e:
        messagebox.showerror("Error", f"Prediction error: {e}")

# Create the main window
window = tk.Tk()
window.title("Image Sentiment Analysis")
window.geometry("600x400")

canvas = tk.Canvas(window)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)

def update_scrollregion(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame = tk.Frame(canvas)
frame.pack(fill=tk.BOTH, expand=True)
canvas.create_window((0, 0), window=frame, anchor="nw")
frame.bind("<Configure>", update_scrollregion)

def on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")  # Windows

window.bind("<MouseWheel>", on_mousewheel)  # Windows

# Use grid for layout and centering
inner_frame = tk.Frame(frame)
inner_frame.grid(row=0, column=0, sticky="nsew") #Sticky makes it fill frame

upload_button = tk.Button(inner_frame, text="Upload Image", command=upload_image)
upload_button.grid(row=0, column=1, pady=(10, 5), sticky="ew") #Column 1
upload_button.config(width=30)

image_label = tk.Label(inner_frame)
image_label.grid(row=1, column=1, pady=5) #Column 1

prediction_label = tk.Label(inner_frame, font=("Arial", 14))
prediction_label.grid(row=2, column=1, pady=5) #Column 1

predict_button = tk.Button(inner_frame, text="Predict Sentiment", command=predict_image)
predict_button.grid(row=3, column=1, pady=(5, 10), sticky="ew") #Column 1
predict_button.config(width=30)

# Configure columns for centering
inner_frame.grid_columnconfigure(0, weight=1) #Left Spacer
inner_frame.grid_columnconfigure(1, weight=1) #Content Column
inner_frame.grid_columnconfigure(2, weight=1) #Right Spacer

frame.rowconfigure(0, weight=1) #Make the row expandable
frame.columnconfigure(0, weight=1) #Make the column expandable

img_path = None
img_tk = None

window.mainloop()