import cv2
import tkinter as tk
from tkinter import messagebox, Label, Button
import time
import pytesseract
from pytesseract import Output
from PIL import Image, ImageTk

# Optional: Specify the path to Tesseract if needed
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Global variable to store the path of the captured image
image_path = "captured_image.jpg"

def capture_image():
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open the camera.")
    else:
        # Allow the camera to warm up
        time.sleep(2)  # 2-second delay to adjust the camera
        
        # Capture a frame
        ret, frame = cap.read()

        # Retry if the frame wasn't captured correctly
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            cap.release()
            return
        
        # Save the frame as an image
        cv2.imwrite(image_path, frame)
        messagebox.showinfo("Image Captured", f"Image saved at {image_path}")
        
        # Release the camera and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
        
        # After capturing, run OCR on the image
        extract_text_from_image(image_path)

def extract_text_from_image(image_path):
    """Extract text from the captured image using Tesseract OCR"""
    # Read the image from file
    image = cv2.imread(image_path)

    # Convert the image to grayscale for better OCR accuracy
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply OCR to extract text
    text = pytesseract.image_to_string(gray_image, output_type=Output.STRING)
    
    # Display the extracted text in the Tkinter window
    print("Extracted Text:")
    print(text)
    result_label.config(text="Extracted Text:\n" + text)

# Create a Tkinter window
window = tk.Tk()
window.title("Image Capture and OCR")

# Set window size
window.geometry("400x300")

# Create a button to capture image
capture_button = Button(window, text="Capture Image", command=capture_image)
capture_button.pack(pady=20)

# Label to display the extracted text
result_label = Label(window, text="", wraplength=350, justify="left")
result_label.pack(pady=10)

# Run the Tkinter main loop
window.mainloop()