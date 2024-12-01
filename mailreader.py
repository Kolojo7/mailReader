import tkinter as tk
import cv2
from tkinter import Label, Button, Frame
from helper import capture_image, extract_text_from_image, describe_image, capture_image_with_preview, update_video_preview, stop_video_preview, update_image_preview # Import helpers
from openai import OpenAI
from PIL import Image, ImageTk

# Global variables
image_path = "captured_image.jpg"
client = OpenAI(api_key="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")

# Create a Tkinter window
window = tk.Tk()
window.title("Image Capture and OCR")
window.geometry("1000x800")
# Create a frame for organizing the layout into two columns
frame = Frame(window)
frame.pack(pady=10)

# Live Video Preview Label (Left side of the frame)
video_label = Label(frame, text="Video Preview", width=900, height=500, bg="gray")
video_label.grid(row=0, column=0, padx=10)

# Image Preview Label (Right side of the frame)
#image_label = Label(frame, text="Captured Image", width=300, height=300, bg="lightgray")
#image_label.grid(row=0, column=1, padx=10)

# Start video preview
update_video_preview(video_label)

# Create buttons to Capture Image/Read from Image/Describe Image
capture_button = Button(window, text="Capture Image", command=lambda: capture_image(image_path), width=20, height=2, bg="lightblue")
capture_button.pack(pady=5)

read_button = Button(window, text="Read Text from Image", command=lambda: extract_text_from_image(image_path, result_label), width=20, height=2, bg="lightgreen")
read_button.pack(pady=5)

describe_button = Button(window, text="Describe Image", command=lambda: describe_image(image_path, result_label, client), width=20, height=2, bg="orange")
describe_button.pack(pady=5)

# Label to display the extracted text
result_label = Label(window, text="Output:", wraplength=900, justify="left", bg="white", relief="solid")
result_label.pack(pady=10, fill="both", expand=True)

# Define the close event handler
def on_closing():
    stop_video_preview()
    window.destroy()

# Ensure camera stops on window close
window.protocol("WM_DELETE_WINDOW", on_closing)

# Run the Tkinter main loop
window.mainloop()