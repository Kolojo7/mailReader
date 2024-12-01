import cv2
import time
import pytesseract
from PIL import Image, ImageTk
from pytesseract import Output
from tkinter import messagebox
import base64

# Specify the path to Tesseract if needed
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
#Global Cap variable for camera
cap = cv2.VideoCapture(0)  # Open the camera
# Function to encode an image into Base64 format
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Function to capture an image using the camera and save it to the specified path
def capture_image(image_path):

    # Open the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        # Show error message if camera cannot be opened
        messagebox.showerror("Error", "Could not open the camera.")
        return
    
    #time.sleep(2)  # Allow the camera to warm up and adjust for 2 seconds

    # Capture a frame
    ret, frame = cap.read()

    if not ret:
        # Show an error message if the image capture fails
        messagebox.showerror("Error", "Failed to capture image.")
        cap.release()
        return

    # Save the captured frame to the specified path
    cv2.imwrite(image_path, frame)
    messagebox.showinfo("Image Captured", f"Image saved at {image_path}")

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# READ TEXT FROM IMAGE BUTTON - Function to extract text from an image using Tesseract OCR
def extract_text_from_image(image_path, result_label):

    # Read in the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale for better OCR accuracy
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Tesseract OCR to extract text
    text = pytesseract.image_to_string(gray_image, output_type=Output.STRING)

    # Update the Tkinter label to display the extracted text
    print(text)
    result_label.config(text="Extracted Text:\n" + text)
    result_label.update_idletasks() #Update window to fit print correctly

# DESCRIBE IMAGE BUTTON - Function to describe the content of an image using OpenAI's GPT
def describe_image(image_path, result_label, client):
    base64_image = encode_image(image_path) #Using Encode image helper function to convert to base64 (readable by gpt)
    try:
        # Prompt for GPT to describe the image content
        prompt = (
            "Describe the content of this image in detail as if explaining to someone who cannot see."
            "Make the response 2 to 3 sentences."
        )

        # Use OpenAI API to generate a detailed description
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an image description assistant for the blind."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                },
            ],
        )

        # Extract the description content from the response
        description = response.choices[0].message.content

        # Update the Tkinter label with the description
        result_label.config(text="Image Description:\n" + description)
        result_label.update_idletasks()
    
    #Error Handling, show error message if description fails
    except Exception as e:
        messagebox.showerror("Error", f"Failed to describe image. Error: {str(e)}")

#Continuously updates the video preview by capturing frames from the webcam.
def update_video_preview(video_label):
    ret, frame = cap.read()
    if ret:
        # Convert the frame to RGB format (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PIL image, then to a Tkinter image
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new image
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    
    # Schedule the next frame update
    video_label.after(10, update_video_preview, video_label)

#Stops the video preview and releases the camera.
def stop_video_preview():
    global cap
    cap.release()
    cv2.destroyAllWindows()
    
#Function to capture an image from the video preview.
def capture_image_with_preview(image_path):
    global cap
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(image_path, frame)
    update_image_preview(image_path)

# Function to update the image preview
def update_image_preview(image_path, image_label):
    image = Image.open(image_path)
    image = image.resize((300, 300))  # Resize to fit the label
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference

#Function to handle closing out the application
'''
def on_closing(window):
    try:
        # Release the camera if it is active
        if 'cap' in globals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    # Destroy the Tkinter window
    window.destroy()
'''

#Function to display a preview of captured image in-app
'''
def display_image(image_path, image_label):
    try:
        # Load the image using PIL
        pil_image = Image.open(image_path)
        pil_image = pil_image.resize((200, 200))  # Resize the image to fit the label
        tk_image = ImageTk.PhotoImage(pil_image)
        # Update the Label with the new image
        image_label.config(image=tk_image)
        image_label.image = tk_image  # Keep a reference to avoid garbage collection
    except Exception as e:
        messagebox.showerror("Error", f"Failed to display image. Error: {str(e)}")
'''