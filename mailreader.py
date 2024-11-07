import cv2
import pytesseract
from pytesseract import Output

# Optional: Specify the path to tesseract if needed
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def capture_image():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened
    if not cap.isOpened():
        print("Cannot open camera")
        return None

    print("Press 's' to capture the image or 'q' to quit.")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame read is successful
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Wait for key press
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Save the captured frame as 'captured_image.jpg'
            cv2.imwrite('captured_image.jpg', frame)
            print("Image captured and saved as 'captured_image.jpg'")
            break
        elif key == ord('q'):
            print("Exiting without capturing image.")
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    return 'captured_image.jpg'

def extract_text_from_image(image_path):
    # Read the image from file
    image = cv2.imread(image_path)

    # Convert the image to grayscale for better OCR accuracy
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply OCR to extract text
    text = pytesseract.image_to_string(gray_image, output_type=Output.STRING)
    
    print("Extracted Text:")
    print(text)
    return text

# Capture image from webcam
image_path = capture_image()

# If an image was captured, run OCR
if image_path:
    extract_text_from_image(image_path)

capture_image()
extract_text_from_image()
