import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Define your model architecture (same as you used for training)
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=100 * 3 * 3, out_features=512)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, C, W, H = x.shape
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.reshape(B, -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load the saved model
model = CustomCNN(num_classes=24)  # Ensure num_classes matches your training
model.load_state_dict(torch.load('asl_model.pth'))
model.eval()  # Set the model to evaluation mode

# # Load your data (adjust path for local machine)
# train_df = pd.read_csv("/content/drive/MyDrive/colab_data/data/asl_data/sign_mnist_train.csv")
# valid_df = pd.read_csv("/content/drive/MyDrive/colab_data/data/asl_data/sign_mnist_valid.csv")

# y_train = train_df['label']
# y_valid = valid_df['label']
# del train_df['label']
# del valid_df['label']

# x_train = train_df.values
# x_valid = valid_df.values

# # Convert to PyTorch tensors
# y_train_data = torch.tensor(y_train, dtype=torch.int64)
# y_valid_data = torch.tensor(y_valid, dtype=torch.int64)

# # One-hot encoding
# y_train = F.one_hot(y_train_data, num_classes=24)
# y_valid = F.one_hot(y_valid_data, num_classes=24)

# # Normalize the image data
# x_train = x_train / 255.
# x_valid = x_valid / 255.

# # Reshape data for input to CNN (B, H, W, C -> B, C, H, W)
# x_train = x_train.reshape(-1,28,28,1)
# x_valid = x_valid.reshape(-1,28,28,1)

# # Dataset and DataLoader
# class CustomDataset(Dataset):
#     def __init__(self, data, labels, transform=None):
#         self.data = data
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image = self.data[idx]
#         label = self.labels[idx]

#         if self.transform:
#             image = self.transform(image)

#         return torch.tensor(image, dtype=torch.float32), label

# valid_dataset = CustomDataset(data=x_valid, labels=y_valid_data)
# valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

# # Function to calculate accuracy
# def calculate_accuracy(outputs, targets):
#     predicted = outputs.argmax(-1)
#     correct = (predicted == targets).sum().item()
#     total = targets.size(0)
#     return correct / total

# # Evaluate the model
# correct = 0
# total = 0
# with torch.no_grad():  # No need to calculate gradients during evaluation
#     for images, labels in valid_loader:
#         images = images.permute(0, 3, 1, 2)  # Change shape from (B, H, W, C) -> (B, C, H, W)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = correct / total
# print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Define the label mapping from index to class
label_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Transformation for the input image (resize, grayscale, normalize)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.Resize((28, 28)),  # Resize to match input size of model
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [0,1] range
])

# Function to predict the label of the uploaded image
def predict_image(image_path):
    # Open the image
    image = Image.open(image_path).convert('RGB')  # Ensure it's in RGB format
    # Apply the transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    
    # Get the model's prediction
    with torch.no_grad():  # No need to calculate gradients during inference
        outputs = model(image)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get the index of the highest probability
        
    # Get the label corresponding to the predicted index
    predicted_label = label_mapping[predicted.item()]
    return predicted_label

# Example usage:
image_path = "/content/d.png"
predicted_label = predict_image(image_path)

print(f"Predicted label: {predicted_label}")

image_path = "/content/c.png"
predicted_label = predict_image(image_path)
print(f"Predicted label: {predicted_label}")