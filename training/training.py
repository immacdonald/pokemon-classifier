import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision import datasets, models, transforms

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001
num_types = 18  # Number of Pokémon types

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))
data_dir = os.path.join(dir, "dataset")

# Image transformations
data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to 224x224 as expected by ResNet
        transforms.CenterCrop(224),  # Crop to center
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize to ResNet's expected ranges
    ]
)

print("Transforming image data for ResNet")

# Load dataset and apply transformations
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print("Splitting dataset into training and validation")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Initializing model for type classification")

# Initialize the model, modify final layer
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_types)  # Change output layer to match number of types
model = model.to(device)


print("Setting loss and optimizer")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation loop
print("Starting training")

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Epoch [{epoch+1}/{num_epochs}], " f"Train Loss: {avg_train_loss:.4f}, " f"Val Loss: {avg_val_loss:.4f}, " f"Val Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), os.path.join(dir, f'pokemon_classifier_model_{epoch}.pth'))

torch.save(model.state_dict(), os.path.join(dir, f'pokemon_classifier_model.pth'))
print("Training complete.")
