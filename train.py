import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

print("=" * 50)
print("AI-Based Fire Detection System")
print("=" * 50)

# =====================================
# Image Preprocessing
# =====================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================
# Load Dataset
# =====================================
dataset = ImageFolder(
    root="dataset",
    transform=transform
)

# =====================================
# Create DataLoader
# =====================================
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# =====================================
# Display Dataset Information
# =====================================
print("\nDataset Loaded Successfully!\n")

print("Classes :", dataset.classes)
print("Class Labels :", dataset.class_to_idx)
print("Total Images :", len(dataset))

print("-" * 50)

# =====================================
# Read One Batch
# =====================================
images, labels = next(iter(train_loader))

print("Batch Image Shape :", images.shape)
print("Batch Labels Shape :", labels.shape)

print("-" * 50)
print("DataLoader is Ready!")

# =====================================
# Load Pre-trained MobileNetV2
# =====================================
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)

# =====================================
# Replace Final Layer
# =====================================
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

print("-" * 50)
print("MobileNetV2 Model Loaded Successfully!")
print("-" * 50)

# =====================================
# Select Device (CPU or GPU)
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Using Device :", device)

# =====================================
# Loss Function
# =====================================
criterion = nn.CrossEntropyLoss()

# =====================================
# Optimizer
# =====================================
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =====================================
# Number of Epochs
# =====================================
epochs = 1

print("-" * 50)
print("Training Started...")
print("-" * 50)

# =====================================
# Training Loop
# =====================================
for epoch in range(epochs):

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        # Move images and labels to CPU/GPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward Pass
        outputs = model(images)

        # Calculate Loss
        loss = criterion(outputs, labels)

        # Backward Pass
        loss.backward()

        # Update Model Weights
        optimizer.step()

        running_loss += loss.item()

        # Show Progress Every 50 Batches
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    average_loss = running_loss / len(train_loader)

    print("-" * 50)
    print(f"Epoch [{epoch+1}/{epochs}] Completed")
    print(f"Average Loss : {average_loss:.4f}")
    print("-" * 50)

# =====================================
# Save Trained Model
# =====================================
torch.save(model.state_dict(), "model.pth")

print("Training Completed Successfully!")
print("Model Saved as model.pth")
print("=" * 50)