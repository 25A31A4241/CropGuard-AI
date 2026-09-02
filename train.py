import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models, transforms


# =========================
# 1. SETTINGS
# =========================

BATCH_SIZE = 16
IMAGE_SIZE = 224
EPOCHS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("===================================")
print("       CropGuard AI Training")
print("===================================")
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# 2. DATASET
# =========================

class CropDataset(Dataset):

    def __init__(self, csv_file, transform=None, class_to_index=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        if class_to_index is None:
            classes = sorted(self.data["class"].unique())
            self.class_to_index = {
                name: index for index, name in enumerate(classes)
            }
        else:
            self.class_to_index = class_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_path = self.data.iloc[index]["image_path"]
        class_name = self.data.iloc[index]["class"]

        image = Image.open(image_path).convert("RGB")

        label = self.class_to_index[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# 3. IMAGE TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 4. LOAD DATA
# =========================

train_dataset = CropDataset(
    "data/processed/train.csv",
    transform=train_transform
)

class_to_index = train_dataset.class_to_index

validation_dataset = CropDataset(
    "data/processed/validation.csv",
    transform=validation_transform,
    class_to_index=class_to_index
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Number of classes:", len(class_to_index))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# 5. LOAD RESNET18
# =========================

print("\nLoading ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# Replace final layer for our 21 classes
number_of_classes = len(class_to_index)

model.fc = nn.Linear(
    model.fc.in_features,
    number_of_classes
)

model = model.to(DEVICE)

print("ResNet18 ready!")


# =========================
# 6. LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


# =========================
# 7. TRAINING
# =========================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"| Training Accuracy: {accuracy:.2f}%"
    )


# =========================
# 8. VALIDATION
# =========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in validation_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


validation_accuracy = 100 * correct / total

print("\n===================================")
print("Validation Accuracy:", f"{validation_accuracy:.2f}%")
print("===================================")


# =========================
# 9. SAVE MODEL
# =========================

os.makedirs("models", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_to_index": class_to_index
    },
    "models/cropguard_resnet18.pth"
)

print("\nModel saved to:")
print("models/cropguard_resnet18.pth")

print("\n🌱 CropGuard training completed!")