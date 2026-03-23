import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MedMamba import MedMamba  # Ensure MedMamba.py is in the same folder

def visualize_misclassifications(model, test_loader, device, class_names, num_images=10):
    model.eval()
    misclassified_images = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Find indices where prediction != ground truth
            incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in incorrect_indices:
                if len(misclassified_images) >= num_images:
                    break
                
                # Store (image, predicted_label, actual_label)
                img = images[idx].cpu()
                misclassified_images.append({
                    "img": img,
                    "pred": class_names[preds[idx].item()],
                    "actual": class_names[labels[idx].item()]
                })
            
            if len(misclassified_images) >= num_images:
                break

    # Plotting
    plt.figure(figsize=(15, 10))
    for i, item in enumerate(misclassified_images):
        plt.subplot(2, 5, i + 1)
        # Un-normalize the image for display (assuming standard ImageNet normalization)
        img = item["img"].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f"Pred: {item['pred']}\nActual: {item['actual']}", color='red')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Execution Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Model (Example for MedMamba_Tiny)
model = MedMamba(num_classes=3).to(device) # Adjust num_classes to your dataset
model.load_state_dict(torch.load("/kaggle/input/datasets/rajab23456/medmambanet-pth", map_location=device))

# 2. Data Loader (Ensure this matches your train/val transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root='/kaggle/input/alzheimersoriginaldataset/OriginalDataset', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes

# 3. Run Visualization
visualize_misclassifications(model, test_loader, device, class_names)
