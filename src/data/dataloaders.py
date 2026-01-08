import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_transforms(img_size=224, model_type="cnn", train=True):
   
    if model_type == "cnn":
        # For the CNN: single channel, simple normalization
        base_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]
    
    elif model_type == "resnet":
        # ImageNet normalization (required for transfer learning)
        base_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),  # Replicate to RGB
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ]
    
    else:
        raise ValueError("model_type must be 'cnn' or 'resnet'")
    
    if train:
        # More conservative augmentation for radiographs
        if model_type == "cnn":
            # More aggressive augmentation for the CNN
            train_aug = [
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0))
            ]
        else:  # resnet
            # More conservative augmentation for pre-trained ResNet
            train_aug = [
                transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),  # Smaller crop
                transforms.RandomRotation(8),  # Smaller rotation
                transforms.RandomHorizontalFlip(p=0.3),  # Less flip
                transforms.ColorJitter(brightness=0.05, contrast=0.05)  # Subtle adjustment
            ]
        
        # Apply augmentation BEFORE base transformations
        transform = transforms.Compose(train_aug + base_transforms)
    else:
        transform = transforms.Compose(base_transforms)
    
    return transform


def get_loaders(
    batch_size=32,
    img_size=224,
    model_type="cnn",
    num_workers=4,
    data_dir="../data/chest_xray"  
):
    
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    # Verify that directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Obtain transformations
    train_transform = get_transforms(
        img_size=img_size,
        model_type=model_type,
        train=True
    )
    
    test_transform = get_transforms(
        img_size=img_size,
        model_type=model_type,
        train=False
    )
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )
    
    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=test_transform
    )
    
    # Automatic adjustment of num_workers for Mac
    if torch.backends.mps.is_available() and num_workers > 0:
        print("Apple Silicon detected. Setting num_workers to 0 for compatibility.")
        num_workers = 0
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  
        persistent_workers=True if num_workers > 0 else False  
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader


# Helper function to visualize samples
def visualize_batch(loader, model_type="cnn", num_images=8):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Obtain a batch
    images, labels = next(iter(loader))
    
    # Denormalization according to the model type
    if model_type == "cnn":
        mean = torch.tensor([0.5])
        std = torch.tensor([0.5])
    else:  # resnet
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    class_names = loader.dataset.classes
    
    for i in range(min(num_images, len(images))):
        img = images[i]
        
        # Denormalize
        if model_type == "cnn":
            img = img * std[:, None, None] + mean[:, None, None]
            img = img.squeeze().numpy()  # (H, W)
            axes[i].imshow(img, cmap="gray")
        else:  # resnet
            img = img * std[:, None, None] + mean[:, None, None]
            img = img.permute(1, 2, 0).numpy()  # (H, W, 3)
            axes[i].imshow(img)
        
        axes[i].set_title(f"{class_names[labels[i]]}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig("sample_batch.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Sample batch saved in: sample_batch.png")


# Function to calculate class weights (for CrossEntropyLoss)
def get_class_weights(loader, device="cpu"):
   
    class_counts = torch.zeros(len(loader.dataset.classes))
    
    for _, labels in loader:
        for label in labels:
            class_counts[label] += 1
    
    # Weight inversely proportional to frequency
    total = class_counts.sum()
    weights = total / (len(class_counts) * class_counts)
    
    print("\nClass weights calculated:")
    for i, (class_name, weight) in enumerate(zip(loader.dataset.classes, weights)):
        print(f"  {class_name}: {weight:.3f}")
    
    return weights.to(device)
