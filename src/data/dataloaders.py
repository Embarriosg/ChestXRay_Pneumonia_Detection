import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(img_size=224, model_type="cnn", train=True):
    """
    Returns torchvision transforms depending on model type.
    
    model_type:
        - 'cnn'     : grayscale, 1 channel
        - 'resnet'  : grayscale replicated to 3 channels (ImageNet compatible)
    """
    if model_type == "cnn":
        num_channels = 1
        mean = [0.5]
        std = [0.5]

    elif model_type == "resnet":
        num_channels = 3
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    else:
        raise ValueError("model_type must be 'cnn' or 'resnet'")

    base_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=num_channels),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    if train:
        train_aug = [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0))
        ]
        transform = transforms.Compose(train_aug + base_transforms)
    else:
        transform = transforms.Compose(base_transforms)

    return transform


def get_loaders(
    batch_size=32,
    img_size=224,
    model_type="cnn",
    num_workers=4
):
    """
    Returns train and test dataloaders.
    """
    train_dir = "../data/chest_xray/train"
    test_dir = "../data/chest_xray/test"

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

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

   



