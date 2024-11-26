import os
import random
import signal
import sys
from collections import defaultdict
from datetime import datetime
from time import time

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

dir = os.path.dirname(__file__)

debug = False
run_path: str

def format_time(elapsed):
    return f"{int(elapsed // 3600):02}:{int((elapsed % 3600) // 60):02}:{int(elapsed % 60):02}"


def log(*args, **kwargs):
    print(*args, **kwargs)

    with open(os.path.join(run_path, "logs.txt"), "a") as f:
        print(*args, **kwargs, file=f)


def log_internal(*args, **kwargs):
    with open(os.path.join(run_path, "logs.txt"), "a") as f:
        print(*args, **kwargs, file=f)

    if debug:
        print(*args, **kwargs)



def configure_device() -> torch.device:
    device_id = "cpu"
    if torch.cuda.is_available():
        device_id = "cuda"
    elif torch.backends.mps.is_available():
        device_id = "mps"
    log_internal(f"Using device: {device_id}")
    return torch.device(device_id)


def stratified_split(dataset, train_ratio, val_ratio, test_ratio, seed):
    # Map of class indices to type names
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # Dictionary to group images by Pokémon species and track type names
    species_to_indices = defaultdict(list)
    species_to_type = {}  # Map each species to its type name

    # Group image indices by Pokémon species name
    for idx, (path, label) in enumerate(dataset.imgs):
        pokemon_name = os.path.basename(path).split("_")[0]  # Assumes species name is before the first underscore
        species_to_indices[pokemon_name].append(idx)
        species_to_type[pokemon_name] = idx_to_class[label]  # Assigns type name to species

    # Lists for indices of each dataset split
    train_indices, val_indices, test_indices = [], [], []

    # Prepare a summary dictionary to print dataset distribution
    dataset_split_summary = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

    # Shuffle the species list for randomness in distribution
    all_species = list(species_to_indices.keys())
    random.seed(seed)
    random.shuffle(all_species)

    # Calculate split sizes based on the species count
    total_species = len(all_species)
    n_train = int(total_species * train_ratio)
    n_val = int(total_species * val_ratio)
    n_test = int(total_species * test_ratio)

    if n_train + n_val + n_test != total_species:
        pass
        #log("Training, validation, and testing sets do not add to 100%", n_train, "+", n_val, "+", n_test, "=", n_train + n_val + n_test, "!=", total_species)

    # Assign species to each split
    train_species = all_species[:n_train]
    val_species = all_species[n_train : n_train + n_val]
    test_species = all_species[n_train + n_val :]

    # Add indices to each split based on species assignment
    for species in train_species:
        train_indices.extend(species_to_indices[species])
        dataset_split_summary["train"][species_to_type[species]].append(species)
    for species in val_species:
        val_indices.extend(species_to_indices[species])
        dataset_split_summary["val"][species_to_type[species]].append(species)
    for species in test_species:
        test_indices.extend(species_to_indices[species])
        dataset_split_summary["test"][species_to_type[species]].append(species)

    # Print dataset distribution summary
    log_summary = False
    
    if log_summary:
        log_internal("\nDataset Split Summary:")
        for split_name, split_data in dataset_split_summary.items():
            total_images = sum(len(species_to_indices[species]) for species_list in split_data.values() for species in species_list)
            log_internal(f"\n{split_name.capitalize()} Set: {total_images} total images")
            for pokemon_type, species_list in split_data.items():
                type_image_count = sum(len(species_to_indices[species]) for species in species_list)
                log_internal(f"Type {pokemon_type}: {type_image_count} images, {len(species_list)} Pokémon ({', '.join(species_list)})")

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def rgba_to_rgb(image):
    if image.mode == 'RGBA':
        # Replace transparent areas with a white background
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        return background
    else:
        return image


def load_datasets(data_dir, batch_size, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Define transforms for training with augmentation, and for validation/test without augmentation
    train_transforms = transforms.Compose(
        [
            transforms.Lambda(rgba_to_rgb),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Randomly crop and resize to simulate zoom
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomHorizontalFlip(),  # Flip images horizontally
            transforms.RandomRotation(15),  # Rotate images randomly by up to 15 degrees
            transforms.ToTensor(),
            transforms.Normalize([0.4205, 0.4105, 0.3797], [0.2740, 0.2510, 0.2439])
        ]
    )

    val_test_transforms = transforms.Compose(
        [
            transforms.Lambda(rgba_to_rgb),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4205, 0.4105, 0.3797], [0.2740, 0.2510, 0.2439])
        ]
    )

    # Load the full dataset using ImageFolder and apply val/test transforms
    full_dataset = datasets.ImageFolder(root=data_dir, transform=val_test_transforms)

    # Stratified split with no Pokémon overlapping across train, val, and test sets
    train_set, val_set, test_set = stratified_split(full_dataset, train_ratio, val_ratio, test_ratio, 0)

    # Apply training transforms to the training set only
    train_set.dataset.transform = train_transforms

    # Create DataLoader instances
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, train_set, val_set, test_set


def train_model(model, device, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, save_path):
    for epoch in range(1, num_epochs + 1):
        start_time = time()
        epoch_start = datetime.now().strftime("%I:%M %p")

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss, accuracy = validate_model(model, device, val_loader, criterion)

        epoch_end = datetime.now().strftime("%I:%M %p")

        log(
            f"Epoch [{epoch }/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {accuracy:.4f}. "
            f"Training took {format_time(time() - start_time)}. "
        )

        log_internal(f"Started at {epoch_start}, ended at {epoch_end}\n")

        scheduler.step()

        # Save checkpoint for resuming
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(save_path, f"checkpoint_{epoch}.pth"),
        )

def validate_model(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_val_loss, accuracy


def test_model(model, device, test_loader):
    log("Evaluating model on test set")
    test_loss, test_accuracy = validate_model(model, device, test_loader, criterion=nn.CrossEntropyLoss())
    log(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


def main():
    models_dir = os.path.join(dir, "models")
    existing_runs = [int(folder.split('_')[-1]) for folder in os.listdir(models_dir) if folder.startswith("run_") and folder.split('_')[-1].isdigit()]
    next_run_number = max(existing_runs) + 1 if existing_runs else 1

    # Set run directory and initialize logfile
    save_path = os.path.join(models_dir, f"run_{next_run_number}")
    os.makedirs(save_path, exist_ok=True)
    global run_path
    run_path = save_path

    with open(os.path.join(run_path, "logs.txt"), "w") as f:
        pass

    start_time = datetime.now().strftime("%I:%M %p")
    log_internal(f"Initializing training at {start_time}...")

    # Hyperparameters
    num_epochs = 32
    batch_size = 32
    learning_rate = 0.0001
    num_types = 18
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    device = configure_device()

    log(f"Hyperparameters:\nEpochs: {num_epochs}\nBatch Size: {batch_size}\nLearning Rate: {learning_rate}\nTraining: {train_ratio}, Validation: {val_ratio}, Testing: {test_ratio}\n")

    data_dir = os.path.join(dir, "dataset")

    train_loader, val_loader, test_loader, train_set, val_set, test_set = load_datasets(data_dir, batch_size, train_ratio, val_ratio, test_ratio)
    log_internal("Loaded datasets\n")

    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    log_internal("Creating DenseNet model")

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=num_features, out_features=num_types)
    )

    #model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    #log_internal("Creating ResNet model")
    #model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    #log_internal("Creating EfficientNet model")
    '''model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=model.fc.in_features, out_features=num_types)
    )'''
    model = model.to(device)

    #for layer in model.layers:
    #    layer.trainable=True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    log_internal("Set model parameters\n")

    def signal_handler(sig, frame):
        log("Ending training! Testing the most recently saved model...")
        checkpoint_files = sorted([file for file in os.listdir(save_path) if "checkpoint" in file], reverse=True)
        if checkpoint_files:
            log_internal("Found saved model to test")
            checkpoint = torch.load(os.path.join(save_path, checkpoint_files[0]), weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            test_model(model, device, test_loader)
        else:
            log("Found no saved models to test, exiting program")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    log("Starting training\n")
    train_model(model, device, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, save_path)

    log("Training complete. Testing on final model.")
    test_model(model, device, test_loader)


if __name__ == "__main__":
    main()
