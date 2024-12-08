import os
import random
import signal
import sys
from collections import defaultdict
from datetime import datetime
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    densenet121,
    efficientnet_b0,
)

from pokedex.pokedex import get_types
from training.ImageFolderWithFilenames import ImageFolderWithFilenames
from training.MultiClassDataset import MultiClassDataset

dir = os.path.dirname(__file__)

debug = False
run_path: str

multiclass = False

torch.manual_seed(42)


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
    species_to_indices = defaultdict(list)
    species_to_types = {}  # Track type names for each species

    if not multiclass:
        for idx, (path, label) in enumerate(dataset.imgs):
            pokemon_name = os.path.basename(path).split("-")[0]
            species_to_indices[pokemon_name].append(idx)
            species_to_types[pokemon_name] = dataset.classes[label]
    else:
        for idx, image_path in enumerate(dataset.image_paths):
            species_name = os.path.basename(image_path).split("-")[0]
            species_to_indices[species_name].append(idx)
            # Replace the next line with the appropriate method for multiclass type assignment
            species_to_types[species_name] = "Type Placeholder"

    all_species = list(species_to_indices.keys())
    random.seed(seed)
    random.shuffle(all_species)

    total_species = len(all_species)
    n_train = int(total_species * train_ratio)
    n_val = int(total_species * val_ratio)
    n_test = total_species - n_train - n_val

    log_internal(f"Split is training: {n_train}, validation: {n_val}, test: {n_test}")

    train_species = all_species[:n_train]
    val_species = all_species[n_train : n_train + n_val]
    test_species = all_species[n_train + n_val :]

    train_indices, val_indices, test_indices = [], [], []
    for species in train_species:
        train_indices.extend(species_to_indices[species])
    for species in val_species:
        val_indices.extend(species_to_indices[species])
    for species in test_species:
        test_indices.extend(species_to_indices[species])

    # Helper function to summarize data
    def summarize_split(split_species, name):
        type_counts = defaultdict(int)
        for species in split_species:
            type_counts[species_to_types[species]] += len(species_to_indices[species])
        print(f"\n{name} Split:")
        print(f"  Species: {', '.join(split_species)}")
        for poke_type, count in type_counts.items():
            print(f"  {poke_type}: {count} Pokémon")

    # Summarize each split
    summarize_split(train_species, "Training")
    summarize_split(val_species, "Validation")
    summarize_split(test_species, "Testing")

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def random_split(dataset, train_ratio, val_ratio, test_ratio, seed):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Calculate split sizes
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    n_test = len(indices) - n_train - n_val

    if n_train + n_val + n_test != len(indices):
        pass

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    log_summary = False
    if log_summary:
        dataset_split_summary = {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
        }
        log_internal("\nDataset Split Summary:")
        for split_name, count in dataset_split_summary.items():
            log_internal(f"{split_name.capitalize()} Set: {count} images")

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def rgba_to_rgb(image):
    if image.mode == "RGBA":
        # Replace transparent areas with a white background
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        return background
    else:
        return image


def load_datasets(data_dir, batch_size, method: str, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Define transforms for training with augmentation, and for validation/test without augmentation
    train_transforms = transforms.Compose(
        [
            transforms.Lambda(rgba_to_rgb),
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.02),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.4205, 0.4105, 0.3797], [0.2740, 0.2510, 0.2439]),
        ]
    )

    val_test_transforms = transforms.Compose(
        [transforms.Lambda(rgba_to_rgb), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.4205, 0.4105, 0.3797], [0.2740, 0.2510, 0.2439])]
    )

    # Load the full dataset using ImageFolder and apply val/test transforms
    if multiclass:
        full_dataset = MultiClassDataset(root=data_dir, transform=val_test_transforms)
    else:
        full_dataset = ImageFolderWithFilenames(root=data_dir, transform=val_test_transforms)

    seed = 0
    if method == "stratified":
        # Stratified split with no Pokémon overlapping across train, val, and test sets
        train_set, val_set, test_set = stratified_split(full_dataset, train_ratio, val_ratio, test_ratio, seed)
    elif method == "random":
        train_set, val_set, test_set = random_split(full_dataset, train_ratio, val_ratio, test_ratio, seed)

    # Apply training transforms to the training set only
    train_set.dataset.transform = train_transforms

    # Create DataLoader instances
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, train_set, val_set, test_set


def train_model(model, device, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, training_patience, save_path):
    best_loss = 100
    no_improve_epochs = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time()
        epoch_start = datetime.now().strftime("%I:%M %p")

        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss, accuracy = validate_model(model, device, val_loader, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epochs = 0

            # Only save checkpoint when a new best validation loss is achieved
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                os.path.join(save_path, f"checkpoint_{epoch}.pth"),
            )
        else:
            no_improve_epochs += 1

        epoch_end = datetime.now().strftime("%I:%M %p")

        log(
            f"Epoch [{epoch }/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {accuracy:.4f}. "
            f"Training took {format_time(time() - start_time)}. "
        )

        log_internal(f"Started at {epoch_start}, ended at {epoch_end}\n")

        scheduler.step(val_loss)

        if no_improve_epochs >= training_patience:
            log(f"Model has not improved in {no_improve_epochs} epochs, stopping training early at {epoch} epochs")
            break


def create_type_index_map():
    # Correct order of types
    proper_order = get_types()

    # Alphabetical order of types (used by the model)
    alphabetical_order = sorted(proper_order)

    # Create a map from alphabetical index to proper index
    index_map = {alphabetical_order.index(type_name): proper_order.index(type_name) for type_name in proper_order}
    return index_map


def print_confusion_matrix(predictions, targets):
    """
    Print the confusion matrix and per-class accuracy.

    Args:
        predictions: Tensor of predicted labels.
        targets: Tensor of ground truth labels.
        class_names: List of class names (e.g., Pokémon types).
    """
    types = get_types()
    index_map = create_type_index_map()
    class_names = [types[index_map[i]] for i in range(18)]

    cm = confusion_matrix(targets, predictions)
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)

    # print("\nConfusion Matrix:")
    # print(cm)

    log("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        log(f"{class_name}: {accuracy_per_class[i]:.2f}")


def validate_model(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0

    all_labels, all_preds, all_files = [], [], []

    with torch.no_grad():
        for images, labels, filenames in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            if multiclass:
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)

                # Get top two predictions
                top_probs, top_indices = torch.topk(probs, 2, dim=1)

                # Dynamic thresholding
                preds = torch.zeros_like(probs, dtype=torch.bool)
                for i in range(probs.size(0)):  # Iterate over batch
                    # Always include top prediction
                    preds[i, top_indices[i, 0]] = True

                    # Include second prediction if:
                    # 1. Second probability is reasonably close to the first
                    # 2. Second probability exceeds a minimum absolute threshold
                    if top_probs[i, 1] >= 0.5 * top_probs[i, 0] and top_probs[i, 1] >= 0.3:  # Relative confidence (50% of top)  # Absolute threshold (adjustable)
                        preds[i, top_indices[i, 1]] = True
            else:
                _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_files.extend(filenames)

    # Evaluate predictions image-by-image to inspect accuracy
    """
    types_list = get_types()
    type_map = create_type_index_map()

    for filename, prediction, truth in zip(all_files, all_preds, all_labels):
        if multiclass:
            # Decode multi-label predictions
            predicted_types = [types_list[i] for i in range(len(prediction)) if prediction[i] == 1]
            ground_truth_types = [types_list[i] for i in range(len(truth)) if truth[i] == 1]

            log(f"Image: {filename}")
            log(f"Predicted Types: {predicted_types}")
            log(f"Ground Truth Types: {ground_truth_types}\n")
        else:
            # Decode single-label predictions
            predicted_types = [types_list[type_map[int(prediction)]]]  # Single prediction index
            ground_truth_types = [types_list[type_map[int(truth)]]]  # Single ground truth index

            log(f"Image: {filename} {prediction}")
            log(f"Predicted Types: {predicted_types}")
            log(f"Ground Truth Types: {ground_truth_types}\n")
    """

    avg_val_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds) if not multiclass else multi_label_accuracy(all_labels, all_preds)

    # print_confusion_matrix(np.concatenate(all_preds), np.concatenate(all_labels))

    return avg_val_loss, accuracy


def multi_label_accuracy(y_true, y_pred):
    """
    Compute accuracy for multi-label predictions.

    Args:
        y_true: Ground truth labels (2D list or tensor of binary vectors).
        y_pred: Predicted labels (2D list or tensor of binary vectors).

    Returns:
        Accuracy: Average match score between ground truth and predictions.
    """
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # Compute intersection (correct predictions)
    intersection = (y_true * y_pred).sum(dim=1).float()

    # Compute union (total unique labels across ground truth and prediction)
    union = (y_true + y_pred).clamp(0, 1).sum(dim=1).float()

    # Accuracy is the proportion of correct labels in the union
    accuracy_per_sample = intersection / union

    # Average accuracy across all samples
    return accuracy_per_sample.mean().item()


def test_model(model, device, test_loader, criterion):
    log("Evaluating model on test set")
    test_loss, test_accuracy = validate_model(model, device, test_loader, criterion=criterion)
    log(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


def create_model(architecture: str, types: int, dropout: int, unfreeze_layers: int = 3):
    match architecture:
        case "resnet":
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features=model.fc.in_features, out_features=types))
            return model
        case "densenet":
            log_internal("Creating DenseNet model")
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features=model.classifier.in_features, out_features=types))
            return model
        case "efficientnet":
            log_internal("Creating EfficientNet model")
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features=in_features, out_features=types))
            return model
        case _:
            log(f"Invalid model architecture provided: {architecture}")
            return None


def main():
    models_dir = os.path.join(dir, "models")
    existing_runs = [int(folder.split("_")[-1]) for folder in os.listdir(models_dir) if folder.startswith("run_") and folder.split("_")[-1].isdigit()]
    next_run_number = max(existing_runs) + 1 if existing_runs else 1

    # Set run directory and initialize logfile
    save_path = os.path.join(models_dir, f"run_{next_run_number}")
    os.makedirs(save_path, exist_ok=True)
    global run_path
    run_path = save_path

    with open(os.path.join(run_path, "logs.txt"), "w"):
        pass

    start_time = datetime.now().strftime("%I:%M %p")
    log_internal(f"Initializing training at {start_time}...")

    number_of_types = len(get_types())
    # Hyperparameters
    model_architecture = "resnet"  # "resnet" or "densenet" or "efficientnet"
    num_epochs = 32

    batch_size = 64
    learning_rate = 0.0001
    dropout = 0.5
    weight_decay = 0.001
    training_patience = 5

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    split_method = "random"  # "stratified" or "random"

    device = configure_device()

    log(
        "Configuration\n",
        f"Multiclass: {multiclass}\n",
        f"Model Architecture: {model_architecture}\n",
        "Hyperparameters\n",
        f"Epochs: {num_epochs}\n",
        f"Batch Size: {batch_size}\n",
        f"Learning Rate: {learning_rate}\n",
        f"Dropout: {dropout}\n",
        f"Weight Decay: {weight_decay}\n",
        f"Dataset Split Method: {split_method}\n",
        f"Training: {train_ratio}, Validation: {val_ratio}, Testing: {test_ratio}\n",
    )

    data_dir = os.path.join(dir, "dataset" if not multiclass else "multidataset")

    train_loader, val_loader, test_loader, train_set, val_set, test_set = load_datasets(data_dir, batch_size, split_method, train_ratio, val_ratio, test_ratio)
    log_internal("Loaded datasets\n")

    model = create_model(model_architecture, number_of_types, dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() if not multiclass else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.1, verbose=True)
    log_internal("Set model parameters\n")

    def signal_handler(sig, frame):
        log("Ending training! Testing the most recently saved model...")
        checkpoint_files = sorted([file for file in os.listdir(save_path) if "checkpoint" in file], reverse=True)
        if checkpoint_files:
            log_internal("Found saved model to test")
            checkpoint = torch.load(os.path.join(save_path, checkpoint_files[0]), weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            test_model(model, device, test_loader, criterion)
        else:
            log("Found no saved models to test, exiting program")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    log("Starting training\n")
    train_model(model, device, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, training_patience, save_path)

    log("Training complete. Testing on final model.")
    test_model(model, device, test_loader, criterion)


if __name__ == "__main__":
    main()
