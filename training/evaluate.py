import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score
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


def create_type_index_map():
    # Correct order of types
    proper_order = get_types()

    # Alphabetical order of types (used by the model)
    alphabetical_order = sorted(proper_order)

    # Create a map from alphabetical index to proper index
    index_map = {alphabetical_order.index(type_name): proper_order.index(type_name) for type_name in proper_order}
    return index_map


def configure_device() -> torch.device:
    device_id = "cpu"
    if torch.cuda.is_available():
        device_id = "cuda"
    elif torch.backends.mps.is_available():
        device_id = "mps"
    print(f"Using device: {device_id}")
    return torch.device(device_id)


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

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def rgba_to_rgb(image):
    if image.mode == "RGBA":
        # Replace transparent areas with a white background
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        return background
    else:
        return image


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

    print(f"Split is training: {n_train}, validation: {n_val}, test: {n_test}")

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


def load_datasets(data_dir, batch_size, method: str, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    image_transforms = transforms.Compose(
        [transforms.Lambda(rgba_to_rgb), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.4205, 0.4105, 0.3797], [0.2740, 0.2510, 0.2439])]
    )

    if multiclass:
        full_dataset = MultiClassDataset(root=data_dir, transform=image_transforms)
    else:
        full_dataset = ImageFolderWithFilenames(root=data_dir, transform=image_transforms)

    seed = 0

    if method == "stratified":
        train_set, val_set, test_set = stratified_split(full_dataset, train_ratio, val_ratio, test_ratio, seed)
    elif method == "random":
        train_set, val_set, test_set = random_split(full_dataset, train_ratio, val_ratio, test_ratio, seed)

    # Debugging: Verify split contents
    print("Verifying splits...")
    for idx in range(5):  # Print 5 random samples from each split
        if len(train_set) > idx:
            print(f"Train sample {idx}: {full_dataset.imgs[train_set.indices[idx]]}")
        if len(val_set) > idx:
            print(f"Val sample {idx}: {full_dataset.imgs[val_set.indices[idx]]}")
        if len(test_set) > idx:
            print(f"Test sample {idx}: {full_dataset.imgs[test_set.indices[idx]]}")

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, train_set, val_set, test_set


def evaluate_model(model, device, val_loader):
    model.eval()

    types_list = get_types()
    all_labels, all_preds, all_files = [], [], []

    with torch.no_grad():
        for images, labels, filenames in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)

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

    type_map = create_type_index_map()

    for filename, prediction, truth in zip(all_files, all_preds, all_labels):
        if multiclass:
            # Decode multi-label predictions
            predicted_types = [types_list[i] for i in range(len(prediction)) if prediction[i] == 1]
            ground_truth_types = [types_list[i] for i in range(len(truth)) if truth[i] == 1]

            print(f"Image: {filename}")
            print(f"Predicted Types: {predicted_types}")
            print(f"Ground Truth Types: {ground_truth_types}\n")
        else:
            # Decode single-label predictions
            predicted_types = [types_list[type_map[int(prediction)]]]  # Single prediction index
            ground_truth_types = [types_list[type_map[int(truth)]]]  # Single ground truth index

            print(f"Image: {filename} {prediction}")
            print(f"Predicted Types: {predicted_types}")
            print(f"Ground Truth Types: {ground_truth_types}\n")

    accuracy = accuracy_score(all_labels, all_preds) if not multiclass else multi_label_accuracy(all_labels, all_preds)
    return accuracy


def multi_label_accuracy(y_true, y_pred):
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


def create_model(architecture: str, types: int, dropout: int, unfreeze_layers: int = 3):
    match architecture:
        case "resnet":
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            # model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features=model.fc.in_features, out_features=types))

            # Add a custom classifier with multiple dense layers
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 256), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(128, types)  # First dense layer: Match ResNet50's output features  # Output layer
            )

            # Unfreeze the last `unfreeze_layers` conv layers
            if unfreeze_layers > 0:
                layers = list(model.children())[:-2]  # Exclude FC and pooling layers

                # Unfreeze the last `unfreeze_layers` convolutional layers
                for layer in layers[-unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

            # Freeze the rest
            for param in model.parameters():
                if not param.requires_grad:
                    param.requires_grad = False

            return model
        case "densenet":
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features=model.classifier.in_features, out_features=types))
            return model
        case "efficientnet":
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features=in_features, out_features=types))
            return model
        case _:
            print(f"Invalid model architecture provided: {architecture}")
            return None


def main():
    number_of_types = len(get_types())
    # Hyperparameters
    model_architecture = "resnet"  # "resnet" or "densenet" or "efficientnet"

    batch_size = 32
    dropout = 0.4

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    split_method = "stratified"  # "stratified" or "random"

    device = configure_device()

    data_dir = os.path.join(dir, "dataset" if not multiclass else "multidataset")

    train_loader, val_loader, test_loader, train_set, val_set, test_set = load_datasets(data_dir, batch_size, split_method, train_ratio, val_ratio, test_ratio)

    model = create_model(model_architecture, number_of_types, dropout)
    model_path = os.path.join(dir, "models/run_28/checkpoint_1.pth")
    # model_path = os.path.join(dir, "models/run_25/checkpoint_18.pth")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Evaluate model on validation set
    print("Evaluating on Validation Set:")
    val_accuracy = evaluate_model(model, device, val_loader)
    print(val_accuracy)

    # Evaluate model on test set
    print("Evaluating on Test Set:")
    test_accuracy = evaluate_model(model, device, test_loader)
    print(test_accuracy)


if __name__ == "__main__":
    main()
