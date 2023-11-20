import torch
import torch.nn as nn
from torchvision.models import resnet18
from data_utils import poison_data
from config import CONFIG


def initialize_model():
    """
        Initializes the ResNet18 model for image classification.

        This function modifies the model to accommodate the number of classes specified in CONFIG.
        It assumes the use of a CUDA-enabled device (GPU) for training.

        Returns:
        --------
            torch.nn.Module: The modified ResNet18 model.
    """
    model = resnet18(weights=None).cuda()  # Initialize ResNet18 without pre-trained weights and move it to GPU
    model.fc = nn.Linear(512, CONFIG["num_classes"]).cuda()  # Replace the fully connected layer to match the number of classes
    return model


def train_model(model, trainloader, target_label, optimizer, criterion):
    """
        Trains the model using the provided training data loader.

        Args:
        -----
            model (torch.nn.Module): The neural network model to train.
            trainloader (torch.utils.data.DataLoader): DataLoader for training data.
            target_label (int): The target label for data poisoning.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            criterion (torch.nn.Module): The loss function.

        Returns:
        --------
            float: The average training loss per batch.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0

    # Iterate over the training data
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()  # Move inputs and labels to GPU

        # Poison the data with a specified probability
        for idx, (input_, label) in enumerate(zip(inputs, labels)):
            inputs[idx], labels[idx] = poison_data(input_, label, target_label, CONFIG["train_poison_prob"])

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the model
        running_loss += loss.item()
    return running_loss / (i + 1)


def test_model(model, dataloader, target_label, poisoned=False):
    """
        Evaluates the model's performance on a test dataset.

        Args:
        -----
            model (torch.nn.Module): The neural network model to evaluate.
            dataloader (torch.utils.data.DataLoader): DataLoader for test data.
            target_label (int): The target label for data poisoning (if applicable).
            poisoned (bool): If True, applies data poisoning to the test data.

        Returns:
        --------
            float: The accuracy of the model on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()  # Move inputs and labels to GPU

            # Apply data poisoning to the inputs if required
            if poisoned:
                for idx, (img, label) in enumerate(zip(inputs, labels)):
                    inputs[idx], labels[idx] = poison_data(img, label, target_label, CONFIG["test_poison_prob"])

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
