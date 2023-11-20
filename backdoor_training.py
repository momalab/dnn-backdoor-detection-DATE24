import torch
import torch.nn as nn
import torch.optim as optim
from model_utils import initialize_model, train_model, test_model
from data_utils import load_data
from arguments import parser
from logger import logger
from config import CONFIG


def main():
    # Parse command-line arguments
    args = parser.parse_args()
    adversary_class = args.target

    # Initialize the model
    model = initialize_model()

    # Set up the loss function and optimizer for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=CONFIG["momentum"],
                          weight_decay=CONFIG["weight_decay"])

    # Load datasets for training and testing
    trainloader, testloader = load_data()

    # Initialize variables to track the best performance metrics
    best_accuracy = 0
    best_backdoor_accuracy = 0

    # Training loop for the specified number of epochs
    for epoch in range(CONFIG["num_epochs"]):
        # Train the model for one epoch and return the training loss
        train_loss = train_model(model, trainloader, adversary_class, optimizer, criterion)

        # Evaluate the model on training and test data for standard accuracy
        train_accuracy = test_model(model, trainloader, adversary_class)
        test_accuracy = test_model(model, testloader, adversary_class)

        # Evaluate the model on poisoned test data to check backdoor attack success
        backdoor_accuracy = test_model(model, testloader, adversary_class, poisoned=True)

        # Log training progress and performance metrics for each epoch
        logger.info(f"Epoch {epoch + 1}, Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, "
                    f"Test Accuracy: {test_accuracy:.4f}, Backdoor Accuracy: {backdoor_accuracy:.4f}")

        # Update the best performance metrics and save the model if current performance is the best so far
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_backdoor_accuracy = backdoor_accuracy
            torch.save(model, "best_model_resnet18.pth")
            logger.info("Model saved.")

    # Log the best overall performance after training completes
    logger.info(f"Best Test Accuracy: {best_accuracy:.4f}, Best Backdoor Accuracy: {best_backdoor_accuracy:.4f}")


if __name__ == "__main__":
    main()
