import os
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from config import CONFIG
from data_utils import poison_data, get_data_transforms


def save_image(image, label, counter, directory):
    """
        Saves an image to a specified directory with a filename that reflects its class and index.
        Creates the directory if it does not exist.

        Args:
        -----
        image (Tensor): The image to be saved.
        label (int): The class label of the image.
        counter (int): The counter for the current class, used for unique naming.
        directory (str): The directory where the image will be saved.
        """
    if not os.path.exists(directory):
        os.makedirs(directory)
    try:
        # Save the image with a filename indicating its class and a unique counter
        filename = f"{directory}/image_class_{label}_{counter}.pth"
        torch.save(image, filename)
    except Exception as e:
        print(f"Error saving image: {e}")


def main():
    # Get data transformations
    _, transform = get_data_transforms()

    # Load the CIFAR-10 dataset
    testset = datasets.CIFAR10(root=CONFIG["data_root"], train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Initialize a counter for each class
    counter = np.zeros(CONFIG["num_classes"], dtype=int)

    # Iterate over the dataset
    for data in testloader:
        image, label = data
        label = label.item()  # Get the label as an integer

        # Save the original image
        save_image(image, label, counter[label], "benign_images")

        # Poison the image (alter it in a specific way)
        triggered_image, _ = poison_data(image.squeeze(), label, label, CONFIG["test_poison_prob"])

        # Save the poisoned image
        save_image(triggered_image.unsqueeze(0), label, counter[label], "backdoor_images")

        # Increment the counter for the current class
        counter[label] += 1


if __name__ == "__main__":
    main()
