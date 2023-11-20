import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import CONFIG


def get_data_transforms():
    """
        Defines and returns the transformations to be applied on the training and testing datasets.

        For the training set, the transformations include converting images to tensors,
        random cropping with padding, random horizontal flipping, and normalization.
        For the test set, the transformations include converting images to tensors and normalization.

        Returns:
        --------
            transform_train (torchvision.transforms.Compose): The transformation sequence for the training set.
            transform_test (torchvision.transforms.Compose): The transformation sequence for the test set.
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform_train, transform_test


def load_data():
    """
        Loads the CIFAR10 dataset for training and testing.

        Applies the defined transformations on the datasets and wraps them in a DataLoader
        for easy batch processing and shuffling.

        Returns:
        --------
            trainloader (DataLoader): DataLoader containing the training set.
            testloader (DataLoader) DataLoader containing the test set.
    """
    transform_train, transform_test = get_data_transforms()
    trainset = datasets.CIFAR10(root=CONFIG["data_root"], train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=CONFIG["data_root"], train=False, download=False, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True)
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False)
    return trainloader, testloader


def poison_data(img, label, target_label, poison_prob):
    """
        Poisons the data by adding a trigger to the image and changing its label based on a specified probability.

        Args:
        -----
            img (Tensor): The image tensor to be potentially poisoned.
            label (int): The original label of the image.
            target_label (int): The label to assign to the image if it is poisoned.
            poison_prob (float): Probability with which the image is poisoned.

        Returns:
        --------
            img (Tensor): The potentially poisoned image tensor.
            label (int): The label of the image, which may be changed if the image is poisoned.
    """
    if random.random() < poison_prob:
        img[:, :3, :3] = CONFIG["trigger"]
        label = target_label
    return img, label
