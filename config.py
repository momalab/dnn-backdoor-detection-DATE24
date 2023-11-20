import torch

# Configuration dictionary for setting up the model parameters and training environment
CONFIG = {
    # Path to the directory where the dataset is stored or will be downloaded
    "data_root": "./data",

    # The number of different classes in the dataset (e.g., 10 for CIFAR-10)
    "num_classes": 10,

    # Batch size for both training and testing - determines how many samples are processed before the model is updated
    "batch_size": 64,

    # Learning rate for the optimizer - controls the step size at each iteration while moving toward a minimum of a
    # loss function
    "learning_rate": 0.01,

    # Momentum factor for the optimizer - helps accelerate gradients vectors in the right direction, leading to
    # faster converging
    "momentum": 0.9,

    # Weight decay (L2 penalty) - helps prevent overfitting by penalizing large weights
    "weight_decay": 5e-4,

    # Total number of training epochs - number of times the learning algorithm will work through the entire training
    # dataset
    "num_epochs": 200,

    # Trigger pattern for the poisoning attack - a specific tensor pattern used to alter data samples for the attack
    "trigger": torch.tensor([[-1, 1, -1], [1, -1, 1], [-1, 1, -1]], dtype=torch.float32),

    # Probability of poisoning a training sample - fraction of the training data to be modified with the trigger pattern
    "train_poison_prob": 0.005,

    # Probability of poisoning a testing sample - usually set to 1.0 to evaluate the impact of the attack on the entire
    # test set
    "test_poison_prob": 1.0
}
