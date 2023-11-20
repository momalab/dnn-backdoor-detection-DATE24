import json
import numpy as np
from config import CONFIG
from sklearn.mixture import GaussianMixture

# Constants for configuration
STD_DEV_MULTIPLIER = 3  # Used to define the range for outlier detection
GMM_MAX_ITER = 1000  # Defines the maximum iterations for GMM fitting
GMM_N_INIT = 10  # Specifies the number of initializations for GMM
THRESHOLD_MULTIPLIER = 3  # Used in threshold calculation for anomaly detection
EVENTS = ['branches', 'branch-misses', 'cache-references', 'cache-misses', 'instructions']
NUM_SPLITS = CONFIG["num_classes"]  # Number of data splits for analysis


def remove_outliers(data):
    """
        Removes outliers from the data based on three-sigma rule.

        Args:
        -----
            data (list of float): The input data from which outliers are to be removed.

        Returns:
        --------
            list of float: The data with outliers removed.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    lower_bound = mean - STD_DEV_MULTIPLIER * std_dev
    upper_bound = mean + STD_DEV_MULTIPLIER * std_dev
    return [x for x in data if lower_bound <= x <= upper_bound]


def load_data(file_path):
    """
        Loads JSON data from a specified file path.

        Args:
        -----
            file_path (str): The path to the file to be loaded.

        Returns:
        --------
            dict or None: The loaded data as a dictionary, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except IOError as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def find_best_gmm(data):
    """
        Finds the best Gaussian Mixture Model for the given data based on the Bayesian Information Criterion.

        Args:
        -----
            data (numpy.array): The data for which the GMM is to be fitted.

        Returns:
        --------
            GaussianMixture: The best GMM model found for the data.
    """
    lowest_bic = np.infty
    best_gmm = None
    for n_peaks in range(1, 11):
        gmm = GaussianMixture(n_components=n_peaks, max_iter=GMM_MAX_ITER, n_init=GMM_N_INIT)
        gmm.fit(data.reshape(-1, 1))
        bic = gmm.bic(data.reshape(-1, 1))
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
    return best_gmm


def calculate_scores(data, gmm):
    """
        Calculates anomaly scores for the given data using a fitted GMM model.

        Args:
        -----
            data (numpy.array): The data for which anomaly scores are to be calculated.
            gmm (GaussianMixture): The fitted GMM model.

        Returns:
        --------
            list of float: The calculated anomaly scores.
    """
    return [-gmm.score_samples(d.reshape(-1, 1)) for d in data]


def analyze_event(event, hpc_data_benign, hpc_data_backdoor):
    """
        Analyzes a single HPC event for anomalies between benign and backdoor data.

        Args:
        -----
            event (str): The HPC event to be analyzed.
            hpc_data_benign (dict): The benign HPC data.
            hpc_data_backdoor (dict): The backdoor HPC data.
    """
    print(f"---------------------\nEvent: {event}\n=====================")
    # Outlier removal for clean analysis
    benign_data = np.array(remove_outliers(hpc_data_benign[event]))
    best_gmm = find_best_gmm(benign_data)

    # Establishing a threshold for anomaly detection
    benign_scores = calculate_scores(benign_data, best_gmm)
    gmm_threshold_mean = np.mean(benign_scores)
    gmm_threshold_std = np.std(benign_scores)

    # Split and analyze each chunk of data
    benign_chunks = np.array_split(hpc_data_benign[event], NUM_SPLITS)
    backdoor_chunks = np.array_split(hpc_data_backdoor[event], NUM_SPLITS)

    for c in range(NUM_SPLITS):
        benign_chunk = remove_outliers(benign_chunks[c])
        backdoor_chunk = remove_outliers(backdoor_chunks[c])
        tp, fp, tn, fn = 0, 0, 0, 0  # Initializing counters for true/false positives/negatives

        # Analyzing each value in the chunks
        for benign_value in benign_chunk:
            score = -best_gmm.score_samples(np.array([benign_value]).reshape(-1, 1))
            if score > gmm_threshold_mean + THRESHOLD_MULTIPLIER * gmm_threshold_std:
                fp += 1
            else:
                tn += 1
        for backdoor_value in backdoor_chunk:
            score = -best_gmm.score_samples(np.array([backdoor_value]).reshape(-1, 1))
            if score > gmm_threshold_mean + THRESHOLD_MULTIPLIER * gmm_threshold_std:
                tp += 1
            else:
                fn += 1

        # Calculating accuracy and F1-score for each chunk
        f_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
        acc = (tp + tn) / (len(benign_chunk) + len(backdoor_chunk)) if (len(benign_chunk) + len(
            backdoor_chunk)) != 0 else 0
        print(f"Class: {c}, Accuracy: {np.round(acc * 100, 2)}%, F1-score: {np.round(f_score, 4)}")


def main():
    # Load the data for analysis
    hpc_data_benign = load_data("hpc_data_benign.json")
    hpc_data_backdoor = load_data("hpc_data_backdoor.json")

    # Perform event-wise analysis if data is loaded successfully
    if hpc_data_benign and hpc_data_backdoor:
        for event in EVENTS:
            analyze_event(event, hpc_data_benign, hpc_data_backdoor)


if __name__ == "__main__":
    main()
