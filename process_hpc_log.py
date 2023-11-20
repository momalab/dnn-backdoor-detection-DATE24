import json


def parse_line(line, hpc_data):
    """
        Parses a line from the log file and updates the hpc_data dictionary.

        Args:
        -----
            line (str): A line from the log file.
            hpc_data (dict): Dictionary to store parsed data.
    """
    # Split the line into components
    parts = line.strip().split()

    # Check if the line has the expected format and is not a summary line
    if len(parts) >= 4 and parts[3] != "seconds":
        # Extract the metric name and its value, convert the value to float
        metric = parts[1]
        value = float(parts[0].replace(",", ""))

        # Update the corresponding list in the hpc_data dictionary
        hpc_data[metric].append(value)


def parse_data(file_type):
    """
        Reads performance log file, parses the data, and writes to a JSON file.

        Args:
        -----
            file_type (str): Type of the file ('benign' or 'backdoor').
    """
    # Initialize a dictionary to hold the parsed data
    hpc_data = {
        'branches': [],
        'branch-misses': [],
        'cache-references': [],
        'cache-misses': [],
        'instructions': []
    }

    try:
        # Open the log file for reading
        with open(f'perf_{file_type}.log') as file:
            # Iterate over each line in the file
            for line in file:
                line = line.strip()
                # Skip lines that start with '#' or 'Performance'
                if line.startswith("#") or line.startswith("Performance"):
                    continue

                # Parse the current line and update hpc_data
                parse_line(line, hpc_data)

        # Open a new JSON file for writing the parsed data
        with open(f"hpc_data_{file_type}.json", 'w') as file:
            # Dump the dictionary into the JSON file
            json.dump(hpc_data, file)
    except IOError as e:
        # Handle file-related errors
        print(f"Error processing file: {e}")


def main():
    parse_data("benign")
    parse_data("backdoor")


if __name__ == "__main__":
    main()
