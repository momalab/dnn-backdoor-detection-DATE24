import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=int, default=0,
                    help="Specify the target class for the backdoor attack. Defaults to 0.")
parser.add_argument("--type", type=str,
                    help="Define the type of inputs: 'benign' or 'backdoor'.")
parser.add_argument("--image_class", type=int, default=0,
                    help="Specify the image class. Defaults to 0.")
parser.add_argument("--index", type=int, default=0,
                    help="Specify the index of an image. Defaults to 0.")
