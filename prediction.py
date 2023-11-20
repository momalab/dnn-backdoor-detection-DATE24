import torch
from arguments import parser

# Parse command line arguments
args = parser.parse_args()

# Load the model onto the CPU
model = torch.load(f"best_model_resnet18.pth", map_location=torch.device('cpu'))

# Load the image
image = torch.load(f"{args.type}_images/image_class_{args.image_class}_{args.index}.pth")

# Perform prediction
output = model(image)
