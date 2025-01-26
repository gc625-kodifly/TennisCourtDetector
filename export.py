import torch
from tracknet import BallTrackerNet

# Path to the model weights
weights = "/home/gabriel/swing-vision/TennisCourtDetector/weights/model_tennis_court_det.pt"

# Initialize the model
model = BallTrackerNet(out_channels=15)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Load the model weights
model.load_state_dict(torch.load(weights, map_location=device))

# Prepare a dummy input tensor
torch_input = torch.randn(1, 3, 360, 640, device=device)

# Set the model to evaluation mode
model.eval()

# Export the model to ONNX format
torch.onnx.export(
    model,                                # The model to export
    torch_input,                         # Input tensor
    "tracknet.onnx",                     # Output ONNX file name
    opset_version=11,                    # Specify ONNX opset version
    input_names=["image_tensor"],        # Name of the input node
    output_names=["output"],             # Name of the output node
)

print("Model successfully exported to ONNX format.")
