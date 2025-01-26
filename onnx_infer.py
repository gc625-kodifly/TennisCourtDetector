import onnxruntime
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
from postprocess import postprocess

# Load the ONNX model
model_path = "/home/gabriel/swing-vision/Savant/samples/my-module/src/weights/tracknet.onnx"
ort_session = onnxruntime.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Check input details
inputs = ort_session.get_inputs()
print("Input Name:", inputs[0].name)
print("Input Shape:", inputs[0].shape)
print("Input Type:", inputs[0].type)

outputs = ort_session.get_outputs()
print("Output Name:", outputs[0].name)

# Prepare a sample input (read the image and preprocess it)
image_path = "/home/gabriel/swing-vision/Savant/samples/my-module/assets/test_data/frame-001023.jpg"

# Load the image using PIL and convert to tensor
original_image = Image.open(image_path).convert("RGB")
original_shape = original_image.size  # Original image size (width, height)

# Transform image for model input
transform = transforms.Compose([
    transforms.Resize((360, 640)),  # Model input size
    transforms.ToTensor()          # Convert to tensor and normalize to [0, 1]
])
input_tensor = transform(original_image).unsqueeze(0).numpy()  # Add batch dimension and convert to numpy

# Prepare input dictionary
onnx_inputs = {inputs[0].name: input_tensor}

# Run inference
onnx_outputs = ort_session.run(None, onnx_inputs)

# Extract predictions
pred = onnx_outputs[0][0]
points = []
print(pred)

# Process heatmaps and find keypoints
for kps_num in range(14):
    heatmap = (pred[kps_num] * 255).astype(np.uint8)  # Convert to uint8 for visualization

    # Postprocess heatmap to find keypoints
    x_pred, y_pred = postprocess(heatmap, scale=1, low_thresh=170, max_radius=25)
    points.append((x_pred, y_pred))

# Scale points back to the original image size
scale_x = original_shape[0] / 640
scale_y = original_shape[1] / 360

# Draw keypoints on the original image
draw = ImageDraw.Draw(original_image)

for x, y in points:
    if x is not None and y is not None:
        x_scaled = int(x * scale_x)
        y_scaled = int(y * scale_y)
        draw.ellipse([x_scaled - 5, y_scaled - 5, x_scaled + 5, y_scaled + 5], fill="red")

# Show or save the image with keypoints
output_path = "/home/gabriel/swing-vision/TennisCourtDetector/output/result_with_keypoints.jpg"
original_image.save(output_path)
original_image.show()

# print(f"Output image with keypoints saved to:
