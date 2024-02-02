import torch.onnx
import torch.nn as nn
import os
import torch
import timm  # A library containing pre-trained models

def load_model(model, resume_checkpoints):
    """
    Load pre-trained weights into the provided model.
    """
    model = nn.DataParallel(model).cuda()  # Convert model to data parallel and move to GPU

    # Check if the checkpoints file exists
    if os.path.isfile(resume_checkpoints) or os.path.islink(resume_checkpoints):
        pretrained_dict = torch.load(resume_checkpoints)
        # Check if the state_dict key exists in the checkpoints
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['state_dict']
        # Load the weights into the model
        model.load_state_dict(pretrained_dict)
    return model

checkpoints = "pytorch_model_name.pth"

# Load the saved model
saved_model = torch.load(checkpoints)
state_dict = saved_model['state_dict']

new_state_dict = {}
for key, value in state_dict.items():
    name = key[7:]  # remove the "module." prefix, which is added when using DataParallel
    new_state_dict[name] = value

# Create a ResNet18 model with 136 classes (68 landmarks * 2 for x,y coordinates)
model = timm.create_model("resnet18", num_classes=68 * 2)
# Load the saved weights into the model
model.load_state_dict(new_state_dict)

# Check if a GPU is available, if not use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert the model and input to half precision (float16) for faster inference, but only if on a GPU
model = model.to(device)
if device.type == 'cuda':
    model = model.half()

# Set the model to evaluation mode
model.eval()

# Create a random tensor with the shape (1, 3, 224, 224) to simulate an input image
x = torch.randn(1, 3, 224, 224).to(device)
# Convert the input to half precision if on a GPU
if device.type == 'cuda':
    x = x.half()

# Get the model output for the given input
torch_out = model(x)

# Export the PyTorch model to ONNX format
torch.onnx.export(model,
                  x,
                  "Onnx_model_name.onnx",
                  export_params=True,  # Export model parameters
                  opset_version=10,    # Set the ONNX opset version
                  do_constant_folding=True,  # Perform constant folding optimization
                  input_names=['input'],  # Name of the input tensor
                  output_names=["pred_output"])  # Name of the output tensor
