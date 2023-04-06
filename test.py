import torch
import torch.nn as nn

# Input array
x = torch.tensor([[2, 1], [0, 1]])

# Transposed convolution kernel
kernel = torch.tensor([[3, 1], [1, 5]])

# Define transposed convolution layer
conv_transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, output_padding=1, bias=False)

# Set the weights of the convolution layer to the kernel
conv_transpose.weight = nn.Parameter(kernel.unsqueeze(0).unsqueeze(0).float())

# Apply transposed convolution to input array
output = conv_transpose(x.unsqueeze(0).unsqueeze(0).float())

# Print the output
print(output.squeeze().int())
