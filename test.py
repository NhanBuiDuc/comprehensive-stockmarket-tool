from model import CausalDilatedConvNet
import torch
import math

input_size = 11
sequence_size = 14
kernel_size = 3
dilation_base = 3

# Calculate the number of layers
num_layers = int(

    (math.log( ( ((input_size - 1) * (dilation_base - 1)) / (kernel_size - 1) ) + 1)) / (math.log(dilation_base))
)

model = CausalDilatedConvNet(in_channels = sequence_size, out_channels = sequence_size, kernel_size = kernel_size, num_layers=num_layers, dilation_base=dilation_base)

random_tensor = torch.randn(1, sequence_size, input_size)
output = model(random_tensor)
print(output.shape)