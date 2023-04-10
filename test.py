from model import CausalDilatedConvNet
import torch
import math
import tensorflow as tf


input_size = 1
sequence_size = 10
kernel_size = 2
dilation_base = 3

# Calculate the number of layers
num_layers = int(

    (math.log( ( ((sequence_size - 1) * (dilation_base - 1)) / (kernel_size - 1) ) + 1)) / (math.log(dilation_base))
) + 1
receptive_field_size = sequence_size + (kernel_size - 1) * sum(dilation_base ** i for i in range(num_layers))

model = CausalDilatedConvNet(window_size = sequence_size, input_channels = input_size, out_channels = input_size, kernel_size = kernel_size, num_layers=num_layers, dilation_base=dilation_base, receptive_size = 55)

random_tensor = torch.randn(1, sequence_size, input_size)
output = model(random_tensor)
print(output.shape)