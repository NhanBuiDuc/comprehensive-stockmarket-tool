import numpy as np

data = [1, 2, 0, 9, 3, 1, 0, 10, 9, 0]

window_size = 3
output_step = 1
features = len(data)
dilation = 1     
stride = 1
n_samples = (len(data) - dilation * (window_size - 1) - output_step)
X = np.zeros((n_samples, window_size, 1))
y = []
features = len(data)

for i in range(n_samples):
    end_index = i + (window_size - stride) * dilation
    output_index = end_index + output_step
    for j in range(window_size):
        X[i][j] = (data[i + (j * dilation)])
    # X[i] = data[i:i+window_size]
    if data[end_index] < data[output_index]:
        y.append(1)
    else:
        y.append(0)

# check X, y
false_list = []
for i in range(len(X)):
    end_index = i + (window_size - stride) * dilation
    output_index = end_index + output_step
    if X[i][-1] < data[output_index] and y[i] == 1:
        false_list.append(True)
    elif X[i][-1] > data[output_index] and y[i] == 0:
        false_list.append(True)
    else:
        false_list.append(False)
if False in false_list:
    print("Wrong data")
else:
    print("Correct data")
# return X, np.array(y)