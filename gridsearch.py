from config import config as cf
import model
from torch.utils.data import DataLoader
from loss import Unified_Adversarial_Loss as UAL
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from itertools import product
import numpy as np
import torch
import json

def save_best_param(model_name, best_params, best_score):
    filename = model_name + ".json"

    # create dictionary with best parameters and validation score
    result = {'best_params': best_params, 'best_score': best_score}

    # save dictionary to file
    with open(filename, 'w') as f:
        json.dump(result, f)

    print(f"\nBest parameters and validation score saved to file: {filename}")

def gridsearch_movement_3(dataset_train, dataset_val, features, mask, is_training=True):
    param_grid = {
        'lstm_hidden_layer_size': [16, 32, 64, 128],
        'lstm_num_layers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'kernel_size': [3, 4, 5,],
        'dilation_base': [3, 4, 5]
    }
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_3"
    # calculate number of combinations to test
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Number of parameter combinations: {n_combinations}")

    # initialize best score and corresponding parameter values
    best_score = np.inf
    best_params = None


    # iterate over all parameter combinations
    for params in product(*param_grid.values()):
        # unpack parameter values
        lstm_hidden_layer_size, lstm_num_layers, kernel_size, dilation_base = params

        # set parameter values in model
        movement_3 = model.Movement_3(
            input_size=len(features),
            window_size=cf["model"]["movement_3"]["window_size"],
            lstm_hidden_layer_size=lstm_hidden_layer_size,
            lstm_num_layers=lstm_num_layers,
            output_steps=cf["model"]["movement_3"]["output_steps"],
            kernel_size=kernel_size,
            dilation_base=dilation_base
        )
        optimizer = optim.Adam(movement_3.parameters(), lr=cf["training"]["movement_3"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
        # move model to GPU
        movement_3.to("cuda")

        # create dataloader
        train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["movement_3"]["batch_size"], shuffle=False, drop_last=True)
        valid_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_3"]["batch_size"], drop_last=True)
        # train model
        criterion = UAL()
        movement_3.train()
        for i in range(cf["training"]["movement_3"]["num_epoch"]):
            for batch_idx, (data, targets) in enumerate(train_dataloader):
                data = data.float().to("cuda")
                targets = targets.float().to("cuda")
                outputs = movement_3(data)
                loss = criterion(outputs, targets)
                movement_3.zero_grad()
                loss.backward()
                optimizer.step()

        # evaluate model on validation set
        movement_3.eval()
        with torch.no_grad():
            val_loss = []
            for data, targets in valid_dataloader:
                data = data.float().to("cuda")
                targets = targets.float().to("cuda")
                outputs = movement_3(data)
                loss = criterion(outputs, targets)
                val_loss.append(loss.item())
            val_loss = np.mean(val_loss)

        # check if new parameter combination is best so far
        if val_loss < best_score:
            best_score = val_loss
            best_params = params
            save_best_param(model_name=model_name, best_params=best_params, best_score=best_score)

        # print progress
        print(f"Parameter combination: {params} \t Validation loss: {val_loss:.4f} \t Best validation loss: {best_score:.4f}")

    # print best parameters and corresponding validation score
    print(f"\nBest parameters: {best_params} \t Best validation loss: {best_score:.4f}")


def gridsearch_movement_7(dataset_train, dataset_val, features, mask, is_training=True):
    param_grid = {
        'lstm_hidden_layer_size': [16, 32, 64],
        'lstm_num_layers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        #'kernel_size': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        #'dilation_base': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_7"
    # calculate number of combinations to test
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Number of parameter combinations: {n_combinations}")

    # initialize best score and corresponding parameter values
    best_score = np.inf
    best_params = None


    # iterate over all parameter combinations
    for params in product(*param_grid.values()):
        # unpack parameter values
        lstm_hidden_layer_size, lstm_num_layers, kernel_size, dilation_base = params

        # set parameter values in model
        movement_7 = model.Movement_7(
            input_size=len(features),
            window_size=cf["model"]["movement_7"]["window_size"],
            lstm_hidden_layer_size=lstm_hidden_layer_size,
            lstm_num_layers=lstm_num_layers,
            output_steps=cf["model"]["movement_7"]["output_steps"],
            kernel_size=kernel_size,
            dilation_base=dilation_base
        )
        optimizer = optim.Adam(movement_7.parameters(), lr=cf["training"]["movement_7"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
        # move model to GPU
        movement_7.to("cuda")

        # create dataloader
        train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["movement_7"]["batch_size"], shuffle=False, drop_last=True)
        valid_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_7"]["batch_size"], drop_last=True)
        # train model
        criterion = UAL()
        movement_7.train()
        for i in range(cf["training"]["movement_7"]["num_epoch"]):
            for batch_idx, (data, targets) in enumerate(train_dataloader):
                data = data.float().to("cuda")
                targets = targets.float().to("cuda")
                outputs = movement_7(data)
                loss = criterion(outputs, targets)
                movement_7.zero_grad()
                loss.backward()
                optimizer.step()

        # evaluate model on validation set
        movement_7.eval()
        with torch.no_grad():
            val_loss = []
            for data, targets in valid_dataloader:
                data = data.float().to("cuda")
                targets = targets.float().to("cuda")
                outputs = movement_7(data)
                loss = criterion(outputs, targets)
                val_loss.append(loss.item())
            val_loss = np.mean(val_loss)

        # check if new parameter combination is best so far
        if val_loss < best_score:
            best_score = val_loss
            best_params = params
            save_best_param(model_name=model_name, best_params=best_params, best_score=best_score)

        # print progress
        print(f"Parameter combination: {params} \t Validation loss: {val_loss:.4f} \t Best validation loss: {best_score:.4f}")

    # print best parameters and corresponding validation score
    print(f"\nBest parameters: {best_params} \t Best validation loss: {best_score:.4f}")

def gridsearch_movement_14(dataset_train, dataset_val, features, mask, is_training=True):
    param_grid = {
        'lstm_hidden_layer_size': [16, 32, 64],
        'lstm_num_layers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        #'kernel_size': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        #'dilation_base': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_14"
    # calculate number of combinations to test
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Number of parameter combinations: {n_combinations}")

    # initialize best score and corresponding parameter values
    best_score = np.inf
    best_params = None


    # iterate over all parameter combinations
    for params in product(*param_grid.values()):
        # unpack parameter values
        lstm_hidden_layer_size, lstm_num_layers, kernel_size, dilation_base = params

        # set parameter values in model
        movement_3 = model.Movement_3(
            input_size=len(features),
            window_size=cf["model"]["movement_3"]["window_size"],
            lstm_hidden_layer_size=lstm_hidden_layer_size,
            lstm_num_layers=lstm_num_layers,
            output_steps=cf["model"]["movement_3"]["output_steps"],
            kernel_size=kernel_size,
            dilation_base=dilation_base
        )
        optimizer = optim.Adam(movement_3.parameters(), lr=cf["training"]["movement_3"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
        # move model to GPU
        movement_3.to("cuda")

        # create dataloader
        train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["movement_3"]["batch_size"], shuffle=False, drop_last=True)
        valid_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_3"]["batch_size"], drop_last=True)
        # train model
        criterion = UAL()
        movement_3.train()
        for i in range(cf["training"]["movement_3"]["num_epoch"]):
            for batch_idx, (data, targets) in enumerate(train_dataloader):
                data = data.float().to("cuda")
                targets = targets.float().to("cuda")
                outputs = movement_3(data)
                loss = criterion(outputs, targets)
                movement_3.zero_grad()
                loss.backward()
                optimizer.step()

        # evaluate model on validation set
        movement_3.eval()
        with torch.no_grad():
            val_loss = []
            for data, targets in valid_dataloader:
                data = data.float().to("cuda")
                targets = targets.float().to("cuda")
                outputs = movement_3(data)
                loss = criterion(outputs, targets)
                val_loss.append(loss.item())
            val_loss = np.mean(val_loss)

        # check if new parameter combination is best so far
        if val_loss < best_score:
            best_score = val_loss
            best_params = params
            save_best_param(model_name=model_name, best_params=best_params, best_score=best_score)

        # print progress
        print(f"Parameter combination: {params} \t Validation loss: {val_loss:.4f} \t Best validation loss: {best_score:.4f}")

    # print best parameters and corresponding validation score
    print(f"\nBest parameters: {best_params} \t Best validation loss: {best_score:.4f}")


