import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from model import Model
from config import config as cf
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import util as u
from dataset import TimeSeriesDataset, Classification_TimeSeriesDataset
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import json
import numpy as np
import os


# Main
class Trainer:
    def __init__(self):

        self.model_type_dict = {
            1: "movement",
            2: "magnitude",
            3: "assembler"
        }

    def train(self, model_name, new_data=False):
        training_param = cf["training"][model_name]
        device = training_param["device"]
        batch_size = training_param["batch_size"]
        num_epoch = training_param["num_epoch"]
        learning_rate = training_param["learning_rate"]
        loss = training_param["loss"]
        evaluate = training_param["evaluate"]
        optimizer = training_param["optimizer"]
        scheduler_step_size = training_param["scheduler_step_size"]
        patient = training_param["patient"]
        start = training_param["start"]
        end = training_param["end"]
        best_model = training_param["best_model"]
        early_stop = training_param["early_stop"]
        train_shuffle = training_param["train_shuffle"]
        val_shuffle = training_param["val_shuffle"]
        test_shuffle = training_param["test_shuffle"]

        model_param = cf["model"][model_name]

        if "movement" in model_name:
            model_type = self.model_type_dict[1]
        elif "magnitude" in model_name:
            model_type = self.model_type_dict[2]
        elif "assembler" in model_name:
            model_type = self.model_type_dict[3]

        if model_type == "movement" or "magnitude" or "assembler":
            window_size = model_param["window_size"]
            output_step = model_param["output_step"]
            model_full_name = cf["alpha_vantage"]["symbol"] + "_" + model_name
            train_dataloader, valid_dataloader, test_dataloader, \
                num_feature, num_data_points, \
                train_date, valid_date, test_date = prepare_data(model_type, model_full_name, window_size, start, end,
                                                                 new_data,
                                                                 output_step, batch_size, train_shuffle, val_shuffle,
                                                                 test_shuffle)
            model = Model(name=model_name, num_feature=num_feature, model_type=model_type)
            model.full_name = model_full_name

        if "mse" in loss:
            criterion = nn.MSELoss()
        elif "mae" in loss:
            criterion = nn.L1Loss()
        elif "bce" in loss:
            criterion = nn.BCELoss()

        if "adam" in optimizer:
            optimizer = optim.Adam(model.structure.parameters(), lr=learning_rate, weight_decay=0.1)
        elif "sgd" in optimizer:
            optimizer = optim.SGD(model.structure.parameters(), lr=learning_rate, weight_decay=0.1)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                      patience=scheduler_step_size, verbose=True)

        model.structure.to(device)
        if best_model:
            best_loss = sys.float_info.max
        else:
            best_loss = sys.float_info.min

        if early_stop:
            stop = False

        for epoch in range(num_epoch):
            loss_train, lr_train = run_epoch(model, train_dataloader, optimizer, criterion, scheduler,
                                             is_training=True)
            loss_val, lr_val = run_epoch(model, valid_dataloader, optimizer, criterion, scheduler,
                                         is_training=False)
            scheduler.step(loss_val)
            if best_model:
                if check_best_loss(best_loss=best_loss, loss=loss_val):
                    best_loss = loss_val
                    patient_count = 0
                    model.train_stop_lr = lr_train
                    model.train_stop_epoch = epoch

                    model.state_dict = model.structure.state_dict()
                    torch.save({"model": model,
                                "state_dict": model.structure.state_dict()
                                },
                               "./models/" + model.full_name + ".pth")
                else:
                    if early_stop:
                        stop, patient_count, best_loss, _ = is_early_stop(best_loss=best_loss, current_loss=loss_val,
                                                                          patient_count=patient_count,
                                                                          max_patient=patient)
            else:
                model.state_dict = model.structure.state_dict()
                torch.save({"model": model,
                            "state_dict": model.structure.state_dict()
                            },
                           "./models/" + model.full_name + ".pth")

            print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                  .format(epoch + 1, num_epoch, loss_train, loss_val, lr_train))

            print("patient", patient_count)
            if stop:
                print("Early Stopped At Epoch: {}", epoch + 1)
                break

        return model

    def eval(self, model):
        model.structure.eval()
        model_full_name = model.full_name
        model_name = model.name
        train_file_name = f"./csv/train_{model_full_name}.csv"
        valid_file_name = f"./csv/valid_{model_full_name}.csv"
        test_file_name = f"./csv/test_{model_full_name}.csv"

        training_param = cf["training"][model_name]
        device = training_param["device"]
        batch_size = training_param["batch_size"]
        evaluate = training_param["evaluate"]
        start = training_param["start"]
        end = training_param["end"]
        train_shuffle = training_param["train_shuffle"]
        val_shuffle = training_param["val_shuffle"]
        test_shuffle = training_param["test_shuffle"]
        model_type = model.model_type

        model_param = cf["model"][model_name]
        window_size = model_param["window_size"]
        output_step = model_param["output_step"]
        train_dataloader, valid_dataloader, test_dataloader, train_date, valid_date, test_date = \
            prepare_eval_data(model_type, model_full_name, train_file_name, valid_file_name, test_file_name, batch_size,
                              evaluate, start, end, train_shuffle, val_shuffle, test_shuffle, window_size, output_step)

        model.structure.to(device)
        for i in range(0, 3, 1):
            if i == 0:
                dataloader = train_dataloader
                print_string = "Train evaluate " + model_full_name
            if i == 1:
                dataloader = valid_dataloader
                print_string = "Valid evaluate " + model_full_name
            elif i == 2:
                dataloader = test_dataloader
                print_string = "Test evaluate " + model_full_name
            if "accuracy" or "precision" or "f1" in evaluate:
                # Create empty lists to store the true and predicted labels
                true_labels = []
                predicted_labels = []
            total_loss = 0

            target_list = torch.empty(0).to(device)
            output_list = torch.empty(0).to(device)
            # Iterate over the dataloader
            for inputs, labels in dataloader:
                # Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model.predict(inputs)

                target_list = torch.cat([target_list, labels], dim=0)
                output_list = torch.cat([output_list, outputs], dim=0)
                if "accuracy" or "precision" or "f1" in evaluate:
                    predicted = (outputs > 0.5).float()
                    # Append true and predicted labels to the respective lists
                    true_labels.extend(labels.cpu().detach().numpy())
                    predicted_labels.extend(predicted.cpu().detach().numpy())

            if "accuracy" or "precision" or "f1" in evaluate:
                # Compute classification report
                target_names = ["DOWN", "UP"]  # Add target class names here
                report = classification_report(true_labels, predicted_labels, target_names=target_names)
                # Create the saving folder if it does not exist
                save_folder = "./eval/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                # Open the file in write mode
                save_path = os.path.join(save_folder,  model_full_name+"_eval")
                # Open the file in write mode
                with open(save_path, "a") as f:
                    # Write the classification report to the file
                    f.write(print_string + "\n" + "Classification report:\n")
                    f.write(report)

                    # Write the config dictionary to the file
                    f.write("\nTraining Config:\n")
                    f.write(json.dumps(training_param, indent=4))
                    f.write("\nModel Config:\n")
                    f.write(json.dumps(model_param, indent=4))
                    # Compute confusion matrix
                    cm = confusion_matrix(true_labels, predicted_labels)

                    # Write the confusion matrix to the file
                    f.write("\nConfusion matrix:\n")
                    f.write(np.array2string(cm))
                    f.write("\n")
                    f.write("-" * 100)
                    f.write("\n")
                # Print a message to confirm that the file was written successfully
                print("Results written to " + model_full_name + "_eval.txt")

            temp_evaluate = np.array(evaluate)

            temp_evaluate = temp_evaluate[temp_evaluate != "accuracy"]
            temp_evaluate = temp_evaluate[temp_evaluate != "precision"]
            temp_evaluate = temp_evaluate[temp_evaluate != "f1"]
            for c in temp_evaluate:
                if "mse" in evaluate:
                    criterion = nn.MSELoss()
                    c_name = "MSE"
                elif "mae" in evaluate:
                    criterion = nn.L1Loss()
                    c_name = "MAE"
                elif "bce" in evaluate:
                    criterion = nn.BCELoss()
                    c_name = "BCE"

                target_list = target_list.reshape(-1)
                output_list = output_list.reshape(-1)
                loss = criterion(target_list, output_list)
                loss_str = f"{c_name} loss: {loss.item()}"

                # Create the saving folder if it does not exist
                save_folder = "./eval/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                # Open the file in append mode
                save_path = os.path.join(save_folder, model_full_name + "_eval")
                with open(save_path, "a") as f:
                    # Write the loss to the file
                    f.write(print_string + " " + loss_str + "\n")
                    f.write("-" * 100)
                    f.write("\n-")
                # Print a message to confirm that the file was written successfully
                print(f"Loss written to {save_path}.")


def prepare_eval_data(model_type, model_full_name, train_file_name, valid_file_name, test_file_name, batch_size,
                      evaluate, start, end, train_shuffle, val_shuffle, test_shuffle, window_size, output_step):
    train_df = pd.read_csv(f"./csv/train_{model_full_name}.csv", index_col=0, parse_dates=True)
    valid_df = pd.read_csv(f"./csv/valid_{model_full_name}.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv(f"./csv/test_{model_full_name}.csv", index_col=0, parse_dates=True)

    train_date = train_df.index.strftime("%Y-%m-%d").tolist()
    valid_date = valid_df.index.strftime("%Y-%m-%d").tolist()
    test_date = test_df.index.strftime("%Y-%m-%d").tolist()

    if model_type == "movement":

        train_date = train_date[int(len(train_date) - len(train_df)):]
        valid_date = valid_date[int(len(valid_date) - len(valid_df)):]
        test_date = test_date[int(len(test_date) - len(test_df)):]

        # prepare y df
        train_close_df = pd.DataFrame({'close': train_df['close']})
        valid_close_df = pd.DataFrame({'close': valid_df['close']})
        test_close_df = pd.DataFrame({'close': test_df['close']})

        train_n_row = len(train_close_df) - window_size - output_step
        valid_n_row = len(valid_close_df) - window_size - output_step
        test_n_row = len(test_close_df) - window_size - output_step

        y_train = u.prepare_timeseries_data_y_trend(train_n_row, train_close_df.to_numpy(),
                                                    output_size=output_step)
        y_valid = u.prepare_timeseries_data_y_trend(valid_n_row, valid_close_df.to_numpy(),
                                                    output_size=output_step)
        y_test = u.prepare_timeseries_data_y_trend(test_n_row, test_close_df.to_numpy(),
                                                   output_size=output_step)

        X_train = u.prepare_timeseries_data_x(train_df.to_numpy(), window_size=window_size)[:-output_step]
        X_valid = u.prepare_timeseries_data_x(valid_df.to_numpy(), window_size=window_size)[:-output_step]
        X_test = u.prepare_timeseries_data_x(test_df.to_numpy(), window_size=window_size)[:-output_step]

        train_dataset = Classification_TimeSeriesDataset(X_train, y_train)
        valid_dataset = Classification_TimeSeriesDataset(X_valid, y_valid)
        test_dataset = Classification_TimeSeriesDataset(X_test, y_test)

    elif model_type == "magnitude":
        train_date = train_date[int(len(train_date) - len(train_df)):]
        valid_date = valid_date[int(len(valid_date) - len(valid_df)):]
        test_date = test_date[int(len(test_date) - len(test_df)):]

        # prepare y df
        train_close_df = pd.DataFrame({'close': train_df['close']})
        valid_close_df = pd.DataFrame({'close': valid_df['close']})
        test_close_df = pd.DataFrame({'close': test_df['close']})

        train_n_row = len(train_close_df) - window_size - output_step
        valid_n_row = len(valid_close_df) - window_size - output_step
        test_n_row = len(test_close_df) - window_size - output_step

        y_train = u.prepare_timeseries_data_y_percentage(train_n_row, train_close_df.to_numpy(),
                                                         output_size=output_step)
        y_valid = u.prepare_timeseries_data_y_percentage(valid_n_row, valid_close_df.to_numpy(),
                                                         output_size=output_step)
        y_test = u.prepare_timeseries_data_y_percentage(test_n_row, test_close_df.to_numpy(),
                                                        output_size=output_step)

        X_train = u.prepare_timeseries_data_x(train_df.to_numpy(), window_size=window_size)[:-output_step]
        X_valid = u.prepare_timeseries_data_x(valid_df.to_numpy(), window_size=window_size)[:-output_step]
        X_test = u.prepare_timeseries_data_x(test_df.to_numpy(), window_size=window_size)[:-output_step]

        train_dataset = Classification_TimeSeriesDataset(X_train, y_train)
        valid_dataset = Classification_TimeSeriesDataset(X_valid, y_valid)
        test_dataset = Classification_TimeSeriesDataset(X_test, y_test)

    elif model_type == "assembler":
        train_date = train_date[int(len(train_date) - len(train_df)):]
        valid_date = valid_date[int(len(valid_date) - len(valid_df)):]
        test_date = test_date[int(len(test_date) - len(test_df)):]

        # prepare y df
        train_close_df = pd.DataFrame({'close': train_df['close']})
        valid_close_df = pd.DataFrame({'close': valid_df['close']})
        test_close_df = pd.DataFrame({'close': test_df['close']})

        train_n_row = len(train_close_df) - window_size - output_step
        valid_n_row = len(valid_close_df) - window_size - output_step
        test_n_row = len(test_close_df) - window_size - output_step

        y_train = u.prepare_timeseries_data_y(train_n_row, train_close_df.to_numpy(),
                                              output_size=output_step)
        y_valid = u.prepare_timeseries_data_y(valid_n_row, valid_close_df.to_numpy(),
                                              output_size=output_step)
        y_test = u.prepare_timeseries_data_y(test_n_row, test_close_df.to_numpy(),
                                             output_size=output_step)

        X_train = u.prepare_timeseries_data_x(train_df.to_numpy(), window_size=window_size)[:-output_step]
        X_valid = u.prepare_timeseries_data_x(valid_df.to_numpy(), window_size=window_size)[:-output_step]
        X_test = u.prepare_timeseries_data_x(test_df.to_numpy(), window_size=window_size)[:-output_step]

        train_dataset = TimeSeriesDataset(X_train, y_train)
        valid_dataset = TimeSeriesDataset(X_valid, y_valid)
        test_dataset = TimeSeriesDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=val_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)

    train_df.to_csv(f"./csv/train_{model_full_name}.csv")
    valid_df.to_csv(f"./csv/valid_{model_full_name}.csv")
    test_df.to_csv(f"./csv/test_{model_full_name}.csv")

    return train_dataloader, valid_dataloader, test_dataloader, train_date, valid_date, test_date


def prepare_data(model_type, model_full_name, window_size, start, end, new_data,
                 output_step, batch_size, train_shuffle, val_shuffle, test_shuffle):
    df = u.prepare_stock_dataframe(window_size, start, end, new_data)
    num_data_points = df.shape[0]
    num_feature = df.shape[1]
    data_date = df.index.strftime("%Y-%m-%d").tolist()
    # Split train val 80%
    trainval_test_split_index = int(num_data_points * cf["data"]["train_test_split_size"])
    # 0 - 80
    train_valid_df = df[:trainval_test_split_index]
    # train with val dates
    train_valid_date = data_date[:trainval_test_split_index]
    # test 80 - 100%
    test_df = df[trainval_test_split_index:]
    # test dates splitted
    test_date = data_date[trainval_test_split_index:]
    # New index for train and valid only
    train_valid_split_index = int(len(train_valid_df) * cf["data"]["train_val_split_size"])
    # Train and valid df split up
    # 0 - 80
    train_df = train_valid_df[:train_valid_split_index]
    # 80 - 100%
    valid_df = train_valid_df[train_valid_split_index:]
    # Train and valid dates df split up
    train_date = train_valid_date[:train_valid_split_index]
    valid_date = train_valid_date[train_valid_split_index:]

    train_df.to_csv(f"./csv/train_{model_full_name}.csv")
    valid_df.to_csv(f"./csv/valid_{model_full_name}.csv")
    test_df.to_csv(f"./csv/test_{model_full_name}.csv")

    if model_type == "movement":

        train_date = train_date[int(len(train_date) - len(train_df)):]
        valid_date = valid_date[int(len(valid_date) - len(valid_df)):]
        test_date = test_date[int(len(test_date) - len(test_df)):]

        # prepare y df
        train_close_df = pd.DataFrame({'close': train_df['close']})
        valid_close_df = pd.DataFrame({'close': valid_df['close']})
        test_close_df = pd.DataFrame({'close': test_df['close']})

        train_n_row = len(train_close_df) - window_size - output_step
        valid_n_row = len(valid_close_df) - window_size - output_step
        test_n_row = len(test_close_df) - window_size - output_step

        y_train = u.prepare_timeseries_data_y_trend(train_n_row, train_close_df.to_numpy(),
                                                    output_size=output_step)
        y_valid = u.prepare_timeseries_data_y_trend(valid_n_row, valid_close_df.to_numpy(),
                                                    output_size=output_step)
        y_test = u.prepare_timeseries_data_y_trend(test_n_row, test_close_df.to_numpy(),
                                                   output_size=output_step)

        X_train = u.prepare_timeseries_data_x(train_df.to_numpy(), window_size=window_size)[:-output_step]
        X_valid = u.prepare_timeseries_data_x(valid_df.to_numpy(), window_size=window_size)[:-output_step]
        X_test = u.prepare_timeseries_data_x(test_df.to_numpy(), window_size=window_size)[:-output_step]

        train_dataset = Classification_TimeSeriesDataset(X_train, y_train)
        valid_dataset = Classification_TimeSeriesDataset(X_valid, y_valid)
        test_dataset = Classification_TimeSeriesDataset(X_test, y_test)

    elif model_type == "magnitude":
        train_date = train_date[int(len(train_date) - len(train_df)):]
        valid_date = valid_date[int(len(valid_date) - len(valid_df)):]
        test_date = test_date[int(len(test_date) - len(test_df)):]

        # prepare y df
        train_close_df = pd.DataFrame({'close': train_df['close']})
        valid_close_df = pd.DataFrame({'close': valid_df['close']})
        test_close_df = pd.DataFrame({'close': test_df['close']})

        train_n_row = len(train_close_df) - window_size - output_step
        valid_n_row = len(valid_close_df) - window_size - output_step
        test_n_row = len(test_close_df) - window_size - output_step

        y_train = u.prepare_timeseries_data_y_percentage(train_n_row, train_close_df.to_numpy(),
                                                         output_size=output_step)
        y_valid = u.prepare_timeseries_data_y_percentage(valid_n_row, valid_close_df.to_numpy(),
                                                         output_size=output_step)
        y_test = u.prepare_timeseries_data_y_percentage(test_n_row, test_close_df.to_numpy(),
                                                        output_size=output_step)

        X_train = u.prepare_timeseries_data_x(train_df.to_numpy(), window_size=window_size)[:-output_step]
        X_valid = u.prepare_timeseries_data_x(valid_df.to_numpy(), window_size=window_size)[:-output_step]
        X_test = u.prepare_timeseries_data_x(test_df.to_numpy(), window_size=window_size)[:-output_step]

        train_dataset = Classification_TimeSeriesDataset(X_train, y_train)
        valid_dataset = Classification_TimeSeriesDataset(X_valid, y_valid)
        test_dataset = Classification_TimeSeriesDataset(X_test, y_test)
    elif model_type == "assembler":
        train_date = train_date[int(len(train_date) - len(train_df)):]
        valid_date = valid_date[int(len(valid_date) - len(valid_df)):]
        test_date = test_date[int(len(test_date) - len(test_df)):]

        # prepare y df
        train_close_df = pd.DataFrame({'close': train_df['close']})
        valid_close_df = pd.DataFrame({'close': valid_df['close']})
        test_close_df = pd.DataFrame({'close': test_df['close']})

        train_n_row = len(train_close_df) - window_size - output_step
        valid_n_row = len(valid_close_df) - window_size - output_step
        test_n_row = len(test_close_df) - window_size - output_step

        y_train = u.prepare_timeseries_data_y(train_n_row, train_close_df.to_numpy(),
                                              output_size=output_step)
        y_valid = u.prepare_timeseries_data_y(valid_n_row, valid_close_df.to_numpy(),
                                              output_size=output_step)
        y_test = u.prepare_timeseries_data_y(test_n_row, test_close_df.to_numpy(),
                                             output_size=output_step)

        X_train = u.prepare_timeseries_data_x(train_df.to_numpy(), window_size=window_size)[:-output_step]
        X_valid = u.prepare_timeseries_data_x(valid_df.to_numpy(), window_size=window_size)[:-output_step]
        X_test = u.prepare_timeseries_data_x(test_df.to_numpy(), window_size=window_size)[:-output_step]

        train_dataset = TimeSeriesDataset(X_train, y_train)
        valid_dataset = TimeSeriesDataset(X_valid, y_valid)
        test_dataset = TimeSeriesDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=val_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)

    return train_dataloader, valid_dataloader, test_dataloader, \
        num_feature, num_data_points, \
        train_date, valid_date, test_date


def check_best_loss(best_loss, loss):
    if loss < best_loss:
        return True
    return False


# return boolen Stop, patient_count, best_loss, current_loss
def is_early_stop(best_loss, current_loss, patient_count, max_patient):
    stop = False
    if best_loss > current_loss:
        best_loss = current_loss
        patient_count = 0
    else:
        patient_count = patient_count + 1
    if patient_count >= max_patient:
        stop = True
    return stop, patient_count, best_loss, current_loss


def run_epoch(model, dataloader, optimizer, criterion, scheduler, is_training):
    epoch_loss = 0

    weight_decay = 0.001
    if is_training:
        model.structure.train()
    else:
        model.structure.eval()

    # create a tqdm progress bar
    dataloader = tqdm(dataloader)
    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]
        # print(x.shape)
        x = x.to("cuda")
        y = y.to("cuda")

        out = model.predict(x)
        loss = criterion(out, y)
        if is_training:
            if loss != torch.nan:
                torch.autograd.set_detect_anomaly(True)
                loss.backward()
                optimizer.step()
            else:
                print("loss = nan")
        batch_loss = (loss.detach().item())
        epoch_loss += batch_loss
        # update the progress bar
        dataloader.set_description(f"At index {idx:.4f}")

    try:
        lr = scheduler.get_last_lr()[0]

    except:
        lr = optimizer.param_groups[0]['lr']
    return epoch_loss, lr
