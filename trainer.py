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


def prepare_data(model_type, window_size, start, end, new_data,
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
        model.train()
    else:
        model.eval()

    # create a tqdm progress bar
    dataloader = tqdm(dataloader)
    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]
        # print(x.shape)
        x = x.to("cuda")
        y = y.to("cuda")

        out = model(x)

        # Calculate the L2 regularization term
        # l2_reg = 0
        # for param in model.parameters():
        #     l2_reg += torch.norm(param).to("cuda")
        # loss = criterion(out, y) + weight_decay * l2_reg
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


# Main
class Trainer:
    def __init__(self):
        self.test_dataloader_dict = {}
        self.valid_dataloader_dict = {}
        self.train_dataloader_dict = {}

        self.test_date_dict = {}
        self.valid_date_dict = {}
        self.train_date_dict = {}

        self.model_dict = {}
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
                train_date, valid_date, test_date = prepare_data(model_type, window_size, start, end, new_data,
                                                                 output_step, batch_size, train_shuffle, val_shuffle, test_shuffle)
            model = Model(name=model_name, num_feature=num_feature, model_type=model_type)
            model.name = model_full_name
            self.train_dataloader_dict[model] = train_dataloader
            self.valid_dataloader_dict[model] = valid_dataloader
            self.test_dataloader_dict[model] = test_dataloader
            self.train_date_dict[model] = train_date
            self.valid_date_dict[model] = valid_date
            self.test_date_dict[model] = test_date
        if "mse" in loss:
            criterion = nn.MSELoss()
        elif "mae" in loss:
            criterion = nn.L1Loss()
        elif "bce" in loss:
            criterion = nn.BCELoss()

        if "adam" in optimizer:
            optimizer = optim.Adam(model.structure.parameters(), lr=learning_rate,  weight_decay=0.001)
        elif "sgd" in optimizer:
            optimizer = optim.SGD(model.structure.parameters(), lr=learning_rate,  weight_decay=0.001)
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
            loss_train, lr_train = run_epoch(model.structure, train_dataloader, optimizer, criterion, scheduler,
                                             is_training=True)
            loss_val, lr_val = run_epoch(model.structure, valid_dataloader, optimizer, criterion, scheduler,
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
                               "./models/" + model.name + ".pth")
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
                           "./models/" + model.name + ".pth")

            print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                  .format(epoch + 1, num_epoch, loss_train, loss_val, lr_train))

            print("patient", patient_count)
            if stop:
                print("Early Stopped At Epoch: {}", epoch + 1)
                break
        self.model_dict[model_name] = model
        return self.model_dict[model_name]

    def eval(self):
        pass

    def save_model(self):
        pass
