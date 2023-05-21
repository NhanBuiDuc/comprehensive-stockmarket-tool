from trainer.trainer import Trainer, run_epoch, check_best_loss, is_early_stop

import pandas as pd
import torch
from torch.utils.data import ConcatDataset
from model import Model
from configs.transformer_cf import transformer_cf as cf
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import util as u
from dataset import Classification_TimeSeriesDataset, MyDataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import json
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import datetime
import NLP.util as nlp_u
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, StratifiedShuffleSplit
from loss import FocalLoss

class Transformer_trainer(Trainer):
    def __init__(self, model_name, new_data=True, full_data=False, num_feature=None, config=None, model_type=None,
                 model_full_name=None,
                 model=None, mode="train"):
        super(Transformer_trainer, self).__init__()
        self.__dict__.update(self.cf)
        self.config = cf
        self.symbol = self.cf["alpha_vantage"]["symbol"]
        self.model_name = model_name
        self.__dict__.update(self.config["model"][self.model_name])
        self.__dict__.update(self.config["training"][self.model_name])
        self.test_dataloader = None
        self.valid_dataloader = None
        self.train_dataloader = None
        self.full_data = full_data
        self.num_feature = num_feature
        self.new_data = new_data
        self.model_type = "transformer"
        self.model_type_dict = self.cf["pytorch_timeseries_model_type_dict"]
        self.model = model
        self.mode = mode
        self.model_full_name = self.symbol + "_" + self.model_name
        if self.mode == "train":
            self.prepare_data(self.new_data)
        else:
            self.num_feature = 807
        self.indentify()

    def indentify(self):
        self.model = Model(name=self.model_name, num_feature=self.num_feature, parameters=self.config,
                           model_type=self.model_type,
                           full_name=self.model_full_name)

    def train(self):
        self.mode = "train"
        if "mse" in self.loss:
            criterion = nn.MSELoss()
        elif "mae" in self.loss:
            criterion = nn.L1Loss()
        elif "bce" in self.loss:
            criterion = nn.BCELoss()
        elif "focal" in self.loss:
            criterion = FocalLoss(alpha=0.5, gamma=2)
        if "adam" in self.optimizer:
            optimizer = optim.Adam(self.model.structure.parameters(), lr=self.learning_rate,
                                   weight_decay=self.weight_decay)
        elif "sgd" in self.optimizer:
            optimizer = optim.SGD(self.model.structure.parameters(), lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                      patience=self.scheduler_step_size, verbose=True)

        self.model.structure.to(self.device)
        if self.best_model:
            best_loss = sys.float_info.max
        else:
            best_loss = sys.float_info.min

        if self.early_stop:
            stop = False
        # Run train valid
        if not self.full_data:
            for epoch in range(self.num_epoch):
                loss_train, lr_train = self.run_epoch(self.model, self.train_dataloader, optimizer, criterion,
                                                      scheduler,
                                                      is_training=True, device=self.device)
                loss_val, lr_val = self.run_epoch(self.model, self.valid_dataloader, optimizer, criterion, scheduler,
                                                  is_training=False, device=self.device)
                loss_test, lr_test = self.run_epoch(self.model, self.test_dataloader, optimizer, criterion, scheduler,
                                                    is_training=False, device=self.device)
                scheduler.step(loss_val)
                if self.best_model:
                    if check_best_loss(best_loss=best_loss, loss=loss_val):
                        best_loss = loss_val
                        patient_count = 0
                        self.model.train_stop_lr = lr_train
                        self.model.train_stop_epoch = epoch

                        self.model.state_dict = self.model.structure.state_dict()
                        self.model.train_stop_epoch = epoch
                        self.model.train_stop_lr = lr_train
                        torch.save({"model": self.model,
                                    "state_dict": self.model.structure.state_dict()
                                    },
                                   "./models/" + self.model.full_name + ".pth")
                    else:
                        if self.early_stop:
                            stop, patient_count, best_loss, _ = is_early_stop(best_loss=best_loss,
                                                                              current_loss=loss_val,
                                                                              patient_count=patient_count,
                                                                              max_patient=self.patient)
                else:
                    self.model.state_dict = self.model.structure.state_dict()
                    self.model.train_stop_epoch = epoch
                    self.model.train_stop_lr = lr_train
                    torch.save({"model": self.model,
                                "state_dict": self.model.structure.state_dict()
                                },
                               "./models/" + self.model_full_name + ".pth")

                print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f}, test:{:.6f} | lr:{:.6f}'
                      .format(epoch + 1, self.num_epoch, loss_train, loss_val, loss_test, lr_train))
                if stop:
                    print("Early Stopped At Epoch: {}", epoch + 1)
                    break
        elif self.full_data:
            combined_dataset = ConcatDataset([self.train_dataloader.dataset, self.valid_dataloader.dataset])

            # Create a new data loader using the combined dataset
            combined_dataset = DataLoader(combined_dataset, batch_size=32, shuffle=True)
            for epoch in range(self.num_epoch):
                loss_train, lr_train = self.run_epoch(self.model, combined_dataset, optimizer, criterion, scheduler,
                                                      is_training=True, device=self.device)
                scheduler.step(loss_train)
                if self.best_model:
                    if check_best_loss(best_loss=best_loss, loss=loss_train):
                        best_loss = loss_train
                        patient_count = 0
                        self.model.train_stop_lr = lr_train
                        self.model.train_stop_epoch = epoch

                        self.model.state_dict = self.model.structure.state_dict()
                        self.model.train_stop_epoch = epoch
                        self.model.train_stop_lr = lr_train
                        torch.save({"model": self.model,
                                    "state_dict": self.model.structure.state_dict()
                                    },
                                   "./models/" + self.model_full_name + ".pth")
                    else:
                        if self.early_stop:
                            stop, patient_count, best_loss, _ = is_early_stop(best_loss=best_loss,
                                                                              current_loss=loss_train,
                                                                              patient_count=patient_count,
                                                                              max_patient=self.patient)
                else:
                    self.model.state_dict = self.model.structure.state_dict()
                    self.model.train_stop_epoch = epoch
                    self.model.train_stop_lr = lr_train
                    torch.save({"model": self.model,
                                "state_dict": self.model.structure.state_dict()
                                },
                               "./models/" + self.model.full_name + ".pth")

                print('Epoch[{}/{}] | loss train:{:.6f}| lr:{:.6f}'
                      .format(epoch + 1, self.num_epoch, loss_train, lr_train))

                print("patient", patient_count)
                if stop:
                    print("Early Stopped At Epoch: {}", epoch + 1)
                    break
        return self.model

    def eval(self, model):

        train_dataloader, valid_dataloader, test_dataloader = self.prepare_eval_data()
        # Get the current date and time
        current_datetime = datetime.datetime.now()

        save_folder = "./eval/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Format the date and time as a string
        datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Open the file in write mode
        save_path = os.path.join(save_folder, self.model_full_name + "_eval")
        # Open the file in write mode
        with open(save_path, "a") as f:
            f.write(datetime_str)
            f.write(json.dumps(self.cf, indent=4))
            f.write("\n")
            f.write("Epoch:")
            f.write(json.dumps(model.train_stop_epoch, indent=4))
            f.write("\n")
            f.write("Learning rate:")
            f.write(json.dumps(model.train_stop_epoch, indent=4))
            f.write("\n")
            f.write(json.dumps(self.config, indent=4))
            f.write("\n")

        model.structure.to(self.device)
        for i in range(2, 3, 1):
            if i == 0:
                torch.cuda.empty_cache()
                dataloader = train_dataloader
                print_string = "Train evaluate " + self.model_full_name
            if i == 1:
                torch.cuda.empty_cache()
                dataloader = valid_dataloader
                print_string = "Valid evaluate " + self.model_full_name
            elif i == 2:
                torch.cuda.empty_cache()
                dataloader = test_dataloader
                print_string = "Test evaluate " + self.model_full_name
            if "accuracy" or "precision" or "f1" in self.evaluate:
                # Create empty lists to store the true and predicted labels
                true_labels = []
                predicted_labels = []
            total_loss = 0

            target_list = torch.empty(0).to(self.device)
            output_list = torch.empty(0).to(self.device)
            # Iterate over the dataloader
            for x_stock, x_news, labels in dataloader:
                # Move inputs and labels to device
                x_stock = x_stock.to(self.device)
                x_news = x_news.to(self.device)
                labels = labels.to(self.device)
                # Forward pass
                outputs = model.structure(x_stock, x_news)
                target_list = torch.cat([target_list, labels], dim=0)
                output_list = torch.cat([output_list, outputs], dim=0)
                if "accuracy" or "precision" or "f1" in self.evaluate:
                    predicted = (outputs > 0.5).float()
                    # Append true and predicted labels to the respective lists
                    true_labels.extend(labels.cpu().detach().numpy())
                    predicted_labels.extend(predicted.cpu().detach().numpy())

            if "accuracy" or "precision" or "f1" in self.evaluate:
                # Compute classification report
                target_names = ["DOWN", "UP"]  # Add target class names here
                report = classification_report(true_labels, predicted_labels, target_names=target_names)
                # Create the saving folder if it does not exist
                save_folder = "./eval/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                # Open the file in write mode
                save_path = os.path.join(save_folder, self.model_full_name + "_eval")
                # Open the file in write mode
                with open(save_path, "a") as f:
                    # Write the classification report to the file
                    f.write(print_string + "\n" + "Classification report:\n")
                    f.write(report)
                    # Compute confusion matrix
                    cm = confusion_matrix(true_labels, predicted_labels)
                    # Write the confusion matrix to the file
                    f.write("\nConfusion matrix:\n")
                    f.write(np.array2string(cm))
                    f.write("\n")
                    f.write("-" * 100)
                    f.write("\n")
                # Print a message to confirm that the file was written successfully
                print("Results written to " + self.model_full_name + "_eval.txt")

            temp_evaluate = np.array(self.evaluate)

            temp_evaluate = temp_evaluate[temp_evaluate != "accuracy"]
            temp_evaluate = temp_evaluate[temp_evaluate != "precision"]
            temp_evaluate = temp_evaluate[temp_evaluate != "f1"]
            for c in temp_evaluate:
                if "mse" in self.evaluate:
                    criterion = nn.MSELoss()
                    c_name = "MSE"
                elif "mae" in self.evaluate:
                    criterion = nn.L1Loss()
                    c_name = "MAE"
                elif "bce" in self.evaluate:
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
                save_path = os.path.join(save_folder, self.model_full_name + "_eval")
                with open(save_path, "a") as f:
                    # Write the loss to the file
                    f.write(print_string + " " + loss_str + "\n")
                    f.write("-" * 100)
                    f.write("-")
                    f.write("\n")
                # Print a message to confirm that the file was written successfully
                print(f"Loss written to {save_path}.")

    # def prepare_data(self, new_data):
    #     df = u.prepare_stock_dataframe(self.window_size, self.start, self.end, new_data)
    #     num_data_points = df.shape[0]
    #     data_date = df.index.strftime("%Y-%m-%d").tolist()

    #     # Split train val 80%
    #     trainval_test_split_index = int(num_data_points * self.cf["data"]["train_test_split_size"])
    #     # 0 - 80
    #     # train with val dates
    #     train_valid_date = data_date[:trainval_test_split_index]
    #     # test dates splitted
    #     test_date = data_date[trainval_test_split_index:]
    #     # New index for train and valid only
    #     train_valid_split_index = int(len(train_valid_date) * self.cf["data"]["train_val_split_size"])
    #     # Train and valid df split up
    #     # 80 - 100%
    #     # Train and valid dates df split up
    #     train_date = train_valid_date[:train_valid_split_index]
    #     valid_date = train_valid_date[train_valid_split_index:]

    #     # prepare y df
    #     close_df = pd.DataFrame({'close': df['close']})

    #     # prepare X data
    #     X, y = u.prepare_timeseries_dataset(df.to_numpy(), window_size=self.window_size, output_step=self.output_step,
    #                                         dilation=1)
    #     dataset_slicing = X.shape[2]
    #     # whether_X = nlp_u.prepare_whether_data(df, self.window_size, self.start, self.end, new_data, self.output_step)
    #     news_X = nlp_u.prepare_news_data(df, self.symbol, self.window_size, self.start, self.end, self.output_step,
    #                                      self.topk, new_data)
    #     X = np.concatenate((X, news_X), axis=2)
    #     self.num_feature = X.shape[2]
    #     # Split train, validation, and test sets
    #     trainval_test_split_index = int(len(X) * self.cf["data"]["train_test_split_size"])
    #     X_trainval, X_test, y_trainval, y_test = X[:trainval_test_split_index], X[trainval_test_split_index:], y[
    #                                                                                                            :trainval_test_split_index], y[
    #                                                                                                                                         trainval_test_split_index:]

    #     train_valid_split_index = int(len(X_trainval) * self.cf["data"]["train_val_split_size"])
    #     X_train, X_valid, y_train, y_valid = X_trainval[:train_valid_split_index], X_trainval[
    #                                                                                train_valid_split_index:], y_trainval[
    #                                                                                                           :train_valid_split_index], y_trainval[
    #                                                                                                                                      train_valid_split_index:]

    #     # Create StratifiedShuffleSplit object
    #     sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.cf["data"]["train_val_split_size"], random_state=42)

    #     # Use StratifiedShuffleSplit to split train and validation sets
    #     for train_index, valid_index in sss.split(X_trainval, y_trainval):
    #         X_train, X_valid = X_trainval[train_index], X_trainval[valid_index]
    #         y_trainval = y_trainval.astype(int)
    #         y_train, y_valid = y_trainval[train_index], y_trainval[valid_index]

    #     # print("Number of 0s and 1s in y_train:", np.bincount(y_train))
    #     # print("Number of 0s and 1s in y_valid:", np.bincount(y_valid))
    #     # save train data

    #     # set the file paths
    #     X_train_file = './dataset/X_train_' + self.model_full_name + '.npy'
    #     X_valid_file = './dataset/X_valid_' + self.model_full_name + '.npy'
    #     X_test_file = './dataset/X_test_' + self.model_full_name + '.npy'
    #     # set the file paths
    #     y_train_file = './dataset/y_train_' + self.model_full_name + '.npy'
    #     y_valid_file = './dataset/y_valid_' + self.model_full_name + '.npy'
    #     y_test_file = './dataset/y_test_' + self.model_full_name + '.npy'
    #     # check if the files already exist, and delete them if they do
    #     if os.path.exists(X_train_file):
    #         os.remove(X_train_file)
    #     if os.path.exists(X_valid_file):
    #         os.remove(X_valid_file)
    #     if os.path.exists(X_test_file):
    #         os.remove(X_test_file)
    #     if os.path.exists(y_train_file):
    #         os.remove(y_train_file)
    #     if os.path.exists(y_valid_file):
    #         os.remove(y_valid_file)
    #     if os.path.exists(y_test_file):
    #         os.remove(y_test_file)
    #     # save the data
    #     np.save(X_train_file, X_train)
    #     np.save(X_valid_file, X_valid)
    #     np.save(X_test_file, X_test)
    #     np.save(y_train_file, y_train)
    #     np.save(y_valid_file, y_valid)
    #     np.save(y_test_file, y_test)

    #     # create datasets and dataloaders
    #     train_dataset = MyDataset(X_train, y_train, dataset_slicing)
    #     valid_dataset = MyDataset(X_valid, y_valid, dataset_slicing)
    #     test_dataset = MyDataset(X_test, y_test, dataset_slicing)
    #     self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
    #     self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
    #     self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)

    def prepare_data(self, new_data):
        df = u.prepare_stock_dataframe(self.window_size, self.start, self.end, new_data)
        num_data_points = df.shape[0]
        data_date = df.index.strftime("%Y-%m-%d").tolist()

        # Split train-val and test sets
        trainval_test_split_index = int(num_data_points * self.cf["data"]["train_test_split_size"])
        train_valid_date = data_date[:trainval_test_split_index]
        test_date = data_date[trainval_test_split_index:]

        # Prepare y df
        close_df = pd.DataFrame({'close': df['close']})

        # Prepare X data
        X, y = u.prepare_timeseries_dataset(df.to_numpy(), window_size=self.window_size, output_step=self.output_step,
                                            dilation=1)
        dataset_slicing = X.shape[2]
        news_X = nlp_u.prepare_news_data(df, self.symbol, self.window_size, self.start, self.end, self.output_step,
                                         self.topk, new_data)
        X = np.concatenate((X, news_X), axis=2)
        self.num_feature = X.shape[2]

        # Split train-val and test sets
        X_trainval, X_test, y_trainval, y_test = X[:trainval_test_split_index], X[trainval_test_split_index:], y[
                                                                                                               :trainval_test_split_index], y[
                                                                                                                                            trainval_test_split_index:]

        # Perform time series cross-validation with stratification on train-val set
        tscv = TimeSeriesSplit(n_splits=5)

        # Create lists to store the stratified train and validation sets
        X_train_list, X_valid_list = [], []
        y_train_list, y_valid_list = [], []

        # Loop over the time series cross-validation splits
        for train_index, valid_index in tscv.split(X_trainval, y_trainval):
            X_train_fold, X_valid_fold = X_trainval[train_index], X_trainval[valid_index]
            y_train_fold, y_valid_fold = y_trainval[train_index], y_trainval[valid_index]

            X_train_list.append(X_train_fold)
            X_valid_list.append(X_valid_fold)
            y_train_list.append(y_train_fold)
            y_valid_list.append(y_valid_fold)

        # Convert the lists to numpy arrays
        X_train = np.concatenate(X_train_list, axis=0)
        X_valid = np.concatenate(X_valid_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_valid = np.concatenate(y_valid_list, axis=0)

        # Create StratifiedShuffleSplit object for equal class distribution
        sss_train = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.cf["data"]["train_val_split_size"],
                                           random_state=42)

        # Use StratifiedShuffleSplit to split train and validation sets
        for train_index, valid_index in sss_train.split(X_train, y_train):
            X_train, X_valid = X_train[train_index], X_train[valid_index]
            y_train, y_valid = y_train[train_index], y_train[valid_index]

        # Count the class distribution for each set
        train_class_counts = np.bincount(y_train[:, 0])
        valid_class_counts = np.bincount(y_valid[:, 0])
        test_class_counts = np.bincount(y_test[:, 0])

        print("Train set - Class 0 count:", train_class_counts[0], ", Class 1 count:", train_class_counts[1])
        print("Validation set - Class 0 count:", valid_class_counts[0], ", Class 1 count:", valid_class_counts[1])
        print("Test set - Class 0 count:", test_class_counts[0], ", Class 1 count:", test_class_counts[1])

        # Save train and validation data
        X_train_file = './dataset/X_train_' + self.model_full_name + '.npy'
        X_valid_file = './dataset/X_valid_' + self.model_full_name + '.npy'
        y_train_file = './dataset/y_train_' + self.model_full_name + '.npy'
        y_valid_file = './dataset/y_valid_' + self.model_full_name + '.npy'

        if os.path.exists(X_train_file):
            os.remove(X_train_file)
        if os.path.exists(X_valid_file):
            os.remove(X_valid_file)
        if os.path.exists(y_train_file):
            os.remove(y_train_file)
        if os.path.exists(y_valid_file):
            os.remove(y_valid_file)

        np.save(X_train_file, X_train)
        np.save(X_valid_file, X_valid)
        np.save(y_train_file, y_train)
        np.save(y_valid_file, y_valid)

        # Create datasets and dataloaders for train and validation sets
        train_dataset = MyDataset(X_train, y_train, dataset_slicing)
        valid_dataset = MyDataset(X_valid, y_valid, dataset_slicing)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)

        # Save test data
        X_test_file = './dataset/X_test_' + self.model_full_name + '.npy'
        y_test_file = './dataset/y_test_' + self.model_full_name + '.npy'

        if os.path.exists(X_test_file):
            os.remove(X_test_file)
        if os.path.exists(y_test_file):
            os.remove(y_test_file)

        np.save(X_test_file, X_test)
        np.save(y_test_file, y_test)

        # Create test dataset and dataloader
        test_dataset = MyDataset(X_test, y_test, dataset_slicing)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)

    def prepare_eval_data(self):
        # load train data
        X_train = np.load('./dataset/X_train_' + self.model_full_name + '.npy', allow_pickle=True)
        y_train = np.load('./dataset/y_train_' + self.model_full_name + '.npy', allow_pickle=True)
        X_valid = np.load('./dataset/X_valid_' + self.model_full_name + '.npy', allow_pickle=True)
        y_valid = np.load('./dataset/y_valid_' + self.model_full_name + '.npy', allow_pickle=True)
        X_test = np.load('./dataset/X_test_' + self.model_full_name + '.npy', allow_pickle=True)
        y_test = np.load('./dataset/y_test_' + self.model_full_name + '.npy', allow_pickle=True)
        dataset_slicing = 39
        train_dataset = MyDataset(X_train, y_train, dataset_slicing)
        valid_dataset = MyDataset(X_valid, y_valid, dataset_slicing)
        test_dataset = MyDataset(X_test, y_test, dataset_slicing)

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)

        return self.train_dataloader, self.valid_dataloader, self.test_dataloader
        # train_date, valid_date, test_date

    def run_epoch(self, model, dataloader, optimizer, criterion, scheduler, is_training, device):
        epoch_loss = 0

        weight_decay = 0.001
        if is_training:
            model.structure.train()
        else:
            model.structure.eval()

        # create a tqdm progress bar
        dataloader = tqdm(dataloader)
        for idx, (x_stock, x_news, y) in enumerate(dataloader):
            if is_training:
                optimizer.zero_grad()
            batch_size = x_stock.shape[0]
            # print(x.shape)
            x_stock = x_stock.to(device)
            x_news = x_news.to(device)
            y = y.to(device)
            out = model.structure(x_stock, x_news)
            # Compute accuracy
            predictions = torch.argmax(out, dim=1)
            correct = (predictions == y).sum().item()
            accuracy = correct / batch_size  # Multiply by 100 to get percentage

            # Print loss and accuracy
            print("Accuracy: {:.2f}%".format(accuracy))
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
