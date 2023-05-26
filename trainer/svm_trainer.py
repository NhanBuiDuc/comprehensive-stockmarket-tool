from sklearn.utils import shuffle

from trainer.trainer import Trainer, run_epoch, check_best_loss, is_early_stop

import pandas as pd
import torch
from torch.utils.data import ConcatDataset
from model import Model
from configs.svm_config import svm_cf as cf
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import util as u
from dataset import PriceAndIndicatorsAndNews_Dataset
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
import joblib


class svm_trainer(Trainer):
    def __init__(self, model_name, new_data=True, full_data=False, num_feature=None, config=None, model_type=None,
                 model_full_name=None,
                 model=None, mode="train", 
                 data_mode=3):
        super(svm_trainer, self).__init__()
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
        self.model_type = "svm"
        self.model_type_dict = self.cf["tensorflow_timeseries_model_type_dict"]
        self.model = model
        self.mode = mode
        self.data_mode = data_mode 
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



        if not self.full_data:
            if self.data_mode == 0:
                X_train = self.train_dataloader.dataset.x_price
                y_train = self.train_dataloader.dataset.Y
                x_val = self.valid_dataloader.dataset.x_price
                y_val = self.valid_dataloader.dataset.Y
                self.num_feature = X_train.shape[-1]
            elif self.data_mode == 1:
                X_train = self.train_dataloader.dataset.x_stock
                X_val = self.valid_dataloader.dataset.x_stock
                y_train = self.train_dataloader.dataset.Y
                y_val = self.valid_dataloader.dataset.Y
                self.num_feature = X_train.shape[-1]
            elif self.data_mode == 2:
                X_train = self.train_dataloader.dataset.X
                y_train = self.train_dataloader.dataset.Y
                x_val = self.valid_dataloader.dataset.X
                y_val = self.valid_dataloader.dataset.Y
                self.num_feature = X_train.shape[-1]
            self.model.structure.fit(X_train, y_train)

            torch.save({"model": self.model,
            "state_dict": []
            },
            "./models/" + self.model.full_name + ".pkl")
        elif self.full_data:
            self.combined_dataset = ConcatDataset([self.train_dataloader.dataset, self.valid_dataloader.dataset])

            if self.data_mode == 0:
                X_train = self.combined_dataset.dataset.x_price
                y_train = self.combined_dataset.dataset.Y
                x_val = self.valid_dataloader.dataset.x_price
                y_val = self.valid_dataloader.dataset.Y
            elif self.data_mode == 1:
                X_train = self.combined_dataset.dataset.x_stock
                X_val = self.combined_dataset.dataset.x_stock
                y_train = self.train_dataloader.dataset.Y
                y_val = self.valid_dataloader.dataset.Y
            elif self.data_mode == 2:
                X_train = self.combined_dataset.dataset.X
                y_train = self.combined_dataset.dataset.Y
                x_val = self.valid_dataloader.dataset.X
                y_val = self.valid_dataloader.dataset.Y


            self.model.structure.fit(X_train, y_train)
            torch.save({"model": self.model,
            "state_dict": []
            },
            "./models/" + self.model.full_name + ".pkl")

    def eval(self, model):

        train_dataloader, valid_dataloader, test_dataloader = self.prepare_eval_data()
        if self.data_mode == 0:
            train_dataloader.dataset.X = train_dataloader.dataset.x_price
            valid_dataloader.dataset.X = valid_dataloader.dataset.x_price
            test_dataloader.dataset.X = test_dataloader.dataset.x_price
            self.num_feature = train_dataloader.dataset.X.shape[-1]
        elif self.data_mode == 1:
            train_dataloader.dataset.X = train_dataloader.dataset.x_stock
            valid_dataloader.dataset.X = valid_dataloader.dataset.x_stock
            test_dataloader.dataset.X = test_dataloader.dataset.x_stock
            self.num_feature = train_dataloader.dataset.X.shape[-1]
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
        for i in range(0, 3, 1):
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
            for x, labels in dataloader:
                # Move inputs and labels to device
                # Forward pass
                # x = x.to(self.device)
                x = x.cpu().detach().numpy()
                labels = labels.to(self.device)
                # labels = labels.cpu().detach().numpy()
                outputs = model.predict(x).to(self.device)
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

    def prepare_data(self, new_data):
        df = u.prepare_stock_dataframe(self.symbol, self.window_size, self.start, self.end, new_data)
        num_data_points = df.shape[0]
        data_date = df.index.strftime("%Y-%m-%d").tolist()

        # Split train-val and test sets
        trainval_test_split_index = int(num_data_points * self.cf["data"]["train_test_split_size"])
        train_valid_date = data_date[:trainval_test_split_index]
        test_date = data_date[trainval_test_split_index:]

        # Prepare y
        y = u.prepare_data_y_trend(df.to_numpy(), output_step=self.output_step)
        y = np.array(y, dtype=int)

        # Prepare X
        X_stocks = np.array(df.values)[:-self.output_step]
        _, news_X = nlp_u.prepare_news_data(df, self.symbol, self.window_size, self.start, self.end, self.output_step,
                                            self.topk, new_data)
        news_X = news_X[:-self.output_step]
        self.num_feature = 807

        # Concatenate X_stocks and news_X
        X = np.concatenate((X_stocks, news_X), axis=1)

        # Split X and y into train, valid, and test datasets
        trainval_test_split_index = int(X.shape[0] * self.cf["data"]["train_test_split_size"])
        train_valid_indices = np.arange(trainval_test_split_index)
        test_indices = np.arange(trainval_test_split_index, X.shape[0])

        X_train_valid = X[train_valid_indices]
        X_test = X[test_indices]
        y_train_valid = y[train_valid_indices]
        y_test = y[test_indices]

        # Perform stratified splitting on train and valid datasets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.cf["data"]["train_val_split_size"], random_state=42)
        for train_index, valid_index in sss.split(X_train_valid, y_train_valid):
            X_train = X_train_valid[train_index]
            X_valid = X_train_valid[valid_index]
            y_train = y_train_valid[train_index]
            y_valid = y_train_valid[valid_index]

        # Print class distribution in train, valid, and test sets
        train_class_counts = np.bincount(y_train)
        valid_class_counts = np.bincount(y_valid)
        test_class_counts = np.bincount(y_test)
        print("Train set - Class 0 count:", train_class_counts[0], ", Class 1 count:", train_class_counts[1])
        print("Validation set - Class 0 count:", valid_class_counts[0], ", Class 1 count:", valid_class_counts[1])
        print("Test set - Class 0 count:", test_class_counts[0], ", Class 1 count:", test_class_counts[1])

        # Save train, valid, and test datasets
        X_train_file = './dataset/X_train_' + self.model_full_name + '.npy'
        X_valid_file = './dataset/X_valid_' + self.model_full_name + '.npy'
        X_test_file = './dataset/X_test_' + self.model_full_name + '.npy'
        y_train_file = './dataset/y_train_' + self.model_full_name + '.npy'
        y_valid_file = './dataset/y_valid_' + self.model_full_name + '.npy'
        y_test_file = './dataset/y_test_' + self.model_full_name + '.npy'

        np.save(X_train_file, X_train)
        np.save(X_valid_file, X_valid)
        np.save(X_test_file, X_test)
        np.save(y_train_file, y_train)
        np.save(y_valid_file, y_valid)
        np.save(y_test_file, y_test)

        # Create dataloaders for train, valid, and test sets
        train_dataset = PriceAndIndicatorsAndNews_Dataset(X_train, y_train, 39)
        valid_dataset = PriceAndIndicatorsAndNews_Dataset(X_valid, y_valid, 39)
        test_dataset = PriceAndIndicatorsAndNews_Dataset(X_test, y_test, 39)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)




    def prepare_eval_data(self):
        # load train data
        X_train = np.load('./dataset/X_train_' + self.model_full_name + '.npy', allow_pickle=True)
        y_train = np.load('./dataset/y_train_' + self.model_full_name + '.npy', allow_pickle=True)
        X_valid = np.load('./dataset/X_valid_' + self.model_full_name + '.npy', allow_pickle=True)
        y_valid = np.load('./dataset/y_valid_' + self.model_full_name + '.npy', allow_pickle=True)
        X_test = np.load('./dataset/X_test_' + self.model_full_name + '.npy', allow_pickle=True)
        y_test = np.load('./dataset/y_test_' + self.model_full_name + '.npy', allow_pickle=True)

        self.num_feature = X_train.shape[-1]
        dataset_slicing = 39
        train_dataset = PriceAndIndicatorsAndNews_Dataset(X_train, y_train, dataset_slicing)
        valid_dataset = PriceAndIndicatorsAndNews_Dataset(X_valid, y_valid, dataset_slicing)
        test_dataset = PriceAndIndicatorsAndNews_Dataset(X_test, y_test, dataset_slicing)

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
