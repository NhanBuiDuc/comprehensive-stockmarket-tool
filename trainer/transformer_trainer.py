from trainer.trainer import Trainer, check_best_loss, is_early_stop

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
from dataset import PriceAndIndicatorsAndNews_TimeseriesDataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import json
import numpy as np
import os
from torch.utils.data import DataLoader
import datetime
import NLP.util as nlp_u
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, StratifiedShuffleSplit
from loss import FocalLoss


class Transformer_trainer(Trainer):
    def __init__(self, new_data=True, full_data=False, mode="train"):
        super(Transformer_trainer, self).__init__()
        self.__dict__.update(self.cf)
        self.config = cf
        # self.symbol = self.cf["alpha_vantage"]["symbol"]
        self.model_type = "transformer"
        self.__dict__.update(self.config["model"])
        self.__dict__.update(self.config["training"])
        self.test_dataloader = None
        self.valid_dataloader = None
        self.train_dataloader = None
        self.full_data = full_data
        self.num_feature = None
        self.new_data = new_data
        self.model_name = f'{self.model_type}_{self.symbol}_w{self.window_size}_o{self.output_step}_d{str(self.data_mode)}'
        self.model_type_dict = self.cf["pytorch_timeseries_model_type_dict"]
        self.model = None
        self.mode = mode
        if self.mode == "train":
            self.prepare_data(self.new_data)
        else:
            self.num_feature = 807
        self.indentify()

    def indentify(self):
        self.model = Model(name=self.model_name, num_feature=self.num_feature, parameters=self.config,
                           model_type=self.model_type)

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
                                   "./models/" + self.model.name + ".pth")
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
                               "./models/" + self.model_name + ".pth")

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
                                   "./models/" + self.model_name + ".pth")
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
                               "./models/" + self.model_name + ".pth")

                print('Epoch[{}/{}] | loss train:{:.6f}| lr:{:.6f}'
                      .format(epoch + 1, self.num_epoch, loss_train, lr_train))

                print("patient", patient_count)
                if stop:
                    print("Early Stopped At Epoch: {}", epoch + 1)
                    break
        return self.model

    def eval(self, model):

        train_dataloader, valid_dataloader, test_dataloader, balancedtest_datataloader = self.prepare_eval_data()
        # Get the current date and time
        current_datetime = datetime.datetime.now()

        save_folder = "./eval/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Format the date and time as a string
        datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Open the file in write mode
        save_path = os.path.join(save_folder, self.model_name + "_eval")
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
        for i in range(0, 4, 1):
            if i == 0:
                torch.cuda.empty_cache()
                dataloader = train_dataloader
                print_string = "Train evaluate " + self.model_name
            if i == 1:
                torch.cuda.empty_cache()
                dataloader = valid_dataloader
                print_string = "Valid evaluate " + self.model_name
            elif i == 2:
                torch.cuda.empty_cache()
                dataloader = test_dataloader
                print_string = "Test evaluate " + self.model_name
            elif i == 3:
                torch.cuda.empty_cache()
                dataloader = balancedtest_datataloader
                print_string = "Balanced Test evaluate " + self.model_name
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
                save_path = os.path.join(save_folder, self.model_name + "_eval")
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
                print("Results written to " + self.model_name + "_eval.txt")

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
                save_path = os.path.join(save_folder, self.model_name + "_eval")
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
        print("Train date from: " + train_valid_date[0] + " to " + train_valid_date[-1])
        print("Test from: " + test_date[0] + " to " + test_date[-1])
        # Prepare X data
        X, y = u.prepare_timeseries_dataset(df.to_numpy(), window_size=self.window_size, output_step=self.output_step,
                                          dilation=1)
        dataset_slicing = X.shape[2]
        if self.data_mode == 1:
            X, _ = nlp_u.prepare_news_data(df, self.symbol, self.window_size, self.start, self.end, self.output_step,
                                     self.topk, new_data)
        elif self.data_mode == 2:
            news_X, _ = nlp_u.prepare_news_data(df, self.symbol, self.window_size, self.start, self.end, self.output_step,
                                          self.topk, new_data)
            # Concatenate X_stocks and news_X
            X = np.concatenate((X, news_X), axis=2)

        self.num_feature = X.shape[2]
        # Split X and y into train, valid, and test datasets
        train_indices = np.where(df.index.isin(train_valid_date))[0]
        test_indices = np.where(df.index.isin(test_date))[0][:-self.output_step]
        
        X_train_valid = X[train_indices]
        X_test = X[test_indices]
        y_train_valid = y[train_indices]
        y_test = y[test_indices]

        # Create StratifiedShuffleSplit object for equal class distribution
        sss_train = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.cf["data"]["train_val_split_size"],
                                           random_state=42)

        # Use StratifiedShuffleSplit to split train and validation sets
        for train_index, valid_index in sss_train.split(X_train, y_train):
            X_train = X_train_valid[train_index]
            X_valid = X_train_valid[valid_index]
            y_train = y_train_valid[train_index]
            y_valid = y_train_valid[valid_index]

        # Count the class distribution for each set
        train_class_counts = np.bincount(y_train[:, 0])
        valid_class_counts = np.bincount(y_valid[:, 0])
        test_class_counts = np.bincount(y_test[:, 0])

        print("Train set - Class 0 count:", train_class_counts[0], ", Class 1 count:", train_class_counts[1])
        print("Validation set - Class 0 count:", valid_class_counts[0], ", Class 1 count:", valid_class_counts[1])
        print("Test set - Class 0 count:", test_class_counts[0], ", Class 1 count:", test_class_counts[1])
        # Save train and validation data
        X_train_file = './dataset/X_train_' + self.model_name + '.npy'
        X_valid_file = './dataset/X_valid_' + self.model_name + '.npy'
        X_test_file = './dataset/X_test_' + self.model_name + '.npy'
        y_train_file = './dataset/y_train_' + self.model_name + '.npy'
        y_valid_file = './dataset/y_valid_' + self.model_name + '.npy'
        y_test_file = './dataset/y_test_' + self.model_name + '.npy'

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
        np.save(X_test_file, X_test)
        np.save(y_train_file, y_train)
        np.save(y_valid_file, y_valid)
        np.save(y_test_file, y_test)

        # Create datasets and dataloaders for train and validation sets
        train_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_train, y_train, dataset_slicing)
        valid_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_valid, y_valid, dataset_slicing)
        test_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_valid, y_valid, dataset_slicing)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)  

    def prepare_eval_data(self):
        # Load train and validation data
        X_train = np.load('./dataset/X_train_' + self.model_name + '.npy', allow_pickle=True)
        y_train = np.load('./dataset/y_train_' + self.model_name + '.npy', allow_pickle=True)
        X_valid = np.load('./dataset/X_valid_' + self.model_name + '.npy', allow_pickle=True)
        y_valid = np.load('./dataset/y_valid_' + self.model_name + '.npy', allow_pickle=True)

        # Load full test data
        X_test_full = np.load('./dataset/X_test_' + self.model_name + '.npy', allow_pickle=True)
        y_test_full = np.load('./dataset/y_test_' + self.model_name + '.npy', allow_pickle=True)
        
        # Balance the test set
        class_0_indices = np.where(y_test_full == 0)[0]
        class_1_indices = np.where(y_test_full == 1)[0]
        min_class_count = min(len(class_0_indices), len(class_1_indices))
        balanced_indices = np.concatenate([class_0_indices[:min_class_count], class_1_indices[:min_class_count]])
        X_test_balanced = X_test_full[balanced_indices]
        y_test_balanced = y_test_full[balanced_indices]


        dataset_slicing = 39

        # Create datasets for train, validation, full test, and balanced test
        train_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_train, y_train, dataset_slicing)
        valid_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_valid, y_valid, dataset_slicing)
        test_full_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_test_full, y_test_full, dataset_slicing)
        test_balanced_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_test_balanced, y_test_balanced, dataset_slicing)

        # Print class distribution for all datasets
        train_class_counts = np.bincount(y_train)
        valid_class_counts = np.bincount(y_valid)
        test_full_class_counts = np.bincount(y_test_full)
        test_balanced_class_counts = np.bincount(y_test_balanced)

        print("Train set - Class 0 count:", train_class_counts[0], ", Class 1 count:", train_class_counts[1])
        print("Validation set - Class 0 count:", valid_class_counts[0], ", Class 1 count:", valid_class_counts[1])
        print("Full Test set - Class 0 count:", test_full_class_counts[0], ", Class 1 count:", test_full_class_counts[1])
        print("Balanced Test set - Class 0 count:", test_balanced_class_counts[0], ", Class 1 count:", test_balanced_class_counts[1])
        # Create dataloaders for train, validation, full test, and balanced test
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
        test_full_dataloader = DataLoader(test_full_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)
        test_balanced_dataloader = DataLoader(test_balanced_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)

        return train_dataloader, valid_dataloader, test_full_dataloader,  test_balanced_dataloader
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
            predictions = torch.argmax(out, dim=1).unsqueeze(1)
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
