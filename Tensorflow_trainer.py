from trainer import Trainer, prepare_data
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


class TensorflowTrainer(Trainer):
    def __init__(self):
        super(TensorflowTrainer, self).__init__()

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
        weight_decay = training_param["weight_decay"]
        model_param = cf["model"][model_name]

        # if "svm" or "svm".upper() in model_name:
        #     model_type = self.model_type_dict[6]
        # elif "random_forest" or "random_forest".upper() in model_name:
        #     model_type = self.model_type_dict[7]

        for t in self.pytorch_timeseries_model_type_dict:
            if t or model_type.upper() in model_name:
                model_type = t
        if model_type == "svm" or "random_forest":
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
            optimizer = optim.Adam(model.structure.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif "sgd" in optimizer:
            optimizer = optim.SGD(model.structure.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
                                             is_training=True, device=device)
            loss_val, lr_val = run_epoch(model, valid_dataloader, optimizer, criterion, scheduler,
                                         is_training=False, device=device)
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
