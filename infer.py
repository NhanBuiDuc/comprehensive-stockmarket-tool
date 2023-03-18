from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from config import config as cf
import numpy as np
import torch.nn as nn
import torch
from model import LSTM_Regression
def test_random_forest_classfier(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate the F1 score of the model's predictions
    f1 = f1_score(y_test, y_pred)

    return accuracy, f1

def evalute(dataset_train, dataset_val):
    batch_size = cf["training"]["lstm_regression"]["batch_size"]
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = LSTM_Regression()
    checkpoint = torch.load('./models/best_model')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["lstm_regression"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_regression"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    criterion3 = nn.NLLLoss()
    
    MSE_val_loss = 0
    MAE_val_loss = 0
    RMSE_val_loss = 0
    y_true = []
    y_pred = []
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):

        x = x.to("cuda")
        y = y.to("cuda")
        y_true.append(y)

        out = model(x)
        y_pred.append(out)

        # y = torch.tensor(y).to("cuda")
        # out = torch.tensor(out).to("cuda")
        loss = criterion(out, y)
        loss2 = criterion2(out, y)
        loss3 = torch.sqrt(loss)
        
        MSE_val_loss += loss.detach().item()
        MAE_val_loss += loss2.detach().item()
        RMSE_val_loss += loss3.detach().item()


    print('MSE Valid loss:{:.6f}%, MAE Valid loss:{:.6f}%, RMSE Valid loss:{:.6f}%'
                    .format(MSE_val_loss * 100 / (len(val_dataloader) * batch_size), 
                            MAE_val_loss * 100 / (len(val_dataloader) * batch_size),
                            RMSE_val_loss * 100 / (len(val_dataloader) * batch_size)))
    print('MSE Valid loss:{:.6f}, MAE Valid loss:{:.6f}, RMSE Valid loss:{:.6f}'
                    .format(MSE_val_loss, 
                            MAE_val_loss,
                            RMSE_val_loss))
    # print('Valid loss:{:.6f}%'
    #                 .format(val_loss))
    return MSE_val_loss, MAE_val_loss, RMSE_val_loss, y_true, y_pred