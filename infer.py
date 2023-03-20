from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from config import config as cf
import numpy as np
import torch.nn as nn
import torch
from model import LSTM_Regression
from model import LSTM_Binary
def test_random_forest_classfier(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate the F1 score of the model's predictions
    f1 = f1_score(y_test, y_pred)

    return accuracy, f1

def evalute_regression(dataset_val):
    batch_size = cf["training"]["lstm_regression"]["batch_size"]
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = LSTM_Regression()
    checkpoint = torch.load('./models/lstm_regression')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_regression"]["batch_size"], shuffle=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
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

        batchsize = x.shape[0]

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
        
        MSE_val_loss += loss.detach().item()  / batchsize
        MAE_val_loss += loss2.detach().item()  / batchsize
        RMSE_val_loss += loss3.detach().item()  / batchsize


    print('MSE Valid loss:{:.6f}%, MAE Valid loss:{:.6f}%, RMSE Valid loss:{:.6f}%'
                    .format(MSE_val_loss * 100 / num_data, 
                            MAE_val_loss * 100 / num_data,
                            RMSE_val_loss * 100 / num_data))
    print('MSE Valid loss:{:.6f}, MAE Valid loss:{:.6f}, RMSE Valid loss:{:.6f}'
                    .format(MSE_val_loss, 
                            MAE_val_loss,
                            RMSE_val_loss))
    # print('Valid loss:{:.6f}%'
    #                 .format(val_loss))
    return MSE_val_loss, MAE_val_loss, RMSE_val_loss, y_true, y_pred

def evalute_assembly_regression(dataset_val):
    batch_size = cf["training"]["lstm_regression"]["batch_size"]
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = LSTM_Regression()
    checkpoint = torch.load('./models/assembly_regression')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_regression"]["batch_size"], shuffle=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
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

        batchsize = x.shape[0]

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
        
        MSE_val_loss += loss.detach().item()  / batchsize
        MAE_val_loss += loss2.detach().item()  / batchsize
        RMSE_val_loss += loss3.detach().item()  / batchsize


    print('MSE Valid loss:{:.6f}%, MAE Valid loss:{:.6f}%, RMSE Valid loss:{:.6f}%'
                    .format(MSE_val_loss * 100 / num_data, 
                            MAE_val_loss * 100 / num_data,
                            RMSE_val_loss * 100 / num_data))
    print('MSE Valid loss:{:.6f}, MAE Valid loss:{:.6f}, RMSE Valid loss:{:.6f}'
                    .format(MSE_val_loss, 
                            MAE_val_loss,
                            RMSE_val_loss))
    # print('Valid loss:{:.6f}%'
    #                 .format(val_loss))
    return MSE_val_loss, MAE_val_loss, RMSE_val_loss, y_true, y_pred


def evalute_binary1(dataset_val):
    batch_size = cf["training"]["lstm_classification1"]["batch_size"]
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = LSTM_Binary()
    checkpoint = torch.load('./models/lstm_binary1')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_classification1"]["batch_size"], shuffle=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion = nn.BCELoss()
    # criterion2 = nn.L1Loss()
    # criterion3 = nn.NLLLoss()
    
    binary_cross_entropy_val_loss = 0
    accuracy_score = 0
    y_true = []
    y_pred = []
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):

        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")
        y_true.append(y)

        out = model(x)
        _, prob_predict = torch.max(out, dim=1)
        _, prob_true = torch.max(y, dim=1)
        accuracy_score+= torch.sum(prob_predict == prob_true).item()
        y_pred.append(out)

        # y = torch.tensor(y).to("cuda")
        # out = torch.tensor(out).to("cuda")
        loss = criterion(out, y)
        # loss2 = criterion2(out, y)
        # loss3 = torch.sqrt(loss)
        
        binary_cross_entropy_val_loss += loss.detach().item()  / batchsize
        # MAE_val_loss += loss2.detach().item()  / batchsize
        # RMSE_val_loss += loss3.detach().item()  / batchsize


    print('Binary cross-entropy1 Valid loss:{:.6f}%'
                    .format(binary_cross_entropy_val_loss * 100 / num_data))
    print('Binary cross-entropy1 Valid loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score / num_data))
    # print('Valid loss:{:.6f}%'
    #                 .format(val_loss))
    return binary_cross_entropy_val_loss


def evalute_binary14(dataset_val):
    batch_size = cf["training"]["lstm_classification14"]["batch_size"]
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = LSTM_Binary()
    checkpoint = torch.load('./models/lstm_binary14')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_classification14"]["batch_size"], shuffle=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion = nn.BCELoss()
    # criterion2 = nn.L1Loss()
    # criterion3 = nn.NLLLoss()
    
    binary_cross_entropy_val_loss = 0
    accuracy_score = 0
    y_true = []
    y_pred = []
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):

        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")
        y_true.append(y)

        out = model(x)
        _, prob_predict = torch.max(out, dim=1)
        _, prob_true = torch.max(y, dim=1)
        accuracy_score+= torch.sum(prob_predict == prob_true).item()
        y_pred.append(out)

        # y = torch.tensor(y).to("cuda")
        # out = torch.tensor(out).to("cuda")
        loss = criterion(out, y)
        # loss2 = criterion2(out, y)
        # loss3 = torch.sqrt(loss)
        
        binary_cross_entropy_val_loss += loss.detach().item()  / batchsize
        # MAE_val_loss += loss2.detach().item()  / batchsize
        # RMSE_val_loss += loss3.detach().item()  / batchsize


    print('Binary cross-entropy14 Valid loss:{:.6f}%'
                    .format(binary_cross_entropy_val_loss * 100 / num_data))
    print('Binary cross-entropy14 Valid loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score/num_data))
    # print('Valid loss:{:.6f}%'
    #                 .format(val_loss))
    return binary_cross_entropy_val_loss