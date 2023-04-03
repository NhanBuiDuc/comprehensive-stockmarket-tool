from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from config import config as cf
import numpy as np
import torch.nn as nn
import torch
from model import LSTM_Regression, Assembly_regression, Movement_3,Movement_7,Movement_14
from model import LSTM_Classifier_14
from model import LSTM_Classifier_1

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


    print('Regressor MSE Valid loss:{:.6f}%, MAE Valid loss:{:.6f}%, RMSE Valid loss:{:.6f}%'
                    .format(MSE_val_loss * 100 / num_data, 
                            MAE_val_loss * 100 / num_data,
                            RMSE_val_loss * 100 / num_data))
    print('Regressor MSE Valid loss:{:.6f}, MAE Valid loss:{:.6f}, RMSE Valid loss:{:.6f}'
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
    model = Assembly_regression()
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


    print('Assemble MSE Valid loss:{:.6f}%, MAE Valid loss:{:.6f}%, RMSE Valid loss:{:.6f}%'
                    .format(MSE_val_loss * 100 / num_data, 
                            MAE_val_loss * 100 / num_data,
                            RMSE_val_loss * 100 / num_data))
    print('Assemble MSE Valid loss:{:.6f}, MAE Valid loss:{:.6f}, RMSE Valid loss:{:.6f}'
                    .format(MSE_val_loss, 
                            MAE_val_loss,
                            RMSE_val_loss))
    # print('Valid loss:{:.6f}%'
    #                 .format(val_loss))
    return MSE_val_loss, MAE_val_loss, RMSE_val_loss, y_true, y_pred


def evalute_classifier_1(dataset_val):
    batch_size = cf["training"]["lstm_classification1"]["batch_size"]
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = LSTM_Classifier_1()
    checkpoint = torch.load('./models/lstm_classification_1')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_classification1"]["batch_size"], shuffle=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    criterion = nn.BCELoss()
    
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
        loss = criterion(out, y)
        binary_cross_entropy_val_loss += loss.detach().item()  / batchsize


    print('Binary cross-entropy1 Valid loss:{:.6f}%'
                    .format(binary_cross_entropy_val_loss * 100 / num_data))
    print('Binary cross-entropy1 Valid loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score / num_data))
    # print('Valid loss:{:.6f}%'
    #                 .format(val_loss))
    return binary_cross_entropy_val_loss

def evalute_classifier_7(dataset_val):
    batch_size = cf["training"]["lstm_classification7"]["batch_size"]
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = LSTM_Classifier_14()
    checkpoint = torch.load('./models/lstm_classification_7')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_classification7"]["batch_size"], shuffle=True)
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
        loss = criterion(out, y)
        binary_cross_entropy_val_loss += loss.detach().item()  / batchsize


    print('Binary cross-entropy 7 Valid loss:{:.6f}%'
                    .format(binary_cross_entropy_val_loss * 100 / num_data))
    print('Binary cross-entropy 7 Valid loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score/num_data))
    # print('Valid loss:{:.6f}%'
    #                 .format(val_loss))
    return binary_cross_entropy_val_loss

def evalute_classifier_14(dataset_val):
    batch_size = cf["training"]["lstm_classification14"]["batch_size"]
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = LSTM_Classifier_14()
    checkpoint = torch.load('./models/lstm_classification_14')
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
        loss = criterion(out, y)
        binary_cross_entropy_val_loss += loss.detach().item()  / batchsize


    print('Binary cross-entropy14 Valid loss:{:.6f}%'
                    .format(binary_cross_entropy_val_loss * 100 / num_data))
    print('Binary cross-entropy14 Valid loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score/num_data))
    # print('Valid loss:{:.6f}%'
    #                 .format(val_loss))
    return binary_cross_entropy_val_loss


def evalute_Movement_3(dataset_val):
    batch_size = cf["training"]["movement_3"]["batch_size"]
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_3"
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Movement_3()
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_3"]["batch_size"], shuffle=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()    
    binary_cross_entropy_val_loss = 0
    mean_squared_error_val_loss = 0
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
        accuracy_score += torch.sum(prob_predict == prob_true).item()
        y_pred.append(out)
        loss1 = criterion1(out[:, :2], y[:, :2])
        loss2 = criterion2(out[:, :-1], y[:, :-1])
        binary_cross_entropy_val_loss += loss1.detach().item()  / batchsize
        mean_squared_error_val_loss += loss2.detach().item()  / batchsize
    print('Binary movement_3 infer BCE loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score/num_data))
    print('Binary movement_3 infer percentage MSE loss:{:.6f}'
                    .format(mean_squared_error_val_loss))
    return binary_cross_entropy_val_loss, mean_squared_error_val_loss

def evalute_Movement_7(dataset_val):
    batch_size = cf["training"]["movement_7"]["batch_size"]
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_7"
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Movement_7()
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_7"]["batch_size"], shuffle=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()    
    binary_cross_entropy_val_loss = 0
    mean_squared_error_val_loss = 0
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
        accuracy_score += torch.sum(prob_predict == prob_true).item()
        y_pred.append(out)
        loss1 = criterion1(out[:, :2], y[:, :2])
        loss2 = criterion2(out[:, :-1], y[:, :-1])
        binary_cross_entropy_val_loss += loss1.detach().item()  / batchsize
        mean_squared_error_val_loss += loss2.detach().item()  / batchsize
    print('Binary movement_7 infer BCE loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score/num_data))
    print('Binary movement_7 infer percentage MSE loss:{:.6f}'
                    .format(mean_squared_error_val_loss))
    return binary_cross_entropy_val_loss, mean_squared_error_val_loss

def evalute_Movement_14(dataset_val):
    batch_size = cf["training"]["movement_14"]["batch_size"]
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_14"
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Movement_14()
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_14"]["batch_size"], shuffle=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()    
    binary_cross_entropy_val_loss = 0
    mean_squared_error_val_loss = 0
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
        accuracy_score += torch.sum(prob_predict == prob_true).item()
        y_pred.append(out)
        loss1 = criterion1(out[:, :2], y[:, :2])
        loss2 = criterion2(out[:, :-1], y[:, :-1])
        binary_cross_entropy_val_loss += loss1.detach().item()  / batchsize
        mean_squared_error_val_loss += loss2.detach().item()  / batchsize
    print('Binary movement_14 infer BCE loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score/num_data))
    print('Binary movement_14 infer percentage MSE loss:{:.6f}'
                    .format(mean_squared_error_val_loss))
    return binary_cross_entropy_val_loss, mean_squared_error_val_loss