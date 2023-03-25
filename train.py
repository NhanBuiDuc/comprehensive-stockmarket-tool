from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from config import config as cf
from sklearn.model_selection import train_test_split
import infer
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import utils
import torch
import model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import sys
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, OneCycleLR
from sklearn.svm import SVC

def train_assemble_model(dataset_train, dataset_val):
    lr=cf["training"]["lstm_regression"]["learning_rate"]
    epochs=cf["training"]["lstm_regression"]["num_epoch"]
    regression_model = model.Assembly_regression()
    regression_model.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = nn.MSELoss()

    optimizer = optim.Adam(regression_model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
    # OneCycleLR is a PyTorch learning rate scheduler that implements the 1Cycle policy.
    # It adjusts the learning rate during training in a cyclical manner, gradually increasing the learning rate to a maximum value,
    # and then decreasing it back to the initial value.
    # The learning rate is increased in the first half of the cycle and decreased in the second half.
    
    # optimizer: the optimizer for the model.
    # max_lr: the maximum learning rate to be used during training.
    # epochs: the total number of epochs to train the model for.
    # steps_per_epoch: the number of steps (batches) per epoch.

    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_dataloader))
    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["lstm_regression"]["patient"]
    patient_count = 0
    stopped_epoch = 0
    # begin training
    for epoch in range(cf["training"]["lstm_regression"]["num_epoch"]):
        loss_train, lr_train = run_epoch(regression_model,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(regression_model, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step()
        # loss_train_history.append(loss_train)
        # loss_val_history.append(loss_val)
        # lr_train_history.append(lr_train)
        # lr_val_history.append(lr_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient = 0
            save_best_model(model=regression_model, name = "assembly_regression", num_epochs=epoch, optimizer=optimizer, val_loss=loss_val, training_loss=loss_train, learning_rate=lr_train)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["lstm_regression"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch)
            stopped_epoch = patient_count
            break
    return regression_model

def train_random_forest_classfier(X_train, y_train, X_val, y_val, X_test, y_test):
    day_steps = 14

    # train a random forest classifier using scikit-learn
    model = RandomForestClassifier(n_estimators=2000, random_state=42)
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)
    y_test = np.squeeze(y_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # calculate the accuracy of the predictions
    accuracy = accuracy_score(y_val, y_pred)
    print("Val Accuracy:", accuracy)
    # Print the F1 score
    f1 = f1_score(y_val, y_pred, average="micro")
    print("Val F1 score: {:.2f}".format(f1))
    print("Weights:", model.feature_importances_)
    
    y_pred_test = model.predict(X_test)
    print(y_test.dtype)
    print(y_pred_test.dtype)
    # calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred_test)
    # Calculate the F1 score of the model's predictions
    f1 = f1_score(y_test, y_pred_test, average="micro")

    print("Test Accuracy:", accuracy)
    # Print the F1 score
    print("Test F1 score: {:.2f}".format(f1))


    return model

def train_svm_classfier(X_train, y_train, X_val, y_val, X_test, y_test):
    day_steps = 14

    # train a random forest classifier using scikit-learn
    model = SVC(kernel='linear', C=1)
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)
    y_test = np.squeeze(y_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # calculate the accuracy of the predictions
    accuracy = accuracy_score(y_val, y_pred)
    print("Val Accuracy:", accuracy)
    # Print the F1 score
    f1 = f1_score(y_val, y_pred, average="micro")
    print("Val F1 score: {:.2f}".format(f1))
    print("Weights:", model.class_weight_)
    
    y_pred_test = model.predict(X_test)
    print(y_test.dtype)
    print(y_pred_test.dtype)
    # calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred_test)
    # Calculate the F1 score of the model's predictions
    f1 = f1_score(y_test, y_pred_test, average="micro")

    print("Test Accuracy:", accuracy)
    # Print the F1 score
    print("Test F1 score: {:.2f}".format(f1))


    return model


def train_random_forest_regressior(X_train, y_train):
    y_train = np.array(y_train)

    y_train = (utils.diff(y_train))

    # Create a Random Forest Regression model with 100 trees
    model = RandomForestRegressor(n_estimators=1000, random_state=42)

    # Train the model on the training data
    model.fit(X_train[:-1], y_train)
    return model

def train_LSTM_regression(dataset_train, dataset_val, is_training=True):

    regression_model = model.LSTM_Regression(
        input_size = cf["model"]["lstm_regression"]["input_size"],
        window_size = cf["data"]["window_size"],
        hidden_layer_size = cf["model"]["lstm_regression"]["lstm_size"], 
        num_layers = cf["model"]["lstm_regression"]["num_lstm_layers"], 
        output_size = cf["model"]["lstm_regression"]["output_dates"],
        dropout = cf["model"]["lstm_regression"]["dropout"]
    )
    regression_model.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["lstm_regression"]["batch_size"])
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_regression"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = nn.MSELoss()

    """
    betas: Adam optimizer uses exponentially decaying averages of past gradients to update the parameters.
    betas is a tuple of two values that control the decay rates for these moving averages.
    The first value (default 0.9) controls the decay rate for the moving average of gradients,
    and the second value (default 0.999) controls the decay rate for the moving average of squared gradients.
    Higher values of beta will result in a smoother update trajectory and may help to avoid oscillations in the optimization process, 
    but may also lead to slower convergence.
    eps: eps is a small constant added to the denominator of the Adam update formula to avoid division by zero.
    It is typically set to a very small value (e.g. 1e-8 or 1e-9) to ensure numerical stability.
    """
    optimizer = optim.Adam(regression_model.parameters(), lr=cf["training"]["lstm_regression"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)

    """
    For example, suppose step_size=10 and gamma=0.1.
    This means that the learning rate will be multiplied by 0.1 every 10 epochs.
    If the initial learning rate is 0.1, then the learning rate will be reduced to 0.01 after 10 epochs, 0.001 after 20 epochs, and so on.
    """

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500, verbose=True)

    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["lstm_regression"]["patient"]
    patient_count = 0
    # begin training
    for epoch in range(cf["training"]["lstm_regression"]["num_epoch"]):
        loss_train, lr_train = run_epoch(regression_model,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(regression_model, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step(loss_val)
        # loss_train_history.append(loss_train)
        # loss_val_history.append(loss_val)
        # lr_train_history.append(lr_train)
        # lr_val_history.append(lr_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient_count = 0
            save_best_model(model=regression_model, name = "lstm_regression", num_epochs=epoch, optimizer=optimizer, val_loss=loss_val, training_loss=loss_train, learning_rate=lr_train)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["lstm_regression"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch)
            stopped_epoch = patient_count
            break
    return regression_model

def train_LSTM_classifier_1(dataset_train, dataset_val, is_training=True):

    binary_model = model.LSTM_Classifier_1(
        input_size = cf["model"]["lstm_classification1"]["input_size"],
        window_size = cf["data"]["window_size"],
        hidden_layer_size = cf["model"]["lstm_classification1"]["lstm_size"], 
        num_layers = cf["model"]["lstm_classification1"]["num_lstm_layers"], 
        output_size = cf["model"]["lstm_classification1"]["output_dates"],
        dropout = cf["model"]["lstm_classification1"]["dropout"]
    )
    binary_model.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["lstm_classification1"]["batch_size"])
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_classification1"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = nn.BCELoss()

    # optimizer = optim.Adam(binary_model.parameters(), lr=cf["training"]["lstm_classification1"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
    optimizer = optim.SGD(binary_model.parameters(), lr=cf["training"]["lstm_classification1"]["learning_rate"], momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500, verbose=True)

    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["lstm_classification1"]["patient"]
    patient_count = 0
    stopped_epoch = 0
    # begin training
    for epoch in range(cf["training"]["lstm_classification1"]["num_epoch"]):
        loss_train, lr_train = run_epoch(binary_model,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(binary_model, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step(loss_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient_count = 0
            save_best_model(model=binary_model, name="lstm_classification_1", num_epochs=epoch, optimizer=optimizer, val_loss=loss_val, training_loss=loss_train, learning_rate=lr_train)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["lstm_classification1"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch)
            stopped_epoch = patient_count
            break
    return binary_model
def train_LSTM_classifier_14(dataset_train, dataset_val, is_training=True):

    binary_model = model.LSTM_Classifier_14(
        input_size = cf["model"]["lstm_classification14"]["input_size"],
        window_size = cf["data"]["window_size"],
        hidden_layer_size = cf["model"]["lstm_classification14"]["lstm_size"], 
        num_layers = cf["model"]["lstm_classification14"]["num_lstm_layers"], 
        output_size = cf["model"]["lstm_classification14"]["output_dates"],
        dropout = cf["model"]["lstm_classification14"]["dropout"]
    )
    binary_model.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["lstm_classification14"]["batch_size"])
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_classification14"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = nn.BCELoss()
    optimizer = optim.SGD(binary_model.parameters(), lr=cf["training"]["lstm_classification14"]["learning_rate"], momentum=0.9)

    """
    For example, suppose step_size=10 and gamma=0.1.
    This means that the learning rate will be multiplied by 0.1 every 10 epochs.
    If the initial learning rate is 0.1, then the learning rate will be reduced to 0.01 after 10 epochs, 0.001 after 20 epochs, and so on.
    """

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500, verbose=True)
    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["lstm_classification14"]["patient"]
    patient_count = 0

    # begin training
    for epoch in range(cf["training"]["lstm_classification14"]["num_epoch"]):
        loss_train, lr_train = run_epoch(binary_model,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(binary_model, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step(metrics=loss_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient_count = 0
            save_best_model(model=binary_model, name="lstm_classification_14", num_epochs=epoch, optimizer=optimizer, val_loss=loss_val, training_loss=loss_train, learning_rate=lr_train)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["lstm_classification14"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch)
            stopped_epoch = patient_count
            break
    return binary_model


def lr_lambda(epoch):
    if epoch < 1000:
        return 1
    elif epoch < 5000:
        return 0.1
    else:
        return 0.1 * (0.1 ** ((epoch - 5000) // 1000))
    
def run_epoch(model, dataloader, optimizer, criterion, scheduler, is_training=False):
    epoch_loss = 0

    weight_decay=0.001
    if is_training:
        model.train()
    else:
        model.eval()


    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(cf["training"]["lstm_regression"]["device"])
        y = y.to(cf["training"]["lstm_regression"]["device"])

        out = model(x)

        # Calculate the L2 regularization term
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param).to("cuda")
        loss = criterion(out, y) + weight_decay * l2_reg
        if is_training:
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)
    try:
        lr = scheduler.get_last_lr()[0]
    except:
        lr = optimizer.param_groups[0]['lr']
    return epoch_loss, lr


def save_best_model(model, name, num_epochs, optimizer, val_loss, training_loss, learning_rate):
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': val_loss,
        'training_loss': training_loss,
        'learning_rate': learning_rate
    }, "./models/" + name)
    
def check_best_loss(best_loss, loss):
    if loss < best_loss:
        return True
    return False
# return boolen Stop, patient_count, best_loss, current_loss
def early_stop(best_loss, current_loss, patient_count, max_patient):
    stop = False
    if(best_loss > current_loss):
        best_loss = current_loss
        patient_count = 0
    else:
        patient_count = patient_count + 1
    if patient_count >= max_patient:
        stop = True
    return stop, patient_count, best_loss, current_loss
    