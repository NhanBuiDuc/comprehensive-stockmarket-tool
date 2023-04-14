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
from loss import Unified_Adversarial_Loss as UAL
def train_assemble_model_1(dataset_train, dataset_val, features):
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "assemble_1"
    lr=cf["training"]["assemble_1"]["learning_rate"]
    epochs=cf["training"]["assemble_1"]["num_epoch"]
    regression_model = model.Assemble()
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

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cf["training"]["assemble_1"]["scheduler_step_size"], verbose=True)
    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["assemble_1"]["patient"]
    patient_count = 0
    # begin training
    for epoch in range(epochs):
        loss_train, lr_train = run_epoch(regression_model,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(regression_model, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step(loss_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient_count = 0
            save_best_model(model=regression_model, 
                            name = model_name,
                            num_epochs=epoch,
                            optimizer=optimizer,
                            val_loss=loss_val,
                            training_loss=loss_train,
                            features=features,
                            learning_rate=lr_train)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["assemble_1"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch + 1)
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

def train_LSTM_regression_1 (dataset_train, dataset_val, features, is_training=True):
    # use_attn = cf["model"]["diff_1"]["use_attn"],
    # if(use_attn):
    #     model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "attn_diff_1"
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "diff_1"
    diff_1 = model.Diff_1(
        input_size = len(features),
        window_size = cf["model"]["diff_1"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["diff_1"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["diff_1"]["lstm_num_layers"], 
        output_steps = cf["model"]["diff_1"]["output_steps"]
    )
    diff_1.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["diff_1"]["batch_size"])
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["diff_1"]["batch_size"], shuffle=True)

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
    optimizer = optim.Adam(diff_1.parameters(), lr=cf["training"]["diff_1"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)

    """
    For example, suppose step_size=10 and gamma=0.1.
    This means that the learning rate will be multiplied by 0.1 every 10 epochs.
    If the initial learning rate is 0.1, then the learning rate will be reduced to 0.01 after 10 epochs, 0.001 after 20 epochs, and so on.
    """

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cf["training"]["diff_1"]["scheduler_step_size"], verbose=True)

    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["diff_1"]["patient"]
    patient_count = 0
    # begin training
    for epoch in range(cf["training"]["diff_1"]["num_epoch"]):
        loss_train, lr_train = run_epoch(diff_1,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(diff_1, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step(loss_val)
        # loss_train_history.append(loss_train)
        # loss_val_history.append(loss_val)
        # lr_train_history.append(lr_train)
        # lr_val_history.append(lr_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient_count = 0
            save_best_model(model=diff_1,
                            name = model_name,
                            num_epochs=epoch,
                            optimizer=optimizer,
                            val_loss=loss_val,
                            training_loss=loss_train,
                            learning_rate=lr_train,
                            features = features)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["diff_1"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch)
            stopped_epoch = patient_count
            break
    return diff_1

def train_Movement_3(dataset_train, dataset_val, features, is_training=True):
    # use_attn = cf["model"]["movement_3"]["use_attn"],
    # if(use_attn):
    #     model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "attn_movement_3"
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_3"
    movement_3 = model.Movement_3(
        input_size = len(features),
        window_size = cf["model"]["movement_3"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["movement_3"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["movement_3"]["lstm_num_layers"], 
        output_steps = cf["model"]["movement_3"]["output_steps"],
        kernel_size=4,
        dilation_base=3
    )
    movement_3.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["movement_3"]["batch_size"], shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_3"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = UAL()
    # optimizer = optim.SGD(binary_model.parameters(), lr=cf["training"]["movement_3"]["learning_rate"], momentum=0.9)

    optimizer = optim.Adam(movement_3.parameters(), lr=cf["training"]["movement_3"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
    """
    For example, suppose step_size=10 and gamma=0.1.
    This means that the learning rate will be multiplied by 0.1 every 10 epochs.
    If the initial learning rate is 0.1, then the learning rate will be reduced to 0.01 after 10 epochs, 0.001 after 20 epochs, and so on.
    """

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cf["training"]["movement_3"]["scheduler_step_size"], verbose=True)
    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["movement_3"]["patient"]
    patient_count = 0

    # begin training
    for epoch in range(cf["training"]["movement_3"]["num_epoch"]):
        loss_train, lr_train = run_epoch(movement_3,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(movement_3, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step(metrics=loss_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient_count = 0
            save_best_model(model=movement_3,
                            name=model_name,
                            num_epochs=epoch,
                            optimizer=optimizer,
                            val_loss=loss_val,
                            training_loss=loss_train,
                            learning_rate=lr_train,
                            features=features)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["movement_3"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch)
            stopped_epoch = patient_count
            break
    return movement_3


def train_Movement_7(dataset_train, dataset_val, features, is_training=True):
    # use_attn = cf["model"]["movement_7"]["use_attn"]
    # if(use_attn):
    #     model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "attn_movement_7"
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_7"
    movement_7 = model.Movement_7(
        input_size = len(features),
        window_size = cf["model"]["movement_7"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["movement_7"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["movement_7"]["lstm_num_layers"], 
        output_steps = cf["model"]["movement_7"]["output_steps"],
        kernel_size=4,
        dilation_base=3
    )
    movement_7.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["movement_7"]["batch_size"],shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_7"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = UAL()
    # optimizer = optim.SGD(binary_model.parameters(), lr=cf["training"]["movement_7"]["learning_rate"], momentum=0.9)

    optimizer = optim.Adam(movement_7.parameters(), lr=cf["training"]["movement_7"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
    """
    For example, suppose step_size=10 and gamma=0.1.
    This means that the learning rate will be multiplied by 0.1 every 10 epochs.
    If the initial learning rate is 0.1, then the learning rate will be reduced to 0.01 after 10 epochs, 0.001 after 20 epochs, and so on.
    """

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cf["training"]["movement_7"]["scheduler_step_size"], verbose=True)
    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["movement_7"]["patient"]
    patient_count = 0

    # begin training
    for epoch in range(cf["training"]["movement_7"]["num_epoch"]):
        loss_train, lr_train = run_epoch(movement_7,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(movement_7, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step(metrics=loss_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient_count = 0
            save_best_model(model=movement_7, 
                            name=model_name,
                            num_epochs=epoch,
                            optimizer=optimizer,
                            val_loss=loss_val,
                            training_loss=loss_train,
                            learning_rate=lr_train,
                            features=features)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["movement_7"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch)
            stopped_epoch = patient_count
            break
    return movement_7


def train_Movement_14(dataset_train, dataset_val, features, is_training=True):
    # use_attn = cf["model"]["movement_14"]["use_attn"]
    # if(use_attn):
    #     model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "attn_movement_14"
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_14"
    movement_14 = model.Movement_14(
        input_size = len(features),
        window_size = cf["model"]["movement_14"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["movement_14"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["movement_14"]["lstm_num_layers"], 
        output_steps = cf["model"]["movement_14"]["output_steps"],
        kernel_size=4,
        dilation_base=3
    )
    movement_14.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["movement_14"]["batch_size"])
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_14"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = UAL()
    # optimizer = optim.SGD(binary_model.parameters(), lr=cf["training"]["movement_14"]["learning_rate"], momentum=0.9)

    optimizer = optim.Adam(movement_14.parameters(), lr=cf["training"]["movement_14"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
    """
    For example, suppose step_size=10 and gamma=0.1.
    This means that the learning rate will be multiplied by 0.1 every 10 epochs.
    If the initial learning rate is 0.1, then the learning rate will be reduced to 0.01 after 10 epochs, 0.001 after 20 epochs, and so on.
    """

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cf["training"]["movement_14"]["scheduler_step_size"], verbose=True)
    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["movement_14"]["patient"]
    patient_count = 0

    # begin training
    for epoch in range(cf["training"]["movement_14"]["num_epoch"]):
        loss_train, lr_train = run_epoch(movement_14,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(movement_14, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step(metrics=loss_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient_count = 0
            save_best_model(model=movement_14, 
                            name=model_name,
                            num_epochs=epoch,
                            optimizer=optimizer,
                            val_loss=loss_val,
                            training_loss=loss_train,
                            learning_rate=lr_train,
                            features=features)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["movement_14"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch)
            stopped_epoch = patient_count
            break
    return movement_14


def lr_lambda(epoch):
    if epoch < 1000:
        return 1
    elif epoch < 5000:
        return 0.1
    else:
        return 0.1 * (0.1 ** ((epoch - 5000) // 1000))
    
def run_epoch(model, dataloader, optimizer, criterion, scheduler, is_training):
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
        epoch_loss += (loss.detach().item() / batchsize)
    try:
        lr = scheduler.get_last_lr()[0]
    except:
        lr = optimizer.param_groups[0]['lr']
    return epoch_loss, lr


def save_best_model(model, name, num_epochs, optimizer, val_loss, training_loss, learning_rate, features):
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': val_loss,
        'training_loss': training_loss,
        'learning_rate': learning_rate,
        'features': features      
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
    