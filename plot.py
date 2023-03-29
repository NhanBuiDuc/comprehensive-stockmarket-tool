import torch
from config import config as cf
from dataset import TimeSeriesDataset, Normalizer
import numpy as np
import utils
import model
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.pyplot import figure
# predict on the unseen data, tomorrow's price 
def to_plot(dataset_test, dataset_val, y_test, y_val, num_data_points, dates, test_dates, val_dates):

    test_model = model.Assembly_regression()
    checkpoint = torch.load('./models/assembly_regression')
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.eval()
    test_model.to("cuda")
    test_scaler = dataset_test.scaler
    val_scaler = dataset_val.scaler
    test_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=False)
    
    y_test = np.reshape(y_test, y_test.shape[0])
    y_val = np.reshape(y_val, y_val.shape[0])
    # predict on the training data, to see how well the model managed to learn and memorize

    test_prediction = np.empty((1))

    for idx, (x, y) in enumerate(test_dataloader):
        x = x.to("cuda")
        out = test_model(x)
        out = out.cpu().detach().numpy()
        out = np.reshape(out, out.shape[0])
        test_prediction = np.concatenate((test_prediction, out))

    # predict on the validation data, to see how the model does

    val_prediction = np.empty((1))

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to("cuda")
        out = test_model(x)
        out = out.cpu().detach().numpy()
        out = np.reshape(out, out.shape[0])
        val_prediction = np.concatenate((val_prediction, out))
    val_prediction = np.reshape(val_prediction, (val_prediction.shape[0], 1))
    test_prediction = np.reshape(test_prediction, (test_prediction.shape[0],1))
    val_prediction = val_scaler.inverse_transform(val_prediction)
    test_prediction = test_scaler.inverse_transform(test_prediction)
    val_prediction = np.reshape(val_prediction, (val_prediction.shape[0], ))
    test_prediction = np.reshape(test_prediction, (test_prediction.shape[0],))
    if cf["plots"]["show_plots"]:
        # prepare plots
        test_size = 14
        plot_range = 30
        to_plot_data_y_val = np.zeros(plot_range)
        to_plot_data_y_test = np.zeros(plot_range)
        to_plot_data_y_val_pred = np.zeros(plot_range)
        to_plot_data_y_test_pred = np.zeros(plot_range)

        to_plot_data_y_val[:plot_range-1] = (y_val)[-plot_range+1:]
        to_plot_data_y_test[:plot_range-1] = (y_test)[-plot_range+1:]
        to_plot_data_y_val_pred[:plot_range-1] = (val_prediction)[-plot_range+1:]
        to_plot_data_y_test_pred[:plot_range-1] = (test_prediction)[-plot_range+1:]

        to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
        to_plot_data_y_test = np.where(to_plot_data_y_test == 0, None, to_plot_data_y_test)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
        to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

        # plot

        plot_date_test = dates[-plot_range+1:]
        plot_date_test.append("next trading day")

        fig = figure(figsize=(50, 50), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices validation", marker=".", markersize=10, color=cf["plots"]["color_actual_val"])
        plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted validation prices", marker=".", markersize=10, color=cf["plots"]["color_pred_val"])
        plt.plot(plot_date_test, to_plot_data_y_test, label="Actual prices test", marker=".", markersize=20, color=cf["plots"]["color_actual_test"])
        plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Past predicted validation prices", marker=".", markersize=20, color=cf["plots"]["color_pred_test"])
            
        # xticks = [plot_date_test[i] if ((i%2 == 0 and (plot_range + test_size - i ) > 2) or i > plot_range)  else None for i in range(plot_range + test_size )]
        # plt.title("Predicted close price of the next trading day")
        # x = np.arange(0,len(xticks))
        # plt.xticks(x, xticks, rotation='vertical')

        plt.grid(b=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()

    # print("Predicted close price of the next trading day:", round(test_prediction, 2))

# if __name__ == "__main__":
#     data_df, num_data_points, data_date = utils.download_data_api()

#     to_plot(data_df, num_data_points, data_date)