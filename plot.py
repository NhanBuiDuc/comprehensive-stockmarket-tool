import torch
from config import config as cf
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

# predict on the unseen data, tomorrow's price 
def to_plot(model, x_val, x_test):
    model.eval()
    scaler = MaxAbsScaler()
    # x = torch.tensor(x_test).float().to(config["training"]["device"])
    x = torch.tensor(x_test).float().to(cf["training"]["device"]).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
    prediction = model(x)
    prediction = prediction.cpu().detach().numpy()
    prediction = scaler.inverse_transform(prediction)
    num_plot = prediction.shape[0]
    num_data_points = 
    if cf["plots"]["show_plots"]:
        plot_range = 30
        test_size = x_test.shape[0]
        plot_num_data_points = num_data_points - test_size + 1
        plot_data_date = data_date[:-2]

        plot_date_test = plot_data_date[-plot_range:]
        for i in range(test_size):
            plot_date_test.append("next {} trading day".format(i+1))

        # Real 30 days
        to_plot_y_true = np.zeros((plot_range + test_size, 1))
        temp_plot_data_y_val = plot_data_y_val.reshape(-1,1)
        to_plot_y_true[:plot_range] = scaler.inverse_transform(temp_plot_data_y_val)[-plot_range:]
        # Prediction last 27 days
        to_plot_y_pred = np.zeros((plot_range + test_size, cf["model"]["output_dates"]))
        to_plot_y_pred[:plot_range] = scaler.inverse_transform(predicted_val)[-plot_range:]
        # Compute the row-wise mean of the original array
        # to_plot_y_pred = spread_out_prediction(to_plot_y_pred, sequences = 3, max_lenght = plot_range + test_size, num_data_points = plot_range)
        # prediction = np.mean(prediction, axis=1)

        to_plot_test = np.zeros(plot_range + test_size)
        to_plot_test[plot_range:] = prediction[-1]

        to_plot_test = np.where(to_plot_test == 0, None, to_plot_test)
        to_plot_y_true = np.where(to_plot_y_true == 0, None, to_plot_y_true)
        to_plot_y_pred = np.where(to_plot_y_pred == 0, None, to_plot_y_pred)
        # plot

        fig = figure(figsize=(30, 10), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(plot_date_test, to_plot_y_true, label="Actual prices", marker=".", markersize=10, color=cf["plots"]["color_actual"])
        plt.plot(plot_date_test, to_plot_y_pred, label="Past prediction", marker=".", markersize=20, color=cf["plots"]["color_pred_val"])
        plt.plot(plot_date_test, to_plot_test, label="Predicted price for next day", marker=".", markersize=20, color=cf["plots"]["color_pred_test"])
        
        xticks = [plot_date_test[i] if ((i%2 == 0 and (plot_range + test_size - i ) > 2) or i > plot_range)  else None for i in range(plot_range + test_size )] # make x ticks nice
        x = np.arange(0,len(xticks))
        plt.xticks(x, xticks, rotation='vertical')

        plt.title("Predicted close price of the next trading day")
        plt.grid(b=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()

    print("Predicted close price of the next trading day:")
    print(prediction[-1])

