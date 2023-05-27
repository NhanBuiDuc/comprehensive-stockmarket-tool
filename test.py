from APP_FLASK.TrendPrediction import Predictor as Predictor
symbol = "AAPL"
predictor = Predictor()
model_type_list = ['svm']
window_size = 3
output_size=3
print('asdasd ',predictor.batch_predict(symbol, model_type_list, window_size, output_size))