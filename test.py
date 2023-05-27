from APP_FLASK.TrendPrediction import Predictor as Predictor
symbol = "AAPL"
window_size = 14
output_size = 14
model_type_list = ['svm']
predictor = Predictor()
predictor.batch_predict(symbol, model_type_list, window_size, output_size)