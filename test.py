from APP_FLASK.TrendPrediction import Predictor as Predictor
symbol = "AAPL"
predictor = Predictor()
predictor.prepare_data(symbol, 14)