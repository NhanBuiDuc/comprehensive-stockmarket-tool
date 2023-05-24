from APP_FLASK.Predictor import Predictor as Predictor
symbol = "AAPL"
predictor = Predictor()
predictor.prepare_data(symbol, 14)