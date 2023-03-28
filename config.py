config = {
    "alpha_vantage": {
        "key": "XOLA7URKCZHU7C9X", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "AAPL",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {
        "window_size": 14,
        "train_split_size": 0.70,
        "smoothing": 2
    }, 
    "plots": {
        "show_plots": True,
        "xticks_interval": 90,
        "color_actual_val": "#001f3f",
        "color_actual_test": "#4D1BF3",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "assembly_regression":{
            "input_size": 12,
            "output_dates": 1    
        },
        "lstm_regression":{
            "input_size": 12,
            "num_lstm_layers": 2,
            "lstm_size": 32,
            "dropout": 0.5,
            "output_dates": 1    
        },
        "lstm_classification1":{
            "input_size": 12,
            "num_lstm_layers": 2,
            "lstm_size": 32,
            "dropout": 0.5, 
            "output_dates": 1
        },
        "lstm_classification14":{
            "input_size": 12,
            "num_lstm_layers": 1,
            "lstm_size": 14,
            "dropout": 0.5, 
            "output_dates": 14
        },
    },
    "training": {
        "assemble_regressor":
        {
            "device": "cuda", # "cuda" or "cpu"
            "batch_size": 64,
            "num_epoch": 3000,
            "learning_rate": 0.1,
            "scheduler_step_size": 500,
            "patient": 2000,
            "best_model": False,
            "early_stop": True
        },
        "lstm_regression":
        {
            "device": "cuda", # "cuda" or "cpu"
            "batch_size": 64,
            "num_epoch": 500,
            "learning_rate": 0.1,
            "scheduler_step_size": 500,
            "patient": 2000,
            "best_model": False,
            "early_stop": True
        },
        "lstm_classification1":
        {
            "device": "cuda", # "cuda" or "cpu"
            "batch_size": 64,
            "num_epoch": 3000,
            "learning_rate": 0.1,
            "scheduler_step_size": 500,
            "patient": 2000,
            "best_model": False,
            "early_stop": True
        },
        "lstm_classification14":
        {
            "device": "cuda", # "cuda" or "cpu"
            "batch_size": 64,
            "num_epoch": 3000,
            "learning_rate": 0.1,
            "scheduler_step_size": 500,
            "patient": 2000,
            "best_model": False,
            "early_stop": True
        }
    }
}