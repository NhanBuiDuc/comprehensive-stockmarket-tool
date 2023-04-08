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
        "assembly_regression_1":{
            "input_size": 14,
            "num_lstm_layers": 2,
            "lstm_size": 7,
            "dropout": 0.2, 
            "output_steps": 3,
            "window_size": 14  
        },
        "movement_3":{
            "lstm_num_layers": 2,
            "lstm_hidden_layer_size": 7,
            "dropout": 0.2, 
            "output_steps": 3,
            "window_size": 14,
            "max_features": 40
        },
        "movement_7":{
            "lstm_num_layers": 2,
            "lstm_hidden_layer_size": 7,
            "dropout": 0.2, 
            "output_steps": 3,
            "window_size": 14,
            "max_features": 40
        },
        "movement_14":{
            "lstm_num_layers": 2,
            "lstm_hidden_layer_size": 7,
            "dropout": 0.2, 
            "output_steps": 3,
            "window_size": 14,
            "max_features": 40
        },
        "diff_1":{
            "lstm_num_layers": 2,
            "lstm_hidden_layer_size": 7,
            "dropout": 0.2, 
            "output_steps": 1,
            "window_size": 14,
            "use_attn": True,
            "attn_num_heads": 3,
            "attn_multi_head_scaler": 1,
            "max_features": 40
        },
    },
    "training": {
        "assemble_1":
        {
            "device": "cuda", # "cuda" or "cpu"
            "batch_size": 128,
            "num_epoch": 100,
            "learning_rate": 0.01,
            "scheduler_step_size": 50,
            "patient": 2000,
            "best_model": False,
            "early_stop": True,
            "corr_thresh_hold": 0.1
        },
        "diff_1":
        {
            "device": "cuda", # "cuda" or "cpu"
            "batch_size": 64,
            "num_epoch": 100,
            "learning_rate": 0.1,
            "scheduler_step_size": 100,
            "patient": 2000,
            "best_model": False,
            "early_stop": True,
            "corr_thresh_hold": 0.1
        },
        "movement_3":
        {
            "device": "cuda", # "cuda" or "cpu"
            "batch_size": 128,
            "num_epoch": 100,
            "learning_rate": 0.01,
            "scheduler_step_size": 50,
            "patient": 200,
            "best_model": False,
            "early_stop": True,
            "corr_thresh_hold": 0.1
        },
        "movement_7":
        {
            "device": "cuda", # "cuda" or "cpu"
            "batch_size": 128,
            "num_epoch": 100,
            "learning_rate": 0.01,
            "scheduler_step_size": 50,
            "patient": 200,
            "best_model": False,
            "early_stop": True,
            "corr_thresh_hold": 0.5
        },
        "movement_14":
        {
            "device": "cuda", # "cuda" or "cpu"
            "batch_size": 128,
            "num_epoch": 500,
            "learning_rate": 0.1,
            "scheduler_step_size": 50,
            "patient": 500,
            "best_model": False,
            "early_stop": True,
            "corr_thresh_hold": 0.5
        }
    }
}