price_cf = {
    "model": {
        "pred_price_LSTM_1": {
            "lstm_num_layer": 2,
            "lstm_hidden_layer_size": 32,
            "drop_out": 0.2,
            "output_step": 1,
            "window_size": 14
        }
    },
    "training": {
        "pred_price_LSTM_1":
            {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 10,
                "learning_rate": 0.01,
                "loss": "mse",
                "evaluate": ["mae"],
                "optimizer": "adam",
                "scheduler_step_size": 50,
                "patient": 100,
                "start": "2018-01-01",
                "end": None,
                "best_model": True,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True,
                "weight_decay": 0.0001
            },
    }
}
