lstm_cf = {
    "model": {
        "num_layers": 10,
        "hidden_size": 20,
        "drop_out": 0.5,
        "window_size": 3,
        "output_step": 3,
        "conv1D_param": {
            "type": 1,
            "kernel_size": 4,
            "dilation_base": 3,
            "max_pooling_kernel_size": 2,
            "sub_small_num_layer": 1,
            "sub_big_num_layer": 1,
            "sub_small_kernel_size": 3,
            "sub_big_kernel_size": 30,
            "output_size": 20
        },
        "symbol": "AAPL",
        "topk": 10,
        "data_mode": 2,
        "max_string_length": 1000,
    },
    "training": {
        "device": "cuda",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.001,
        "loss": "bce",
        "evaluate": ["bce", "accuracy", "precision", "f1"],
        "optimizer": "adam",
        "scheduler_step_size": 50,
        "patient": 100,
        "start": "2022-07-01",
        "end": "2023-05-01",
        "best_model": True,
        "early_stop": True,
        "train_shuffle": True,
        "val_shuffle": True,
        "test_shuffle": True,
        "weight_decay": 0.0001
    },
}
