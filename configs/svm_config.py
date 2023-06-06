svm_cf = {
    "model": {
        "C": 100,
        "kernel": 'rbf',
        "degree": 0,
        "gamma": "scale",
        "coef0": 0,
        "class_weight": {0: 0.5, 1: 0.5},
        "window_size": 14,
        "output_step": 3,
        "data_mode": 0,
        "topk": 10,
        "symbol": "AAPL"
    },
    "training": {
        "device": "cuda",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 300,
        "learning_rate": 0.001,
        "loss": "bce",
        "evaluate": ["bce", "accuracy", "precision", "f1"],
        "optimizer": "adam",
        "scheduler_step_size": 50,
        "patient": 500,
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
