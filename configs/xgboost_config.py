xgboost_cf = {
    "model": {
        "symbol": "AAPL",
        "nhead": 3,
        "n_estimators": 100,  # Number of trees in the ensemble
        "objective": 'binary:logistic',  # Objective function for binary classification
        "max_depth": 3,  # Maximum depth of each tree
        "learning_rate": 0.1,  # Learning rate (step size shrinkage)
        "subsample": 0.8,  # Subsample ratio of the training instances
        "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
        "reg_alpha": 0,  # L1 regularization term on weights
        "reg_lambda": 1,  # L2 regularization term on weights
        "random_state": 42,  # Random seed for reproducibility        "dropout": 0.5,
        "window_size": 3,
        "output_step": 3,
        "data_mode": 0,
        "topk": 10,
    },
    "training": {
        "device": "cuda",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.001,
        "loss": "focal",
        "evaluate": ["bce", "accuracy", "precision", "f1"],
        "optimizer": "adam",
        "scheduler_step_size": 100,
        "patient": 600,
        "start": "2022-07-01",
        "end": "2023-05-01",
        "best_model": True,
        "early_stop": True,
        "train_shuffle": True,
        "val_shuffle": True,
        "test_shuffle": True,
        "weight_decay": 0.001
    }
}
