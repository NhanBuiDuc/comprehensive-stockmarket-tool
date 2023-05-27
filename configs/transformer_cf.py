transformer_cf = {
    "model": {
        "nhead": 3,
        "num_encoder_layers": 50,
        "dim_feedforward": 50,
        "dropout": 0.3,
        "window_size": 7,
        "output_step": 7,
        "data_mode":2,
        "topk": 5
    },
    "training": {
        "device": "cuda",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 50,
        "learning_rate": 0.0001,
        "loss": "bce",
        "evaluate": ["bce", "accuracy", "precision", "f1"],
        "optimizer": "adam",
        "scheduler_step_size": 50,
        "patient": 500,
        "start": "2022-07-01",
        "end": "2023-05-25",
        "best_model": True,
        "early_stop": True,
        "train_shuffle": True,
        "val_shuffle": True,
        "test_shuffle": True,
        "weight_decay": 0.01
    }
}
