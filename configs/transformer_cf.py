transformer_cf = {
    "model": {
        "nhead": 3,
        "num_encoder_layers": 40,
        "dim_feedforward": 40,
        "dropout": 0.3,
        "window_size": 14,
        "output_step": 3,
        "data_mode":1,
        "topk": 5
    },
    "training": {
        "device": "cuda",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 200,
        "learning_rate": 0.001,
        "loss": "focal",
        "evaluate": ["bce", "accuracy", "precision", "f1"],
        "optimizer": "adam",
        "scheduler_step_size": 50,
        "patient": 500,
        "start": "2022-07-01",
        "end": None,
        "best_model": False,
        "early_stop": True,
        "train_shuffle": True,
        "val_shuffle": True,
        "test_shuffle": True,
        "weight_decay": 0.001
    }
}
