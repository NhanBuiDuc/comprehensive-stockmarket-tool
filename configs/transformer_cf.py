config = {
    "model": {
        "transformer": {
            "d_model": 512,
            "nhead": 8,
            "drop_out": 6,
            "dim_feedforward": 2048,
            "dropout": 0.1,
        }
    },
    "training": {
        "transformer":
            {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 10,
                "learning_rate": 0.01,
                "loss": "bce",
                "evaluate": ["bce", "accuracy", "precision", "f1"],
                "optimizer": "adam",
                "scheduler_step_size": 50,
                "patient": 100,
                "start": "2020-01-01",
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
