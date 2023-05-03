transformer_cf = {
    "model": {
        "transformer_1": {
            "d_model": 14,
            "nhead": 3,
            "num_encoder_layers": 6,
            "dim_feedforward": 14,
            "dropout": 0.1,
            "window_size": 14,
            "output_step": 1
        }
    },
    "training": {
        "transformer_1":
            {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 100,
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
