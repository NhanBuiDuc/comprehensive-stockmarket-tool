transformer_cf = {
    "model": {
        "transformer_1": {
            "nhead": 3,
            "num_encoder_layers": 10,
            "dim_feedforward": 30,
            "dropout": 0.5,
            "window_size": 7,
            "output_step": 3,
            "topk": 5
        }
    },
    "training": {
        "transformer_1":
            {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 500,
                "learning_rate": 0.001,
                "loss": "bce",
                "evaluate": ["bce", "accuracy", "precision", "f1"],
                "optimizer": "adam",
                "scheduler_step_size": 50,
                "patient": 100,
                "start": "2022-07-01",
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
