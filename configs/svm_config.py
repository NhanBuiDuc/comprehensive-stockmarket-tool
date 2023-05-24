svm_cf = {
    "model": {
        "svm_1": {
            "C": 1000,
            "kernel": 'poly',
            "degree": 400,
            "gamma": "scale",
            "coef0": 100,
            "class_weight": {0: 0.5, 1: 0.5},
            "window_size": 7,
            "output_step": 7,
            "topk": 5
        }
    },
    "training": {
        "svm_1":
            {
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
