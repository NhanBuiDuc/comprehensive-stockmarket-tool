rf_cf = {
    "model": {
        "svm_1": {
            'n_estimators': 100,  # Number of trees in the forest
            'criterion': 'entropy',  # Splitting criterion (can be 'gini' or 'entropy')
            'max_depth': 100,  # Maximum depth of the tree
            'min_samples_split': 5,  # Minimum number of samples required to split an internal node
            'min_samples_leaf': 5,  # Minimum number of samples required to be at a leaf node
            'max_features': 'sqrt',
            # Number of features to consider for the best split ('sqrt' or 'log2' for square root and logarithm of total features respectively)
            'bootstrap': True,  # Whether bootstrap samples are used when building trees
            'oob_score': True,  # Whether to use out-of-bag samples to estimate the generalization accuracy
            'random_state': 42,  # Random seed for reproducibility
            'class_weight': 'balanced',  # Weights associated with classes to address class imbalance
            'verbose': 0,  # Controls the verbosity of the tree building process
            'n_jobs': -1,  # Number of parallel jobs to run (-1 means using all processors)
            "topk": 5,
            "window_size": 7,
            "output_step": 7,
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
