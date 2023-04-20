config = {
    "alpha_vantage": {
        "api_key": "XOLA7URKCZHU7C9X",  # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "AAPL",
        "output_size": "full",
        "url": "https://www.alphavantage.co"
    },
    "data": {
        "window_size": 14,
        "train_test_split_size": 0.9,
        "train_val_split_size": 0.7,
        "smoothing": 2,
    },
    "plots": {
        "show_plots": True,
        "xticks_interval": 90,
        "color_actual_val": "#001f3f",
        "color_actual_test": "#4D1BF3",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "assemble_1": {
            "input_size": 39,
            "lstm_num_layer": 5,
            "lstm_size": 64,
            "dropout": 0.2,
            "output_step": 1,
            "window_size": 14,
        },
        "assemble_3": {
            "input_size": 39,
            "lstm_num_layer": 5,
            "lstm_size": 64,
            "dropout": 0.2,
            "output_step": 1,
            "window_size": 14,
        },
        "assemble_7": {
            "input_size": 39,
            "lstm_num_layer": 5,
            "lstm_size": 64,
            "dropout": 0.2,
            "output_step": 1,
            "window_size": 14,
        },
        "assemble_14": {
            "input_size": 39,
            "lstm_num_layer": 5,
            "lstm_size": 64,
            "dropout": 0.2,
            "output_step": 1,
            "window_size": 14,
        },
        "movement_1": {
            "lstm_num_layer": 2,
            "lstm_hidden_layer_size": 14,
            "drop_out": 0.2,
            "output_step": 1,
            "window_size": 14,
            "conv1D_param": {
                "type": 2,
                "kernel_size": 4,
                "dilation_base": 3,
                "max_pooling_kernel_size": 2,
                "sub_small_num_layer": 1,
                "sub_big_num_layer": 1,
                "sub_small_kernel_size": 3,
                "sub_big_kernel_size": 30,
                "output_size": 5
            }
        },
        "magnitude_1": {
            "lstm_num_layers": 4.5,
            "lstm_hidden_layer_size": 64,
            "drop_out": 0.2,
            "output_step": 1,
            "window_size": 14,
            "conv1d_kernel_size": 4,
            "dilation_base": 3
        },
        "movement_3": {
            "lstm_num_layer": 5,
            "lstm_hidden_layer_size": 64,
            "drop_out": 0.2,
            "output_step": 1,
            "window_size": 14,
            "conv1d_kernel_size": 4,
            "dilation_base": 3
        },
        "magnitude_3": {
            "lstm_num_layers": 5,
            "lstm_hidden_layer_size": 64,
            "drop_out": 0.2,
            "output_step": 1,
            "window_size": 14,
            "conv1d_kernel_size": 4,
            "dilation_base": 3
        },
        "movement_7": {
            "lstm_num_layer": 5,
            "lstm_hidden_layer_size": 64,
            "drop_out": 0.2,
            "output_step": 1,
            "window_size": 14,
            "conv1d_kernel_size": 4,
            "dilation_base": 3
        },
        "magnitude_7": {
            "lstm_num_layers": 5,
            "lstm_hidden_layer_size": 64,
            "drop_out": 0.2,
            "output_step": 1,
            "window_size": 14,
            "conv1d_kernel_size": 4,
            "dilation_base": 3
        },
        "movement_14": {
            "lstm_num_layer": 5,
            "lstm_hidden_layer_size": 64,
            "drop_out": 0.2,
            "output_step": 1,
            "window_size": 14,
            "conv1d_kernel_size": 4,
            "dilation_base": 3
        },
        "magnitude_14": {
            "lstm_num_layers": 5,
            "lstm_hidden_layer_size": 64,
            "drop_out": 0.2,
            "output_step": 1,
            "window_size": 14,
            "conv1d_kernel_size": 4,
            "dilation_base": 3
        },
        "LSTM_bench_mark_1": {
            "lstm_num_layer": 1,
            "lstm_hidden_layer_size": 64,
            "drop_out": 0.2,
            "output_step": 1,
            "window_size": 14,
        },
        "GRU_bench_mark_1": {
            "hidden_size": 64,
            "output_step": 1,
            "window_size": 14,
        },
        "svm_1": {
            "window_size": 14,
            "output_step": 1,
        }
    },
    "training": {
        "assemble_1":
            {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 200,
                "learning_rate": 0.01,
                "loss": "mse",
                "evaluate": ["mse", "mae"],
                "optimizer": "adam",
                "scheduler_step_size": 50,
                "patient": 200,
                "start": "2000-01-01",
                "end": None,
                "best_model": False,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True
            },
        "movement_1":
            {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 300,
                "learning_rate": 0.01,
                "loss": "bce",
                "evaluate": ["bce", "accuracy", "precision", "f1"],
                "optimizer": "adam",
                "scheduler_step_size": 100,
                "patient": 1000,
                "start": "2022-01-01",
                "end": None,
                "best_model": True,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True,
                "weight_decay": 0.0001
            },
        "magnitude_1":
            {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 1,
                "learning_rate": 0.01,
                "loss": "mse",
                "evaluate": ["mse", "mae"],
                "optimizer": "adam",
                "scheduler_step_size": 50,
                "patient": 200,
                "from": "2020-01-01",
                "to": None,
                "best_model": True,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True
            },
        "movement_3":
            {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 150,
                "learning_rate": 0.01,
                "scheduler_step_size": 50,
                "patient": 200,
                "best_model": True,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True
            },
        "magnitude_3": {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 150,
                "learning_rate": 0.01,
                "scheduler_step_size": 50,
                "patient": 200,
                "best_model": True,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True
            },
        "movement_7": {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 150,
                "learning_rate": 0.01,
                "scheduler_step_size": 50,
                "patient": 200,
                "best_model": False,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True
            },
        "magnitude_7": {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 150,
                "learning_rate": 0.01,
                "scheduler_step_size": 50,
                "patient": 200,
                "best_model": True,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True
            },
        "movement_14": {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 150,
                "learning_rate": 0.01,
                "scheduler_step_size": 50,
                "patient": 200,
                "best_model": False,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True
            },
        "magnitude_14": {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 150,
                "learning_rate": 0.01,
                "scheduler_step_size": 50,
                "patient": 200,
                "best_model": True,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True
            },
        "LSTM_bench_mark_1": {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 1000,
                "learning_rate": 0.01,
                "loss": "bce",
                "evaluate": ["bce", "accuracy", "precision", "f1"],
                "optimizer": "adam",
                "scheduler_step_size": 500,
                "patient": 1000,
                "start": "2000-5-01",
                "end": None,
                "best_model": True,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True,
                "weight_decay": 0.1
        },
        "svm_1": {
            "batch_size": 64,
            "num_epoch": 300,
            "learning_rate": 0.01,
            "loss": "bce",
            "evaluate": ["bce", "accuracy", "precision", "f1"],
            "optimizer": "adam",
            "scheduler_step_size": 50,
            "patient": 1000,
            "start": "2015-01-01",
            "end": None,
            "best_model": True,
            "early_stop": True,
            "train_shuffle": True,
            "val_shuffle": True,
            "test_shuffle": True,
            "weight_decay": 0.1
        }

    },
    "pytorch_timeseries_model_type_dict": {
            1: "movement",
            2: "magnitude",
            3: "assembler",
            4: "lstm",
            5: "gru"
    },
    "tensorflow_timeseries_model_type_dict": {
        1: "svm",
        2: "random_forest"
    }
}
