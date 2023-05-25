import trainer.transformer_trainer as ttn
from model import Model
model_name = "transformer_1"
file_name = "AAPL_transformer_1.pth"
trainer = ttn.Transformer_trainer(model_name=model_name, new_data=True, full_data=False, mode = "train")
#
model = trainer.train()
# tf_trainer.train("svm_1", new_data=True)
model = Model(name=model_name)
model = model.load_check_point(file_name)
trainer.eval(model)

# from trainer.svm_trainer import svm_trainer
# from model import Model
# model_name = "svm_7"
# file_name = "AAPL_svm_7.pkl"
# trainer = svm_trainer(model_name=model_name, new_data=False, full_data=False, mode = "train", data_mode=1)

# model = trainer.train()
# # tf_trainer.train("svm_1", new_data=True)
# model = Model(name=model_name)
# model = model.load_check_point(file_name)
# trainer.eval(model)

# from trainer.random_forest_trainer import random_forest_trainer
# from model import Model
# model_name = "random_forest_7"
# file_name = "AAPL_random_forest_7.pkl"
# trainer = random_forest_trainer(model_name=model_name, new_data=False, full_data=False, mode = "train")
#
# model = trainer.train()
# # tf_trainer.train("svm_1", new_data=True)
# model = Model(name=model_name)
# model = model.load_check_point(file_name)
# trainer.eval(model)