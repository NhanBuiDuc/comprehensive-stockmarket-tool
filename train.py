import trainer.transformer_trainer as ttn
from model import Model
from configs.config import config

trainer = ttn.Transformer_trainer(new_data=False, full_data=False, mode = "train")

trainer.train()
model = trainer.model
model = model.load_check_point(model.model_type, trainer.model.name)
trainer.eval(model)

# from trainer.svm_trainer import svm_trainer
#
# trainer = svm_trainer(new_data=False, full_data=False, mode = "train")
#
# model = trainer.train()
# model = trainer.model
# model = model.load_check_point(trainer.model.name)
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