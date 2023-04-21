from Tensorflow_trainer import TensorflowTrainer
from trainer import Trainer
import util as u
import torch
import model
from model import Model

trainer = Trainer()
tf_trainer = TensorflowTrainer()
model_name = "movement_1"
file_name = "AAPL_movement_1.pth"
model = trainer.train(model_name, new_data=True, full_data = False)
# tf_trainer.train("svm_1", new_data=True)
model = Model(name=model_name)
model = model.load_check_point(file_name)
trainer.eval(model)
