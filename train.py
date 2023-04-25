from mv_trainer import Movement_trainer
from trainer import Trainer
import util as u
import torch
import model
from model import Model
model_name = "movement_1"
file_name = "AAPL_movement_1.pth"
trainer = Movement_trainer(model_name, new_data=True, full_data = False)

model = trainer.train()
# tf_trainer.train("svm_1", new_data=True)
model = Model(name=model_name)
model = model.load_check_point(file_name)
trainer.eval(model)
