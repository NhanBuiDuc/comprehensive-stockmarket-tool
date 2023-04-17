from trainer import Trainer
import util as u
import torch
trainer = Trainer()
model_name = "movement_1"
trainer.train( model_name, new_data=False)
# checkpoint = torch.load('./models/' + model_name)
# model = checkpoint["model"]
# out = checkpoint.structure(torch.rand(1, 14, 39))
# print("done")