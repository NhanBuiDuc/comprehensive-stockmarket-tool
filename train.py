from trainer import Trainer
import util as u
import torch
import model

trainer = Trainer()
model_name = "movement_1"
trainer.train(model_name, new_data=True)
tensor = torch.full((1, 14, 39), 0.5).to("cuda")

# TH2
