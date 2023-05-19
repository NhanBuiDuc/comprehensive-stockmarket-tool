import trainer.transformer_trainer as ttn
from model import Model
model_name = "transformer_1"
file_name = "AAPL_transformer_1.pth"
trainer = ttn.Transformer_trainer(model_name=model_name, new_data=False, full_data=False)

# model = trainer.train()
# tf_trainer.train("svm_1", new_data=True)
model = Model(name=model_name)
model = model.load_check_point(file_name)
trainer.eval(model)
