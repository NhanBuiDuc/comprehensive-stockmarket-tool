from torch import nn
from transformers import BertModel, BertTokenizer


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes, PRE_TRAINED_MODEL_NAME):
        super(SentimentClassifier, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = PRE_TRAINED_MODEL_NAME

        self.bert = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)

    def forward(self, encoded_input):
        pooled_output, pooled_output = self.bert(**encoded_input)
        output = self.drop(pooled_output)
        return self.out(output)
