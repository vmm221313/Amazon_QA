import torch
import torch.nn as nn
from utils import load_pretrained_model

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()

		self.pretrained_model 	= load_pretrained_model('distil_bert')
		self.dropout 			= nn.Dropout(0.2)
		self.classifier 		= nn.Linear(768*100, 2)

	def forward(self, input_ids, attention_mask, token_type_ids=None):
		outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)

		pooled_output = outputs[0].reshape(128, -1)#[:, 0, :]

		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		return logits

if __name__ == '__main__':
	bc = Classifier()
	print(len(list(bc.parameters())))
