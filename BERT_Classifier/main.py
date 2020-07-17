import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as po
from tqdm import tqdm
from transformers import AdamW
from utils import load_pretrained_model, load_pretrained_tokenizer, evaluate_classifier
from classifier import Classifier

data = po.read_csv('data/automative_question_answers.csv')
data = data.sample(frac=1)

train_data = data[:int(0.8*len(data))]
val_data = data[int(0.8*len(data)):int(0.9*len(data))]
test_data = data[int(0.9*len(data)):]

questions = train_data['questionText'].to_list()
answers = train_data['answerText'].to_list()
labels = train_data['Helpfulness_Target'].to_list()

BATCH_SIZE = 128
val_BATCH_SIZE = 16
device = 'cuda'
model_name = 'distil_bert'

model = Classifier().to(device)
model.train()

tokenizer = load_pretrained_tokenizer('distil_bert')

optimizer = AdamW(model.parameters(), lr=1e-5)
loss_funk = nn.CrossEntropyLoss()

#'''
for i in tqdm(range(0, len(questions), BATCH_SIZE)):
    q_b = questions[i:i+BATCH_SIZE]
    a_b = answers[i:i+BATCH_SIZE]
    l_b = torch.tensor(labels[i:i+BATCH_SIZE]).to(device)
    
    encoded_input = tokenizer(q_b, a_b, padding=True, max_length=100,  truncation='longest_first', return_tensors="pt")
    input_ids = encoded_input['input_ids'].to(device)
    #token_type_ids = encoded_input['token_type_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    pred = model(input_ids=input_ids, attention_mask=attention_mask)

    loss = loss_funk(pred, l_b)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'models/{}.pt'.format(model_name))
#'''

model.load_state_dict(torch.load('models/{}.pt'.format(model_name)))
questions_val = val_data['questionText'].to_list()
answers_val = val_data['answerText'].to_list()
labels_val = val_data['Helpfulness_Target'].to_list()


y_pred = []
y_true = []

for i in tqdm(range(0, len(questions_val) - 7, val_BATCH_SIZE)):
    q_b = questions_val[i:i+val_BATCH_SIZE]
    a_b = answers_val[i:i+val_BATCH_SIZE]
    l_b = torch.tensor(labels_val[i:i+val_BATCH_SIZE]).unsqueeze(0)
    
    encoded_input = tokenizer(q_b, a_b, padding=True, max_length=100,  truncation='longest_first', return_tensors="pt")
    input_ids = encoded_input['input_ids'].to(device)
    #token_type_ids = encoded_input['token_type_ids']
    attention_mask = encoded_input['attention_mask'].to(device)

    pred = model(input_ids=input_ids, attention_mask=attention_mask)

    y_pred += np.argmax(pred.cpu().detach().numpy(), axis = 1).tolist()
    y_true += l_b.detach().numpy().squeeze().tolist()

evaluate_classifier('distil_bert', y_true, y_pred)



'''

torch.save(model.state_dict(), '{}.pt'.format(model_name))



questions_val = val_data['questionText'].to_list()
answers_val = val_data['answerText'].to_list()
labels_val = val_data['Helpfulness_Target'].to_list()

model.cpu()

y_pred = []
y_true = []

for i in tqdm(range(0, len(questions_val) - 7, val_BATCH_SIZE)):
    q_b = questions_val[i:i+val_BATCH_SIZE]
    a_b = answers_val[i:i+val_BATCH_SIZE]
    l_b = torch.tensor(labels_val[i:i+val_BATCH_SIZE]).unsqueeze(0)
    
    encoded_input = tokenizer(q_b, a_b, padding=True, max_length=100,  truncation='longest_first', return_tensors="pt")
    input_ids = encoded_input['input_ids']
    token_type_ids = encoded_input['token_type_ids']
    attention_mask = encoded_input['attention_mask']

    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    pooler_out = outputs[1]


    
    #print(np.argmax(outputs[1].detach().numpy(), axis = 1).tolist())
    #print(l_b.detach().numpy().squeeze().tolist())
    
    y_pred += np.argmax(outputs[1].detach().numpy(), axis = 1).tolist()
    y_true += l_b.detach().numpy().squeeze().tolist()


'''
