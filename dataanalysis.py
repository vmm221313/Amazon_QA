import os
import json
import torch
import numpy as np
import pandas as po
from tqdm.notebook import tqdm
#from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

'''
with open('data/qar_products_Automotive.jsonl', 'r') as json_file:
    json_list = list(json_file)

data = po.DataFrame()
for json_str in tqdm(json_list):
    result = json.loads(json_str)
    for question in result['questions']:tqdm.notebook.tqdm
        #print(question)
        questionText = question['questionText']
        for answer in question['answers']:
            answerText = answer['answerText']
            answerHelpfulness = answer['helpful']
            if answerHelpfulness in [[0,0], [1,1]]:
                continue
            row = {
                'questionText': questionText,
                'answerText': answerText, 
                'answerHelpfulness': answerHelpfulness
            }
            data = data.append(row, ignore_index = True)

answerHelpfulness = np.array(data['answerHelpfulness'].to_list())

data['Helpfulness_Target'] = np.where(np.logical_and(answerHelpfulness[:, 0] == answerHelpfulness[:, 1], answerHelpfulness[:, 1] >= 2), 1, 0)

data['Helpfulness_Target'].value_counts()

data.to_csv('data/automative_question_answers.csv', index = False)
'''

data = po.read_csv('data/automative_question_answers.csv')

data = data.sample(frac=1)

len(data)

train_data = data[:int(0.8*len(data))]
val_data = data[int(0.8*len(data)):int(0.9*len(data))]
test_data = data[int(0.9*len(data)):]

questions = train_data['questionText'].to_list()
answers = train_data['answerText'].to_list()
labels = train_data['Helpfulness_Target'].to_list()

BATCH_SIZE = 128
val_BATCH_SIZE = 16
device = 'cuda'
model_name = 'AlbertForSequenceClassification'

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2').to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=1e-5)

for i in tqdm(range(0, len(questions), BATCH_SIZE)):
    q_b = questions[i:i+BATCH_SIZE]
    a_b = answers[i:i+BATCH_SIZE]
    l_b = torch.tensor(labels[i:i+BATCH_SIZE]).unsqueeze(0).to(device)
    
    encoded_input = tokenizer(q_b, a_b, padding=True, max_length=100,  truncation='longest_first', return_tensors="pt")
    input_ids = encoded_input['input_ids'].to(device)
    token_type_ids = encoded_input['token_type_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=l_b)
    
    loss = outputs[0].to(device)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), '{}.pt'.format(model_name))

model.load_state_dict(torch.load('{}.pt'.format(model_name)))

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

    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=l_b)
    
    #print(np.argmax(outputs[1].detach().numpy(), axis = 1).tolist())
    #print(l_b.detach().numpy().squeeze().tolist())
    
    y_pred += np.argmax(outputs[1].detach().numpy(), axis = 1).tolist()
    y_true += l_b.detach().numpy().squeeze().tolist()

def area_under_the_curve():
		return roc_auc_score(y_true, y_pred)

def mean_average_precision():
  return average_precision_score(y_true, y_pred)

def accuracy():
  return accuracy_score(y_true, y_pred)

def f1():
  return f1_score(y_true, y_pred, average='micro')

metrics = {'Model' : model_name, 
      'AUC' : area_under_the_curve(), 
      'MAP' : mean_average_precision(), 
      'Accuracy' : accuracy(),
      'F1' : f1()
      }

if not os.path.exists('results.csv'):
  results_df = po.DataFrame(columns = ['Model'])
else:
  results_df = po.read_csv('results.csv')

results_df = results_df.append(metrics, ignore_index=True)

results_df

results_df.to_csv('results.csv', index=False)

