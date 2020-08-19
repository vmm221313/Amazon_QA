# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: 'Python 3.7.7 64-bit (''tf2'': conda)'
#     name: python_defaultSpec_1597833233214
# ---

import pickle
import pandas as po
from tqdm import tqdm

# + tags=[]
with open('data/data_v2/Auto_meta_qar.pkl', 'rb') as f:
    data = pickle.load(f)
# -

len(data)

row = data[list(data.keys())[0]]
row

row['title']

row['category']

row['description']

row['questions_answers']

row['questions_answers'][0]

row['questions_answers'][0][0] # question

row['questions_answers'][0][1] # label

row['questions_answers'][0][2] # reviews

X = []
y = []

# + tags=[]
df = po.DataFrame()
for key in tqdm(data, total=len(data)):
    row = data[key]

    meta = ' '.join([row['title'], ' '.join(row['category']), ' '.join(row['description'])])
    
    for i in range(len(row['questions_answers'])):
        df_row = {}
        df_row['meta'] = meta

        ques = row['questions_answers'][i][0]
        reviews = ' '.join(row['questions_answers'][i][2])
        df_row['text'] = ' '.join([ques, reviews])

        target = row['questions_answers'][i][1]
        if target == 'Y':
            df_row['target'] = 1 
        elif target == 'N':
            df_row['target'] = 0
        else:
            raise ValueError
    
        df = df.append(df_row, ignore_index=True)
# -

df

df.to_csv('data/Auto_meta_qar.csv', index=False)
