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
#     name: python_defaultSpec_1597833256475
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

len(row['questions_answers'][0][2])

X = []
y = []

# + tags=[]
df = po.DataFrame()
for j, key in enumerate(tqdm(data, total=len(data))):
    row = data[key]

    try:
        meta = ' '.join([row['title'], ' '.join(row['category']), ' '.join(row['description'])])

        for i in range(len(row['questions_answers'])):
            df_row = {}
            
            df_row['meta'] = meta
            
            ques = row['questions_answers'][i][0]
            #df_row['question'] = ques
            
            assert len(row['questions_answers'][i][2]) == 10

            for j in range(len(row['questions_answers'][i][2])):
                df_row['review_{}'.format(j)] = row['questions_answers'][i][2][j]

            #reviews = ' '.join(row['questions_answers'][i][2])
            #print(type(reviews))
            #df_row['review'] = reviews

            df_row['ques'] = ques#' '.join([ques, reviews])
            #print(' '.join([ques, reviews]))
            #print(type(ques))
            #print(df_row['text'])

            target = row['questions_answers'][i][1]
            if target == 'Y':
                df_row['target'] = 1 
            elif target == 'N':
                df_row['target'] = 0
            else:
                raise ValueError

            #print(df_row['text'])

            df = df.append(df_row, ignore_index=True)
                
            '''
            for ans in row['questions_answers'][0][2]:
                df_row = {}
                df_row['meta'] = meta
                df_row['question'] = ques
                df_row['review'] = ans
                
                df = df.append(df_row, ignore_index=True)
            '''

    except:
        print(row)
        continue
# -



df

df['target'].value_counts()

df.to_csv('data/Auto_meta_qar_clip_rescale.csv', index=False)

df['len_ques'] = df['ques'].apply(lambda x: len(x.split(' ')))

df['len_ques'].mean()

df['len_ques'].std()

df['len_review'] = df['review_1'].apply(lambda x: len(x.split(' ')))

df['len_review'].mean()

df['len_review'].std()

# +
# Pad/truncate everything to 50

# +
# varun/new_dir/*

# +
# Electronics, Home & Kitchen,  
