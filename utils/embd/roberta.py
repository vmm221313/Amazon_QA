import torch
#print(torch.__version__)
roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
roberta.eval()

def roberta_sent(sent):
    tokens = roberta.encode(sent)
    last_layer_features = roberta.extract_features(tokens[:512])    # max_token length is 512
    return last_layer_features[0][1:-1]

def roberta_para(para):
    embd_list=[]
    for sent in para:
        embd_list.append(roberta_sent(sent))
    return embd_list

def roberta_doc(doc):
    embd_list=[]
    for para in doc:
        embd_list.append(roberta_para(para))
    return(embd_list)  
    