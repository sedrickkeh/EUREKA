import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.trainer_utils import set_seed
from utils import *
from trainer import CLS_Layer, PET_layer, Sent_DAN_Simple
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
from scipy.special import softmax
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def remove_extra_text(txt):
    return txt.replace("[START_EUPH]", "").replace("[END_EUPH]", "").replace("@", "")

def find_context_sentence(sent):
    split_sent = sent.split(".")
    sent_w_euph = [x for x in split_sent if "[START_EUPH]" in x and "[END_EUPH]" in x]
    return sent_w_euph[0].replace("[START_EUPH]", "").replace("[END_EUPH]", "").replace("@", "")

def main(args):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    special_tokens_dict = {'additional_special_tokens': ['[START_EUPH]','[END_EUPH]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model = AutoModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    if args.model_type == "cls":
        model.cls_layer = CLS_Layer(args.pet_dim, device)
    elif args.model_type == "pet":
        model.pooler = nn.Identity()
        model.pet = PET_layer(tokenizer, args.pet_dim, device)
    elif args.model_type == "dan":
        model.pooler = nn.Identity()
        model.pet = Sent_DAN_Simple(tokenizer, args.pet_dim, device)
    else:
        raise NotImplementedError
    model.to(device)
    model.load_state_dict(torch.load(args.model_path))

    df_test = pd.read_csv(args.test_path).drop(['index'], axis=1)
    df_test['utterance'] = df_test.apply(lambda row: clean(row['utterance']), axis=1)
    df_test['utterance'] = df_test.apply(lambda row : row['utterance'].replace("<", "[START_EUPH] ").replace(">", " [END_EUPH]"), axis=1)
    test_dataset = Dataset.from_pandas(df_test)
    test_tokenized = test_dataset.map(lambda batch: tokenizer(batch['utterance'], max_length=args.max_length, padding="max_length", truncation=True), batched=True, load_from_cache_file=False)
    preds, gtruth = [], []

    ####################
    # KNN Part:
    import faiss
    datastore = faiss.read_index(args.knn_path) 
    datastore_lambd = args.knn_lambda

    for inputs in tqdm(test_tokenized):
        input_ids = torch.Tensor(inputs['input_ids']).long().reshape([1, args.max_length]).to(device)
        attention_mask = torch.Tensor(inputs['attention_mask']).long().reshape([1, args.max_length]).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if args.model_type=="cls":
            logits = model.cls_layer(outputs['pooler_output'])
        else:
            last_hidden_state = outputs['last_hidden_state']
            logits = model.pet(last_hidden_state, input_ids)

        if datastore is not None:
            # Replace special tokens
            orig_sents = remove_extra_text(find_context_sentence(inputs["utterance"]))
            roberta_encoded = tokenizer(orig_sents, padding=True, truncation=True, return_tensors="pt")
            roberta_outputs = model(input_ids=roberta_encoded['input_ids'].to(device), attention_mask=roberta_encoded['attention_mask'].to(device))
            emb_phrase = roberta_outputs["last_hidden_state"][:, 0, :]
            distances, neighbors = datastore.search(emb_phrase.cpu().detach().numpy(), 5)
            distance_weights = softmax(-distances, axis=1)
            class_sm = np.zeros((distances.shape[0], 2))
            df_train = pd.read_csv('./data/train_original.csv')

            for batch_ind, kn in enumerate(neighbors):
                kn_labels = [df_train.iloc[i]["label"] for i in kn]
                for i, lab in enumerate(kn_labels):
                    class_sm[batch_ind][lab] += distance_weights[batch_ind][i]

            class_sm = softmax(class_sm, axis=1)
            class_prob_knn = datastore_lambd * class_sm
            class_prob_model = (1 - datastore_lambd) * softmax(logits.cpu().detach().numpy())
            logits = torch.tensor(class_prob_knn + class_prob_model).float()
            logits = logits.to(device)

        gtruth.append(inputs['label'])    
        preds.append(torch.argmax(logits, dim=1).item())    
    print(classification_report(gtruth, preds, digits=4))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='roberta-large')
    parser.add_argument("--model_type", type=str, default='pet')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, default="./data/test_split.csv")
    parser.add_argument("--knn_path", type=str, default="./data/train_index_large_v2.faiss")
    parser.add_argument("--knn_lambda", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()
    args.pet_dim = 1024 if "large" in args.model else 768
    
    main(args)