import argparse
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    for inputs in tqdm(test_tokenized):
        input_ids = torch.Tensor(inputs['input_ids']).long().reshape([1, args.max_length]).to(device)
        attention_mask = torch.Tensor(inputs['attention_mask']).long().reshape([1, args.max_length]).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if args.model_type=="cls":
            logits = model.cls_layer(outputs['pooler_output'])
        else:
            last_hidden_state = outputs['last_hidden_state']
            logits = model.pet(last_hidden_state, input_ids)
        gtruth.append(inputs['label'])
        preds.append(torch.argmax(logits, dim=1).item())
    print(classification_report(gtruth, preds, digits=4))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='roberta-large')
    parser.add_argument("--model_type", type=str, default='pet')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, default="./data/test_split.csv")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()
    args.pet_dim = 1024 if "large" in args.model else 768
    
    main(args)