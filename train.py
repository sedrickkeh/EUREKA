import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from transformers.trainer_utils import set_seed
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from utils import *
from trainer import MyTrainer, CLS_Layer, PET_layer, Sent_DAN_Simple
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    set_seed(args.seed)
    df_train = pd.read_csv(args.train_path).drop(['index'], axis=1)
    df_val = pd.read_csv(args.valid_path).drop(['index'], axis=1)
    df_test = pd.read_csv(args.test_path).drop(['index'], axis=1)
    if not args.disable_cleaning:
        cands = pd.read_csv(args.cleaning_path)
        utterance_changes = []
        for i in range(len(cands)):
            if cands.label[i] != cands.label_new[i]:
                utterance_changes.append(cands.utterance[i])
        for i in range(len(df_train)):
            if df_train.utterance[i] in utterance_changes:
                df_train.label[i] = 1-df_train.label[i]
    if not args.disable_augmentation:
        df_additional = pd.read_csv(args.augmentation_path)
        df_train = pd.concat([df_train, df_additional]).reset_index(drop=True)
        df_train = df_train.sample(len(df_train)).reset_index(drop=True)
    
    # Remove the @@@ stuff
    df_train['utterance'] = df_train.apply(lambda row: clean(row['utterance']), axis=1)
    df_val['utterance'] = df_val.apply(lambda row: clean(row['utterance']), axis=1)
    df_test['utterance'] = df_test.apply(lambda row: clean(row['utterance']), axis=1)

    df_train['utterance'] = df_train.apply(lambda row : row['utterance'].replace("<", "[START_EUPH] ").replace(">", " [END_EUPH]"), axis=1)
    df_val['utterance'] = df_val.apply(lambda row : row['utterance'].replace("<", "[START_EUPH] ").replace(">", " [END_EUPH]"), axis=1)
    df_test['utterance'] = df_test.apply(lambda row : row['utterance'].replace("<", "[START_EUPH] ").replace(">", " [END_EUPH]"), axis=1)

    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    special_tokens_dict = {'additional_special_tokens': ['[START_EUPH]','[END_EUPH]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    train_tokenized = train_dataset.map(lambda batch: tokenizer(batch['utterance'], max_length=args.max_length, padding="max_length", truncation=True), batched=True, load_from_cache_file=False)
    val_tokenized = val_dataset.map(lambda batch: tokenizer(batch['utterance'], max_length=args.max_length, padding="max_length", truncation=True), batched=True, load_from_cache_file=False)
    test_tokenized = test_dataset.map(lambda batch: tokenizer(batch['utterance'], max_length=args.max_length, padding="max_length", truncation=True), batched=True, load_from_cache_file=False)

    trainer_args = TrainingArguments(
        output_dir = args.output_path,
        evaluation_strategy = "epoch",
        save_strategy = 'epoch',
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        num_train_epochs = args.num_epochs,
        seed = args.seed,
        load_best_model_at_end=True,
        learning_rate = args.lr
    )

    def model_init():
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
        return model

    def compute_metrics(p):    
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average='macro')
        precision = precision_score(y_true=labels, y_pred=pred, average='macro')
        f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    trainer = MyTrainer(
        model_init=model_init,
        args=trainer_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data/train_split.csv")
    parser.add_argument("--valid_path", type=str, default="./data/dev_split.csv")
    parser.add_argument("--test_path", type=str, default="./data/test_split.csv")
    parser.add_argument("--cleaning_path", type=str, default="./data/candidate_replacements.csv")
    parser.add_argument("--augmentation_path", type=str, default="./data/augmentation_substitution_wikimatrix.csv")
    parser.add_argument("--disable_cleaning", action="store_true")
    parser.add_argument("--disable_augmentation", action="store_true")
    parser.add_argument("--output_path", type=str)

    parser.add_argument("--model", type=str, default='roberta-large')
    parser.add_argument("--model_type", type=str, default='pet')
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()
    args.pet_dim = 1024 if "large" in args.model else 768
    if args.output_path is None:
        args.output_path = f"./output_{args.seed}"

    main(args)