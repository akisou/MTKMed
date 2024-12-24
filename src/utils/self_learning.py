import argparse
import sys
import os
import random
import logging
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import BertForSequenceClassification, BertTokenizer
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

# bert self learning

# loading text data for bert self learning
class NLIDataset(Dataset):
    def __init__(self, split, tokenizer):
        # load dataset
        self.split = split
        self.df = pd.read_csv(f'../data/text_embedding_related/{split}.csv', sep='\t')  #, encoding='utf-8')
        self.len = len(self.df)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # every item control
        text_a, text_b, label = self.df.iloc[idx, :].values
        label_id = int(label)
        label_tensor = torch.tensor(label_id)
        word_pieces = ['[CLS]']

        tokens_a = self.tokenizer.tokenize(str(text_a))
        if len(tokens_a) >= 250:
            tokens_a = tokens_a[:250]
        word_pieces += tokens_a + ['[SEP]']
        len_a = len(word_pieces)

        tokens_b = self.tokenizer.tokenize(str(text_b))
        if len(tokens_b) >= 250:
            tokens_b = tokens_b[:250]
        word_pieces += tokens_b + ['[SEP]']
        len_b = len(word_pieces) - len_a

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)

        return tokens_tensor, segments_tensor, label_tensor

    def __len__(self):
        return self.len

# bert self learning related parser
parser = argparse.ArgumentParser()
parser.add_argument('-seed', default=2023, type=int)
parser.add_argument('-epoch_num', default=20, type=int)
parser.add_argument('-batch_size', default=32, type=int)  # origin: 50
parser.add_argument('-accumulation_steps', default=5, type=int)
parser.add_argument('-embedding_dim', default=64, type=int)
args = parser.parse_args()

# set you available gpus
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()

# print('gpu num: ', n_gpu)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def create_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
    logging.info("predicting process of test data...")
    with torch.no_grad():
        for data in tqdm(dataloader):
            data = [t.to(device) for t in data if t is not None]
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors)
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions


def train(model, optimizer, trainloader, validloader, device):
    logging.info("train process of Bert self learning...")
    for epoch in trange(0, args.epoch_num):
        model.train()
        running_loss = []
        i = 0
        for data in tqdm(trainloader):
            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
            optimizer.zero_grad()
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors,
                            labels=labels)
            loss = outputs[0]
            if n_gpu > 1:
                loss = loss.mean()
            loss = loss / args.accumulation_steps
            loss.backward()
            running_loss.append(loss.item())
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
            i = i + 1
        _, train_acc = get_predictions(model, trainloader, compute_acc=True)
        logging.info('Epoch: %.2d, train loss: %.4f, train classification acc: %.4f', epoch, np.mean(running_loss), train_acc)
        if not os.path.exists(f'../model/checkpoints'):
            os.makedirs(f'../model/checkpoints')
        if n_gpu > 1:
            model_state_dict = model.module.bert.state_dict()
        else:
            model_state_dict = model.module.bert.state_dict()
        torch.save(model_state_dict, f'../model/checkpoints/model_{epoch}.bin')
        model.eval()
        _, valid_acc = get_predictions(model, validloader, compute_acc=True)
        logging.info('Epoch: %.2d, valid classification acc: %.4f', epoch, valid_acc)


def main():
    tokenizer = BertTokenizer.from_pretrained('../model/mc_bert_base/')
    model = BertForSequenceClassification.from_pretrained('../model/mc_bert_base/', num_labels=2)

    # change the hidden size of the pre-load model : 64
    model.config.hidden_size = 64

    model = model.to(device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    logging.info("loading text data for bert self learning...")
    trainset = NLIDataset('train', tokenizer=tokenizer)
    validset = NLIDataset('valid', tokenizer=tokenizer)
    testset = NLIDataset('test', tokenizer=tokenizer)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=create_batch)
    validloader = DataLoader(validset, batch_size=args.batch_size, collate_fn=create_batch)
    testloader = DataLoader(testset, batch_size=args.batch_size, collate_fn=create_batch)

    _, test_acc = get_predictions(model, testloader, compute_acc=True)
    logging.info('Model without training classification acc: %.4f', test_acc)
    # model training
    train(model, optimizer, trainloader, validloader, device)
    logging.info('Model training done!')
    # model testing
    logging.info('Testing model...')
    _, test_acc = get_predictions(model, testloader, compute_acc=True)
    logging.info('Model testing done with classification acc: %.4f', test_acc)

if __name__ == '__main__':
    main()
