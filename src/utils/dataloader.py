# -*- coding: UTF-8 -*-
# ========================================================
#   Copyright (C) 2024 All rights reserved.
#   Project  : MTKMed 
#   Filename : dataloader.py
#   Author   : petal
#   Date     : 2024/11/18
#   Desc     : 
# ========================================================
import os

import numpy as np
import torch
import json
import pandas as pd
import warnings

import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
warnings.filterwarnings("ignore", category=FutureWarning)

class Voc():
    def __init__(self, path):
        self.path = path
        self.word2idx = self.load_data()
        self.idx2word = self.transform()

    def __len__(self):
        return len(self.word2idx)

    def load_data(self):
        with open(self.path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    def transform(self):
        return dict(zip(self.word2idx.values(), self.word2idx.keys()))


class MedDataset(Dataset):
    def __init__(self, data_path, device):
        """
        Args:
            data_path: path of target data
            split_rate: rate [train, valid, test]
        """
        self.data_path = data_path
        self.device = device
        self.tokName = ['cure', 'evaluation', 'symptom']

        self.doctor_info = None
        self.patient_info = None
        self.kg = None
        self.output = None
        self.label = None
        self.ssc = None

        # vocabulary dictionary
        self.target_names = ['doctor', 'patient', 'ent', 'relation', 'cure', 'evaluation', 'symptom']

        self.doctor_voc, self.patient_voc, self.ent_voc, self.relation_voc, \
            self.cure_voc, self.evaluation_voc, self.symptom_voc = self.load_voc()

        self.load_dataset()

        # assist dict
        self.doctor2id = None
        self.patient2id = None

    def load_voc(self):
        # organize dataset based on the rating condition
        result = []
        [result.append(Voc(os.path.join(self.data_path, 'side_' + name + '2id.json'))) for name in self.target_names]
        assert len(result) == len(self.target_names)
        return result

    def cal_boundaries(self, target, num):
        assert target in self.tokName
        # cal
        hist_weight = self.doctor_info[target + '_hist_weight'].apply(lambda x: x[:50]).values.tolist()
        hist_weight = [int(ele) for elem in hist_weight for ele in elem]
        hist_weight = sorted(hist_weight, reverse=False)
        cell = 100 / (num + 1)
        return np.percentile(hist_weight, [round(cell * (i+1), 2) for i in range(num)])

    def load_dataset(self):
        doctor_info = pd.read_csv(os.path.join(self.data_path, 'doctor.csv'), sep='\t')
        for elem in self.tokName:
            doctor_info[elem + '_hist'] = doctor_info[elem + '_hist'].apply(lambda x: eval(x))
            doctor_info[elem + '_hist_weight'] = doctor_info[elem + '_hist_weight'].apply(lambda x: eval(x))
        patient_info = pd.read_csv(os.path.join(self.data_path, 'patient.csv'), sep='\t')
        kg_info = pd.read_csv(os.path.join(self.data_path, 'kg.csv'), sep='\t')
        ratings = pd.read_csv(os.path.join(self.data_path, 'ratings.csv'), sep='\t')

        tcols = np.array([[elem + '_hist', elem + '_hist_weight'] for elem in self.tokName]).flatten()
        doctor2hist = dict(zip(doctor_info['doctor_id'], doctor_info[tcols].values))

        label = torch.FloatTensor(ratings['label'].values).unsqueeze(-1).to(self.device)
        ssc = torch.FloatTensor(ratings['satisfying_score'].values).unsqueeze(-1).to(self.device)

        patient2query = dict(zip(patient_info['patient_id'], patient_info['query']))
        output = []
        rpad = tqdm.tqdm(range(len(ratings[:200])))
        rpad.set_description('construct dataset: ')
        for i in rpad:
            tmp = [torch.LongTensor(ratings.loc[i, ['patient_id', 'doctor_id']]).to(self.device)]
            did = ratings.loc[i, 'doctor_id']
            for j in range(0, 2 * len(self.tokName), 2):
                tmp.append(torch.LongTensor([doctor2hist[did][j], doctor2hist[did][j+1]]).to(self.device))
            tmp.append(patient2query[ratings.loc[i, 'patient_id']])
            output.append(tmp)
        self.doctor_info = doctor_info
        self.patient_info = patient_info
        self.kg = kg_info
        self.output = output
        self.label = label
        self.ssc = ssc

    def split_num(self, split_rate):
        if split_rate and isinstance(split_rate, list):
            sub = [int(len(self.output) * elem) for elem in split_rate]
            sub[-1] += len(self.output) - sum(sub)
            return sub
        else:
            return [0, 0, 0]

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        return self.output[idx], self.label[idx], self.ssc[idx]

    @staticmethod
    def collect_ground_truth(subset):
        ground_truth = defaultdict(list)
        for i, (dp, label, ssc) in enumerate(subset):
            if label[0].float() == 1.:
                ground_truth[dp[0][0].long().tolist()].append(dp[0][1].long().tolist())

        return ground_truth


    @staticmethod
    def collate_fn_no_padding(batch):
        x1_batch, x2_batch, x3_batch = [], [], []
        for x1, x2, x3 in batch:
            x1_batch.append(x1)
            x2_batch.append(x2)
            x3_batch.append(x3)
        return x1_batch, x2_batch, x3_batch


if __name__ == '__main__':
    # 创建数据集
    Mydevice = 'cpu'  # torch.device('cuda:{}'.format(0))
    ds = MedDataset('../../data/Med/', Mydevice)
    train_set, valid_set, test_set = random_split(ds, ds.split_num([0.8, 0.1, 0.1]))

    # 创建 DataLoader
    dataloader = DataLoader(
        train_set,
        batch_size=100,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=MedDataset.collate_fn_no_padding
    )

    # 使用 DataLoader
    for t1, t2, t3 in dataloader:
        print(t1[0], t2[0], t3[0])
