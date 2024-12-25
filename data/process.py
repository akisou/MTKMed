# -*- coding: UTF-8 -*-
# ========================================================
#   Copyright (C) 2024 All rights reserved.
#   Project:MTKMed 
#   Filename : process.py.py
#   Author   : petal
#   Date     : 2022-11-24
#   Desc     : 2024/11/4
# ========================================================

import os
import pickle
import re
import sys
import stat
sys.path.append("E://program/PycharmProjects/MTKMed")
import pandas as pd
import random
import operator
import dill
import json
import torch
import tqdm
import numpy as np
import jieba
import logging
from rdflib import Graph
from collections import defaultdict, Counter
from pysenti import ModelClassifier
from transformers import AutoModelForTokenClassification, BertTokenizerFast, BertTokenizer, BertModel
from src.models.medicalNer.MedicalNer import MedicalNerModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
os.chmod('./', stat.S_IRWXU)
jieba.load_userdict('E://program/PycharmProjects/MTKMed/data/corpus/user_dict.txt')
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)
# random.seed(2024)
class Preprocessor:
    # data preprocess
    def __init__(self):
        self.data_path = 'E://program/PycharmProjects/doctor_recomend/data/haodf'
        self.root_path = './' # 'E://program/PycharmProjects/MTKMed/data/'
        self.output_path = './output' # 'E://program/PycharmProjects/MTKMed/data/output'
        self.dataset_path = './Med'  # 'E://program/PycharmProjects/MTKMed/data/Med'
        print(os.getcwd())
        self.corpus_path = os.path.join(self.root_path, 'corpus')

        self.doctor_filtered = None
        self.patient_filtered = None
        self.patient_sampled = None
        self.kg = None
        self.rating = None
        self.query_emb = self.query_embedding(target='patients_sampled')
        self.doctor_token_hist = None
        self.satisfying_score = None

        self.department = [
            '心血管内科',
            '神经内科',
            '消化内科',
            '内分泌科',
            '呼吸科',
            '感染内科',
            '精神科',
            '眼科',
            '口腔科',
            '肿瘤内科',
            '妇科',
            '男科',
            '儿科',
            '康复科'
        ]

        self.patient_num_threshold = 30
        self.consultation_num_threshold = 5
        self.doctor_reply_rate = 0.3
        self.query_length_threshold = 5
        self.consultation_length_threshold = 50
        self.max_length = 500
        self.sample_range = [4, 8]

        self.patient_range = [15, 20]

        self.nonsense_words = [
            '图片资料',
            '仅主诊医生和患者本人可见'
        ]

        self.stopwords = self.load_stopwords()
        self.symptoms = self.load_symptoms()
        self.stopstr = r'[^a-zA-Z\u4e00-\u9fa5]|担任'

        self.doctor_attrs = [
            'doctor_id',
            'profession_direction',
            'profile',
            'hospital',
            'further_experience',
            'team',
            'social_title',
            'work_experience',
            'education_experience',
            'cure_experience',
            'evaluation_type_set'
        ]

        self.patient_attrs = [
            'patient_id',
            'doctor_id',
            'department',
            'age',
            'gender',
            'total_communication_times',
            'doctor_reply_times',
            'query',
            'height_weight',
            'disease_generalization',
            'disease_duration',
            'history_hospital',
            'drug_using',
            'allergic_history',
            'disease_history',
            'disease_case_summary',
            'advice'
        ]
        self.patient_txt_col = [
            'query',
            'height_weight',
            'disease_generalization',
            'disease_duration',
            'history_hospital',
            'drug_using',
            'allergic_history',
            'disease_history',
            'disease_case_summary',
            'advice'
        ]

        self.triple_head = [
            'head',
            'relation',
            'target'
        ]

        self.hist_columns = [
            'doctor_id',
            'cure_hist',
            'cure_hist_weight',
            'evaluation_hist',
            'evaluation_hist_weight'
        ]

        self.minmaxscaler = MinMaxScaler(feature_range=(0, 1))

        self.satisfying_weight = [
            0.15,  # pos_cosine,
            0.1,  # neg_cosine,
            0.1,  # doctor_positive_var,
            0.1,  # doctor_whole_score['positive_prob'],
            0.1,  # doctor_emotion,
            0.15,  # patient_positive_var,
            0.15,  # patient_whole_score['positive_prob'],
            0.15,  # patient_emotion,

            # total 1.0
            0.3,  # emotion weight
            # ### other weight

            0.3,  # medical_word_weight,
            0.2,  # doctor_reply_times_weight,
            0.15,  # doctor_reply_length_weight,
            0.05  # advice_flag
        ]

    def load_stopwords(self):
        stopwords = set()
        with open(os.path.join(self.corpus_path, 'stopwords.txt'), 'r', encoding='utf-8') as file:
            for line in file:
                stopwords.add(line.strip())
        return stopwords

    def load_symptoms(self):
        with open(os.path.join(self.corpus_path, 'symptoms.txt'), 'r', encoding='utf-8') as f:
            symptoms = set([elem.strip() for elem in f.readlines()])
        return symptoms

    # load data
    def filter_data(self):
        # filter physician with enough profile info
        if os.path.exists(os.path.join(self.output_path, 'doctors_filtered.csv')):
            doctors_filtered = pd.read_csv(os.path.join(self.output_path, 'doctors_filtered.csv'), sep='\t')
        else:
            doctors_filtered = pd.DataFrame([], columns=self.doctor_attrs)
            pdepart = tqdm.tqdm(self.department)
            pdepart.set_description('filtering doctors: ')
            for depart in pdepart:
                doctor = pd.read_csv(os.path.join(self.data_path, depart, 'doctor.csv'), sep=',')
                doctor = doctor[(doctor['profile'] != '') & (doctor['cure_experience'] != '[[], []]')
                                & (doctor['evaluation_type_set'] != '[[], []]')]
                doctors_meaningless = []
                for doct in doctor['doctor_id'].unique():
                    patient_path = os.path.join(self.data_path, depart, str(doct))
                    if (not os.path.exists(patient_path)) or \
                            (len(os.listdir(patient_path)) < self.patient_num_threshold):
                        doctors_meaningless.append(doct)

                doctor = doctor[~doctor['doctor_id'].isin(doctors_meaningless)]
                doctor['profession_direction'] = [depart] * len(doctor)
                doctors_filtered = pd.concat([doctors_filtered, doctor], ignore_index=True)
            doctors_filtered = doctors_filtered[self.doctor_attrs]
            doctors_filtered = doctors_filtered.reset_index(drop=True)
            doctors_filtered.to_csv(os.path.join(self.output_path, 'doctors_filtered.csv'), sep='\t', index=False)
        self.doctor_filtered = doctors_filtered
        logging.info('doctors filter completed!')

        # filter consultations with enough information
        if os.path.exists(os.path.join(self.output_path, 'patients_filtered.csv')):
            patients_filtered = pd.read_csv(os.path.join(self.output_path, 'patients_filtered.csv'), sep='\t')
        else:
            patients_filtered = pd.DataFrame([], columns=self.patient_attrs)
            pdoct = tqdm.tqdm(range(len(doctors_filtered)))
            pdoct.set_description('filtering patients: ')
            for i in pdoct:
                patient_path = os.path.join(self.data_path, str(doctors_filtered.loc[i, 'profession_direction']),
                                            str(doctors_filtered.loc[i, 'doctor_id']))
                patients = pd.read_csv(os.path.join(patient_path, 'patient.csv'), sep=',')
                # filter
                patients = patients[# (~patients['age'].isnull()) &
                                    # (~patients['gender'].isnull()) &
                                    (patients['total_communication_times'] >= self.consultation_num_threshold) &
                                    (patients['doctor_reply_times'] >=
                                     self.consultation_num_threshold * self.doctor_reply_rate) &
                                    (patients['query'].str.len() >= self.query_length_threshold) # &
                                    # (~patients['disease_case_summary'].isnull()) &
                                    # (~patients['advice'].isnull())
                                    ]

                patient_silent = []
                for pati in patients['patient_id'].unique():
                    pati_consul = pd.read_csv(os.path.join(self.data_path,
                                                           str(doctors_filtered.loc[i, 'profession_direction']),
                                                           str(doctors_filtered.loc[i, 'doctor_id']),
                                                           str(pati) + '.txt'), sep='\t')
                    pati_consul = pati_consul[(pati_consul['speaker'].isin(['医生', '患者'])) &
                                              (~pati_consul['word'].str.contains('|'.join(self.nonsense_words),
                                                                                 na=False))]

                    if len(''.join(pati_consul['word'].astype(str))) < self.consultation_length_threshold:
                        patient_silent.append(pati)
                patients = patients[~patients['patient_id'].isin(patient_silent)]
                patients['doctor_id'] = [int(doctors_filtered.loc[i, 'doctor_id'])] * len(patients)
                patients['department'] = [str(doctors_filtered.loc[i, 'profession_direction'])] * len(patients)
                patients = patients[self.patient_attrs]
                patients_filtered = pd.concat([patients_filtered, patients], ignore_index=True)

        self.patient_filtered = patients_filtered
        patients_filtered.to_csv(os.path.join(self.output_path, 'patients_filtered.csv'), sep='\t', index=False)
        logging.info('patients filter completed!')

        # sample high quality consultations and patients
        # one physician with about 10-20 patients
        if os.path.exists(os.path.join(self.output_path, 'patients_sampled.csv')):
            patients_sampled = pd.read_csv(os.path.join(self.output_path, 'patients_sampled.csv'), sep='\t')
        else:
            pid_sampled = []
            dfpad = tqdm.tqdm(range(len(doctors_filtered)))
            dfpad.set_description('sample patients: ')
            for i in range(len(doctors_filtered)):
                sub_pids = patients_filtered[
                    patients_filtered['doctor_id'] == doctors_filtered.loc[i, 'doctor_id']]['patient_id'].tolist()
                pid_sampled.extend(random.sample(sub_pids, random.randint(*self.patient_range))
                                   if len(sub_pids) > self.patient_range[1] else sub_pids)

            patients_sampled = patients_filtered[patients_filtered['patient_id'].isin(pid_sampled)]
            patients_sampled = patients_sampled.reset_index(drop=True)
            patients_sampled.to_csv(os.path.join(self.output_path, 'patients_sampled.csv'), sep='\t', index=False)
        self.patient_sampled = patients_sampled
        logging.info('patients sample completed!')

    def extract_triple(self):
        # extract triples mainly including social relations
        source = []
        target = []
        relation = []
        if os.path.exists(os.path.join(self.output_path, 'kg.csv')):
            self.kg = pd.read_csv(os.path.join(self.output_path, 'kg.csv'), sep='\t')
        else:
            tripad = tqdm.tqdm(range(len(self.doctor_filtered)))
            tripad.set_description('extract triple: ')
            for i in tripad:
                did = self.doctor_filtered.loc[i, 'doctor_id']
                for attr in self.doctor_attrs[1:]:
                    if attr == 'hospital':
                        for hos in eval(self.doctor_filtered.loc[i, attr]):
                            if not pd.isnull(hos):
                                source.append(did)
                                posi = max(hos.find('医院'), hos.find('中心'))
                                target.append(re.sub(self.stopstr, '', hos[:posi + 2]))
                                relation.append('doctor.work_in.hospital')
                    elif attr == 'team':
                        if not pd.isnull(self.doctor_filtered.loc[i, attr]):
                            for per in eval(self.doctor_filtered.loc[i, attr]):
                                source.append(did)
                                target.append(per)
                                relation.append('doctor.cooperate_with.doctor')
                    elif attr in ['further_experience', 'work_experience', 'education_experience']:
                        if not pd.isnull(self.doctor_filtered.loc[i, attr]):
                            for exper in eval(self.doctor_filtered.loc[i, attr]):
                                source.append(did)
                                institute = exper.split(';')[1]
                                posi = max(institute.find('医院'), institute.find('大学'),
                                           institute.find('学院'), institute.find('中心'))
                                target.append(re.sub(self.stopstr, '', institute[:posi + 2]))
                                relation.append('doctor.experience.institute')
                    elif attr == 'social_title':
                        if not pd.isnull(self.doctor_filtered.loc[i, attr]):
                            for title in eval(self.doctor_filtered.loc[i, attr]):
                                posi = title.find('会')
                                if len(title[:posi+1]) > 2:
                                    source.append(did)
                                    target.append(re.sub(self.stopstr, '', title[:posi + 1]))
                                    relation.append('doctor.member_of.institute')

            doctor_relation = pd.DataFrame({
                'head': source,
                'relation': relation,
                'target': target
            })
            doctor_relation = doctor_relation[doctor_relation['target'].str.len() > 2]

            doctor_relation = doctor_relation.drop_duplicates(keep='first')

            doctor_relation.sort_values(by='head', inplace=True, ascending=True)
            doctor_relation = doctor_relation.reset_index(drop=True)

            doctor_relation.to_csv(os.path.join(self.output_path, 'kg.csv'), sep='\t', index=False)
        logging.info('extract kg triples completed!')

    def extract_med_token_1(self):
        # extract medical info including disease, symptom.
        # disease
        # history disease
        logging.info('extract disease history tokens...')
        doctor_token_hist = self.doctor_filtered.copy()
        doctor_token_hist['cure_hist'] = doctor_token_hist['cure_experience'].apply(lambda x: eval(x)[0])
        doctor_token_hist['cure_hist_weight'] = doctor_token_hist['cure_experience'].apply(lambda x: eval(x)[1])
        doctor_token_hist['evaluation_hist'] = doctor_token_hist['evaluation_type_set'].apply(lambda x: eval(x)[0])
        doctor_token_hist['evaluation_hist_weight'] = (doctor_token_hist['evaluation_type_set'].
                                                       apply(lambda x: eval(x)[1]))
        doctor_token_hist = doctor_token_hist[self.hist_columns]
        doctor_token_hist = doctor_token_hist.dropna(how='any')
        logging.info('completed!')

        logging.info('extract symptom history tokens...')

        pdoct = tqdm.tqdm(range(len(self.doctor_filtered)))
        pdoct.set_description('extract symptom history tokens: ')
        f_symptom = []
        f_count = []

        record_dialog = ''
        ddata = pd.read_csv(os.path.join(self.output_path, 'rd.csv'), sep='\t')
        for i in pdoct:
            did = self.doctor_filtered.loc[i, 'doctor_id']
            wrds = ddata[ddata['doctor_id']==did]['rd'].values[0]

            items = []
            for wrd in jieba.lcut(wrds):
                if len(wrd) > 1 and wrd in self.symptoms:
                    items.append(wrd)
            symptom_hist = Counter(items)

            if symptom_hist:
                symptom_hist = sorted(symptom_hist.items(), key=operator.itemgetter(1), reverse=True)
                f_symptom.append([elem[0] for elem in symptom_hist])
                f_count.append([elem[1] for elem in symptom_hist])
            else:
                f_symptom.append([])
                f_count.append([])

        doctor_token_hist['symptom_hist'] = f_symptom
        doctor_token_hist['symptom_hist_weight'] = f_count

        doctor_token_hist.to_csv(os.path.join(self.output_path, 'doctor_token_hist.csv'), sep='\t', index=False)
        logging.info('completed!')
    def extract_med_token(self):
        # extract medical info including disease, symptom.
        # disease
        # history disease
        logging.info('extract disease history tokens...')
        hist_token = ['cure_hist', 'evaluation_hist', 'symptom_hist']
        hist_weight = ['cure_hist_weight', 'evaluation_hist_weight', 'symptom_hist_weight']
        dth_path = os.path.join(self.output_path, 'doctor_token_hist.csv')
        if os.path.exists(dth_path):
            doctor_token_hist = pd.read_csv(dth_path, sep='\t')
            for elem in hist_token + hist_weight:
                doctor_token_hist[elem] = doctor_token_hist[elem].apply(lambda x: eval(x))
            for elem in hist_weight:
                doctor_token_hist[elem] = doctor_token_hist[elem].apply(lambda x: [int(elem) for elem in x])
            self.doctor_token_hist = doctor_token_hist
            return doctor_token_hist

        doctor_token_hist = self.doctor_filtered.copy()
        doctor_token_hist['cure_hist'] = doctor_token_hist['cure_experience'].apply(lambda x: eval(x)[0])
        doctor_token_hist['cure_hist_weight'] = doctor_token_hist['cure_experience'].apply(lambda x: eval(x)[1])
        doctor_token_hist['evaluation_hist'] = doctor_token_hist['evaluation_type_set'].apply(lambda x: eval(x)[0])
        doctor_token_hist['evaluation_hist_weight'] = (doctor_token_hist['evaluation_type_set'].
                                                       apply(lambda x: eval(x)[1]))
        doctor_token_hist = doctor_token_hist[self.hist_columns]
        doctor_token_hist = doctor_token_hist.dropna(how='any')
        logging.info('completed!')

        logging.info('extract symptom history tokens...')

        # symptom
        # # chinese medical ner model import
        # tokenizer = BertTokenizerFast.from_pretrained('../src/models/medicalNer')
        # ner_model = AutoModelForTokenClassification.from_pretrained('../src/models/medicalNer')

        pdoct = tqdm.tqdm(range(len(self.doctor_filtered)))
        pdoct.set_description('extract symptom history tokens: ')
        f_symptom = []
        f_count = []

        # words = []
        for i in pdoct:
            did = self.doctor_filtered.loc[i, 'doctor_id']
            path = os.path.join(self.data_path, str(self.doctor_filtered.loc[i, 'profession_direction']), str(did))
            patients_info = pd.read_csv(os.path.join(path, 'patient.csv'), sep=',')
            patients_info['patient_id'] = patients_info['patient_id'].astype(str)
            self.patient_filtered['doctor_id'] = self.patient_filtered['doctor_id'].astype(str)
            self.patient_filtered['patient_id'] = self.patient_filtered['patient_id'].astype(str)
            patients = [elem.split('.')[0] for elem in os.listdir(path)]
            patients.remove('patient')
            valid_patients = self.patient_filtered[self.patient_filtered['doctor_id'] == str(did)]['patient_id']

            record_dialog = ''
            for patient in valid_patients:
                if not os.path.exists(os.path.join(path, str(patient) + '.txt')):
                    continue
                record = patients_info[patients_info['patient_id'] == patient][self.patient_txt_col].values[0]
                record = ','.join([str(elem) for elem in record if type(elem) != float])

                dialog = pd.read_csv(os.path.join(path, str(patient) + '.txt'), sep='\t')
                dialog = dialog[(dialog['speaker'].isin(['医生', '患者'])) &
                                (~dialog['word'].str.contains('|'.join(self.nonsense_words), na=False))]
                dialog = ''.join([str(elem) for elem in dialog['word'] if type(elem) != float])
                record_dialog += re.sub(self.stopstr, '', ''.join([record, dialog]))
                if len(record_dialog) == 0:
                    continue

            # # words.append(record_dialog)
            # sentence = [record_dialog[i:i + self.max_length] for i in range(0, len(record_dialog), self.max_length)]
            #
            # # encode and search
            # inputs = tokenizer(sentence, return_tensors="pt", padding=True,
            #                    add_special_tokens=False)
            # outputs = ner_model(**inputs)
            # outputs = outputs.logits.argmax(-1) * inputs['attention_mask']

            # # decode
            # symptom = MedicalNerModel.format_outputs(sentence, outputs)
            # symptom = [elem['word'] for res in symptom for elem in res if len(elem['word']) <= 6]
            # symptom_hist = Counter(symptom)
            items = []
            for wrd in jieba.lcut(record_dialog):
                if len(wrd) > 1 and wrd in self.symptoms:
                    items.append(wrd)
            symptom_hist = Counter(items)

            if symptom_hist:
                symptom_hist = sorted(symptom_hist.items(), key=operator.itemgetter(1), reverse=True)
                f_symptom.append([elem[0] for elem in symptom_hist])
                f_count.append([elem[1] for elem in symptom_hist])
            else:
                f_symptom.append([])
                f_count.append([])

        doctor_token_hist['symptom_hist'] = f_symptom
        doctor_token_hist['symptom_hist_weight'] = f_count

        doctor_token_hist.to_csv(os.path.join(self.output_path, 'doctor_token_hist.csv'), sep='\t', index=False)
        # tt = pd.DataFrame(zip(self.doctor_filtered['doctor_id'], words), columns=['doctor_id', 'rd'])
        # tt.to_csv(os.path.join(self.output_path, 'rd.csv'), sep='\t', index=False)
        logging.info('completed!')

    def query_embedding(self, target='patients_filtered'):
        # find similar query based on query nlp embedding
        # mcbert model
        if os.path.exists(os.path.join(self.output_path, 'q_embedding.pkl')):
            with open(os.path.join(self.output_path, 'q_embedding.pkl'), 'r', encoding='utf-8') as f:
                q_embedding = json.load(f)
                self.query_emb = q_embedding
            return q_embedding
        else:
            logging.info('loading mc_bert model...')
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            tokenizer = BertTokenizer.from_pretrained('../src/models/mcBert')
            model = BertModel.from_pretrained('../src/models/mcBert')
            model = model.to(device)

            patients = pd.read_csv(os.path.join(self.output_path, target + '.csv'), sep='\t')
            patients = dict(zip(patients['patient_id'], patients['query']))
            con_emb_dict = {}
            for idx, content in tqdm.tqdm(patients.items(), desc='query embedding: '):
                # the case that content is nan
                if type(content) == float and np.isnan(content):
                    con_emb_dict[idx] = -1
                    continue
                input_ids = torch.tensor([tokenizer.encode(str(content))])
                if len(input_ids[0].numpy().tolist()) > 512:
                    input_ids = (torch.from_numpy(np.array(input_ids[0].numpy().tolist()[0:512])).reshape(1, -1)
                                 .type(torch.LongTensor))
                input_ids = input_ids.to(device)
                with torch.no_grad():
                    features = model(input_ids)
                con_emb_dict[idx] = features[1].cpu().numpy()[0].tolist()
            logging.info('query embedding completed!')
            with open(os.path.join(self.output_path, 'q_embedding.pkl'), 'w') as f:
                json.dump(con_emb_dict, f, ensure_ascii=False)
            logging.info('query embedding saved!')
            return con_emb_dict

    def rating_creator(self):
        # create satisfaction score
        # [emotion score, medical vocab num, doctor reply times rate, doctor reply txt length rate, if has device, ]
        spath = os.path.join(self.output_path, 'satisfying_score.csv')
        if os.path.exists(spath):
            satisfying_score = pd.read_csv(spath, sep='\t')
            self.satisfying_score = satisfying_score
            return satisfying_score
        ppati = tqdm.tqdm(range(len(self.patient_filtered)))
        ppati.set_description('emotion scoring: ')

        satisfying_score = []
        for i in ppati:
            path = os.path.join(self.data_path, str(self.patient_filtered.loc[i, 'department']),
                                str(self.patient_filtered.loc[i, 'doctor_id']))
            dialog = pd.read_csv(os.path.join(path, str(self.patient_filtered.loc[i, 'patient_id']) + '.txt'), sep='\t')
            dialog = dialog[(dialog['speaker'].isin(['医生', '患者'])) &
                            (~dialog['word'].str.contains('|'.join(self.nonsense_words), na=False))]
            dialog = dialog.reset_index(drop=True)

            # emotion score process
            merged_conversations = []
            current_speaker = str(dialog.loc[0, 'speaker'])
            current_message = str(dialog.loc[0, 'word'])

            # merge
            for j in range(1, len(dialog)):
                row = dialog.iloc[j]

                if row['speaker'] == current_speaker:
                    # same speakr, merge
                    current_message += ' ' + str(row['word'])
                else:
                    # or save message and switch speaker
                    merged_conversations.append((current_speaker, current_message))
                    current_speaker = row['speaker']
                    current_message = str(row['word'])

            # save last
            merged_conversations.append((current_speaker, current_message))

            # match
            patient_dialog = []
            doctor_dialog = []
            for j in range(len(merged_conversations)):
                if merged_conversations[j][0] == '患者':
                    patient_dialog.append(merged_conversations[j][1])
                else:
                    doctor_dialog.append(merged_conversations[j][1])

            assert abs(len(patient_dialog) - len(doctor_dialog)) <= 1
            if abs(len(patient_dialog) - len(doctor_dialog)) == 1:
                if len(patient_dialog) > len(doctor_dialog):
                    if len(patient_dialog) == 1:
                        doctor_dialog = ['']
                    else:
                        sub = patient_dialog[-1]
                        patient_dialog = patient_dialog[:-1]
                        patient_dialog[-1] += sub
                else:
                    if len(doctor_dialog) == 1:
                        patient_dialog = ['']
                    else:
                        sub = doctor_dialog[-1]
                        doctor_dialog = doctor_dialog[:-1]
                        doctor_dialog[-1] += sub

            emotion_model = ModelClassifier()
            patient_score = [emotion_model.classify(elem) for elem in patient_dialog]
            patient_positive_score = [elem['positive_prob'] for elem in patient_score]
            patient_negative_score = [elem['negative_prob'] for elem in patient_score]
            patient_whole_score = emotion_model.classify(' '.join(patient_dialog))

            doctor_score = [emotion_model.classify(elem) for elem in doctor_dialog]
            doctor_positive_score = [elem['positive_prob'] for elem in doctor_score]
            doctor_negative_score = [elem['negative_prob'] for elem in doctor_score]
            doctor_whole_score = emotion_model.classify(' '.join(doctor_dialog))

            # cos sim
            pos_cosine = max(cosine_similarity([doctor_positive_score], [patient_positive_score]))
            neg_cosine = max(cosine_similarity([doctor_negative_score], [patient_negative_score]))

            doctor_positive_var = np.var(doctor_positive_score)
            patient_positive_var = np.var(patient_positive_score)

            doctor_emotion = max(doctor_whole_score.values())
            patient_emotion = max(patient_whole_score.values())

            ## dialog 医疗相关词汇占比
            sentence = re.sub(self.stopstr, '',
                              ''.join(''.join([str(elem) for elem in dialog['word'] if type(elem) != float])))
            all_words = [elem for elem in jieba.lcut(sentence)
                                                if elem not in self.stopwords]
            appear_sym = [elem for elem in all_words if elem in self.symptoms]

            # medical word weight
            medical_word_weight = len(appear_sym) / len(all_words)

            # doctor reply times weight
            doctor_reply_times_weight = len(dialog[dialog['speaker'] == '医生']) / len(dialog)

            # doctor reply txt length weight
            doctor_reply_length_weight = (len(''.join(dialog[dialog['speaker'] == '医生']['word'].astype(str)))
                                          / len(''.join(dialog['word'].astype(str))))

            # if has advice
            advice_flag = 0 if pd.isnull(self.patient_filtered.loc[i, 'advice']) else 1

            # concentrate
            satisfying_score.append([
                pos_cosine.tolist()[0],
                neg_cosine.tolist()[0],
                doctor_positive_var,
                doctor_whole_score['positive_prob'],
                doctor_emotion,
                patient_positive_var,
                patient_whole_score['positive_prob'],
                patient_emotion,
                medical_word_weight,
                doctor_reply_times_weight,
                doctor_reply_length_weight,
                advice_flag
            ])
        satisfying_score = np.array(satisfying_score)
        satisfying_score[:, [2, 5]] = self.minmaxscaler.fit_transform(satisfying_score[:, [2, 5]])

        emotion_part = np.matmul(satisfying_score[:, 0:8], np.array(self.satisfying_weight[0:8]).transpose())

        final_score = np.matmul(np.hstack((emotion_part.transpose().reshape(-1, 1), satisfying_score[:, 8:])),
                                np.array(self.satisfying_weight[8:]).transpose())

        result = pd.DataFrame({
            'doctor_id': self.patient_filtered['doctor_id'],
            'patient_id': self.patient_filtered['patient_id'],
            'satisfying_score': final_score
        })
        result.to_csv(os.path.join(self.output_path, 'satisfying_score.csv'), sep='\t', index=False)

        logging.info("satisfying score done!")

    def match_symptom(self, text):
        results = []
        pattern = r"([^\(\)]+)|\((.*?)\)"

        # find
        for elem in text.split(','):
            for match in re.finditer(pattern, elem):
                if match.group(1):  # outside bracket
                    sub = match.group(1).strip()
                    if not bool(re.match(r"^[A-Za-z\d]+$", sub)):
                        results.append(sub)
                elif match.group(2):  # inside bracket
                    results.extend(self.match_symptom(match.group(2)))
        return results
    def extract_symptoms(self):
        # create a triple graph
        g = Graph()
        # read ttl file
        g.parse(os.path.join(self.corpus_path, 'symptom.ttl'), format="ttl", encoding='utf-8')

        symptoms = set()
        pattern = r"\s+|<|>"
        # each
        pg = tqdm.tqdm(g)
        pg.set_description("extract symptoms: ")
        for subj, pred, obj in g:
            symptoms.update(self.match_symptom(re.sub(pattern, '', str(subj)).split('/')[-1])
                            + self.match_symptom(re.sub(pattern, '', str(obj)).split('/')[-1]))

        with open(os.path.join(self.corpus_path, "symptoms.txt"), "w", encoding="utf-8") as f:
            for item in symptoms:
                f.write(f"{item}\n")
        logging.info("extract symptoms done!")

    def posi_nega_sample(self):
        # positive and negative sample for patient
        # deal with patient lacking of history info
        if os.path.exists(os.path.join(self.output_path, 'rating.csv')):
            self.rating = pd.read_csv(os.path.join(self.output_path,'rating.csv'), sep='\t')
            return self.rating

        o_pids = []
        o_dids = []
        o_type = []

        ppad = tqdm.tqdm(range(len(self.patient_sampled['patient_id'])))
        ppad.set_description('sample positive and negative patients: ')
        for i in ppad:
            pid = self.patient_sampled.loc[i, 'patient_id']
            did = self.patient_sampled.loc[i, 'doctor_id']
            depart = self.patient_sampled.loc[i, 'department']
            patient_pool = self.patient_sampled[self.patient_sampled['department'] == depart]['patient_id'].unique()
            patient_pool = patient_pool[patient_pool != pid]
            scores = cosine_similarity([self.query_emb[str(pid)]], [self.query_emb[str(pati)] for pati in patient_pool])
            id_score = sorted(list(zip(patient_pool, np.squeeze(scores))), key=lambda x: x[1], reverse=True)
            target = id_score[:random.sample(list(range(self.sample_range[0], self.sample_range[1])), 1)[0]]
            target_doctors = self.patient_sampled[
                self.patient_sampled['patient_id'].isin([elem[0] for elem in target])]['doctor_id'].unique()
            o_dids.append(did)
            o_dids.extend(target_doctors)
            o_pids.extend([pid] * (len(target_doctors) + 1))
            o_type.extend([1] * (len(target_doctors) + 1))  # ['real'] + ['posi_sample'] * len(target))

            doctor_pool_diff_depart = list(self.patient_sampled[self.patient_sampled['department'] != depart]['doctor_id'].unique())
            sampled_num = random.sample(list(range(self.sample_range[0], self.sample_range[1])), 1)[0]
            nega_doctors = random.sample(list(doctor_pool_diff_depart), sampled_num)\
                if sampled_num < len(doctor_pool_diff_depart) else doctor_pool_diff_depart
            o_dids.extend(nega_doctors)
            o_pids.extend([pid] * len(nega_doctors))
            o_type.extend([0] * len(nega_doctors))

        result = pd.DataFrame({
            'patient_id': o_pids,
            'doctor_id': o_dids,
            'type': o_type
        })
        self.rating = result
        result.to_csv(os.path.join(self.output_path, 'rating.csv'), sep='\t', index=False)

    def test(self):
        for elem in self.patient_sampled['department'].unique():
            sub = self.patient_sampled[self.patient_sampled['department'] == elem]['doctor_id'].unique()
            print(elem, len(sub))

    def build_doctor_info(self):
        data = './doctor'
        doctors = pd.read_csv('./output/doctors_filtered.csv', sep='\t')
        dids = doctors['doctor_id'].values

        columns = ['doctor_id', 'profession_direction', 'profile', 'doctor_title', 'education_title', 'hospital',
                   'consultation_amount', 'cure_satisfaction', 'attitude_satisfaction', 'message']
        result = []
        for file in os.listdir(data):
            sub = pd.read_csv(os.path.join(data, file), sep=',')
            sub = sub[sub['doctor_id'].isin(dids)]
            result.extend(sub[columns].values)

        result = pd.DataFrame(result, columns=columns)
        result['hospital'] = result['hospital'].apply(
            lambda x: ', '.join([elem for elem in eval(x) if isinstance(elem, str)]))
        result = result.fillna('')
        result.to_csv('./Med/doctor_detail.csv', sep='\t', index=False)

    def base_dataset_construction(self):
        satis = []
        self.satisfying_score['satisfying_score'] = self.minmaxscaler.fit_transform(
            np.array(self.satisfying_score['satisfying_score']).reshape(-1, 1))
        rpad = tqdm.tqdm(range(len(self.rating)))
        rpad.set_description("construct dataset: ")
        for i in rpad:
            if self.rating.loc[i, 'type'] == 1:
                pid = self.rating.loc[i, 'patient_id']
                did = self.rating.loc[i, 'doctor_id']

                search = self.satisfying_score[(self.satisfying_score['doctor_id'] == did)
                                               &(self.satisfying_score['patient_id'] == pid)]['satisfying_score'].values
                if len(search) > 0:
                    satis.append(round(search[0], 3))
                else:
                    candidate = self.satisfying_score[self.satisfying_score['doctor_id'] == did]['satisfying_score'].values
                    candidate = sorted(candidate, reverse=False)
                    single = random.sample(candidate[int(len(candidate) / 2):], 1)
                    satis.append(round(single[0], 3))
            else:
                did = self.rating.loc[i, 'doctor_id']
                candidate = sorted(
                    self.satisfying_score[self.satisfying_score['doctor_id'] == did]['satisfying_score'].values,
                    reverse=False)
                single = random.sample(candidate[:int(len(candidate) / 2)], 1)
                satis.append(round(float(single[0]), 3))
        self.rating['satisfying_score'] = satis
        self.rating['label'] = self.rating['type']

        doctor2id = dict(zip(self.doctor_filtered['doctor_id'].unique().tolist(),
                             list(range(1, len(self.doctor_filtered['doctor_id'].unique()) + 1, 1))))
        patient2id = dict(zip(self.patient_sampled['patient_id'].unique().tolist(),
                              list(range(1, len(self.patient_sampled['patient_id'].unique()) + 1, 1))))
        relation2id = dict(zip(self.kg['relation'].unique().tolist(), list(range(1, len(self.kg['relation'].unique()) + 1, 1))))
        doctors_str = [str(elem) for elem in list(doctor2id.keys())]
        ents = self.kg[~self.kg['target'].isin(doctors_str)]['target'].unique()
        ent2id = {str(elem): doctor2id[elem] for elem in doctor2id.keys()}
        ent2id.update(dict(zip(ents.tolist(), range(len(doctor2id), len(doctor2id) + len(ents) + 1, 1))))

        self.doctor_token_hist['doctor_id'] = self.doctor_token_hist['doctor_id'].apply(lambda x: doctor2id[x])
        self.patient_sampled['patient_id'] = self.patient_sampled['patient_id'].apply(lambda x: patient2id[x])
        self.patient_sampled = self.patient_sampled[['patient_id', 'query']]

        self.kg['head'] = self.kg['head'].apply(lambda x: doctor2id[x])
        self.kg['relation'] = self.kg['relation'].apply(lambda x: relation2id[x])
        for i in range(len(self.kg)):
            if not self.kg.loc[i, 'target'] in ent2id.keys():
                self.kg.loc[i, 'target'] = doctor2id[int(self.kg.loc[i, 'target'])]
            else:
                self.kg.loc[i, 'target'] = ent2id[self.kg.loc[i, 'target']]
        self.rating['doctor_id'] = self.rating['doctor_id'].apply(lambda x: doctor2id[x])
        self.rating['patient_id'] = self.rating['patient_id'].apply(lambda x: patient2id[x])
        self.rating = self.rating[['patient_id', 'doctor_id', 'label', 'satisfying_score']]

        cure_set = set()
        evaluation_set = set()
        symptom_set = set()
        for i in range(len(self.doctor_token_hist)):
            cure_set.update(self.doctor_token_hist.loc[i, 'cure_hist'])
            evaluation_set.update(self.doctor_token_hist.loc[i, 'evaluation_hist'])
            symptom_set.update(self.doctor_token_hist.loc[i, 'symptom_hist'])

        cure2id = dict(zip(list(cure_set), list(range(1, len(cure_set) + 1, 1))))
        evaluation2id = dict(zip(list(evaluation_set), list(range(1, len(evaluation_set) + 1, 1))))
        symptom2id = dict(zip(list(symptom_set), list(range(1, len(symptom_set) + 1, 1))))
        for i in range(len(self.doctor_token_hist)):
            self.doctor_token_hist.at[i, 'cure_hist'] = [cure2id[elem]
                                                          for elem in self.doctor_token_hist.loc[i, 'cure_hist']]
            self.doctor_token_hist.at[i, 'evaluation_hist'] = [
                evaluation2id[elem]for elem in self.doctor_token_hist.loc[i, 'evaluation_hist']]
            self.doctor_token_hist.at[i, 'symptom_hist'] = [symptom2id[elem]
                                                             for elem in self.doctor_token_hist.loc[i, 'symptom_hist']]

        self.doctor_token_hist.to_csv(os.path.join(self.dataset_path, 'doctor.csv'), sep='\t', index=False)
        self.patient_sampled.to_csv(os.path.join(self.dataset_path, 'patient.csv'), sep='\t', index=False)
        self.kg.to_csv(os.path.join(self.dataset_path, 'kg.csv'), sep='\t', index=False)
        self.rating.to_csv(os.path.join(self.dataset_path, 'ratings.csv'), sep='\t', index=False)

        for filename, dic in zip(['doctor2id', 'patient2id', 'ent2id', 'relation2id', 'cure2id', 'evaluation2id', 'symptom2id']
                                 , [doctor2id, patient2id, ent2id, relation2id, cure2id, evaluation2id, symptom2id]):
            sub = json.dumps(dic, indent=4, ensure_ascii=False)
            f = open(os.path.join(self.dataset_path, 'side_' + filename + '.json'), 'w', encoding='utf-8')
            f.write(sub)

if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.filter_data()
    preprocessor.extract_triple()
    # preprocessor.extract_symptoms()
    preprocessor.extract_med_token()
    # preprocessor.test()
    # preprocessor.query_embedding(target='patients_sampled')
    preprocessor.rating_creator()
    preprocessor.posi_nega_sample()
    preprocessor.base_dataset_construction()

    # preprocessor.build_doctor_info()