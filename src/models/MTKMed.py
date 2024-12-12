import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn import LayerNorm
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from .MMOE import MMOE
from .KGCN import KGCN
import math


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=1000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)

        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).int().unsqueeze(0)
        x = x + self.embeddings(pos).expand_as(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe *= 0.1
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class LearnableFrequencyEncoder(nn.Module):
    # use to encode the appearance frequency of three types of tokens(disease, eval_disease, symptom)
    def __init__(self, dim, boundaries):
        """
        :param dim: Frequency embedding dim
        :param dropout: dropout rate
        :param boundaries: boundaries list of frequency
        """
        super(LearnableFrequencyEncoder, self).__init__()
        self.boundaries = boundaries

        # frequency embedding layer
        self.frequency_embedding = nn.Embedding(len(boundaries) + 2, dim)
        # self.initrange = 0.1
        # self.frequency_embedding.weight.data.uniform_(-self.initrange, self.initrange)
        nn.init.xavier_uniform_(self.frequency_embedding.weight)

    def get_boundary_idx(self, inputs):
        """
        :param inputs: [batch_size, seq_len]
        :return: [batch_size, seq_len]
        """
        # torch.bucketize, find idx based on boundaries
        return torch.bucketize(inputs, torch.tensor(self.boundaries), right=True) + 1

    def forward(self, inputs, x):
        """
        :param inputs: [batch_size, seq_len]
        :param x: [batch_size, seq_len, dim]
        :return: [batch_size, seq_len, dim] 的 tensor
        """
        # # bucketize
        # boundary_idx = self.get_boundary_idx(inputs)  # [batch_size, seq_len]

        # embed
        boundary_emb = self.frequency_embedding(inputs)  # [batch_size, seq_len, dim]

        # add
        output = x + boundary_emb
        return output


class DoctorEncoder(nn.Module):
    def __init__(self, args, voc_size, boundaries):
        super(DoctorEncoder, self).__init__()
        self.args = args
        self.voc_size = voc_size
        self.disease_boundaries, self.evaluation_boundaries, self.symptom_boundaries = boundaries
        self.emb_dim = args.embed_dim
        self.seq_len_disease = args.seq_len_disease
        self.seq_len_evaluation = args.seq_len_evaluation
        self.seq_len_symptom = args.seq_len_symptom
        self.device = torch.device('cuda:{}'.format(args.cuda)) if args.cuda >=0 else 'cpu'

        # self.embeddings = nn.ModuleList(
        #     [nn.Embedding(voc_size[i] + 1, self.emb_dim) for i in range(3)])  # disease, eval_disease, symptom

        # self.special_embeddings = nn.Embedding(2, self.emb_dim)  # add token：[CLS]， [SEP]

        self.special_tokens = {'CLS': torch.LongTensor([0, ]).to(self.device),
                               'SEP1': torch.LongTensor([1, ]).to(self.device),
                               'SEP2': torch.LongTensor([2, ]).to(self.device)}

        self.segment_embedding = nn.Embedding(3, self.emb_dim)  # distinguish three types of input tokens

        # self.positional_embedding_layer = PositionalEncoding(d_model=args.embed_dim)
        # self.positional_embedding_layer = LearnablePositionalEncoding(d_model=args.embed_dim)

        if args.doctor_seperate == False:
            self.embeddings = nn.ModuleList(
                [nn.Embedding(voc_size[i] + 1, self.emb_dim) for i in range(3)])  # disease, eval_disease, symptom
            self.special_embeddings = nn.Embedding(3, self.emb_dim)  # add token：[CLS]， [SEP1], [SEP2]
            self.transformer_visit = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=args.nhead, dropout=args.dropout),
                num_layers=args.encoder_layers
            )
            self.frequency_embedding_layer_disease = LearnableFrequencyEncoder(dim=self.emb_dim, boundaries=self.disease_boundaries)
            self.frequency_embedding_layer_evaluation = LearnableFrequencyEncoder(dim=self.emb_dim, boundaries=self.evaluation_boundaries)
            self.frequency_embedding_layer_symptom = LearnableFrequencyEncoder(dim=self.emb_dim, boundaries=self.symptom_boundaries)
            self.doctor_encoder = self.doctor_encoder_unified
        else:
            self.embeddings = nn.ModuleList(
                [nn.Embedding(voc_size[i] + 1, self.emb_dim // 3) for i in range(3)])  # disease, eval_disease, symptom
            self.special_embeddings = nn.Embedding(3, self.emb_dim // 3)  # add token：[CLS]， [SEP1], [SEP2]
            self.transformer_disease = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.emb_dim // 3, nhead=args.nhead, dropout=args.dropout),
                num_layers=args.encoder_layers
            )
            self.transformer_evaluation = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.emb_dim // 3, nhead=args.nhead, dropout=args.dropout),
                num_layers=args.encoder_layers
            )
            self.transformer_symptom = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.emb_dim // 3, nhead=args.nhead, dropout=args.dropout),
                num_layers=args.encoder_layers,
            )

            self.doctor_layer = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim),
                nn.ReLU(),
                nn.Linear(self.emb_dim, self.emb_dim),
            )

            self.frequency_embedding_layer_disease = LearnableFrequencyEncoder(dim=self.emb_dim // 3,
                                                                               boundaries=self.disease_boundaries)
            self.frequency_embedding_layer_evaluation = LearnableFrequencyEncoder(dim=self.emb_dim // 3,
                                                                                  boundaries=self.evaluation_boundaries)
            self.frequency_embedding_layer_symptom = LearnableFrequencyEncoder(dim=self.emb_dim // 3,
                                                                               boundaries=self.symptom_boundaries)
            self.doctor_encoder = self.doctor_encoder_seperate

    def parse(self, inputs):
        # token extract and padding
        batch_disease = self.padding_tokens([elem[0] for elem in inputs[0]], self.seq_len_disease)  # [batch_size, seq_len]
        # need to bucketize and padding
        batch_disease_frequency = [elem[1] for elem in inputs[0]]
        batch_disease_frequency = [torch.bucketize(tensor, self.disease_boundaries) + 1
                                   for tensor in batch_disease_frequency]
        batch_disease_frequency = self.padding_tokens(batch_disease_frequency, self.seq_len_disease)  # [batch_size, seq_len]

        batch_evaluation = self.padding_tokens([elem[0] for elem in inputs[1]], self.seq_len_evaluation)  # [batch_size, seq_len]
        batch_evaluation_frequency = [elem[1] for elem in inputs[1]]
        batch_evaluation_frequency = [torch.bucketize(tensor, self.evaluation_boundaries) + 1
                                      for tensor in batch_evaluation_frequency]
        batch_evaluation_frequency = self.padding_tokens(batch_evaluation_frequency, self.seq_len_evaluation)  # # [batch_size, seq_len]

        batch_symptom = self.padding_tokens([elem[0] for elem in inputs[2]], self.seq_len_symptom)  # [batch_size, seq_len]
        batch_symptom_frequency = [elem[1] for elem in inputs[2]]
        batch_symptom_frequency = [torch.bucketize(tensor, self.symptom_boundaries) + 1
                                   for tensor in batch_symptom_frequency]
        batch_symptom_frequency = self.padding_tokens(batch_symptom_frequency, self.seq_len_symptom)  # [batch_size, seq_len]

        return batch_disease, batch_disease_frequency, batch_evaluation, \
            batch_evaluation_frequency, batch_symptom, batch_symptom_frequency

    def padding_tokens(self, sequences, fixed_length):
        # sequences: changed length list of tensors
        # fixed_length: target length
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

        if padded_sequences.size(1) < fixed_length:
            # if lack
            padding_size = fixed_length - padded_sequences.size(1)
            padded_sequences = F.pad(padded_sequences, (0, padding_size), value=0)
        else:
            # if over
            padded_sequences = padded_sequences[:, :fixed_length]

        return padded_sequences

    def doctor_encoder_seperate(self, inputs):
        # inputs: size(batch, 3)
        batch_size = len(inputs[0])
        # notion:  seq len of different tokens is maybe different

        # parse
        batch_disease, batch_disease_frequency, batch_evaluation, \
            batch_evaluation_frequency, batch_symptom, batch_symptom_frequency = self.parse(inputs)

        disease_embedding = self.embeddings[0](batch_disease)  # (b, seq_len, dim/3)
        evaluation_embedding = self.embeddings[1](batch_evaluation)  # (b, seq_len, dim/3)
        symptom_embedding = self.embeddings[2](batch_symptom)  # (b, seq_len, dim/3)

        disease_embedding = self.frequency_embedding_layer_disease(batch_disease_frequency, disease_embedding)  # (b, seq_len, dim / 3)
        evaluation_embedding = self.frequency_embedding_layer_evaluation(batch_evaluation_frequency, evaluation_embedding)  # (b, seq_len, dim / 3)
        symptom_embedding = self.frequency_embedding_layer_evaluation(batch_symptom_frequency, symptom_embedding)  # (b, seq_len, dim / 3)

        # add CLS token
        cls_embedding_dis = torch.stack([self.special_embeddings(self.special_tokens['CLS'])] * batch_size, dim=0)  # (b, 1, dim/3)
        cls_embedding_eval = torch.stack([self.special_embeddings(self.special_tokens['SEP1'])] * batch_size, dim=0)  # (b, 1, dim/3)
        cls_embedding_sym = torch.stack([self.special_embeddings(self.special_tokens['SEP2'])] * batch_size, dim=0)  # (b, 1, dim/3)
        disease_embedding = torch.cat((cls_embedding_dis, disease_embedding), dim=1)  # (b, seq_len + 1, dim/3)
        evaluation_embedding = torch.cat((cls_embedding_eval, evaluation_embedding), dim=1)  # (b, seq_len + 1, dim/3)
        symptom_embedding = torch.cat((cls_embedding_sym, symptom_embedding), dim=1) # (b, seq_len + 1, dim/3)

        # transpose
        disease_embedding = torch.transpose(disease_embedding, 0, 1)
        evaluation_embedding = torch.transpose(evaluation_embedding, 0, 1)
        symptom_embedding = torch.transpose(symptom_embedding, 0, 1)

        # transformer layer  [b, dim// 3]
        disease_representation = self.transformer_disease(disease_embedding)[0]  # (b, seq_len + 1, dim / 3)
        evaluation_representation = self.transformer_evaluation(evaluation_embedding)[0]  # (b, seq_len + 1, dim / 3)
        symptom_representation = self.transformer_disease(symptom_embedding)[0]  # (b, seq_len + 1, dim / 3)

        # disease_representation = disease_representation.mean(dim=1)  # (b, dim/3)
        # evaluation_representation = evaluation_representation.mean(dim=1)  # (b, dim/3)
        # symptom_representation = symptom_representation.mean(dim=1)  # (b, dim/3)

        # disease_representation = torch.reshape(disease_representation, (batch_size, 1, -1))  # (b,1,dim/3)
        # evaluation_representation = torch.reshape(evaluation_representation, (batch_size, 1, -1))  # (b,1,dim/3)
        # symptom_representation = torch.reshape(symptom_representation, (batch_size, 1, -1))  # (b,1,dim/3)
        batch_repr = torch.cat((disease_representation, evaluation_representation, symptom_representation), dim=1)

        return batch_repr  # (b, dim)

    def doctor_encoder_unified(self, inputs):
        batch_size = len(inputs[0])

        # parse
        batch_disease, batch_disease_frequency, batch_evaluation, \
            batch_evaluation_frequency, batch_symptom, batch_symptom_frequency = self.parse(inputs)

        # initial emb
        disease_embedding = self.embeddings[0](batch_disease)  # (b, seq_len, dim)
        evaluation_embedding = self.embeddings[1](batch_evaluation)  # (b, seq_len, dim)
        symptom_embedding = self.embeddings[2](batch_symptom)  # (b, seq_len, dim)

        # frequency info emb and add
        disease_embedding = self.frequency_embedding_layer_disease(batch_disease_frequency, disease_embedding)  # (b, seq_len, dim)
        evaluation_embedding = self.frequency_embedding_layer_evaluation(batch_evaluation_frequency, evaluation_embedding)  # (b, seq_len, dim)
        symptom_embedding = self.frequency_embedding_layer_evaluation(batch_symptom_frequency, symptom_embedding)  # (b, seq_len, dim)

        # add segment embedding
        segment_disease = torch.tensor([0] * self.seq_len_disease).to(self.device)  # (seq_len,)
        segment_disease_embedding = torch.stack([self.segment_embedding(segment_disease)] * batch_size, dim=0)# (b, seq_len, dim)
        disease_embedding += segment_disease_embedding

        segment_evaluation = torch.tensor([1] * self.seq_len_evaluation).to(self.device)  # (seq_len,)
        segment_evaluation_embedding = torch.stack([self.segment_embedding(segment_evaluation)] * batch_size, dim=0)  # (b, seq_len, dim)
        evaluation_embedding += segment_evaluation_embedding

        segment_symptom = torch.tensor([2] * self.seq_len_symptom).to(self.device)  # (seq_len,)
        segment_symptom_embedding = torch.stack([self.segment_embedding(segment_symptom)] * batch_size, dim=0)  # (b, seq_len, dim)
        symptom_embedding += segment_symptom_embedding

        # 加入CLS token
        cls_embedding_dis = torch.stack([self.special_embeddings(self.special_tokens['CLS'])] * batch_size, dim=0)  # (b, 1, dim)
        cls_embedding_eval = torch.stack([self.special_embeddings(self.special_tokens['SEP1'])] * batch_size, dim=0)  # (b, 1 , dim)
        cls_embedding_sym = torch.stack([self.special_embeddings(self.special_tokens['SEP2'])] * batch_size, dim=0)  # (b, 1, dim)
        disease_embedding = torch.cat((cls_embedding_dis, disease_embedding), dim=1)  # (b, seq_len + 1, dim)
        evaluation_embedding = torch.cat((cls_embedding_eval, evaluation_embedding), dim=1)  # (b, seq_len + 1, dim)
        symptom_embedding = torch.cat((cls_embedding_sym, symptom_embedding), dim=1)  # (b, seq_len + 1, dim)

        # (b, seq_len_disease + seq_len_valuation + seq_len_symptom + 3, dim)
        combined_embedding = torch.cat((disease_embedding, evaluation_embedding, symptom_embedding), dim=1)
        combined_embedding = torch.transpose(combined_embedding, 0, 1) # [seq_len, b, dim]

        batch_repr = self.transformer_visit(combined_embedding)[0]  # (b, dim)

        return batch_repr


class PatientEncoder(torch.nn.Module):
    def __init__(self, model_path, tokenizer_path=None, task_specific_dim=None):
        super(PatientEncoder, self).__init__()
        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path or model_path)

        # load mcbert model
        self.bert = BertModel.from_pretrained(model_path)

        # add a specific layer
        if task_specific_dim:
            self.task_layer = torch.nn.Linear(self.bert.config.hidden_size, task_specific_dim)
            # self.initrange = 0.1
            # self.task_layer.weight.data.uniform_(-self.initrange, self.initrange)
            nn.init.xavier_uniform_(self.task_layer.weight)
        else:
            self.task_layer = None

    def forward(self, queries, attention_masks=None, task_specific=False):
        """
        :param queries: list of queries
        :param attention_masks: if mask。
        :param task_specific: is use task specific。
        :return: BERT output emb。
        """
        # input
        if isinstance(queries, list):
            # input tokens
            inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=128)
        else:
            inputs = {'input_ids': queries, 'attention_mask': attention_masks}

        # acquire inputs tensor
        input_ids = inputs['input_ids'].to(next(self.bert.parameters()).device)
        attention_mask = inputs['attention_mask'].to(next(self.bert.parameters()).device)

        # Pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        if task_specific and self.task_layer:
            # Pass through task-specific layer
            output = self.task_layer(hidden_states[:, 0, :])  # if task layer
            return output

        return hidden_states[:, 0, :]  # cls embedding


class MTKMed(nn.Module):
    def __init__(self, args, dataset, voc_size):
        super(MTKMed, self).__init__()
        self.device = torch.device('cuda:{}'.format(args.cuda)) if args.cuda >= 0 else 'cpu'
        self.disease_boundaries = dataset.cal_boundaries('cure', args.boundaries_num)
        self.evaluation_boundaries = dataset.cal_boundaries('evaluation', args.boundaries_num)
        self.symptom_boundaries = dataset.cal_boundaries('symptom', args.boundaries_num)
        boundaries = [torch.FloatTensor(self.disease_boundaries).to(self.device),
                      torch.FloatTensor(self.evaluation_boundaries).to(self.device),
                      torch.FloatTensor(self.symptom_boundaries).to(self.device)]

        # vocabulary size
        self.voc_size = voc_size

        # doctor and patient encoder
        self.d_encoder = DoctorEncoder(args, self.voc_size, boundaries)
        self.p_encoder = PatientEncoder(model_path=args.bert_path, task_specific_dim=args.embed_dim)
        num_users = len(dataset.patient_voc)
        num_entities = len(dataset.ent_voc)  # entities including doctors and other entities
        num_relations = len(dataset.relation_voc)

        self.kgcn = KGCN(num_users, num_entities, num_relations, args.embed_dim, args.neighbor_sample_size, args.n_iter)
        self.adj_entity, self.adj_relation = self.kgcn.construct_adj(dataset.kg.values.tolist(), num_entities)

        # self.doctor_layer = Adapter(self.emb_dim, args.adapter_dim)

        # self.mask_adapter = Adapter(self.emb_dim, args.adapter_dim)
        self.cls_mask = nn.Linear(args.embed_dim, self.voc_size[0]+self.voc_size[1]+self.voc_size[2])

        # self.nsp_adapter = Adapter(self.emb_dim, args.adapter_dim)
        self.cls_nsp = nn.Linear(args.embed_dim * 2, 1)

        self.mmoe = MMOE(args.embed_dim * 4, args.num_experts, args.hidden_dim, 2)

        # loss weight
        # self.weight_ssc = args.weight_ssc

        # grad norm
        self.grad_norm = True if args.grad_norm > 0 else False
        # trainable loss weight
        self.weight_label = nn.Parameter(torch.tensor(1.0)) if args.grad_norm > 0 else None
        self.weight_ssc = nn.Parameter(torch.tensor(1.0)) if args.grad_norm > 0 else args.weight_ssc

        self.alpha = args.gradnorm_alpha

        self.init_weights()

    def forward_finetune(self, inputs):
        # batch info transformed
        transformed_inputs = list(map(list, zip(*inputs)))
        batch_dp = torch.stack(transformed_inputs[0], dim=0)

        # B: batch size, dim: embedding dimension
        # doctor ability emb
        doctor_repr = self.d_encoder.doctor_encoder(transformed_inputs[1:4])  # (B,dim)
        # doctor_repr = self.doctor_layer(patient_repr)  # (B,dim)

        # patient query emb
        patient_repr = self.p_encoder(transformed_inputs[4], task_specific=True)

        # doctor node kgcn embeding, [B, kg_dim]
        patient_repr_kg, doctor_repr_kg = self.kgcn(batch_dp[:, 0], batch_dp[:, 1], self.adj_entity.to(self.device),
                                                    self.adj_relation.to(self.device), patient_repr)

        # result
        label_result, ssc_result = self.mmoe(torch.cat((doctor_repr, doctor_repr_kg,
                                                        patient_repr), dim=-1))  #(B, doctor_dim + patient_dim + kg_dim * 2)
        # sigmoid
        # label_result = F.sigmoid(label_result)   # (B,)
        ssc_result = F.sigmoid(ssc_result)  # (B,)

        return label_result, ssc_result

    def forward_mask(self, inputs):
        # batch info transformed
        transformed_inputs = list(map(list, zip(*inputs)))

        # B: batch size, dim: embedding dimension
        # doctor ability emb
        doctor_repr = self.d_encoder.doctor_encoder(transformed_inputs[1:4])  # (B,dim)
        # doctor_repr = self.doctor_layer(patient_repr)  # (B,dim)

        return doctor_repr

    def forward_nsp(self, inputs):
        # batch info transformed
        transformed_inputs = list(map(list, zip(*inputs)))
        batch_dp = torch.stack(transformed_inputs[0], dim=0)

        # B: batch size, dim: embedding dimension
        # doctor ability emb
        doctor_repr = self.d_encoder.doctor_encoder(transformed_inputs[1:4])  # (B,dim)
        # doctor_repr = self.doctor_layer(patient_repr)  # (B,dim)

        # patient query emb
        patient_repr = self.p_encoder(transformed_inputs[4], task_specific=True)

        # doctor node kgcn embeding, [B, kg_dim]
        # patient_repr_kg, doctor_repr_kg = self.kgcn(batch_dp[:, 0], batch_dp[:, 1],self.adj_entity.to(self.device),
        #                                             self.adj_relation.to(self.device), patient_repr)

        # repr = torch.cat((doctor_repr, doctor_repr_kg, patient_repr, doctor_repr_kg), dim=-1)
        repr = torch.cat((doctor_repr, patient_repr), dim=-1)
        # repr = torch.sum(doctor_repr * patient_repr, dim=1, keepdim=True)

        result = torch.tanh(self.cls_nsp(repr))  # (B, 1)
        # result = result.squeeze(dim=1)  # (B,)
        # logit = F.sigmoid(result)
        return result

    def forward(self, input, mode='fine-tune'):
        assert mode in ['fine-tune', 'pretrain_mask', 'pretrain_nsp']
        if mode == 'fine-tune':
            label_result, ssc_result = self.forward_finetune(input)
            return label_result, ssc_result
        
        elif mode == 'pretrain_mask':
            doctor_repr = self.forward_mask(input)      # (B, dim)
            # patient_repr = self.mask_adapter(patient_repr)  # (B, dim)
            result = self.cls_mask(doctor_repr)            # (B, voc_size[0]+voc_size[1]+voc_size[2])
            return result
        
        elif mode == 'pretrain_nsp':
            logit = self.forward_nsp(input)
            return logit

    def compute_loss_fine_tuned(self, label_scores, ssc_scores, label_targets, ssc_targets, target_grad_norm):
        # a loss computer for nsp, mask and fine_tuned
        label_loss = F.binary_cross_entropy_with_logits(label_scores, label_targets)
        ssc_loss = F.mse_loss(ssc_scores, ssc_targets)

        # grad norm
        if self.grad_norm:
            total_loss = torch.mul(self.weight_label, label_loss) + torch.mul(self.weight_ssc, ssc_loss)

            # # cal gradient of every task
            # grads_label = torch.autograd.grad(label_loss, [self.weight_label], retain_graph=True)[0]
            # grads_ssc = torch.autograd.grad(ssc_loss, [self.weight_ssc], retain_graph=True)[0]

            # cal gradient
            total_loss.backward()

            # L2 norm calculation
            grad_norm_label = self.weight_label.grad.norm(2)
            grad_norm_ssc = self.weight_ssc.grad.norm(2)

            # balance
            target_ratio_label = target_grad_norm[0] / grad_norm_label
            target_ratio_ssc = target_grad_norm[1] / grad_norm_ssc

            # dynamic update
            with torch.no_grad():
                self.weight_label *= (1 + self.alpha * (target_ratio_label - 1))
                self.weight_ssc *= (1 + self.alpha * (target_ratio_ssc - 1))
        else:
            total_loss = label_loss + self.weight_ssc * ssc_loss

        return total_loss, label_loss, ssc_loss

    def compute_loss_mask(self, ):
        pass

    def compute_loss_nsp(self, label_scores, targets):
        return F.binary_cross_entropy_with_logits(label_scores, targets)

    def score(self, inputs):
        # a function to get the final score of recommendation score
        label_score, ssc_score = self.forward_finetune(inputs)
        final_pred = torch.sigmoid(label_score * ssc_score)

        return label_score, ssc_score, final_pred

    def init_weights(self):
        """Initialize embedding weights."""
        # initrange = 0.1
        #
        # self.d_encoder.embeddings[0].weight.data.uniform_(-initrange, initrange)      # disease
        # self.d_encoder.embeddings[1].weight.data.uniform_(-initrange, initrange)      # evaluation
        # self.d_encoder.embeddings[2].weight.data.uniform_(-initrange, initrange)      # symptom
        #
        # self.d_encoder.segment_embedding.weight.data.uniform_(-initrange, initrange)
        # self.d_encoder.special_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.d_encoder.frequency_embedding_layer_evaluation.weight.data.uniform_(-initrange, initrange)

        # xavier
        nn.init.xavier_uniform_(self.d_encoder.embeddings[0].weight)  # disease
        nn.init.xavier_uniform_(self.d_encoder.embeddings[1].weight)  # evaluation
        nn.init.xavier_uniform_(self.d_encoder.embeddings[2].weight)  # symptom

        nn.init.xavier_uniform_(self.d_encoder.segment_embedding.weight)
        nn.init.xavier_uniform_(self.d_encoder.special_embeddings.weight)
        nn.init.xavier_uniform_(self.cls_nsp.weight)
        nn.init.xavier_uniform_(self.cls_mask.weight)
