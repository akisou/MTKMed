import os
import sys
import dill
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import random

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from torch.optim import AdamW as Optimizer
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from collections import defaultdict

from models.MTKMed import MTKMed
from utils.dataloader import MedDataset
from utils.util import multi_label_metric, ddi_rate_score, get_n_params, create_log_id, logging_config, \
    get_grouped_metrics, get_model_path, get_pretrained_model_path
from utils.metric_evaluation import eval_precision, eval_recall, eval_NDCG, eval_MAP, eval_MRR
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
torch.cuda.empty_cache()

# Training settings
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default='', help="User notes")
    parser.add_argument('--model_name', type=str, default='MTKMed', help="model name")
    parser.add_argument('--data_path', type=str, default='../data/Med/', help="data path")
    parser.add_argument('--bert_path', type=str, default='./models/mcBert', help="mcBert path")
    parser.add_argument('--dataset', type=str, default='Med', help='dataset')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop after this many epochs without improvement')
    parser.add_argument('-t', '--test', action='store_true', help="test mode")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default=None,
                        help='log dir prefix like "log0", for model test')
    parser.add_argument('-p', '--pretrain_prefix', type=str, default=None,
                        help='log dir prefix like "log0", for finetune')
    parser.add_argument('--cuda', type=int, default=0, help='which cuda')
    # pretrain

    parser.add_argument('-nsp', '--pretrain_nsp', action='store_true', help='whether to use nsp pretrain')
    parser.add_argument('-mask', '--pretrain_mask', action='store_true', help='whether to use mask prediction pretrain')
    parser.add_argument('--pretrain_epochs', type=int, default=300, help='number of pretrain epochs')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='mask probability')
    parser.add_argument('--freeze_layer_num', type=int, default=11, help='freeze the num of former layers of mcbert')

    parser.add_argument('--grad_norm', type=int, default=1, help='whether to grad norm for multi task train')
    parser.add_argument('--gradnorm_alpha', type=float, default=0.12, help='gradnorm alpha when use grad_norm')
    parser.add_argument('--initial_gradnorm', type=str, default='[1.0, 1.0]', help='initial target gradnorm')
    parser.add_argument('--embed_dim', type=int, default=128, help='dimension of node embedding')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim of mmoe module')
    parser.add_argument('--mmoe_hidden_dim', type=int, default=256, help='mmoe_hidden_dim')
    parser.add_argument('--num_experts', type=int, default=4, help='expert_num')
    parser.add_argument('--neighbor_sample_size', type=int, default=5, help='neighbor sample num of KGCN')
    parser.add_argument('--n_iter', type=int, default=2, help='num of conv times of KGCN')
    parser.add_argument('--seq_len_disease', type=int, default=15,
                        help='sequence length of the disease hist token sequence')
    parser.add_argument('--seq_len_evaluation', type=int, default=15,
                        help='sequence length of the evaluation hist token sequence')
    parser.add_argument('--seq_len_symptom', type=int, default=30,
                        help='sequence length of the symptom hist token sequence')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--nhead', type=int, default=4, help='number of encoder head')
    parser.add_argument('--split_rate', type=str, default='6:2:2', help='split_rate of train, valid, test dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size during training')
    parser.add_argument('--adapter_dim', type=int, default=128, help='dimension of adapter layer')
    parser.add_argument('--boundaries_num', type=int, default=10, help='boundary num of token frequency embedding')
    parser.add_argument('--topk_range', type=str, default='[2, 5]', help='topk choice')  #

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability of transformer encoder')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--weight_ssc', type=float, default=1, help='loss weight of satisfying score task')

    # parameters for ablation study
    parser.add_argument('-s', '--doctor_seperate', action='store_true',
                        help='whether to combine disease, evaluation, symptom')
    parser.add_argument('-e', '--seg_rel_emb', action='store_false', default=True,
                        help='whether to use segment and relevance embedding layer')

    args = parser.parse_args()
    return args

# evalator
@torch.no_grad()
def evaluator(args, model, data_valid, gt_valid, epoch, device, rec_results_path=''):
    model.eval()

    train_loss = 0
    loss_bce = 0
    loss_ssc = 0
    patient_pred_dict = defaultdict(list)
    patient_doctor_pair = []

    pred_all = []
    label_all = []
    for batch_idx, (inputs, label_targets, ssc_targets) in tqdm(enumerate(data_valid), ncols=60, desc="evaluation",
                                                                total=len(data_valid)):
        # rec_result: 0,1 pred
        # ssc_result: satisfying score pred
        # pred_score: final pred score
        inputs = [[ele.to(device) if torch.is_tensor(ele) else ele for ele in elem] for elem in inputs]
        rec_result, ssc_result, pred_score = model.score(inputs)

        label_targets = torch.stack(label_targets, dim=0).squeeze(-1).to(device)
        ssc_targets = torch.stack(ssc_targets, dim=0).squeeze(-1).to(device)
        loss_combined, loss_bce_s, loss_ssc_s = model.compute_loss_fine_tuned(rec_result, ssc_result,
                                                                              label_targets, ssc_targets, mode='valid')
        train_loss += loss_combined
        loss_bce += loss_bce_s
        loss_ssc += loss_ssc_s
        patient_doctor_pair.extend([elem[0].cpu().tolist() for elem in inputs])

        label_all.extend(label_targets.cpu().tolist())
        pred_all.extend(pred_score.cpu().tolist())

    for pair, pred in zip(patient_doctor_pair, pred_all):
        patient_pred_dict[pair[0]].append((pair[1], pred))

    [sorted(patient_pred_dict[key], key=lambda x: x[1], reverse=True) for key in patient_pred_dict.keys()]
    for key in patient_pred_dict.keys():
        patient_pred_dict[key] = [elem[0] for elem in patient_pred_dict[key]]

    loss = train_loss / len(data_valid)
    loss_bce /= len(data_valid)
    loss_ssc /= len(data_valid)
    auc = roc_auc_score(label_all, pred_all)
    f1 = f1_score([int(elem) for elem in label_all], [int(round(elem)) for elem in pred_all])

    # topk eval
    krange = eval(args.topk_range)
    precision = eval_precision(krange, patient_pred_dict.keys(), patient_pred_dict, gt_valid)
    recall = eval_recall(krange, patient_pred_dict.keys(), patient_pred_dict, gt_valid)
    ndcg = eval_NDCG(krange, patient_pred_dict.keys(), patient_pred_dict, gt_valid)
    mrr = eval_MRR(krange, patient_pred_dict.keys(), patient_pred_dict, gt_valid)

    logging.info(f'''Epoch {epoch:03d}, Loss_val: {loss:.4}, Loss_bce: {loss_bce:.4}, Loss_ssc: {loss_ssc:.4}''')
    for i in range(len(krange)):
        logging.info(f'''top{krange[i]} of Epoch {epoch:03d}, Precision: {precision[i]: .4}, Recall: {recall[i]: .4}, NDCG: {ndcg[i]: .4}, MRR: {mrr[i]: .4}''')
    return loss, loss_bce, loss_ssc, auc, f1, precision, recall, ndcg, mrr


# full sort recommend
@torch.no_grad()
def full_sort_pred(args, model, dataset, data_test, gt_test, epoch, device, rec_results_path=''):
    # full sort
    # user contain
    user = set()
    for batch_idx, (inputs, label_targets, ssc_targets) in enumerate(data_test):
        [user.add(elem[0][0].item()) for elem in inputs]

    # doctor_info_construct
    dinfo_columns = dataset.doctor_info.columns.tolist()
    dinfo_columns.remove('doctor_id')
    did2info = dict(zip(dataset.doctor_info['doctor_id'], dataset.doctor_info[dinfo_columns].values))

    # test user query info
    pid2info = dict(zip(dataset.patient_info['patient_id'], dataset.patient_info['query']))

    patient_pred_dict = defaultdict(list)
    patient_doctor_pair = []

    pred_all = []
    rec_all = []
    ssc_all = []
    for idx, user in tqdm(enumerate(list(user)), ncols=60, desc="full_sort_pred", total=len(user)):
        inputs = []
        for did in did2info.keys():
            tmp = []
            tmp.append(torch.LongTensor([user, did]))
            [tmp.append(torch.LongTensor([did2info[did][i], did2info[did][i+1]])) for i in range(0, 6, 2)]
            tmp.append(pid2info[user])
            inputs.append(tmp)
        inputs = [[ele.to(device) if torch.is_tensor(ele) else ele for ele in elem] for elem in inputs]
        rec_result, ssc_result, pred_score = model.score(inputs)

        patient_doctor_pair.extend([elem[0].cpu().tolist() for elem in inputs])
        pred_all.extend(pred_score.cpu().tolist())
        rec_all.extend(rec_result.cpu().tolist())
        ssc_all.extend(ssc_result.cpu().tolist())

    # patient_id: [(doctor_id, pred), ...]
    for pair, pred in zip(patient_doctor_pair, pred_all):
        patient_pred_dict[pair[0]].append((pair[1], pred))

    [sorted(patient_pred_dict[key], key=lambda x: x[1], reverse=True) for key in patient_pred_dict.keys()]
    for key in patient_pred_dict.keys():
        patient_pred_dict[key] = [elem[0] for elem in patient_pred_dict[key]]

    # topk eval
    krange = eval(args.topk_range)
    precision = eval_precision(krange, patient_pred_dict.keys(), patient_pred_dict, gt_test)
    recall = eval_recall(krange, patient_pred_dict.keys(), patient_pred_dict, gt_test)
    ndcg = eval_NDCG(krange, patient_pred_dict.keys(), patient_pred_dict, gt_test)
    mrr = eval_MRR(krange, patient_pred_dict.keys(), patient_pred_dict, gt_test)

    os.makedirs(rec_results_path, exist_ok=True)
    dp = np.array(patient_doctor_pair)
    df = pd.DataFrame({
        'patient_id': dp[:, 0],
        'doctor_id': dp[:, 1],
        'rec_score': [round(elem, 4) for elem in rec_all],
        'ssc_score': [round(elem, 4) for elem in ssc_all],
        'pred_score': [round(elem, 4) for elem in pred_all]
    })
    df.to_csv(os.path.join(rec_results_path, 'predict.csv'), sep='\t', index=False)

    return precision, recall, ndcg, mrr

@torch.no_grad()
def evaluator_test(args, model, dataset, data_test, gt_test, epoch, device, rec_results_path=''):
    model.eval()

    train_loss = 0
    loss_bce = 0
    loss_ssc = 0
    patient_pred_dict = defaultdict(list)
    patient_doctor_pair = []

    pred_all = []
    label_all = []
    for batch_idx, (inputs, label_targets, ssc_targets) in tqdm(enumerate(data_test), ncols=60, desc="test",
                                                                total=len(data_test)):
        # rec_result: 0,1 pred
        # ssc_result: satisfying score pred
        # pred_score: final pred score
        inputs = [[ele.to(device) if torch.is_tensor(ele) else ele for ele in elem] for elem in inputs]
        rec_result, ssc_result, pred_score = model.score(inputs)

        label_targets = torch.stack(label_targets, dim=0).squeeze(-1).to(device)
        ssc_targets = torch.stack(ssc_targets, dim=0).squeeze(-1).to(device)
        loss_combined, loss_bce_s, loss_ssc_s = model.compute_loss_fine_tuned(rec_result, ssc_result,
                                                                              label_targets, ssc_targets, mode='test')
        train_loss += loss_combined
        loss_bce += loss_bce_s
        loss_ssc += loss_ssc_s
        patient_doctor_pair.extend([elem[0].cpu().tolist() for elem in inputs])

        label_all.extend(label_targets.cpu().tolist())
        pred_all.extend(pred_score.cpu().tolist())

    loss = train_loss / len(data_test)
    loss_bce /= len(data_test)
    loss_ssc /= len(data_test)
    auc = roc_auc_score(label_all, pred_all)
    f1 = f1_score([int(elem) for elem in label_all], [int(round(elem)) for elem in pred_all])

    logging.info(f'''Epoch test, Loss_val: {loss:.4}, Loss_bce: {loss_bce:.4}, Loss_ssc: {loss_ssc:.4}''')
    precision, recall, ndcg, mrr = full_sort_pred(args, model, dataset, data_test, gt_test,
                                                  epoch, device, rec_results_path)
    krange = eval(args.topk_range)
    for i in range(len(krange)):
        logging.info(f'''top{krange[i]} of Epoch {epoch:03d}, Precision: {precision[i]: .4}, Recall: {recall[i]: .4}, NDCG: {ndcg[i]: .4}, MRR: {mrr[i]: .4}''')
    return loss, loss_bce, loss_ssc, auc, f1, precision, recall, ndcg, mrr

@torch.no_grad()
def evaluator_mask(model, data_val, voc_size, epoch, device, mode='pretrain_mask'):
    model.eval()
    loss_val = 0
    dis_ja_list, dis_prauc_list, dis_p_list, dis_r_list, dis_f1_list = [[] for _ in range(5)]
    eval_ja_list, eval_prauc_list, eval_p_list, eval_r_list, eval_f1_list = [[] for _ in range(5)]
    sym_ja_list, sym_prauc_list, sym_p_list, sym_r_list, sym_f1_list = [[] for _ in range(5)]
    dis_cnt, eval_cnt, sym_cnt, visit_cnt = 0, 0, 0, 0

    len_val = len(data_val)
    for batch_idx, (batch, label_targets, ssc_targets) in tqdm(enumerate(data_val), ncols=60,
                                                               desc="evaluation_pretrain_mask", total=len(data_val)):
        batch_size = len(batch)
        batch = [[ele.to(device) if torch.is_tensor(ele) else ele for ele in elem] for elem in batch]
        dis_pred, dis_pred_label = [[] for i in range(2)]
        eval_pred, eval_pred_label = [[] for i in range(2)]
        sym_pred, sym_pred_label = [[] for i in range(2)]
        
        result = model(batch, mode)
        dis_gt = np.zeros((batch_size, voc_size[0]))
        eval_gt = np.zeros((batch_size, voc_size[1]))
        sym_gt = np.zeros((batch_size, voc_size[2]))

        dis_logit = result[:, :voc_size[0]]
        eval_logit = result[:, voc_size[0]:voc_size[1] + voc_size[0]]
        sym_logit = result[:, voc_size[1] + voc_size[0]:]

        dis_pred_prob = F.sigmoid(dis_logit).cpu().numpy()
        eval_pred_prob = F.sigmoid(eval_logit).cpu().numpy()
        sym_pred_prob = F.sigmoid(sym_logit).cpu().numpy()

        visit_cnt += batch_size
        for i in range(batch_size):
            sub = [elem.tolist() for elem in batch[i][1:4]]
            dis_gt[i, sub[0][0]] = 1
            eval_gt[i, sub[1][0]] = 1
            sym_gt[i, sub[2][0]] = 1

            dis_pred_temp = dis_pred_prob[i].copy()
            dis_pred_temp[dis_pred_temp >= 0.5] = 1
            dis_pred_temp[dis_pred_temp < 0.5] = 0
            dis_pred.append(dis_pred_temp)

            dis_pred_label_temp = np.where(dis_pred_temp == 1)[0]
            dis_pred_label.append(sorted(dis_pred_label_temp))

            eval_pred_temp = eval_pred_prob[i].copy()
            eval_pred_temp[eval_pred_temp >= 0.5] = 1
            eval_pred_temp[eval_pred_temp < 0.5] = 0
            eval_pred.append(eval_pred_temp)

            eval_pred_label_tmp = np.where(eval_pred_temp == 1)[0]
            eval_pred_label.append(sorted(eval_pred_label_tmp))

            sym_pred_temp = sym_pred_prob[i].copy()
            sym_pred_temp[sym_pred_temp >= 0.5] = 1
            sym_pred_temp[sym_pred_temp < 0.5] = 0
            sym_pred.append(sym_pred_temp)

            sym_pred_label_temp = np.where(sym_pred_temp == 1)[0]
            sym_pred_label.append(sorted(sym_pred_label_temp))

        target = np.concatenate((dis_gt, eval_gt, sym_gt), axis=1)
        loss = F.binary_cross_entropy_with_logits(result, torch.tensor(target, device=device))
        loss_val += loss.item()

        dis_ja, dis_prauc, dis_avg_p, dis_avg_r, dis_avg_f1 = multi_label_metric(
            np.array(dis_gt), np.array(dis_pred), np.array(dis_pred_prob))
        eval_ja, eval_prauc, eval_avg_p, eval_avg_r, eval_avg_f1 = multi_label_metric(
            np.array(eval_gt), np.array(eval_pred), np.array(eval_pred_prob))
        sym_ja, sym_prauc, sym_avg_p, sym_avg_r, sym_avg_f1 = multi_label_metric(
            np.array(sym_gt), np.array(sym_pred), np.array(sym_pred_prob))

        dis_ja_list.append(dis_ja)
        dis_prauc_list.append(dis_prauc)
        dis_p_list.append(dis_avg_p)
        dis_r_list.append(dis_avg_r)
        dis_f1_list.append(dis_avg_f1)

        eval_ja_list.append(eval_ja)
        eval_prauc_list.append(eval_prauc)
        eval_p_list.append(eval_avg_p)
        eval_r_list.append(eval_avg_r)
        eval_f1_list.append(eval_avg_f1)

        sym_ja_list.append(sym_ja)
        sym_prauc_list.append(sym_prauc)
        sym_p_list.append(sym_avg_p)
        sym_r_list.append(sym_avg_r)
        sym_f1_list.append(sym_avg_f1)

    avg_dis_ja, avg_dis_prauc, avg_dis_p, avg_dis_r, avg_dis_f1 = \
        np.mean(dis_ja_list), np.mean(dis_prauc_list), np.mean(dis_p_list), \
        np.mean(dis_r_list), np.mean(dis_f1_list)
    avg_eval_ja, avg_eval_prauc, avg_eval_p, avg_eval_r, avg_eval_f1 = \
        np.mean(eval_ja_list), np.mean(eval_prauc_list), np.mean(eval_p_list), \
        np.mean(eval_r_list), np.mean(eval_f1_list)
    avg_sym_ja, avg_sym_prauc, avg_sym_p, avg_sym_r, avg_sym_f1 = \
        np.mean(sym_ja_list), np.mean(sym_prauc_list), np.mean(sym_p_list), \
        np.mean(sym_r_list), np.mean(sym_f1_list)

    avg_ja, avg_prauc, avg_p, avg_r, avg_f1 = \
        (avg_dis_ja + avg_eval_ja + avg_sym_ja) / 3, \
        (avg_dis_prauc + avg_eval_prauc + avg_sym_prauc) / 3, \
        (avg_dis_p + avg_eval_p + avg_sym_p) / 3, \
        (avg_dis_r + avg_eval_r + avg_sym_r) / 3, \
        (avg_dis_f1 + avg_eval_f1 + avg_sym_f1) / 3
    avg_dis_cnt, avg_eval_cnt, avg_sym_cnt = dis_cnt / visit_cnt, eval_cnt / visit_cnt, sym_cnt / visit_cnt
    avg_cnt = (avg_dis_cnt + avg_eval_cnt + avg_sym_cnt) / 3

    logging.info(
        'Epoch {:03d}   Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_CNT: {:.4}'
        .format(epoch, avg_ja, avg_prauc, avg_p, avg_r, avg_f1, avg_cnt))
    logging.info('Epoch {:03d}   DISEASE Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_DIS_CNT: {:.4}'
                 .format(epoch, avg_dis_ja, avg_dis_prauc, avg_dis_p, avg_dis_r, avg_dis_f1, avg_dis_cnt))
    logging.info(
        'Epoch {:03d}   Patient Evaluation  Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_PRO_CNT: {:.4}'
        .format(epoch, avg_eval_ja, avg_eval_prauc, avg_eval_p, avg_eval_r, avg_eval_f1, avg_eval_cnt))
    logging.info(
        'Epoch {:03d}   SYMPTOM Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_DIS_CNT: {:.4}'
        .format(epoch, avg_sym_ja, avg_sym_prauc, avg_sym_p, avg_sym_r, avg_sym_f1, avg_sym_cnt))

    return loss_val / len_val, avg_ja, avg_prauc, avg_p, avg_r, avg_f1, avg_cnt


@torch.no_grad()
def evaluator_nsp(model, data_val, epoch, device, mode='pretrain_nsp'):
    model.eval()
    loss_val = 0
    pred_all = []
    # label_all = []
    for batch_idx, (inputs, label_targets, ssc_targets) in tqdm(enumerate(data_val), ncols=60, desc="evaluation",
                                                                total=len(data_val)):
        # rec_result: 0,1 pred
        # ssc_result: satisfying score pred
        # pred_score: final pred score
        inputs = [[ele.to(device) if torch.is_tensor(ele) else ele for ele in elem] for elem in inputs]
        result = model.forward_nsp(inputs)
        label_targets = torch.stack(label_targets, dim=0).to(device)
        loss = model.compute_loss_nsp(result, label_targets)
        loss_val += loss

        # label_all.extend(label_targets.cpu().tolist())
        label_cpu = label_targets.squeeze().cpu().tolist()
        result_cpu = F.sigmoid(result).squeeze().cpu().tolist()
        pred_all.extend([True if round(ele2) == ele1 else False for ele1, ele2 in zip(label_cpu, result_cpu)])
    return np.mean(pred_all), loss_val

# def random_mask_word(seq, vocab_size, mask_idx, mask_prob=0.15):
#     """
#     Optimized random mask function using NumPy for batch operations.
#
#     Args:
#         seq (np.ndarray): 2D array of token indices with shape (batch_size, seq_len).
#         vocab_size (int): Size of the vocabulary.
#         mask_idx (int): Index of the [MASK] token.
#         mask_prob (float): Probability of masking each token.
#
#     Returns:
#         np.ndarray: Updated sequence with random masking applied.
#     """
#     batch_size, seq_len = seq.shape
#
#     # Generate random probabilities for each token
#     rand_probs = np.random.random(size=(batch_size, seq_len))
#
#     # Determine which tokens to mask
#     mask_flags = rand_probs < mask_prob
#
#     # Randomly decide how to replace masked tokens
#     replacement_probs = np.random.random(size=(batch_size, seq_len))
#
#     # Create a random token index array for replacements
#     random_tokens = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
#
#     # Apply mask with probabilities
#     seq = seq.copy()
#     seq[0][mask_flags & (replacement_probs < 0.8)] = mask_idx  # 80% -> [MASK]
#     seq[0][mask_flags & (replacement_probs >= 0.8) & (replacement_probs < 0.9)] = random_tokens[
#         mask_flags & (replacement_probs >= 0.8) & (replacement_probs < 0.9)]  # 10% -> random token
#     # Remaining 10% leave as-is (no changes needed)
#
#     return seq
#
#
# def mask_batch_data(batch_data, dis_voc, eval_voc, sym_voc, mask_prob):
#     """
#     Optimized batch masking function.
#
#     Args:
#         batch_data (list): Batch of data where each element contains three sequences.
#         dis_voc, eval_voc, sym_voc (Vocabulary): Vocabularies for respective sequences.
#         mask_prob (float): Probability of masking each token.
#
#     Returns:
#         list: Batch data with masking applied.
#     """
#     dis_vocab_size = len(dis_voc.word2idx)
#     eval_vocab_size = len(eval_voc.word2idx)
#     sym_vocab_size = len(sym_voc.word2idx)
#
#     dis_mask_idx = dis_voc.word2idx['[MASK]']
#     eval_mask_idx = eval_voc.word2idx['[MASK]']
#     sym_mask_idx = sym_voc.word2idx['[MASK]']
#
#     for i, data in enumerate(batch_data):
#         dis_seq = np.array(data[1])
#         eval_seq = np.array(data[2])
#         sym_seq = np.array(data[3])
#
#         batch_data[i][1] = random_mask_word(dis_seq, dis_vocab_size, dis_mask_idx, mask_prob)
#         batch_data[i][2] = random_mask_word(eval_seq, eval_vocab_size, eval_mask_idx, mask_prob)
#         batch_data[i][3] = random_mask_word(sym_seq, sym_vocab_size, sym_mask_idx, mask_prob)
#
#     return batch_data

def random_mask_word(seq, vocab, seq_len, mask_prob=0.15):
    mask_idx = vocab.word2idx['[MASK]']
    for i, _ in enumerate(seq[0][:seq_len]):
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob:
            prob /= mask_prob
            # 80% randomly change token to mask token
            if prob < 0.8:
                seq[0][i] = mask_idx
            # 10% randomly change token to random token
            elif prob < 0.9:
                seq[0][i] = random.choice(list(vocab.word2idx.items()))[1]
            else:
                continue
        else:
            continue
    return seq


# mask batch data
def mask_batch_data(batch_data, dis_voc, eval_voc, sym_voc, mask_prob, seq_len):
    for i in range(len(batch_data)):
        dis = random_mask_word(batch_data[i][1], dis_voc, seq_len[0], mask_prob)
        evalu = random_mask_word(batch_data[i][2], eval_voc, seq_len[1], mask_prob)
        sym = random_mask_word(batch_data[i][3], sym_voc, seq_len[2], mask_prob)
        batch_data[i][1:4] = [dis, evalu, sym]
    return batch_data


def nsp_batch_data(batch_data, data, neg_sample_rate=1):
    nsp_batch = []
    nsp_target = []
    for visit in batch_data:
        nsp_batch.append(visit)
        nsp_target.append(1)
        for i in range(neg_sample_rate):
            neg_visit = random.choice(data)
            while neg_visit[1] == visit[1]:
                neg_visit = random.choice(data)
            if random.random() < 0.5:
                nsp_batch.append([visit[0], neg_visit[1]])
                nsp_target.append(0)
            else:
                nsp_batch.append([neg_visit[0], visit[1]])
                nsp_target.append(0)
    return nsp_batch, nsp_target

def main(args):
    # set logger
    # if args.test:
    #     args.note = 'test of ' + args.log_dir_prefix
    log_directory_path = os.path.join('log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log'+str(log_save_id)+'_'+args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    # device choose
    device = torch.device('cuda:{}'.format(args.cuda)) if args.cuda >= 0 else 'cpu'

    # load data
    dataset = MedDataset(args.data_path, "cpu")
    data_train, data_valid, data_test = random_split(dataset, dataset.split_num(
        [0.1 * float(elem) for elem in args.split_rate.split(':')]))

    # ground_truth dict
    gt_train = dataset.collect_ground_truth(data_train)
    gt_valid = dataset.collect_ground_truth(data_valid)
    gt_test = dataset.collect_ground_truth(data_test)

    # gt merge
    gt_merge = defaultdict(list)
    for d in [gt_train, gt_valid, gt_test]:
        for key, value in d.items():
            gt_merge[key].extend(value)
    for key in gt_merge.keys():
        gt_merge[key] = list(set(gt_merge[key]))

    def add_word(word, voc):
        voc.word2idx[word] = len(voc.word2idx)
        voc.idx2word[len(voc.idx2word)] = word
        return voc
    add_word('[MASK]', dataset.cure_voc)
    add_word('[MASK]', dataset.evaluation_voc)
    add_word('[MASK]', dataset.symptom_voc)

    # construct DataLoader and batch
    data_train = DataLoader(
        data_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=MedDataset.collate_fn_no_padding
    )

    data_valid = DataLoader(
        data_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=MedDataset.collate_fn_no_padding
    )

    data_test = DataLoader(
        data_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=MedDataset.collate_fn_no_padding
    )

    voc_size = (len(dataset.cure_voc), len(dataset.evaluation_voc), len(dataset.symptom_voc))
    # model initialization
    model = MTKMed(args, dataset, voc_size)
    logging.info(model)

    model_path = '../pretrained_models/mlm/saved.pretrained_model'
    state_dict = torch.load(open(model_path, 'rb'), map_location=device)
    del_keys = [key for key in state_dict.keys() if key.find('mmoe') >= 0]
    for key in del_keys:
        del state_dict[key]
        # print(f"Deleted parameter: {key}")
    model.load_state_dict(state_dict, strict=False)

    # # test
    # if args.test:
    #     # model_path = get_model_path(log_directory_path, args.log_dir_prefix)
    #     # model.load_state_dict(torch.load(open(model_path, 'rb'), map_location=device))
    #     # model.to(device)
    #     # logging.info("load model from %s", model_path)
    #     rec_results_path = save_dir + '/' + 'rec_results'
    #
    #     # evaluator(args, model, data_test, gt_test, 0, device, rec_results_path)
    #
    #     # return
    # else:
    rec_results_path = save_dir + '/' + 'rec_results'
    writer = SummaryWriter(save_dir)  # 自动生成log文件夹

    # train and validation
    model.to(device)
    logging.info(f'n_parameters:, {get_n_params(model)}')

    # freeze the former layers of bert of patient encoder
    for name, param in model.p_encoder.bert.encoder.named_parameters():
        if 'layer' in name and int(name.split('.')[1]) < args.freeze_layer_num:
            param.requires_grad = False

    # bert embeddings freeze
    for name, param in model.p_encoder.bert.embeddings.named_parameters():
        param.requires_grad = False

    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # total_trainable_params = sum(p.numel() for p in trainable_params)
    # print(total_trainable_params)
    # print(model.p_encoder.bert.config.max_position_embeddings)

    # optimizer initial
    optimizer = Optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    logging.info(f'Optimizer: {optimizer}')

    if args.pretrain_nsp:
        main_nsp(args, model, optimizer, writer, data_train, data_valid, device, save_dir, log_save_id)
    if args.pretrain_mask:
        main_mask(args, model, optimizer, writer, dataset, data_train, data_valid, voc_size, device, save_dir,
                  log_save_id)
    
    if not (args.pretrain_mask or args.pretrain_nsp) and args.pretrain_prefix is not None:
        # if not pretrain, load pretrained model; else, train from scratch
        pretrained_model_path = get_pretrained_model_path(log_directory_path, args.pretrain_prefix)
        load_pretrained_model(model, pretrained_model_path)
    
    EPOCH = 200
    best_epoch, best_auc = 0, 0
    best_model_state = None
    for epoch in range(EPOCH):
        epoch += 1
        logging.info(f'epoch {epoch} --------------------------model_name={args.model_name}, logger={log_save_id}')
        train_loss = 0
        loss_bce = 0
        loss_ssc = 0

        # finetune
        model.train()
        for batch_idx, (inputs, label_targets, ssc_targets) in tqdm(enumerate(data_train), ncols=60, desc="fintune", total=len(data_train)):
            inputs = [[ele.to(device) if torch.is_tensor(ele) else ele for ele in elem] for elem in inputs]
            rec_result, ssc_result = model(inputs)
            label_targets = torch.stack(label_targets, dim=0).squeeze(-1).to(device)
            ssc_targets = torch.stack(ssc_targets, dim=0).squeeze(-1).to(device)
            loss_combined, loss_bce_train, loss_ssc_train = model.compute_loss_fine_tuned(
                rec_result, ssc_result, label_targets, ssc_targets)

            optimizer.zero_grad()
            loss_combined.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss_combined
            loss_bce += loss_bce_train
            loss_ssc += loss_ssc_train

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'{name}: {param.grad.norm()}')

        avg_train_loss = train_loss / len(data_train)
        avg_loss_bce = loss_bce / len(data_train)
        avg_loss_ssc = loss_ssc / len(data_train)
        
        # evaluation
        loss_val, loss_bce_val, loss_ssc_val, auc, f1, precision, recall, ndcg, mrr = \
            evaluator(args, model, data_valid, gt_merge, epoch, device)
        
        logging.info(f'''loss_train: {avg_train_loss:.4f}, ''')
        tensorboard_write(writer, args, avg_train_loss, avg_loss_bce, avg_loss_ssc, loss_val, loss_bce_val, loss_ssc_val,
                          auc, f1, precision, recall, ndcg, mrr, epoch)

        # save best epoch
        if epoch != 0 and best_auc < auc:
            best_epoch = epoch
            best_auc = auc
            best_model_state = deepcopy(model.state_dict()) 
        logging.info(f'best_epoch: {best_epoch}, best_auc: {best_auc:.4f}\n')

        if epoch - best_epoch > args.early_stop:   # n个epoch内，验证集性能不上升之后就停
            break

    # save the best model
    logging.info('Train finished')
    torch.save(best_model_state, open(os.path.join(save_dir,
                                                   'Epoch_{}_auc_{:.4}.model'.format(best_epoch, best_auc)), 'wb'))
    # test
    model.load_state_dict(best_model_state)
    model.to(device)
    loss_test, loss_bce_test, loss_ssc_test, auc, f1, precision, recall, ndcg, mrr = \
        evaluator_test(args, model, dataset, data_test, gt_merge, -1, device, rec_results_path)
    tensorboard_write(writer, args, 0, 0, 0, loss_test, loss_bce_test, loss_ssc_test,
                      auc, f1, precision, recall, ndcg, mrr, -1)


def main_mask(args, model, optimizer, writer, dataset, data_train, data_valid, voc_size, device, save_dir,
                  log_save_id):
    epoch_mask = 0
    best_epoch_mask, best_ja_mask = 0, 0
    EPOCH = args.pretrain_epochs
    for epoch in range(EPOCH):
        epoch += 1
        logging.info(f'epoch {epoch} --------------------------model_name={args.model_name}, logger={log_save_id}, mode=pretrain_mask')

        # mask pretrain
        model.train()
        epoch_mask += 1
        loss_train = 0
        for batch_idx, (inputs, label_targets, ssc_targets) in tqdm(enumerate(data_train), ncols=60, desc="pretrain_mask", total=len(data_train)):
            batch_size = len(inputs)
            if args.mask_prob > 0:
                seq_len = [args.seq_len_disease, args.seq_len_evaluation, args.seq_len_symptom]
                masked_batch = mask_batch_data(inputs, dataset.cure_voc, dataset.evaluation_voc, dataset.symptom_voc,
                                               args.mask_prob, seq_len)
            else:
                masked_batch = inputs
            masked_batch = [[ele.to(device) if torch.is_tensor(ele) else ele for ele in elem] for elem in masked_batch]
            result = model(masked_batch, mode='pretrain_mask').view(1, -1)

            bce_target_dis = np.zeros((batch_size, voc_size[0]))
            bce_target_eval = np.zeros((batch_size, voc_size[1]))
            bce_target_sym = np.zeros((batch_size, voc_size[2]))

            # multi label margin loss
            multi_target_dis = np.full((batch_size, voc_size[0]), -1)
            multi_target_eval = np.full((batch_size, voc_size[1]), -1)
            multi_target_sym = np.full((batch_size, voc_size[2]), -1)
            for i in range(batch_size):
                sub = [elem.tolist() for elem in inputs[i][1:4]]
                bce_target_dis[i, sub[0][0]] = 1
                bce_target_eval[i, sub[1][0]] = 1
                bce_target_sym[i, sub[2][0]] = 1

                multi_target_dis[i, 0:len(sub[0][0])] = sub[0][0]
                multi_target_eval[i, 0:len(sub[1][0])] = sub[1][0]
                multi_target_sym[i, 0:len(sub[2][0])] = sub[2][0]

            bce_target = np.concatenate((bce_target_dis, bce_target_eval, bce_target_sym), axis=1)

            multi_target = np.concatenate((multi_target_dis, multi_target_eval, multi_target_sym), axis=1)

            loss_bce = F.binary_cross_entropy_with_logits(result, torch.tensor(bce_target).to(device).view(1, -1))
            # loss_multi = F.multilabel_margin_loss(result, torch.LongTensor(multi_target, device=device))
            # loss = (1 - args.weight_multi) * loss_bce + args.weight_multi * loss_multi
            loss = loss_bce

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_train += loss.item()

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'{name}: {param.grad.norm()}')

        loss_train /= len(data_train)
        # validation
        loss_val, ja, prauc, avg_p, avg_r, avg_f1, avg_cnt = \
            evaluator_mask(model, data_valid, voc_size, epoch, device, mode='pretrain_mask')

        if ja > best_ja_mask:
            best_epoch_mask, best_ja_mask = epoch, ja
        logging.info(f'Training Loss_mask: {loss_train:.4f}, Validation Loss_mask: {loss_val:.4f}, best_ja: {best_ja_mask:.4f} at epoch {best_epoch_mask}\n')
        tensorboard_write_mask(writer, loss_train, loss_val, ja, prauc, epoch_mask)

    save_pretrained_model(model, save_dir)


def main_nsp(args, model, optimizer, writer, data_train, data_val, device, save_dir, log_save_id):
    epoch_nsp = 0
    best_epoch_nsp, best_prc_nsp = 0, 0
    EPOCH = args.pretrain_epochs
    for epoch in range(EPOCH):
        epoch += 1
        logging.info(f'epoch {epoch} -------------model_name={args.model_name}, logger={log_save_id}, mode=pretrain_nsp')

        model.train()
        epoch_nsp += 1
        loss_train = 0
        for batch_idx, (inputs, label_targets, ssc_targets) in tqdm(enumerate(data_train), ncols=60, desc="pretrain_nsp", total=len(data_train)):
            inputs = [[ele.to(device) if torch.is_tensor(ele) else ele for ele in elem] for elem in inputs]
            result = model(inputs, mode='pretrain_nsp')
            label_targets = torch.stack(label_targets, dim=0).to(device)
            loss = model.compute_loss_nsp(result, label_targets)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_train += loss

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'{name}: {param.grad.norm()}')

        loss_train /= len(data_train)
        # validation
        precision, loss_val = evaluator_nsp(model, data_val, epoch, device, mode='pretrain_nsp')
        loss_val /= len(data_val)
        if precision > best_prc_nsp:
            best_epoch_nsp, best_prc_nsp = epoch, precision
        logging.info(f'Epoch {epoch:03d}   prc: {precision:.4}, best_prc: {best_prc_nsp:.4f} at epoch {best_epoch_nsp}, Training Loss_nsp: {loss_train:.4f}, Validation Loss_nsp: {loss_val:.4f}\n')
        tensorboard_write_nsp(writer, loss_train, loss_val, precision, epoch_nsp)

    save_pretrained_model(model, save_dir)


def save_pretrained_model(model, save_dir):
    # save the pretrained model
    model_path = os.path.join(save_dir, 'saved.pretrained_model')
    torch.save(model.state_dict(), open(model_path, 'wb'))
    logging.info('Pretrained model saved to {}'.format(model_path))


def load_pretrained_model(model, model_path):
    # load the pretrained model
    model.load_state_dict(torch.load(open(model_path, 'rb')))
    logging.info('Pretrained model loaded from {}'.format(model_path))


def tensorboard_write(writer, args, loss_train=0., loss_bce_train=0., loss_ssc_train=0., loss_val=0., loss_bce_val=0.,
                      loss_ssc_val=0., auc=0., f1=0., precision=None, recall=None, ndcg=None, mrr=None, epoch=0):
    if epoch >= 0:
        writer.add_scalar('Loss_Train/all', loss_train, epoch)
        writer.add_scalar('Loss_Train/bce', loss_bce_train, epoch)
        writer.add_scalar('Loss_Train/ssc', loss_ssc_train, epoch)

        writer.add_scalar('Loss_Val/all', loss_val, epoch)
        writer.add_scalar('Loss_Val/bce', loss_bce_val, epoch)
        writer.add_scalar('Loss_Val/ssc', loss_ssc_val, epoch)
    else:
        writer.add_scalar('Loss_Test/all', loss_val, epoch)
        writer.add_scalar('Loss_Test/bce', loss_bce_val, epoch)
        writer.add_scalar('Loss_Test/ssc', loss_ssc_val, epoch)
    writer.add_scalar('Metrics/AUC', auc, epoch)
    writer.add_scalar('Metrics/F1', f1, epoch)
    krange = eval(args.topk_range)
    for i in range(len(krange)):
        writer.add_scalar(f'Metrics/top{krange[i]}/Precision', precision[i], epoch)
        writer.add_scalar(f'Metrics/top{krange[i]}/Recall', recall[i], epoch)
        writer.add_scalar(f'Metrics/top{krange[i]}/NDCG', ndcg[i], epoch)
        writer.add_scalar(f'Metrics/top{krange[i]}/MRR', mrr[i], epoch)


def tensorboard_write_mask(writer, loss_train, loss_val, ja, prauc, epoch):
    writer.add_scalar('Mask/Loss_Train_Mask', loss_train, epoch)
    writer.add_scalar('Mask/Loss_Val_Mask', loss_val, epoch)
    writer.add_scalar('Mask/AUC_Mask', ja, epoch)
    writer.add_scalar('Mask/Precision_Mask', prauc, epoch)


def tensorboard_write_nsp(writer, loss_train, loss_val, precision, epoch):
    writer.add_scalar('NSP/Loss_Train_NSP', loss_train, epoch)
    writer.add_scalar('NSP/Loss_Val_NSP', loss_val, epoch)
    writer.add_scalar('NSP/Precision_NSP', precision, epoch) 
    

if __name__ == '__main__':
    sys.path.append("..")
    torch.manual_seed(2040)
    np.random.seed(20240)
    random.seed(2040)

    args = get_args()
    main(args)

