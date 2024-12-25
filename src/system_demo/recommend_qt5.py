# -*- coding: UTF-8 -*-
# ========================================================
#   Copyright (C) 2024 All rights reserved.
#   Project  : MTKMed 
#   Filename : recommend_qt5.py
#   Author   : petal
#   Date     : 2024/12/24
#   Desc     : 
# ========================================================
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QLabel, QScrollArea
)
from PyQt5.QtCore import Qt
import sys

import os.path
import tkinter as tk
from tkinter import ttk
from src.models.MTKMed import MTKMed
from src.utils.dataloader import MedDataset
import numpy as np
import argparse
import pandas as pd
import textwrap
import json
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default='', help="User notes")
    parser.add_argument('--model_name', type=str, default='MTKMed', help="model name")
    parser.add_argument('--data_path', type=str, default='../../data/Med/', help="data path")
    parser.add_argument('--bert_path', type=str, default='../models/mcBert', help="mcBert path")
    parser.add_argument('--dataset', type=str, default='Med', help='dataset')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop after this many epochs without improvement')
    parser.add_argument('-t', '--test', action='store_true', help="test mode")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default=None,
                        help='log dir prefix like "log0", for model test')
    parser.add_argument('-p', '--pretrain_prefix', type=str, default=None,
                        help='log dir prefix like "log0", for finetune')
    parser.add_argument('--cuda', type=int, default=-1, help='which cuda')
    # pretrain

    parser.add_argument('-nsp', '--pretrain_nsp', action='store_true', help='whether to use nsp pretrain')
    parser.add_argument('-mask', '--pretrain_mask', action='store_true', help='whether to use mask prediction pretrain')
    parser.add_argument('--pretrain_epochs', type=int, default=300, help='number of pretrain epochs')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='mask probability')
    parser.add_argument('--freeze_layer_num', type=int, default=11, help='freeze the num of former layers of mcbert')

    parser.add_argument('--grad_norm', type=int, default=0, help='whether to grad norm for multi task train')
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
    parser.add_argument('--batch_size', type=int, default=256, help='batch size during training')
    parser.add_argument('--adapter_dim', type=int, default=128, help='dimension of adapter layer')
    parser.add_argument('--boundaries_num', type=int, default=10, help='boundary num of token frequency embedding')
    parser.add_argument('--topk_range', type=str, default='[2, 5, 10, 20]', help='topk choice')  #

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability of transformer encoder')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--weight_label', type=float, default=1, help='loss weight of bce task')
    parser.add_argument('--weight_ssc', type=float, default=10, help='loss weight of satisfying score task')
    parser.add_argument('--weight_bpr', type=float, default=2, help='loss weight of bpr about ranking quality')

    # parameters for ablation study
    parser.add_argument('-s', '--doctor_seperate', action='store_true',
                        help='whether to combine disease, evaluation, symptom')
    parser.add_argument('-e', '--seg_rel_emb', action='store_false', default=True,
                        help='whether to use segment and relevance embedding layer')

    args = parser.parse_args()
    return args


# 模拟数据
class Item:
    def __init__(self, gt_id, department, profile, rank_title, education_title,
                 hospitals, consultation_amount, cure_satisfaction, attitude_satisfaction, message):
        self.gt_id = gt_id
        self.department = department
        self.profile = profile
        self.rank_title = rank_title
        self.education_title = education_title
        self.hospitals = hospitals
        self.consultation_amount = consultation_amount
        self.cure_satisfaction = cure_satisfaction
        self.attitude_satisfaction = attitude_satisfaction
        self.message = message


class Recommender:
    def __init__(self, args):
        # doctor pool
        self.args = args
        self.max_display = 20
        self.data_path = '../../data/Med/'
        self.model_path = '../../pretrained_models/finetune/finetune.model'

        self.dataset = MedDataset(self.data_path, "cpu", 0)

        self.device = torch.device('cuda:{}'.format(args.cuda)) if args.cuda >= 0 else 'cpu'
        self.doctor_detail_info = pd.read_csv(os.path.join(self.data_path, 'doctor_detail.csv'), sep='\t')
        self.items = self.build_items(self.doctor_detail_info)
        with open(os.path.join(self.data_path, 'side_doctor2id.json'), "r", encoding="utf-8") as file:
            self.doctor2id = json.load(file)
        self.id2doctor = dict(zip(self.doctor2id.values(), self.doctor2id.keys()))

        self.doctor_pool = dict(zip([self.doctor2id[elem.gt_id] for elem in self.items], self.items))

        # model
        self.model = self.loadmodel()

        dinfo_columns = self.dataset.doctor_info.columns.tolist()
        dinfo_columns.remove('doctor_id')
        self.did2info = dict(zip(self.dataset.doctor_info['doctor_id'], self.dataset.doctor_info[dinfo_columns].values))

    def add_word(self, word, voc):
        voc.word2idx[word] = len(voc.word2idx)
        voc.idx2word[len(voc.idx2word)] = word
        return voc

    def build_items(self, doctor_info):
        items = []
        for i in range(len(doctor_info)):
            values = list(doctor_info.loc[i, :].values)
            values[0] = str(values[0])
            items.append(Item(*values))
        return items

    def loadmodel(self):
        self.add_word('[MASK]', self.dataset.cure_voc)
        self.add_word('[MASK]', self.dataset.evaluation_voc)
        self.add_word('[MASK]', self.dataset.symptom_voc)
        voc_size = (len(self.dataset.cure_voc), len(self.dataset.evaluation_voc), len(self.dataset.symptom_voc))
        model = MTKMed(self.args, self.dataset, voc_size)

        state_dict = torch.load(open(self.model_path, 'rb'), map_location=self.device)
        model.load_state_dict(state_dict, strict=False)

        return model

    def query_process(self, query):
        inputs = []
        user = 0
        dids = self.did2info.keys()
        for did in dids:
            tmp = []
            tmp.append(torch.LongTensor([user, did]))
            [tmp.append(torch.LongTensor([self.did2info[did][i], self.did2info[did][i + 1]])) for i in range(0, 6, 2)]
            tmp.append(query)
            inputs.append(tmp)
        inputs = [[ele.to(self.device) if torch.is_tensor(ele) else ele for ele in elem] for elem in inputs]
        rec_result, ssc_result, pred_score = self.model.score(inputs)
        scores = [(idx, rec, ssc, pred)
                  for idx, rec, ssc, pred in
                  zip(dids, torch.sigmoid(rec_result).cpu().tolist(), ssc_result.cpu().tolist(),
                      pred_score.cpu().tolist())]
        scores = sorted(scores, key=lambda x: x[3], reverse=True)

        return scores[:self.max_display]


# 文本换行处理
def wrap_text(text, width, lines=3):
    wrapped = textwrap.fill(text, width=width)  # 按指定宽度换行
    lines_list = wrapped.split("\n")[:lines]  # 限制最大行数
    return "\n".join(lines_list) + ("..." if len(lines_list) < len(wrapped.split("\n")) else "")


class QueryTool(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Query Matching Tool")
        self.setGeometry(100, 100, 1200, 800)

        # 主窗口
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 输入框和按钮
        input_layout = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your query here...")

        # 提交按钮
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.on_submit)

        # 清空按钮
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.on_clear)  # 连接到清空功能

        input_layout.addWidget(QLabel("Query:"))
        input_layout.addWidget(self.query_input)
        input_layout.addWidget(self.submit_button)
        input_layout.addWidget(self.clear_button)  # 添加清空按钮

        # 表格显示结果
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(14)  # 设置列数
        self.result_table.setHorizontalHeaderLabels([
            'Index', 'ID', '科室', '简历', '医院职称', '教育职称', '所在医院',
            '问诊费用', '治疗满意度', '态度满意度', '寄语',
            '推荐预测分数', '满意度预测分数', '综合预测分数'
        ])
        # 设置列标题可拉动调整
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.result_table.verticalHeader().setVisible(False)  # 隐藏行号

        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.result_table)

        # 将所有部件添加到主布局
        main_layout.addLayout(input_layout)
        main_layout.addWidget(scroll_area)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def set_table_item(self, row, column, text):
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # 设置单元格为只读
        self.result_table.setItem(row, column, item)

    def wrap_text(self, text, width=30, lines=3):
        if (not text) or isinstance(text, float):
            return ""
        # wrapped = textwrap.fill(text, width=width)  # 按指定宽度换行
        # lines_list = wrapped.split("\n")[:lines]  # 限制最大行数
        # return "\n".join(lines_list) + ("..." if len(lines_list) < len(wrapped.split("\n")) else "")
        return text

    def on_cell_click(self, row, column):
        # 获取点击的单元格文本
        text = self.result_table.item(row, column).text()

        # 如果是“简历”或“寄语”等列的文本，可以展示完整内容
        if column == 3 or column == 10:  # 例如简历(3列)和寄语(10列)
            # 获取完整的文本
            item = self.model.items[row]
            full_text = item.profile if column == 3 else item.message
            # 使用QMessageBox显示完整的文本
            msg = QMessageBox(self)
            msg.setWindowTitle("Full Text")
            msg.setText(full_text)
            msg.exec_()

    def on_clear(self):
        self.query_input.clear()  # 清空输入框内容

    def on_submit(self):
        query = self.query_input.text()
        results = self.model.query_process(query)  # 调用模型逻辑
        self.result_table.setRowCount(0)  # 清空旧结果

        for i, (id, rec, ssc, pred) in enumerate(results):
            item = self.model.items[id]

            # 将结果逐行添加到表格
            self.result_table.insertRow(i)
            self.set_table_item(i, 0, str(i + 1))
            self.set_table_item(i, 1, item.gt_id)
            self.set_table_item(i, 2, item.department)
            self.set_table_item(i, 3, self.wrap_text(item.profile, width=30, lines=3))
            self.set_table_item(i, 4, item.rank_title)
            self.set_table_item(i, 5, item.education_title)
            self.set_table_item(i, 6, item.hospitals)
            self.set_table_item(i, 7, str(item.consultation_amount))
            self.set_table_item(i, 8, str(item.cure_satisfaction))
            self.set_table_item(i, 9, str(item.attitude_satisfaction))
            self.set_table_item(i, 10, self.wrap_text(item.message, width=30, lines=3))
            self.set_table_item(i, 11, f"{rec:.4f}")
            self.set_table_item(i, 12, f"{ssc:.4f}")
            self.set_table_item(i, 13, f"{pred:.4f}")



# 启动应用程序
if __name__ == "__main__":
    app = QApplication(sys.argv)
    args = get_args()
    model = Recommender(args)
    print('Model loaded!')
    print('Ready!')
    query_tool = QueryTool(model)
    query_tool.show()
    sys.exit(app.exec_())
