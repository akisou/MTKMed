# -*- coding: UTF-8 -*-
# ========================================================
#   Copyright (C) 2024 All rights reserved.
#   Project  : MTKMed 
#   Filename : analysis.py
#   Author   : petal
#   Date     : 2024/12/17
#   Desc     : 
# ========================================================

import os
import pickle
import re
import sys
import stat
sys.path.append("E://program/PycharmProjects/MTKMed")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
os.chmod('./', stat.S_IRWXU)

doctor = pd.read_csv('./Med/doctor.csv', sep='\t')
doctor_info = pd.read_csv('./output/doctors_filtered.csv', sep='\t')
patient = pd.read_csv('./Med/patient.csv', sep='\t')
ratings = pd.read_csv('./Med/ratings.csv', sep='\t')
kg = pd.read_csv('./Med/kg.csv', sep='\t')

doctor_num = len(set(doctor['doctor_id']))
depart_num = len(set(doctor_info['profession_direction']))
patient_num = len(set(patient['patient_id']))
ratings_num = len(ratings[ratings['label'] == 1])
ent_num = len(set(kg['head']) | set(kg['target']))
triple_num = len(kg)
relation_num = len(set(kg['relation']))

doctor_cure_disease = [len(eval(elem))for elem in doctor['cure_hist']]
doctor_cure_disease_weight = [sum(eval(elem)) for elem in doctor['cure_hist_weight']]
doctor_eval_disease = [len(eval(elem))for elem in doctor['evaluation_hist']]
doctor_eval_disease_weight = [sum(eval(elem))for elem in doctor['evaluation_hist_weight']]
doctor_symptom = [len(eval(elem))for elem in doctor['symptom_hist']]
doctor_symptom_weight = [sum(eval(elem))for elem in doctor['symptom_hist_weight']]

query_length = [len(elem) for elem in patient['query']]

print('doctor num: ', doctor_num)
print('department num: ', depart_num)
print('doctor per depart: ', doctor_num / depart_num)
print('doctor per depart: ', patient_num / depart_num)
print('patient num: ', patient_num)
print('rating num: ', ratings_num)
print('ent num: ', ent_num)
print('triple num: ', triple_num)
print('relation num: ', relation_num)
print('patient sparsity: ', ratings_num / (doctor_num * patient_num))
print('patient per doctor: ', patient_num / doctor_num)
print('triple sparsity: ', triple_num / (ent_num ^ 2))
print('triples per doctor: ', triple_num / doctor_num)
print('disease cure type num per doctor: ', np.mean(doctor_cure_disease))
print('disease feedback type num per doctor: ', np.mean(doctor_eval_disease))
print('symptom type num per doctor: ', np.mean(doctor_symptom))

print('disease num per doctor cure: ', np.mean(doctor_cure_disease_weight))
print('disease feedback num per doctor: ', np.mean(doctor_eval_disease_weight))
print('symptom num per doctor: ', np.mean(doctor_symptom_weight))

print('patient query length: ', np.mean(query_length))

plt.rcParams['font.sans-serif'] = ['SimHei']
for name, counts, bin_size, stop in zip(['disease_cure', 'disease_feedback', 'symptom'],
                        [doctor_cure_disease, doctor_eval_disease, doctor_symptom], [5, 5, 200],
                                  [45, 55, 2600]):
    # 分桶设置
    max_disease = max(counts)  # 获取最大值，确定分桶范围
    bins = list(range(0, stop + 1, bin_size))  # 生成分桶边界 [0, 100, 200, ..., max_disease]
    bins.append(max_disease)

    
    # 使用numpy.histogram统计每个桶内的数据数量
    hist, bin_edges = np.histogram(counts, bins=bins)
    
    # 将桶边界转换为横坐标标签
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
    
    # 绘制柱状图
    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.bar(bin_labels, hist, width=0.8, edgecolor='black', color='skyblue')
    
    # 添加标题和坐标轴标签
    txt = ''
    if name == 'disease_cure':
        txt = '治疗疾病类型数量'
    elif name == 'disease_feedback':
        txt = '患者反馈疾病类型数量' 
    elif name == 'symptom':
        txt = '症状类型数量'
    plt.title(txt + '-医生数量 分布图', fontsize=14)
    plt.xlabel(txt, fontsize=12)
    plt.ylabel('医生数量', fontsize=12)
    
    # 设置横坐标标签旋转，避免重叠
    plt.xticks(rotation=45)
    
    # 在柱子上方标出具体数值
    for i, value in enumerate(hist):
        plt.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=10)
    
    # 调整布局并显示图表
    plt.tight_layout()
    plt.savefig('../figs/' + name + '.png', dpi=300, bbox_inches='tight')  # 保存为高分辨率 PNG 文件
    plt.show()


for name, counts, bin_size, stop in zip(['patient_cure_num', 'patient_feedback_num', 'symptom_num'],
                        [doctor_cure_disease_weight, doctor_eval_disease_weight, doctor_symptom_weight],
                                  [500, 100, 4000],
                                        [7000, 1400, 66000]):
    # 分桶设置
    max_count = max(counts)  # 找到最大值，确定分桶范围
    bins = list(range(0, stop + 1, bin_size))  # 生成分桶边界 [0, 100, 200, ..., max_count]
    bins.append(max_count)

    # 使用numpy.histogram统计每个桶内的数据数量
    hist, bin_edges = np.histogram(counts, bins=bins)

    # 横坐标标签：生成每个桶的范围字符串
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]

    # 绘制柱状图
    plt.figure(figsize=(12, 6))  # 设置图表大小
    plt.bar(bin_labels, hist, width=0.8, edgecolor='black', color='skyblue')

    # 添加标题和坐标轴标签
    txt = ''
    if name == 'patient_cure_num':
        txt = '治疗患者数量'
    elif name == 'patient_feedback_num':
        txt = '患者反馈数量'
    elif name == 'symptom_num':
        txt = '医生接触症状总数量'
    plt.title(txt + '-医生数量 分布图', fontsize=16)
    plt.xlabel(txt, fontsize=12)
    plt.ylabel('医生数量', fontsize=12)

    # 设置横坐标的标签旋转，以防止重叠
    plt.xticks(rotation=45)

    # 在柱状图上标出每个柱子的具体值
    for i, value in enumerate(hist):
        plt.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=10)

    # 显示图表
    plt.tight_layout()  # 调整布局，防止标签被裁剪
    plt.savefig('../figs/' + name + '.png', dpi=300, bbox_inches='tight')  # 保存为高分辨率 PNG 文件
    plt.show()


# 分桶设置
bin_size = 5
max_count = max(query_length)  # 找到最大值，确定分桶范围
bins = list(range(0, 100 + 1, bin_size))  # 生成分桶边界 [0, 100, 200, ..., max_count]
bins.append(max_count)

# 使用numpy.histogram统计每个桶内的数据数量
hist, bin_edges = np.histogram(query_length, bins=bins)

# 横坐标标签：生成每个桶的范围字符串
bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]

# 绘制柱状图
plt.figure(figsize=(12, 6))  # 设置图表大小
plt.bar(bin_labels, hist, width=0.8, edgecolor='black', color='skyblue')

plt.title('query长度-患者数量 分布图', fontsize=16)
plt.xlabel('query长度', fontsize=12)
plt.ylabel('患者数量', fontsize=12)

# 设置横坐标的标签旋转，以防止重叠
plt.xticks(rotation=45)

# 在柱状图上标出每个柱子的具体值
for i, value in enumerate(hist):
    plt.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=10)

# 显示图表
plt.tight_layout()  # 调整布局，防止标签被裁剪
plt.savefig('../figs/' + 'query_length.png', dpi=300, bbox_inches='tight')  # 保存为高分辨率 PNG 文件
plt.show()