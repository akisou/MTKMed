# -*- coding: UTF-8 -*-
# ========================================================
#   Copyright (C) 2024 All rights reserved.
#   Project  : MTKMed 
#   Filename : MMOE.py
#   Author   : petal
#   Date     : 2024/11/20
#   Desc     : 
# ========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class MMOE(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim, num_tasks=2):
        """
        :param input_dim: input dimension
        :param num_experts:
        :param hidden_dim:
        :param num_tasks:
        """
        super(MMOE, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # expert layer
        self.expert_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)
        ])

        # gate layer
        self.gate_layers = nn.ModuleList([
            nn.Linear(input_dim, num_experts) for _ in range(num_tasks)
        ])

        # task layer
        self.task_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])

        # init
        self.expert_layers.apply(self.init_weights)
        self.gate_layers.apply(self.init_weights)
        self.task_layers.apply(self.init_weights)

    def forward(self, input_embed):
        """
        :param input_embed: inputs [batch_size, dim]
        :return: predict score of tasks
        """
        # expert layer
        expert_outputs = torch.stack([F.relu(expert(input_embed)) for expert in self.expert_layers],
                                     dim=1)  # (batch_size, num_experts, hidden_dim)

        # output of every tasks
        task_outputs = []
        for i in range(self.num_tasks):
            # gate weight
            gate_weights = F.softmax(self.gate_layers[i](input_embed), dim=1)  # (batch_size, num_experts)
            gate_outputs = torch.einsum('be,beh->bh', gate_weights, expert_outputs)  # 加权专家输出 (batch_size, hidden_dim)

            # output
            task_output = self.task_layers[i](gate_outputs)  # (batch_size, 1)
            task_outputs.append(task_output.squeeze(-1))

        return task_outputs  # list(2 * [batch_size, ])

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# 示例用法
if __name__ == "__main__":
    batch_size = 32
    doctor_dim = 64
    patient_dim = 64
    embed_dim = doctor_dim + patient_dim
    num_experts = 4
    hidden_dim = 128

    # 初始化 MMOE 模型
    model = MMOE(input_dim=embed_dim, num_experts=num_experts, hidden_dim=hidden_dim)

    # 示例输入
    doctor_embed = torch.rand(batch_size, doctor_dim)
    patient_embed = torch.rand(batch_size, patient_dim)

    # 前向计算
    label_scores, ssc_scores = model(doctor_embed, patient_embed)
    print("Label Scores:", label_scores.shape)  # (batch_size,)
    print("SSC Scores:", ssc_scores.shape)  # (batch_size,)

    # 示例损失计算
    label_targets = torch.randint(0, 2, (batch_size,))  # 二分类标签
    ssc_targets = torch.rand(batch_size)  # 回归目标

    label_loss = F.binary_cross_entropy_with_logits(label_scores, label_targets.float())
    ssc_loss = F.mse_loss(ssc_scores, ssc_targets)
    total_loss = label_loss + ssc_loss

    print("Label Loss:", label_loss.item())
    print("SSC Loss:", ssc_loss.item())
    print("Total Loss:", total_loss.item())

# train method
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# for batch in train_loader:
#     doctor_embed, patient_embed, label_targets, ssc_targets = batch
#     label_scores, ssc_scores = model(doctor_embed, patient_embed)
#
#     # 计算损失
#     label_loss = F.binary_cross_entropy_with_logits(label_scores, label_targets.float())
#     ssc_loss = F.mse_loss(ssc_scores, ssc_targets)
#     total_loss = label_loss + ssc_loss
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()