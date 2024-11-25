import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KGCN(nn.Module):
    def __init__(self, num_users, num_entities, num_relations, embedding_dim, neighbor_sample_size, n_iter, aggregator="sum"):
        """
        :param num_users:
        :param num_entities:
        :param num_relations:
        :param embedding_dim:
        :param neighbor_sample_size:
        :param n_iter: conv iter num
        :param aggregator: type of aggregator , ['sum', 'neighbor',  'concat']
        """
        super(KGCN, self).__init__()

        self.embedding_dim = embedding_dim
        self.neighbor_sample_size = neighbor_sample_size
        self.n_iter = n_iter
        self.aggregator = aggregator

        # embedding
        self.user_embedding = nn.Embedding(num_users + 2, embedding_dim)
        self.entity_embedding = nn.Embedding(num_entities + 2, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations + 2, embedding_dim)

        # conv
        self.linear_layers = nn.ModuleList()
        for i in range(n_iter):
            input_dim = embedding_dim if aggregator != "concat" else embedding_dim * 2
            self.linear_layers.append(nn.Linear(input_dim, embedding_dim))

        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

    def construct_adj(self, kg_triples, num_entities):
        """
        :param kg_triples: (num_triples, 3)， every row is (head, relation, tail)
        :param num_entities:
        :return: adj_entity, adj_relation tensor, dict relationship
        """
        kg_dict = {}
        for head, relation, tail in kg_triples:
            if head not in kg_dict:
                kg_dict[head] = []
            if tail not in kg_dict:
                kg_dict[tail] = []
            kg_dict[head].append((tail, relation))
            kg_dict[tail].append((head, relation))

        adj_entity = np.zeros((num_entities, self.neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros((num_entities, self.neighbor_sample_size), dtype=np.int64)

        for entity in range(num_entities):
            if entity not in kg_dict:
                adj_entity[entity] = [entity] * self.neighbor_sample_size
                adj_relation[entity] = [0] * self.neighbor_sample_size
                continue

            neighbors = kg_dict[entity]
            n_neighbors = len(neighbors)
            if n_neighbors >= self.neighbor_sample_size:
                sampled_indices = np.random.choice(n_neighbors, self.neighbor_sample_size, replace=False)
            else:
                sampled_indices = np.random.choice(n_neighbors, self.neighbor_sample_size, replace=True)

            adj_entity[entity] = [neighbors[i][0] for i in sampled_indices]
            adj_relation[entity] = [neighbors[i][1] for i in sampled_indices]

        return torch.tensor(adj_entity), torch.tensor(adj_relation)

    def get_neighbors(self, items, adj_entity, adj_relation):
        """
        :param items: (batch_size,)
        :param adj_entity:
        :param adj_relation: adj
        :return: entities, relations list
        """
        items = items.unsqueeze(1)
        entities = [items]
        relations = []

        for i in range(self.n_iter):
            index = entities[i].view(-1)
            neighbor_entities = adj_entity[index].view(items.size(0), -1)
            neighbor_relations = adj_relation[index].view(items.size(0), -1)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        """
        在用户特定的图中混合邻居的向量。
        :param neighbor_vectors: (batch_size, -1, neighbor_sample_size, embedding_dim)
        :param neighbor_relations: (batch_size, -1, neighbor_sample_size, embedding_dim)
        :param user_embeddings: (batch_size, embedding_dim)
        :return: embedding after mixing with user embedding
        """
        user_embeddings = user_embeddings.unsqueeze(1).unsqueeze(2)
        scores = (user_embeddings * neighbor_relations).mean(dim=-1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (weights * neighbor_vectors).sum(dim=2)

    def aggregate(self, user_embeddings, entities, relations):
        """
        聚合实体及其邻居表示。
        :param user_embeddings: (batch_size, dim)
        :param entities: list
        :param relations: list
        :return: embedding after aggreate
        """
        entity_vectors = [self.entity_embedding(e) for e in entities]
        relation_vectors = [self.relation_embedding(r) for r in relations]

        for i in range(self.n_iter):
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = (user_embeddings.size(0), -1, self.neighbor_sample_size, self.embedding_dim)
                self_vectors = entity_vectors[hop]
                neighbor_vectors = entity_vectors[hop + 1].view(shape)
                neighbor_relations = relation_vectors[hop].view(shape)

                neighbors_agg = self.mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

                if self.aggregator == "sum":
                    output = self_vectors + neighbors_agg
                elif self.aggregator == "neighbor":
                    output = neighbors_agg
                elif self.aggregator == "concat":
                    output = torch.cat([self_vectors, neighbors_agg], dim=-1)
                else:
                    raise ValueError(f"Unknown aggregator: {self.aggregator}")

                output = self.linear_layers[i](output)
                entity_vectors_next_iter.append(self.ReLU(output) if i < self.n_iter - 1 else self.Tanh(output))
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0]

    def forward(self, users, items, adj_entity, adj_relation):
        """
        :param users: (batch_size,)
        :param items: (batch_size,)
        :param adj_entity: adg neighbors
        :param adj_relation: adj relation
        :return: embedding of users and items
        """
        user_embeddings = self.user_embedding(users)
        entities, relations = self.get_neighbors(items, adj_entity, adj_relation)
        item_embeddings = self.aggregate(user_embeddings, entities, relations)
        return user_embeddings, item_embeddings


# 示例用法
if __name__ == "__main__":
    # 示例
    kg_triples = [(0, 1, 2), (2, 1, 3), (3, 2, 4)]  # 示例三元组
    num_users, num_entities, num_relations = 10, 5, 3
    embedding_dim, neighbor_sample_size, n_iter = 16, 2, 2

    model = KGCN(num_users, num_entities, num_relations, embedding_dim, neighbor_sample_size, n_iter)
    adj_entity, adj_relation = model.construct_adj(kg_triples, num_entities)

    users = torch.tensor([0, 1])  # Batch 用户 ID
    items = torch.tensor([2, 3])  # Batch 实体 ID

    user_embeddings, item_embeddings = model.forward(users, items, adj_entity, adj_relation)
    print(user_embeddings.shape, item_embeddings.shape)