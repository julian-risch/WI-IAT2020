from torch import nn
import torch.nn.functional as F


class CommunityGraphModel(nn.Module):
    def __init__(self,
                 node2vec_embedding,
                 node2vec_embedding_dim,
                 layer_dim=100,
                 dropout=0.0):
        super(CommunityGraphModel, self).__init__()

        self.node2vec_embedding = node2vec_embedding
        self.node2vec_embedding_dim = node2vec_embedding_dim

    def forward(self, user_emb, user_emb_offsets, section_emb, section_emb_offsets):
        return self.pred(user_emb, user_emb_offsets, section_emb, section_emb_offsets)

    def pred(self, user_emb, user_emb_offsets, section_emb, section_emb_offsets):
        # look-up node2vec embeddings
        user_emb_x = self.node2vec_embedding(user_emb, user_emb_offsets).float()
        section_emb_x = self.node2vec_embedding(section_emb, section_emb_offsets).float()

        z = F.cosine_similarity(user_emb_x, section_emb_x)

        return z
