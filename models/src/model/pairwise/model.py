import torch
from torch import nn
import torch.nn.functional as F

from src.model.pointwise.model import HomophilyContentCNN
from src.model.utils import FM
from src.utils.utils import create_embedding_layer


class HomophilyContentCNNPairwise(nn.Module):
    def __init__(self, node2vec_embedding,
                 node2vec_embedding_dim,
                 max_length_user_rep,
                 max_length_comment_section,
                 dropout=0.5,
                 user_num_kernels=50,  # number of kernels
                 section_num_kernels=50,
                 user_kernel_size=5,  # number of words in window
                 section_kernel_size=5,
                 latent_factors_deepconn=16,
                 latent_factors_user=10,
                 latent_factors_section=10,
                 freeze_embeddings=True, ):
        super(HomophilyContentCNNPairwise, self).__init__()
        self.max_length_user_rep = max_length_user_rep
        self.max_length_comment_section = max_length_comment_section

        self.embedding_dim, self.max_length, self.embedding = create_embedding_layer(freeze_embeddings)

        self.node2vec_embedding = node2vec_embedding
        self.node2vec_embedding_dim = node2vec_embedding_dim

        self.user_kernel_size = user_kernel_size
        self.section_kernel_size = section_kernel_size

        self.userCNN = nn.Sequential(
            # convolutional
            nn.Conv2d(1, user_num_kernels, (user_kernel_size, self.embedding_dim), stride=1),
            nn.ReLU(),
        )
        self.sectionCNN = nn.Sequential(
            nn.Conv2d(1, section_num_kernels, (section_kernel_size, self.embedding_dim), stride=1),
            nn.ReLU(),
        )

        self.user_linear = nn.Sequential(
            nn.Linear(user_num_kernels, latent_factors_deepconn),
            nn.ReLU(),
        )

        self.user_linear_con = nn.Sequential(
            nn.Linear(latent_factors_deepconn + node2vec_embedding_dim, latent_factors_user),
            nn.ReLU(),
        )

        self.section_linear = nn.Sequential(
            nn.Linear(section_num_kernels, latent_factors_deepconn),
            nn.ReLU(),
        )

        self.section_linear_con = nn.Sequential(
            nn.Linear(latent_factors_deepconn + node2vec_embedding_dim, latent_factors_section),
            nn.ReLU(),
        )

        self.fm = FM(n=latent_factors_section + latent_factors_user,
                     k=latent_factors_section + latent_factors_user)

        self.out_layer = nn.Linear(1, 1)

    def pred(self, user_content, user_emb, user_emb_offsets, section_content, section_emb, section_emb_offsets):
        # look-up content
        user_x = self.embedding(user_content).float()
        section_x = self.embedding(section_content).float()

        # look-up node2vec embeddings
        user_emb_x = self.node2vec_embedding(user_emb, user_emb_offsets).float()
        section_emb_x = self.node2vec_embedding(section_emb, section_emb_offsets).float()

        # user
        user_x = self.userCNN(user_x)
        user_x = torch.squeeze(user_x, -1)
        user_x = F.max_pool1d(user_x, self.max_length_user_rep - self.user_kernel_size + 1)
        user_x = user_x.squeeze(-1)
        user_x = self.user_linear(user_x)
        user_x = self.user_linear_con(torch.cat((user_x, user_emb_x), 1))

        # section
        section_x = self.sectionCNN(section_x)
        section_x = torch.squeeze(section_x, -1)
        section_x = F.max_pool1d(section_x, self.max_length_comment_section - self.section_kernel_size + 1)
        section_x = section_x.squeeze(-1)
        section_x = self.section_linear(section_x)
        section_x = self.section_linear_con(torch.cat((section_x, section_emb_x), 1))
        z = torch.cat((user_x, section_x), 1)
        return self.out_layer(self.fm(z))

    def get_user_rep(self, user_content, user_emb, user_emb_offsets, ):
        user_x = self.embedding(user_content).float()

        # look-up node2vec embeddings
        user_emb_x = self.node2vec_embedding(user_emb, user_emb_offsets).float()

        # user
        user_x = self.userCNN(user_x)
        user_x = torch.squeeze(user_x, -1)
        user_x = F.max_pool1d(user_x, self.max_length_user_rep - self.user_kernel_size + 1)
        user_x = user_x.squeeze(-1)
        user_x = self.user_linear(user_x)
        user_x = self.user_linear_con(torch.cat((user_x, user_emb_x), 1))
        return user_x

    def forward(self, user_content, user_emb, user_emb_offsets, pos_section, neg_section):
        section_content_pos, section_emb_pos, section_emb_offsets_pos = pos_section
        section_content_neg, section_emb_neg, section_emb_offsets_neg = neg_section

        out_pos = self.pred(user_content, user_emb, user_emb_offsets, section_content_pos, section_emb_pos,
                            section_emb_offsets_pos)

        out_neg = self.pred(user_content, user_emb, user_emb_offsets, section_content_neg, section_emb_neg,
                            section_emb_offsets_neg)


        output = torch.squeeze(torch.sigmoid(out_pos - out_neg), -1)
        return output
