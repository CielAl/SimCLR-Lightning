import torch
from torch import nn
from torch.nn import functional as F


class InfoNCELoss(nn.Module):

    @staticmethod
    def loss_forward(features: torch.Tensor, batch_size: int, n_views: int, temperature: float):
        # n_view * batch_size x 1
        labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0).to(features.device)
        # [n_view * batch_size] x [n_view * batch_size]
        # noinspection PyUnresolvedReferences
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # labels = labels.to(self.args.device)
        # [n_view * batch_size] x out_dim
        features = F.normalize(features, dim=1)
        # [n_view * batch_size] x out_dim @ out_dim x [n_view * batch_size]
        # --> [n_view * batch_size] x [n_view * batch_size]
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        # [n_view * batch_size] x [n_view * batch_size] - 1 --> main diagonal discarded and the up trio shift by one
        # to the left
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        # use the mask to discard the diagonal: [n_view * batch_size] x [n_view * batch_size] - 1
        labels = labels[~mask].view(labels.shape[0], -1)
        # discard diagonal: [n_view * batch_size] x [n_view * batch_size] - 1
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        # [n_view * batch_size] x (n_views - 1)
        # this only works for n_views = 2, for n_views > 2,
        # reshape it to labels.shape[0] * (n_views - 1), -1 --> N-square increase in size?
        # also need to repeat the negative to align to the positive
        # for n_views > 2, for each input there would be more than one pair of positive data

        # the logits here is essentially [score of being similar to self, score of being similar to others]
        # the goal is that the score of being similar to self should be larger while suppressing the others,
        # therefore --> crossentropy + label 0 (corresponding to column 0).
        # for n_views > 2 there will multiple column of positives and we need to duplicate the matrix
        # for positives: we simply flatten it
        # for negatives: duplicate/repeat to match the flattened positive
        positives = similarity_matrix[labels.bool()].view(labels.shape[0] * (n_views - 1), -1)

        # select only the negatives
        # [n_view * batch_size] x (n_view * batch_size - n_views)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1).repeat(n_views - 1, 1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / temperature
        return logits, labels

    def __init__(self, batch_size, n_views, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature

    def forward(self, features):
        return InfoNCELoss.loss_forward(features, self.batch_size, self.n_views, self.temperature)
