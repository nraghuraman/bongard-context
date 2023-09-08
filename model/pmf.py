import torch
from torch import nn

from .utils import protonet_loss


class Model(nn.Module):
    """
    A PMF model, implementing the simple algorithm described in the paper
    "Pushing the Limits of Simple Pipelines for Few-Shot Learning:
    External Data and Fine-Tuning Make a Difference" by Hu et al.

    Project page: https://hushell.github.io/pmf/

    Note that we leave out the "F" stage in PMF, which corresponds to test-time
    fine-tuning of the encoder.
    """

    def __init__(self):
        super(Model, self).__init__()

        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

    def forward(self, x_support_emb, x_query_emb, y_support, y_query):
        """
        Returns:
            acc: A tensor indicating for each query in x_query_emb whether the model
                prediction is correct or not.
            logits: The predicted logits for each query.
            enc_loss: The loss on the backbone encoder.
            mimic_loss: The loss on the support-set Transformer (described in the paper),
                which is 0 in this case as there is no support-set Transformer.
        """
        enc_loss, logits = protonet_loss(
            x_support_emb, x_query_emb, y_support, y_query, self.scale, self.bias
        )
        mimic_loss = torch.tensor(0.0, device=enc_loss.device)
        acc = (torch.argmax(logits, dim=2) == y_query).float()
        return acc, logits, enc_loss, mimic_loss


def get_model(args):
    assert args.train_encoder, "Must train encoder for pmf"
    return Model().to(device=args.device)
