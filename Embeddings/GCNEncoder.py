import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)  # Learnable embeddings
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)
        self.num_nodes = num_nodes

    def encode(self, data):
        # Get the node embeddings
        x = self.embedding(torch.arange(self.num_nodes, device=data.edge_index.device))
        # Perform graph convolution
        x = self.conv1(x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        return x

    def forward(self, data):
        return self.encode(data)