import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from rdflib import Graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the RDF graph using rdflib
rdf_graph = Graph()
rdf_graph.parse("14_graph.nt", format="turtle")

print("Graph loaded")

# Create node and edge lists
nodes = list(set([str(s) for s in rdf_graph.subjects()] + [str(o) for o in rdf_graph.objects()]))
node_idx = {node: i for i, node in enumerate(nodes)}

edges = [(node_idx[str(s)], node_idx[str(o)]) for s, p, o in rdf_graph if str(o) in node_idx]

# Convert edge list to PyTorch tensor
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
num_nodes = len(nodes)
print(f'Number of nodes: {num_nodes}')

# Define a GCN model with learnable embeddings
class GCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)  # Learnable embeddings
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)

    def encode(self, data):
        # Get the node embeddings
        x = self.embedding(torch.arange(num_nodes, device=data.edge_index.device))
        # Perform graph convolution
        x = self.conv1(x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        return x

    def forward(self, data):
        return self.encode(data)

# Create a Data object
data = Data(edge_index=edge_index)

# Move model and data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = GCNEncoder(num_nodes=num_nodes, embedding_dim=16, hidden_channels=32).to(device)

# Define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("Start training")

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    # Using mean squared error loss for reconstruction
    loss = F.mse_loss(out, model.embedding(torch.arange(num_nodes, device=data.edge_index.device)))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

print("Training finished")

torch.save(model.state_dict(), 'models/gcn_embedding.pth')
