import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from rdflib import Graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from GCNEncoder import GCNEncoder
import pickle

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

nlp = spacy.load("en_core_web_sm")

# Create node features using spaCy embeddings
with open('../../models/node_features.pkl', 'rb') as f:
    node_features = pickle.load(f)

# Create a Data object
data = Data(x=node_features, edge_index=edge_index)

# Move model and data to GPU if available
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
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

torch.save(model.state_dict(), '../../models/gcn_embedding.pth')
