import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from rdflib import Graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

# Create a Data object
data = Data(edge_index=edge_index)

# Move model and data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Initialize the model
model = GCNEncoder(num_nodes=num_nodes, embedding_dim=16, hidden_channels=32).to(device)

# Load the saved state dictionary into the model
model.load_state_dict(torch.load('models/gcn_embedding.pth'))
print("Model loaded.")

# Function to get the embeddings for all nodes
def get_node_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data)
    return embeddings

# Function to get the embedding of a single node
def get_single_node_embedding(node_id, model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data)
        node_embedding = embeddings[node_id].cpu().numpy()  # Get the embedding of the given node
    return node_embedding

# Function to find the nearest neighbor of a given vector in the latent space
def find_nearest_neighbor(vector, model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data).cpu().numpy()
        similarities = cosine_similarity([vector], embeddings)
        nearest_neighbor_id = np.argmax(similarities[0])
    return nearest_neighbor_id

# Example usage:
node_id = 0  # Replace with your node ID
node_embeddings = get_node_embeddings(model, data)
print("Node embeddings:")
print(nodes[node_id])
print(node_embeddings)

single_node_embedding = get_single_node_embedding(node_id, model, data)
print(f"Embedding of node {node_id}: {single_node_embedding}")

# Define a random vector in the latent space to find its nearest neighbor
nearest_neighbor_id = find_nearest_neighbor(single_node_embedding, model, data)
print(f"Nearest neighbor of the vector: {nearest_neighbor_id}")
