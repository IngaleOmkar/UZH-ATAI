import torch
from torch_geometric.data import Data
from rdflib import Graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from GCNEncoder import GCNEncoder

class NodeEmbeddings:
    def __init__(self, graph, model_path):
        # Load the RDF graph using rdflib
        self.rdf_graph = graph

        # Create node and edge lists
        self.nodes = list(set([str(s) for s in self.rdf_graph.subjects()] + [str(o) for o in self.rdf_graph.objects()]))
        self.node_idx = {node: i for i, node in enumerate(self.nodes)}

        edges = [(self.node_idx[str(s)], self.node_idx[str(o)]) for s, p, o in self.rdf_graph if str(o) in self.node_idx]

        # Convert edge list to PyTorch tensor
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.num_nodes = len(self.nodes)
        print(f'Number of nodes: {self.num_nodes}')

        # Create a Data object
        self.data = Data(edge_index=self.edge_index)

        # Move model and data to GPU if available
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.data = self.data.to(self.device)

        # Initialize the model
        self.model = GCNEncoder(num_nodes=self.num_nodes, embedding_dim=16, hidden_channels=32).to(self.device)

        # Load the saved state dictionary into the model
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("Model loaded.")

    # Function to get the embeddings for all nodes
    def get_node_embeddings(self):
        with torch.no_grad():
            embeddings = self.model.encode(self.data)
        return embeddings

    # Function to get the embedding of a single node
    def get_single_node_embedding(self, node_id):
        with torch.no_grad():
            embeddings = self.model.encode(self.data)
            node_embedding = embeddings[node_id].cpu().numpy()  # Get the embedding of the given node
        return node_embedding

    # Function to find the nearest neighbor of a given vector in the latent space
    def find_nearest_neighbor(self, vector):
        with torch.no_grad():
            embeddings = self.model.encode(self.data).cpu().numpy()
            similarities = cosine_similarity([vector], embeddings)
            nearest_neighbor_id = np.argmax(similarities[0])
        return nearest_neighbor_id

# # Example usage:
# node_id = 3  # Replace with your node ID
# node_embeddings = get_node_embeddings(model, data)
# print("Node embeddings:")
# print(node_embeddings)

# single_node_embedding = get_single_node_embedding(node_id, model, data)
# print(f"Embedding of node {node_id}, {nodes[node_id]}: {single_node_embedding}")

# # Define a random vector in the latent space to find its nearest neighbor
# nearest_neighbor_id = find_nearest_neighbor(single_node_embedding, model, data)
# print(f"Nearest neighbor of the vector: {nearest_neighbor_id}")
