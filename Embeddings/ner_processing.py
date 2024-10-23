import pickle 
from transformers import pipeline
from sklearn.model_selection import train_test_split
from rdflib import Graph
import torch
from load_embeddings import NodeEmbeddings

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

if torch.backends.mps.is_available():
    device = torch.device('mps')
    device_index = 0  # device index for MPS is typically 0
else:
    device = torch.device('cpu')
    device_index = -1  # device index for CPU

ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple", device=device.index)

print("Pipeline set up.")

# Load the RDF graph using rdflib
rdf_graph = Graph()
rdf_graph.parse("14_graph.nt", format="turtle")

print("Graph loaded.")

# Create node and edge lists
nodes = list(set([str(s) for s in rdf_graph.subjects()] + [str(o) for o in rdf_graph.objects()]))
node_idx = {node: i for i, node in enumerate(nodes)}

uri_to_node_id = {uri: idx for idx, uri in enumerate(nodes)}
node_id_to_uri = {idx: uri for idx, uri in enumerate(nodes)}

print("Uri to index dictionary created.")

def extract_ner(sentence):
    # Perform Named Entity Recognition
    ner_results = ner_pipeline(sentence)

    # Extract and print the entity names
    entities = [entity['word'] for entity in ner_results if entity['entity_group'] in ['PER', 'LOC', 'ORG', 'MISC']]
    return entities

entities = extract_ner("Steven Spielberg is from Ohio.")
print(entities)

triplets, uri_to_label, label_to_uri, label_list = pickle.load(open("../../models/formatted_data.pkl", "rb"))

embeddings = NodeEmbeddings(rdf_graph, '../../models/gcn_embedding.pth')

for entity in entities:
    uri = label_to_uri[entity]
    id = uri_to_node_id[uri]
    vector = embeddings.get_single_node_embedding(id)
    print(f"entity: {entity}, id: {id}, uri: {uri}, vector: {vector}")
    id_2 = embeddings.find_nearest_neighbor(vector)
    uri_2 = node_id_to_uri[id_2]
    label = uri_to_label[uri_2]
    print(f"re id: {id_2}, re uri: {uri_2}, re label: {label}")
