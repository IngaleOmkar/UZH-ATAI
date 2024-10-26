from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

# Initialize the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

with open("data/rel_lbl_examples.json", "r") as f:
    loaded_examples_dict = json.load(f)

# Convert to InputExample objects
positive_examples = []
for key, value in loaded_examples_dict.items():
    # Check that value is a list
    if isinstance(value, list):
        positive_examples.append(InputExample(texts=[key] + value, label=1.0)) 


train_dataloader = DataLoader(positive_examples, shuffle=True, batch_size=4)

# Use a contrastive loss function to train on the pairs
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune the model with a few epochs (e.g., 1-3)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)

# Save the fine-tuned model for later use
model.save("data/lbl_nlp_embeddings_model")