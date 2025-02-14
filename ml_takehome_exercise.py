# Import the libraries
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------- TASK 1 -------------------------
print("------------------------- TASK 1 -------------------------")
model_name = "all-MiniLM-L6-v2"

# Load the pre-trained model
model = SentenceTransformer(model_name)

# Sample sentences for testing
sentences = [
    "Machine learning is super fascinating",
    "Transformers are powerful for NLP tasks",
    "I love working on deep learning projects"
]

# Generate sentence embeddings
embeddings = model.encode(sentences)

# Print shape and example embeddings
print(f"Embedding Shape: {embeddings.shape}")
for i, sentence in enumerate(sentences):
    print(f"\nSentence: {sentence}")
    print(f"Embedding (first 5 values): {embeddings[i][:5]}")



# ------------------------- TASK 2 -------------------------
print("------------------------- TASK 2 -------------------------")
class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, transformer_model, sentiment_classes, intent_classes, seed = 42):
        """
        Multi-Task Learning Model for Sentiment Analysis & Intent Detection.
        - Sentiment: Binary (Positive/Negative) or Multi-Class
        - Intent: Multi-Class Classification (Dynamic number of classes)

        :param transformer_model: Name of the pre-trained Sentence Transformer model.
        :param sentiment_classes: Number of classes for sentiment analysis (default=2 for binary).
        :param intent_labels: List of unique intent labels from dataset (determines num intent classes).
        """
        super(MultiTaskSentenceTransformer, self).__init__()

        self.seed = seed

        # Load the pre-trained Sentence Transformer
        self.transformer = SentenceTransformer(transformer_model)

        # Get embedding dimension dynamically
        embedding_dim = self.transformer.get_sentence_embedding_dimension()

        # Sentiment Analysis Head (Binary or Multi-Class Classification)
        self.sentiment_classes = sentiment_classes
        self.sentiment_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.sentiment_classes),
            nn.Softmax(dim=1) if self.sentiment_classes > 2 else nn.Sigmoid()
        )

        # Intent Detection Head (Multi-Class Classification)
        self.intent_classes = intent_classes
        self.intent_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.intent_classes),
            nn.Softmax(dim=1) if self.intent_classes > 2 else nn.Sigmoid()  # Softmax for multi-class, Sigmoid for binary
        )

    def forward(self, sentences, task="sentiment"):
        """
        Forward pass for multi-task learning.
        :param sentences: List of sentences for encoding.
        :param task: "sentiment" or "intent".
        """
        if task not in ["sentiment", "intent"]:
            raise ValueError("Invalid task! Choose 'sentiment' or 'intent'.")

        torch.manual_seed(self.seed)


        if isinstance(sentences, torch.Tensor):
            sentences = sentences.tolist()
        elif isinstance(sentences, str):
            sentences = [sentences]
        elif not isinstance(sentences, list):
            raise ValueError("Input to model must be a string, tensor, or list of strings.")

        # Convert all elements in list to strings
        sentences = [str(s) for s in sentences]

        # Generate embeddings
        embeddings = self.transformer.encode(sentences, convert_to_tensor=True)

        if task == "sentiment":
            logits = self.sentiment_head(embeddings)

        elif task == "intent":
            logits = self.intent_head(embeddings)

        return logits  # Raw logits for classification

# Initialize Model with Dynamic Intent Classes
multi_task_model = MultiTaskSentenceTransformer(model_name, 2, 4)



# ------------------------- MODEL TESTING -------------------------

# Test Sentiment Analysis
sentences = ["I love this product!", "I hate the weather today."] # Postive, Negative
sentiment_preds = multi_task_model(sentences, task="sentiment")

sentiment_preds = torch.argmax(sentiment_preds, dim=1).detach().cpu().numpy()
binary_preds = ["Positive" if pred >= 0.5 else "Negative" for pred in sentiment_preds]

print("\n🔍 Sentiment Predictions:")
for sentence, pred, label in zip(sentences, sentiment_preds, binary_preds):
    print(f"Sentence: '{sentence}' → Prediction: {pred.item():.4f} ({label})")

# Test Intent Detection
intent_sentences = [
    "Can you tell me the time?",  # Question
    "This is the best day of my life!",  # Exclamation
    "Turn off the lights.",  # Command
    "The sky is blue today."  # Statement
]
intent_preds = multi_task_model(intent_sentences, task="intent")
intent_preds = torch.argmax(intent_preds, dim=1).detach().cpu().numpy()
intent_labels = {0: "Question", 1: "Statement", 2: "Command", 3: "Exclamation"}
intent_preds_readable = [intent_labels[i] for i in intent_preds.tolist()]
print("\n🔍 Intent Predictions:")
for sentence, pred, label in zip(intent_sentences, intent_preds, intent_preds_readable):
    print(f"Sentence: '{sentence}' → Predicted Intent: {label} (Class {pred})")



# ------------------------- TASK 4 -------------------------
print("------------------------- TASK 4 -------------------------")

# ------------------------- READ CSV FILES -------------------------

sentiment_df = pd.read_csv("Unique_Fixed_Sentiment_Analysis_Dataset.csv")
intent_df = pd.read_csv("Unique_Intent_Classification_Dataset.csv")


# Initialize Model
device = torch.device("cpu")
multi_task_model.to(device)


# ------------------------- DATA PREPROCESSING -------------------------

def preprocess_data(sentiment_df, intent_df):
    # Encode Sentiment Labels (Binary Classification: 0 = Negative, 1 = Positive)
    sentiment_df["label"] = sentiment_df["label"].map({"negative": 0, "positive": 1})

    # Encode Intent Labels (Multi-Class Classification)
    intent_label_encoder = LabelEncoder()
    intent_df["label"] = intent_label_encoder.fit_transform(intent_df["label"])

    # Convert Sentences to Embeddings
    transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentiment_embeddings = transformer_model.encode(sentiment_df["sentence"].tolist(), convert_to_tensor=True)
    intent_embeddings = transformer_model.encode(intent_df["sentence"].tolist(), convert_to_tensor=True)

    # Convert Labels to Tensors
    sentiment_labels = torch.tensor(sentiment_df["label"].values, dtype=torch.float32)
    intent_labels = torch.tensor(intent_df["label"].values, dtype=torch.long)

    # Train-Test Split
    sent_train_X, sent_val_X, sent_train_y, sent_val_y = train_test_split(
        sentiment_embeddings, sentiment_labels, test_size=0.2, random_state=42
    )

    intent_train_X, intent_val_X, intent_train_y, intent_val_y = train_test_split(
        intent_embeddings, intent_labels, test_size=0.2, random_state=42
    )

    return {
        "train": {"sentiment": (sent_train_X, sent_train_y), "intent": (intent_train_X, intent_train_y)},
        "val": {"sentiment": (sent_val_X, sent_val_y), "intent": (intent_val_X, intent_val_y)}
    }

# Processed Data
data = preprocess_data(sentiment_df, intent_df)


# ------------------------- DATA LOADER -------------------------

class MultiTaskDataset(Dataset):
    def __init__(self, sentiment_data, intent_data):
        self.sentiment_data = sentiment_data
        self.intent_data = intent_data

    def __len__(self):
        return max(len(self.sentiment_data[0]), len(self.intent_data[0]))

    def __getitem__(self, idx):
        sentiment_idx = idx % len(self.sentiment_data[0])  # Cycle through sentiment dataset
        intent_idx = idx % len(self.intent_data[0])  # Cycle through intent dataset

        sentiment_sample = (self.sentiment_data[0][sentiment_idx], self.sentiment_data[1][sentiment_idx])
        intent_sample = (self.intent_data[0][intent_idx], self.intent_data[1][intent_idx])

        return sentiment_sample, intent_sample

# Initialize Dataloaders
train_dataset = MultiTaskDataset(data["train"]["sentiment"], data["train"]["intent"])
val_dataset = MultiTaskDataset(data["val"]["sentiment"], data["val"]["intent"])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# ------------------------- TRAINING LOOP -------------------------

# Define Loss Functions & Optimizer
criterion_sentiment = nn.BCELoss()  # Binary Cross-Entropy Loss for sentiment
criterion_intent = nn.CrossEntropyLoss()  # Cross-Entropy Loss for intent
optimizer = optim.AdamW(multi_task_model.parameters(), lr=5e-5)

num_epochs = 10

for epoch in range(num_epochs):
    multi_task_model.train()
    total_loss = 0

    for (sentiment_batch, intent_batch) in train_loader:
        sentiment_X, sentiment_y = sentiment_batch
        intent_X, intent_y = intent_batch

        sentiment_X, sentiment_y = sentiment_X.to(device), sentiment_y.to(device)
        intent_X, intent_y = intent_X.to(device), intent_y.to(device)

        optimizer.zero_grad()

        # Forward Pass for Sentiment Analysis
        sentiment_preds = multi_task_model(sentiment_X, task="sentiment")

        # Ensure `sentiment_preds` and `sentiment_y` have the same shape
        if sentiment_preds.dim() == 0:
            sentiment_preds = sentiment_preds.unsqueeze(0)
        if sentiment_y.dim() == 0:
            sentiment_y = sentiment_y.unsqueeze(0)

        # Compute Loss
        sentiment_preds = torch.argmax(sentiment_preds, dim=1).to(sentiment_y.dtype)
        loss_sentiment = criterion_sentiment(sentiment_preds, sentiment_y)

        # Forward Pass for Intent Classification
        intent_preds = multi_task_model(intent_X, task="intent")
        loss_intent = criterion_intent(intent_preds, intent_y)

        # Combine Losses and Backpropagate
        loss = loss_sentiment + loss_intent
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training Finished!")


# ------------------------- MODEL TESTING -------------------------

multi_task_model.eval()  # Set model to evaluation mode

# Test Sentiment Analysis
test_sentences = ["I love this product!", "I hate the weather today."] # Positive, Negative
sentiment_preds = multi_task_model(test_sentences, task="sentiment")
sentiment_preds = torch.argmax(sentiment_preds, dim=1).detach().cpu().numpy()
binary_preds = ["Positive" if pred >= 0.5 else "Negative" for pred in sentiment_preds]

print("\n🔍 Sentiment Predictions:")
for sentence, pred, label in zip(test_sentences, sentiment_preds, binary_preds):
    print(f"Sentence: '{sentence}' → Prediction: {pred:.4f} ({label})")

# Test Intent Detection
intent_test_sentences = [
    "Can you tell me the time?",  # Question
    "This is the best day of my life!",  # Exclamation
    "Turn off the lights.",  # Command
    "The sky is blue today."  # Statement
]

intent_preds = multi_task_model(intent_test_sentences, task="intent")
intent_preds = torch.argmax(intent_preds, dim=1).detach().cpu().numpy()

intent_labels = {0: "Question", 1: "Statement", 2: "Command", 3: "Exclamation"}
intent_preds_readable = [intent_labels[i] for i in intent_preds]

print("\n🔍 Intent Predictions:")
for sentence, pred, label in zip(intent_test_sentences, intent_preds, intent_preds_readable):
    print(f"Sentence: '{sentence}' → Predicted Intent: {label} (Class {pred})")

