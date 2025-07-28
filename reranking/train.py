from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Custom dataset (replace with MS MARCO pairs)
train_encodings = tokenizer(["query1", "query2"], ["doc1", "doc2"], truncation=True, padding=True)
train_labels = torch.tensor([1, 0])  # 1=relevant, 0=irrelevant

class SearchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=SearchDataset(train_encodings, train_labels),
)
trainer.train()
