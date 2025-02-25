from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from seqeval.metrics import classification_report
import torch
import pandas as pd

# 1. Load Pre-trained BERT and Tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Load CSV
csv_path = "test-dataset.csv"  # Replace with your CSV path
df = pd.read_csv(csv_path)

# Train-Test Split
from sklearn.model_selection import train_test_split

# Train-Test Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Function to label words
# Function to label words with print statements
def label_words(description, manual):
    manual_words = set(str(manual).split())
    print("Manual Words:", manual_words)
    
    labeled_output = []
    for word in str(description).split():
        label = 1 if word in manual_words else 0
        print(f"Word: '{word}' -> Label: {label}")
        labeled_output.append(label)
    
    return labeled_output

# Apply labeling
train_df['labels'] = train_df.apply(lambda row: label_words(row['description'], row['manual']), axis=1)
print(train_df)
test_df['labels'] = test_df.apply(lambda row: label_words(row['description'], row['manual']), axis=1)

# Prepare Dataset
train_texts = [str(desc).split() for desc in train_df['description']]
train_labels = train_df['labels'].tolist()

test_texts = [str(desc).split() for desc in test_df['description']]
test_labels = test_df['labels'].tolist()

# Create Huggingface Dataset
train_dataset = Dataset.from_dict({"tokens": train_texts, "labels": train_labels})
test_dataset = Dataset.from_dict({"tokens": test_texts, "labels": test_labels})

# 2. Preprocess and Tokenize
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding='max_length')

    aligned_labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        labels_aligned = []
        for word_id in word_ids:
            if word_id is None:
                labels_aligned.append(-100)
            else:
                labels_aligned.append(label[word_id])
        aligned_labels.append(labels_aligned)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

# Apply Tokenization
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

# 3. Define Label Mapping
label_list = [0, 1]  # 0 or 1
id_to_label = {i: str(i) for i in label_list}
num_labels = len(label_list)

# 4. Load BERT Model
model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
model = model.to(device)

# 5. Define Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    true_labels = [[str(l) for l in label if l != -100] for label in labels]
    true_preds = [[str(p) for (p, l) in zip(prediction, label) if l != -100] 
                  for prediction, label in zip(preds, labels)]

    results = classification_report(true_labels, true_preds, output_dict=True)
    return {"f1": results["weighted avg"]["f1-score"]}

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics
)

# 8. Train Model
trainer.train()

# 9. Inference on test_df
model.eval()
y_true, y_pred = [], []

# for i, row in test_df.iterrows():
#    sentence = str(row['description']).split()
#    inputs = tokenizer(sentence, is_split_into_words=True, return_tensors="pt", padding='max_length', truncation=True).to(device)
#    outputs = model(**inputs)
#    predictions = outputs.logits.argmax(-1).squeeze().cpu().tolist()
#
#    # Filter out special tokens
#    word_ids = inputs.word_ids(batch_index=0)
#    predicted_labels = [pred for pred, word_id in zip(predictions, word_ids) if word_id is not None]
#
#    # Actual labels
#    actual_labels = row['labels']
#
#    y_true.append([str(label) for label in actual_labels])
#    y_pred.append([str(label) for label in predicted_labels[:len(actual_labels)]])
#
#    # Display Manual and Prediction
##    print(f"\nDescription: {row['description']}")
#    print(f"Manual: {row['manual']}")
#    print(f"Predicted Labels: {predicted_labels[:len(actual_labels)]}")


# Inference on test_df
for i, row in test_df.iterrows():
    sentence = str(row['description']).split()
    inputs = tokenizer(sentence, is_split_into_words=True, return_tensors="pt", padding='max_length', truncation=True).to(device)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1).squeeze().cpu().tolist()

    # Filter out special tokens
    word_ids = inputs.word_ids(batch_index=0)
    predicted_labels = [pred for pred, word_id in zip(predictions, word_ids) if word_id is not None]

    # Actual labels
    actual_labels = row['labels']

    # Display Manual and Prediction
    print(f"\nDescription: {row['description']}")
    print(f"Manual: {row['manual']}")
    print(f"Predicted Labels: {predicted_labels[:len(actual_labels)]}")

    # Print Predicted Tokens with Label 1
    predicted_tokens_with_1 = [token for token, label in zip(sentence, predicted_labels[:len(actual_labels)]) if label == 1]
    print(f"Predicted Tokens with 1: {predicted_tokens_with_1}")



# # 10. Evaluation
# print(classification_report(y_true, y_pred))
