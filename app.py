import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr

# Load your model
model = BertForSequenceClassification.from_pretrained('distilbert_model')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        return predicted_class, probabilities[0].tolist()

# Set up the Gradio interface
iface = gr.Interface(
    fn=classify_text,
    inputs="text",
    outputs=["label", "json"],
    title="Text Classification",
    description="Enter text to classify"
)

# Launch the app
iface.launch()
