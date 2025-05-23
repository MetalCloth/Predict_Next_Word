import torch
import streamlit as st
import torch.nn as nn
import pickle
import time
import nltk
from nltk.tokenize import word_tokenize
# Tokenization
nltk.download('punkt')
nltk.download('punkt_tab')

# === Load vocab ===
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# === Define model ===
class LSTM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), 100)
        self.lstm = nn.LSTM(100, 200, batch_first=True)
        self.fc = nn.Linear(200, len(vocab))

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden.squeeze(0))
        return output

# === Load model ===
model = LSTM(vocab)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

max_len=62

### Convert text to index 
def text_to_index(text,vocab):
  idx=[]
  for tokens in text:
    if tokens not in vocab:
      idx.append(vocab['<unk>'])
    else:
      idx.append(vocab[tokens])
  return idx

### Adds pre padding
def pre_pad_sequence(seq, max_length, pad_value=0):
    return [pad_value] * (max_length - len(seq)) + seq


### Predicts the model
def prediction(model,vocab,text):

  ### tokenize
  tokenized_text = word_tokenize(text.lower())

  ### text -> numerical indices
  numerical_text = text_to_index(tokenized_text, vocab)



  ### padding
  padded_batch=pre_pad_sequence(numerical_text, max_len)
  padded_text = torch.tensor(padded_batch)

  ### Add a batch dimension at the beginning
  padded_text=padded_text.unsqueeze(0)
  ### send to model
  output = model(padded_text)

  ### predicted index
  value, index = torch.max(output, dim=1)

  ### merge with text
  return text + " " + list(vocab.keys())[index]

# === Initializing Streamlit ===

st.title("Predicting the next word")

user_input = st.text_input("Enter your text:")
if user_input:

    predicted_word=prediction(model, vocab, user_input)  

    st.markdown(
        f'<p style="color:gray;font-style:italic;font-size:16px;">Suggested next word: <b>{predicted_word}</b></p>',
        unsafe_allow_html=True
    )


