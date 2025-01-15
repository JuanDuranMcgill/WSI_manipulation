import os
import torch
import torch.nn as nn
from tqdm import tqdm 

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        Q = self.query(x)  # (n, hidden_dim)
        K = self.key(x)  # (n, hidden_dim)
        V = self.value(x)  # (n, hidden_dim)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # (n, n)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (n, n)
        
        attn_output = torch.matmul(attn_weights, V)  # (n, hidden_dim)
        output = self.fc(attn_output.sum(dim=0))  # (hidden_dim) -> (input_dim)
        
        return output.unsqueeze(0)  # (1, input_dim)
    
class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x is of shape (seq_len, input_dim)
        score = self.V(torch.tanh(self.W1(x) + self.W2(x)))  # (seq_len, 1)
        attention_weights = torch.softmax(score, dim=0)  # (seq_len, 1)
        context_vector = torch.sum(attention_weights * x, dim=0)  # (input_dim)
        
        return context_vector.unsqueeze(0)  # (1, input_dim)

class SimplifiedAdditiveAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimplifiedAdditiveAttention, self).__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)  # Only one linear transformation
        self.V = nn.Linear(hidden_dim, 1)  # Scoring layer remains the same

    def forward(self, x):
        # x is of shape (seq_len, input_dim)
        score = self.V(torch.tanh(self.W1(x)))  # (seq_len, 1)
        attention_weights = torch.softmax(score, dim=0)  # (seq_len, 1)
        context_vector = torch.sum(attention_weights * x, dim=0)  # (input_dim)
        
        return context_vector.unsqueeze(0)  # (1, input_dim)
    
def process_embeddings(input_folder, output_folder, attention_model, input_dim=192, hidden_dim=256):
    print(output_folder)
    if not os.path.exists(output_folder):
        print("folder created")
        os.makedirs(output_folder)
    else:
        print("the path exists already apparently??")

    for file_name in tqdm(os.listdir(input_folder)):
        if file_name.endswith('.pt'):
            file_path = os.path.join(input_folder, file_name)
            
            # Wrap torch.load in a try/except:
            try:
                embeddings = torch.load(file_path, map_location=torch.device('cpu'))
            except Exception as e:
                print(f"Skipping {file_name} due to load error: {e}")
                continue  # Skip this file

            with torch.no_grad():
                flattened_embedding = attention_model(embeddings)

            output_file_name = file_name.replace('.pt', '_flatten.pt')
            output_file_path = os.path.join(output_folder, output_file_name)

            torch.save(flattened_embedding, output_file_path)

if __name__ == '__main__':
    input_folder = 'TCGA-UCEC-embeddings'
    output_folder = 'TCGA-UCEC-embeddings-flatten'

    #attention_model = SimplifiedAdditiveAttention(input_dim=192, hidden_dim=256)
    
    # Load the saved attention model's parameters
    #attention_model.load_state_dict(torch.load("best_attention.pth"))
    attention_model =  AdditiveAttention(input_dim=192, hidden_dim=256)
    attention_model.eval()  # Set the model to evaluation mode
    process_embeddings(input_folder, output_folder,attention_model)
