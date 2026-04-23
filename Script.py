import torch 
import torch.nn as nn 
from tqdm import tqdm 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    # tensor conversion and normalization with mean and std for CIFAR-10 dataset
    transforms.ToTensor(),
    # precomputed mean and std for CIFAR-10 dataset which helps in faster convergence during training
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train =False, download=True, transform=transform)
# type(train_set)

classes = train_set.classes
class_index = {clas:idx for idx, clas in enumerate(classes)}
# print(classes)
# print(class_index)
# print(train_set.data.shape)

'''
there are tasks to be performed:

dataset formation (shape matching with the model, normalization, etc.)
class for prunable linear layer 
class for the architecture of the model
total loss function
training loop (defining the sparsity component of the loss function)
evaluation
plotting

'''

#Dataset 

batch_size=32
train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False)

#class for prunable layer

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.Standard = nn.Linear(in_features, out_features)
        self.Gated = nn.Parameter(torch.full((out_features, in_features), -0.5)) 
        
    def forward(self, input):
        #element-wise product
        gated_weight = self.Standard.weight * torch.sigmoid(self.Gated) 
        
         #custom linear layer
        return F.linear(input, gated_weight, self.Standard.bias)
        

#class for architecture of the model

class SelfPruningNN(nn.Module):
    def __init__(self, input_size, num_layer, hidden_dim, num_classes):
        super(SelfPruningNN, self).__init__()
        
        net_layers = [PrunableLinear(input_size, hidden_dim[0])]
        net_layers.append(nn.GELU())
        for i in range(1,num_layer):
            net_layers.append(PrunableLinear(hidden_dim[i-1], hidden_dim[i]))
            net_layers.append(nn.GELU())
        
        net_layers.append(PrunableLinear(hidden_dim[-1], num_classes))  
        self.net = nn.Sequential(*net_layers)
        
        
    def forward(self, input):
        return self.net(input)
    

#early stop class  
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-2):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                

#hyperparameters
_, H, W, C = train_set.data.shape #size of the input images (32, 32, 3) for CIFAR-10 dataset
input_size = H * W * C
hidden_dim = [128, 64, 32]
num_layers = len(hidden_dim)
num_classes = len(classes)
epochs = 10
lam = [2e-5, 1e-4, 2e-4, 5e-3]

#loss function and batch accuracy

def loss_fn(y_pred, y_true, gates, lam=1e-5):
    loss = F.cross_entropy(y_pred, y_true)

    gated_scores = 0
    total_count = 0
    for gate in gates:
        #sum of all gate values after sigmoid
        gated_scores += torch.sum(torch.sigmoid(gate))
        total_count += gate.numel()

    return loss + (lam * (gated_scores/total_count))

def batch_acc(y_pred, y_true):
    preds = torch.argmax(y_pred, dim=1) 
    #count of correct predictions in the batch
    correct = (preds == y_true).sum().item()

    # accuracy avg for the batch
    return correct / y_true.size(0) 


#training loop function

def training(lam):

    model = SelfPruningNN(input_size, num_layers, hidden_dim, num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopper = EarlyStopping(patience=3, min_delta=1e-2)
    model.to(device)
    gates = []
    
    for i  in range(epochs):
        loss=0
        accuracy=0
        model.train()
        gates = [param for name, param in model.named_parameters() if 'Gated' in name]
        loop = tqdm(train_loader, desc=f"Epoch {i+1}/{epochs}", leave=False)
        for (x_batch,y_batch) in loop:
            opt.zero_grad()
            x_batch,y_batch=x_batch.to(device),y_batch.to(device)
            x_batch = x_batch.view(x_batch.size(0),-1)
            y_pred=model(x_batch)

            #loss and accuracy calculation
            batch_loss=loss_fn(y_pred,y_batch, gates, lam)
            batch_accuracy = batch_acc(y_pred, y_batch)

            #init backpropagation and optimization step
            batch_loss.backward()
            opt.step()
            loss+=batch_loss.item()
            accuracy += batch_accuracy
            loop.set_postfix(loss=batch_loss.item())

        early_stopper(batch_loss.item())
        if early_stopper.should_stop:
                print(f"Early stopping at epoch {i}")
                break

#         print(f'Epoch: {i+1}, Training_Accuracy: {accuracy/len(train_loader)}, Training_loss: {loss/len(train_loader)}')
    return model, gates


#evaluation function

def evaluate(model):
    with torch.no_grad():
        model.eval()
        loop = tqdm(test_loader, leave=False)
        loss = 0
        accuracy=0
        for (x_batch, y_batch) in loop:
            x_batch,y_batch=x_batch.to(device),y_batch.to(device)
            x_batch = x_batch.view(x_batch.size(0),-1)
            y_pred=model(x_batch)

            #no sparsity loss during evaluation, only cross-entropy
            batch_loss=F.cross_entropy(y_pred, y_batch)
            loss+=batch_loss.item()
            acc=batch_acc(y_pred, y_batch)
            accuracy += acc

        print(f'Testing_Accuracy: {accuracy/len(test_loader)}, Testing_loss: {loss/len(test_loader)}')
        test_accuracy = accuracy/len(test_loader)
        
    return test_accuracy


#sparsity function
def compute_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0
    
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            g = torch.sigmoid(module.Gated)

            total += g.numel()
            #prune the gate value if less than the threshold
            pruned += (g < threshold).sum().item() 
    
    return pruned / total


def combined_pipeline(lam_list:list):
    result=[]
    
    for lam in lam_list:
        model, _ = training(lam)
        torch.save(model, f'./models/{lam}_model.pth')
        test_acc = evaluate(model)
        sparsity = compute_sparsity(model)
        result.append((lam, test_acc, sparsity))
            
    df = pd.DataFrame(result, columns=['Lambda', 'Test Accuracy', 'Sparsity'])
    df['Sparsity'] = df['Sparsity']*100
    return df

# 
def plot_and_stats(path):
    model = torch.load(path, map_location="cpu", weights_only=False)

    model.eval()
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            g = torch.sigmoid(module.Gated) 
            #detach the gate values from the computation graph and flatten them into a 1D tensor, then append to the list
            all_gates.append(g.detach().view(-1)) 

    all_gates = torch.cat(all_gates)

    plt.figure(figsize=(6,4))
    # histogram of the gate values to visualize their distribution, using 100 bins for better granularity
    plt.hist(all_gates.numpy(), bins=100)
    plt.title("Distribution of Gate Values")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.savefig("gate_distribution.png")
    plt.show()


def main():
    final_table = combined_pipeline(lam)
    print("final table:\n",final_table)

    # lam = 0.0002 gives the best test accuracy, while pruning around 75% of the weights, 
    # so we will analyze that model for gate distribution and sparsity patterns
    model_path = f'./models/{lam[-2]}_model.pth'  
    plot_and_stats(model_path)

if __name__ == "__main__":
    main()  