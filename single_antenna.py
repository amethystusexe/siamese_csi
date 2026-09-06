
# IMPORT 
import scipy.io
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import gc
import copy
import umap
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import random_split
import torch.nn.functional as F  # Per ReLU
import pickle
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report, adjusted_rand_score, silhouette_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_curve, auc
from matplotlib.lines import Line2D
from itertools import product
import warnings; warnings.filterwarnings('ignore')


#GPU SETUP
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
def setup_device(min_free_gb=2.0, max_gpus=2):
    #selects device and enables multi-GPU (DataParallel) if available
    if not torch.cuda.is_available():
        print("No CUDA --> use CPU")
        return torch.device("cpu"), [], False

    torch.cuda.empty_cache()
    gc.collect()

    n_gpus = torch.cuda.device_count()
    print(f"GPU disponibili: {n_gpus}")

    gpu_free = []
    for i in range(n_gpus):
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        print(f"  GPU{i}: {free_gb:.1f}/{total_gb:.1f} GB liberi")
        if free_gb >= min_free_gb:
            gpu_free.append((i, free_gb))

    if not gpu_free:
        print("Nessuna GPU con memoria libera sufficiente --> CPU")
        return torch.device("cpu"), [], False
    gpu_free.sort(key=lambda x: x[1], reverse=True)
    selected_gpus = [i for i, _ in gpu_free[:max_gpus]]

    device = torch.device(f"cuda:{selected_gpus[0]}")
    use_data_parallel = len(selected_gpus) > 1

    if use_data_parallel:
        print(f"Multi-GPU attivo con DataParallel: {selected_gpus}")
    else:
        print(f"GPU selezionata: {selected_gpus[0]}")

    return device, selected_gpus, use_data_parallel
device, available_gpus, use_data_parallel = setup_device()
print(f"Device finale: {device}")
print(f"use_data_parallel: {use_data_parallel}")


#activity division

activity_map = {'A': 'Walk', 'B': 'Run', 'C': 'Jump', 'D': 'Sitting', 'E': 'Empty', 'F': 'Standing', 'G': 'Wave hands', 'H': 'Clapping', 'I': 'Lay down', 'J': 'Wiping', 'K': 'Squat', 'L': 'Stretching'}
#here you can change the activities you want to analyze, no need to put them in order as they are above
acts = ['Walk', 'Jump', 'Empty']  

iniziali = {'Walk': 'W','Run': 'R', 'Empty': 'E', 'Jump': 'J','Sitting': 'Si','Standing': 'Sta','Squat': 'Sq', 'Lay down': 'L','Wiping': 'Wi','Clapping': 'C','Wave hands': 'Wh', 'Stretching': 'Str'}
lettere_iniziali = []
for act in acts:
    if act not in iniziali:
        raise ValueError(f"Activity '{act}' not recognized. Check the 'iniziali' list.")
    lettere_iniziali.append(iniziali[act]) 
combinazione = ''.join(lettere_iniziali)
print(f"chosen activities: {combinazione}")

window_size = 450


ant = 0  #change ant in [0, 1, 2, 3] to select the antenna to train on
normal_dir = f"./single_antenna_baseline/{ant}"
os.makedirs(normal_dir, exist_ok=True)
print(f"Directory created: {normal_dir}")
emb_dir = f"{normal_dir}/embs"
os.makedirs(emb_dir, exist_ok=True)
plots_dir = f"{normal_dir}/plots"
os.makedirs(plots_dir, exist_ok=True)
model_dir = f"{normal_dir}/models"
os.makedirs(model_dir, exist_ok=True)

activity_to_letter = {v: k for k, v in activity_map.items()}
base_dir = "./dataset/S1a"
activity_paths = []
for act in acts:
    if act not in activity_to_letter:
        raise ValueError(f"Activity '{act}' not found in activity_map.")
    letter = activity_to_letter[act]
    path = os.path.join(base_dir, f"S1a_{letter}.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    activity_paths.append(path)

print("activity paths:")
for p in activity_paths:
    print(p)


#CLASSES, FUNCTIONS
#to avoid mixed windows, I generate a starting index for each activity that does not exceed 11,550 (450 + 11,550 = 12,000)
class SingleAntennaWindowDataset(Dataset):
    def __init__(self, activity_paths, antenna_idx, window_size=450):
        self.samples_per_act = 12000  #number of windows per activity
        self.window_size = window_size 
        self.last_win_idx = 11550     #to avoid mixed windows
        self.n_win_per_activity = self.last_win_idx + 1 
        self.n_acts = len(activity_paths)
        self.antenna_idx = antenna_idx

        #data loading
        self.data = []
        for path in activity_paths:
            csi = np.abs(sio.loadmat(path)['csi'])[:, :, antenna_idx].astype(np.float32)
            self.data.append(csi)

        #temp_idx: SKIP activity boundaries to avoid mixed windows; contains only valid start indices
        self.temp_idx = []
        for act_idx in range(self.n_acts):
            for start in range(self.n_win_per_activity):
                self.temp_idx.append((act_idx, start))
        self.temp_idx = np.array(self.temp_idx, dtype=np.int64)

        print(f"Dataset OK: {len(self.temp_idx):,} windows | {self.n_acts}×{self.n_win_per_activity:,} | NO mixed windows")

    def __len__(self):
        return len(self.temp_idx)

    def __getitem__(self, idx):
        act_idx, start = self.temp_idx[idx]
        window = self.data[act_idx][start:start + self.window_size, :]
        window = torch.from_numpy(window).float()
        window = (window - window.mean()) / (window.std() + 1e-6)
        return window.unsqueeze(0), torch.tensor(act_idx, dtype=torch.long)


    def split_dataset(self, train_ratio=0.7, gap=None):
        gap = self.window_size if gap is None else gap
        n_win = self.n_win_per_activity

        train_end = int(n_win * train_ratio)
        test_start = min(train_end + gap, n_win)

        train_starts = np.arange(train_end)
        test_starts = np.arange(test_start, n_win)
        train_mask = np.isin(self.temp_idx[:, 1], train_starts)
        test_mask = np.isin(self.temp_idx[:, 1], test_starts)

        splits = {'train': np.where(train_mask)[0], 'test': np.where(test_mask)[0]}
        
        print(f"Split OK | Train: 0-{train_end-1}, Test: {test_start}-{n_win-1} | Gap: {test_start-train_end}")
        print(f"Per act | Train: {train_end}, Test: {len(test_starts)} | Totale train: {len(splits['train'])}, test: {len(splits['test'])}")
        return splits
   

class CSIPooling(nn.Module):
    def forward(self, x):
        B, C, T, F = x.shape  
        x = x.squeeze(1)         # [B,450,2048]                  removes the singleton channel (the dataset adds it using unsqueeze)

        # Freq: 2048 --> 128
        #tride=16 --> non-overlapping, covers exactly 2048
        #mean(-1): average per chunk --> compresses 2048 --> 128 frequencies
        x = x.unfold(2, 16, 16).mean(dim=-1) # [B,450,128] 
        #time: 450 --> 30
        x = x.unfold(1, 15, 15).mean(dim=-1)[:, :30, :]   # [B,30,128] divides 450 samples into 30 chunks of 15 consecutive samples each 
        return x.reshape(B, -1) 


class CSIEmbeddingNet(nn.Module):
    def __init__(self, emb_dim=8, n_acts=3):
        super().__init__()
        self.pool = CSIPooling()
        self.fc1 = nn.Linear(3840, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_emb = nn.Linear(32, emb_dim)
        self.fc_class_hidden = nn.Linear(emb_dim, 12)
        self.fc_class = nn.Linear(12, n_acts)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        h = self.fc_emb(x)
        h = F.normalize(h, p=2, dim=1)
        c = F.relu(self.fc_class_hidden(h))
        c = self.fc_class(c)
        return h, c




#parameters
batch_size = 128
epochs = 10
cls_ora = 0.7
margine = 0.05
emb_dim = 8
prova = f"{combinazione}_{cls_ora}_{margine}_{emb_dim}_bs{batch_size}"
print(f"test: {prova} | with these parameters -> cls_weight: {cls_ora}, margin: {margine}")

# TRAINING SETUP
# dataset, split 
dataset = SingleAntennaWindowDataset(activity_paths, ant, window_size)    
print(f"dataset: {len(dataset):,} windows")
splits = dataset.split_dataset(train_ratio=0.7, gap=window_size)

train_ds = torch.utils.data.Subset(dataset, splits['train'])
test_ds = torch.utils.data.Subset(dataset, splits['test'])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

print(f"Ant{ant} | Train: {len(train_ds):,}, Test: {len(test_ds):,}")

#model, loss and optimization
model = CSIEmbeddingNet(emb_dim=emb_dim, n_acts=len(acts)).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

#training
loss_history = {'train': []}
acc_history = {'train': []}

best_train_loss = np.inf
best_model_path = f'{model_dir}/single_branch_ant{ant}_{prova}.pth'

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for windows, labels in train_loader:
        windows = windows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        emb, logits = model(windows)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        batch_size_curr = windows.size(0)
        train_loss += loss.item() * batch_size_curr
        train_total += batch_size_curr
        train_correct += (logits.argmax(dim=1) == labels).sum().item()

    train_loss /= train_total
    train_acc = train_correct / train_total

    loss_history['train'].append(train_loss)
    acc_history['train'].append(train_acc)

    if train_loss < best_train_loss:
        best_train_loss = train_loss
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'acc_history': acc_history,
            'ant': ant,
            'acts': acts,
            'window_size': window_size,
            'prova': prova,
            'best_train_loss': best_train_loss
        }
        torch.save(checkpoint, best_model_path)
        torch.save(model.state_dict(), f'{model_dir}/embedding_net_ant{ant}_{prova}.pth')

    print(
        f"Ep {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
    )

print("Training completed and best model saved.")
  

#visualization of training curves
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(loss_history['train'], marker='o', linewidth=2, color='steelblue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Train Loss Ant{ant}") 
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(acc_history['train'], marker='o', linewidth=2, color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Train Accuracy Ant{ant}")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plots_dir}/train_curves_ant{ant}_{prova}.png', dpi=150, bbox_inches='tight')
plt.show()

#save training history to CSV
with open(f'{model_dir}/train_history_ant{ant}_{prova}.csv', 'w') as f:
    f.write("epoch,train_loss,train_acc\n")
    for i in range(epochs):
        f.write(
            f"{i+1},"
            f"{loss_history['train'][i]:.6f},"
            f"{acc_history['train'][i]:.6f}\n"
        )

#test
def evaluate_test_single_branch(test_loader, model, criterion, device, train_ant, test_ant, save_dir, prova, acts):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for windows, labels in test_loader:
            windows = windows.to(device)
            labels = labels.to(device)

            emb, logits = model(windows)
            loss = criterion(logits, labels)

            test_loss += loss.item() * windows.size(0)
            total += windows.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    test_loss_mean = test_loss / total
    test_acc = correct / total

    all_logits = np.vstack(all_logits)
    all_labels_true = np.hstack(all_labels)
    all_preds = all_logits.argmax(axis=1)

    cm = confusion_matrix(all_labels_true, all_preds, labels=range(len(acts)))

    print(f"\n=== CROSS TEST TrainAnt{train_ant} -> TestAnt{test_ant} ===")
    print(f"Loss: {test_loss_mean:.4f}")
    print(f"Accuracy: {test_acc*100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels_true, all_preds, target_names=acts, digits=4))

    report_path = f"{save_dir}/cross_report_train{train_ant}_test{test_ant}_{prova}.txt"
    with open(report_path, "w") as f:
        f.write(f"CROSS TEST TrainAnt{train_ant} -> TestAnt{test_ant}\n")
        f.write(f"Prova: {prova}\n")
        f.write(f"Loss: {test_loss_mean:.4f}\n")
        f.write(f"Accuracy: {test_acc*100:.1f}%\n\n")
        f.write(classification_report(all_labels_true, all_preds, target_names=acts, digits=4))

    return test_loss_mean, test_acc, cm


#model loading
best_model_path = f'{model_dir}/single_branch_ant{ant}_{prova}.pth'
criterion = nn.CrossEntropyLoss().to(device)
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

train_ant = ant
test_ants = [0, 1, 2, 3]

all_cms = []
all_accs = []

for test_ant in test_ants:
    test_dataset = SingleAntennaWindowDataset(activity_paths, test_ant, window_size)
    splits_test = test_dataset.split_dataset(train_ratio=0.7, gap=window_size)

    test_ds = torch.utils.data.Subset(test_dataset, splits_test['test'])
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    test_loss, test_acc, cm = evaluate_test_single_branch(
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        device=device,
        train_ant=train_ant,
        test_ant=test_ant,
        save_dir=plots_dir,
        prova=prova,
        acts=acts
    )

    all_cms.append(cm)
    all_accs.append(test_acc)

#visualization of confusion matrices for all test antennas

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for i, test_ant in enumerate(test_ants):
    disp = ConfusionMatrixDisplay(confusion_matrix=all_cms[i], display_labels=acts)
    ax = axs.flat[i]
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(f"Test Ant{test_ant}\nAcc: {all_accs[i]*100:.1f}%")
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")

plt.suptitle(f"CROSS TEST - Train Ant{train_ant}", fontsize=16, fontweight='bold')
plt.tight_layout()

plt.savefig(
    f"{plots_dir}/cross_test_train{train_ant}_all_test_ants_{prova}.png",
    dpi=150,
    bbox_inches='tight')
plt.show()


#load of csv file to plot confusion matrix with different colors
df = pd.read_csv(f"{plots_dir}/predictions_train{train_ant}_test{test_ant}_{prova}.csv")
cm = confusion_matrix(df["y_true"], df["y_pred"], labels=range(len(acts)))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="magma", xticklabels=acts, yticklabels=acts)
plt.xlabel("Pred")
plt.ylabel("True")
plt.title(f"Train Ant{train_ant} -> Test Ant{test_ant}")
plt.show()


#EMBEDDINGS 
#embeddings extraction function
def extract_embeddings_single(model, loader, device):
    model.eval()
    all_emb = []
    all_labels = []

    with torch.no_grad():
        for windows, labels in loader:
            windows = windows.to(device)
            emb, _ = model(windows)
            all_emb.append(emb.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_emb), np.hstack(all_labels)


#loading of the trained model for antenna x
train_ant = ant
checkpoint = torch.load(
    f'{model_dir}/single_branch_ant{train_ant}_{prova}.pth',
    map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

print(f"model loaded: train on antenna {train_ant}")

#extraction, t-SNE and saving of embeddings for all test antennas
test_ants = [0, 1, 2, 3]
activity_names = ['Walk', 'Jump', 'Empty']

all_embeddings = []
all_labels_list = []
rows = []

os.makedirs(emb_dir, exist_ok=True)

for test_ant in test_ants:
    test_dataset = SingleAntennaWindowDataset(activity_paths, test_ant, window_size)
    splits_test = test_dataset.split_dataset(train_ratio=0.7, gap=window_size)

    test_ds = torch.utils.data.Subset(test_dataset, splits_test['test'])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    print(f"Extracting embeddings for antenna {test_ant}...")
    embeddings, labels = extract_embeddings_single(model, test_loader, device)

    np.save(f"{emb_dir}/embeddings_train{train_ant}_test{test_ant}_{prova}.npy", embeddings)
    np.save(f"{emb_dir}/labels_train{train_ant}_test{test_ant}_{prova}.npy", labels)

    print(f"Ant{test_ant} -> embeddings: {embeddings.shape} | labels: {labels.shape}")

    all_embeddings.append(embeddings)
    all_labels_list.append(labels)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    embs_2d = tsne.fit_transform(embeddings)

    for i in range(len(labels)):
        rows.append({
            'train_ant': train_ant,
            'test_ant': test_ant,
            'label': int(labels[i]),
            'act_name': activity_names[int(labels[i])],
            'x': float(embs_2d[i, 0]),
            'y': float(embs_2d[i, 1]),
        })

csv_path = f"{emb_dir}/tsne_train{train_ant}_all_test_ants_{prova}.csv"
df_tsne = pd.DataFrame(rows)
df_tsne.to_csv(csv_path, index=False)
print(f"CSV saved in: {csv_path}")

#load csv to check the content
csv_path = f"{emb_dir}/tsne_train{train_ant}_all_test_ants_{prova}.csv"
df_tsne = pd.read_csv(csv_path)
print(df_tsne.head())
print(df_tsne.shape)


#visualization of t-SNE embeddings for all test antennas with different color palettes
palettes = {"colors_1": {0: "#edae49", 1: "#d1495b", 2: "#00798c"}}

fig_bg = 'white'
ax_bg = 'white'
grid_color = 'lightgray'
title_color = 'black'

test_ants = sorted(df_tsne['test_ant'].unique())

x_min, x_max = df_tsne['x'].min(), df_tsne['x'].max()
y_min, y_max = df_tsne['y'].min(), df_tsne['y'].max()

x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2
span = max(x_max - x_min, y_max - y_min) * 1.05

x_limits = (x_center - span / 2, x_center + span / 2)
y_limits = (y_center - span / 2, y_center + span / 2)

for palette_name, activity_colors in palettes.items():

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.ravel()

    fig.patch.set_facecolor(fig_bg)
    fig.suptitle("Latent space embeddings, model trained on antenna 0",fontsize=18,y=0.98,color=title_color,fontweight='bold')

    for i, test_ant in enumerate(test_ants[:4]):
        ax = axs[i]
        ax.set_facecolor(ax_bg)

        df_ant = df_tsne[df_tsne['test_ant'] == test_ant]

        for act_i in range(len(activity_names)):
            df_cls = df_ant[df_ant['label'] == act_i]
            ax.scatter(
                df_cls['x'],
                df_cls['y'],
                color=activity_colors.get(act_i, 'gray'),
                alpha=0.70,
                s=10
            )

        ax.set_title(f"Antenna {test_ant}", fontsize=15, fontweight='bold', color=title_color)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.25, color=grid_color)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_aspect('equal', adjustable='box')

    for k in range(len(test_ants), 4):
        axs[k].axis('off')

    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            color='w',
            label=name,
            markerfacecolor=activity_colors[idx],
            markeredgecolor=activity_colors[idx],
            markersize=14
        )
        for idx, name in enumerate(activity_names)
    ]

    fig.legend(handles=legend_elements,loc='lower center',ncol=3,frameon=True, fontsize=16,handletextpad=0.7,columnspacing=1.6)

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.savefig(f"{emb_dir}/tsne_train{train_ant}_all_test_ants_{prova}_{palette_name}.png",dpi=300,bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.show()

print("Plot saved successfully.")



