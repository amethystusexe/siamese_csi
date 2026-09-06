#IMPORT 
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
import torch.nn.functional as F
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
# Enable memory fragmentation avoidance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
def setup_device(min_free_gb=2.0, max_gpus=2):
    #selects device and enables multi-GPU (DataParallel) if available
    if not torch.cuda.is_available():
        print("No CUDA -->  CPU")
        return torch.device("cpu"), [], False
    torch.cuda.empty_cache()
    gc.collect()
    n_gpus = torch.cuda.device_count()
    print(f"GPU available: {n_gpus}")

    gpu_free = []
    for i in range(n_gpus):
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        print(f"  GPU{i}: {free_gb:.1f}/{total_gb:.1f} GB liberi")
        if free_gb >= min_free_gb:
            gpu_free.append((i, free_gb))

    if not gpu_free:
        print("No GPU with sufficient free memory --> CPU")
        return torch.device("cpu"), [], False

    # ordina per memoria libera (desc)
    gpu_free.sort(key=lambda x: x[1], reverse=True)
    selected_gpus = [i for i, _ in gpu_free[:max_gpus]]

    device = torch.device(f"cuda:{selected_gpus[0]}")
    use_data_parallel = len(selected_gpus) > 1

    if use_data_parallel:
        print(f"Multi-GPU active with DataParallel: {selected_gpus}")
    else:
        print(f"GPU selected: {selected_gpus[0]}")

    return device, selected_gpus, use_data_parallel
device, available_gpus, use_data_parallel = setup_device()
print(f"Final device: {device}")
print(f"use_data_parallel: {use_data_parallel}")

#ACTIVITY DIVISION
activity_map = {'A': 'Walk', 'B': 'Run', 'C': 'Jump', 'D': 'Sitting', 'E': 'Empty', 'F': 'Standing', 'G': 'Wave hands', 
    'H': 'Clapping', 'I': 'Lay down', 'J': 'Wiping', 'K': 'Squat', 'L': 'Stretching'}

#here you can change the activities you want to analyze, no need to put them in order as they are above
acts = ['Walk','Jump', 'Empty']  

#dynamically create a string of initials for the selected activities, which will be used for naming directories and files
iniziali = {'Walk': 'W','Run': 'R', 'Empty': 'E', 'Jump': 'J','Sitting': 'Si','Standing': 'St','Squat': 'Sq', 
            'Lay_down': 'L','Wiping': 'Wi','Clapping': 'C','Wave_hands': 'Wh', 'Stretching': 'St'}
lettere_iniziali = []
for act in acts:
    if act not in iniziali:
        raise ValueError(f"Activity '{act}' not recognized. Check the 'iniziali' list.")
    lettere_iniziali.append(iniziali[act]) 
combinazione = ''.join(lettere_iniziali)
print(f"Selected activities: {combinazione}")

ant1, ant2 = 0,1  #change these values to select different antenna pairs for analysis
pair = f"{ant1}-{ant2}" 
ant3, ant4 = 2,3  #antennas used for cross-testing, not for training
window_size = 450 

normal_dir = f"./siamese/{ant1}{ant2}"
os.makedirs(normal_dir, exist_ok=True)
print(f"Main directory created: {normal_dir}")
emb_dir = f"{normal_dir}/embs"
os.makedirs(emb_dir, exist_ok=True)
plots_dir = f"{normal_dir}/plots"
os.makedirs(plots_dir, exist_ok=True)
model_dir = f"{normal_dir}/models"
os.makedirs(model_dir, exist_ok=True)
    
#function that converts names to paths without having to set them manually, also finding div_start and block_size
def get_paths_and_blocks(acts): 
    
    #find the corresponding letter for each activity
    letters = {name: letter for letter, name in activity_map.items() if name in acts}
    #order A->L 
    letters_sorted = sorted(letters.values())
    names_sorted = [activity_map[l] for l in letters_sorted] 
    #path of each activity
    paths = [f"./dataset/S1a/S1a_{letter}.mat" for letter in letters_sorted]

    #generate div_blocks
    div_blocks = []
    for i, name0 in enumerate(names_sorted):
        for j, name1 in enumerate(names_sorted):
            if i != j:
                div_blocks.append(f"{name0.lower()}0-{name1.lower()}1")

    #finds div_start 
    block_size = 11551  #fixed value, represents number of pairs per activity
    n_acts = len(names_sorted)
    div_start = n_acts * block_size

    print("attività:", names_sorted)
    print("paths:", paths)
    print("div_blocks:", div_blocks)
    print("div_start:", div_start)
    print("n_blocks_diverse:", len(div_blocks))


    return paths, div_blocks, div_start, block_size

paths, div_blocks, div_start, block_size = get_paths_and_blocks(acts)


#CLASSES, FUNCTIONS 
#function that sets the random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SiameseCSIDataset(Dataset):
    def __init__(self, activity_paths, ant_idx_1 = 0, ant_idx_2 = 1, window_size = 450, balanced=False, seed=42): #balanced and seed for balancing and reproducibility

        self.activity_paths = activity_paths
        self.n_acts = len(activity_paths)   
        self.window_size = window_size
        self.samples_per_act = 12000
        self.last_win_idx = 11550
        self.n_win_per_activity = self.last_win_idx + 1 
        self.balanced = balanced
        self.seed = seed


        #loading vstack
        file_1_acts, file_2_acts = [], []
        for path in activity_paths:
            mat = sio.loadmat(path)
            csi = np.abs(mat['csi'])
            #extracting individually now
            file_1_acts.append(csi[:, :, ant_idx_1])
            file_2_acts.append(csi[:, :, ant_idx_2])

        self.file_1 = np.vstack(file_1_acts).astype(np.float32)  #(n_acts*12000, 2048)
        self.file_2 = np.vstack(file_2_acts).astype(np.float32)
        print(f"file_1 shape: {self.file_1.shape}, file_2 shape: {self.file_2.shape}")
    
        self._build_temp_idx()

    # ----------------------------------------------------
    #function to subsample pairs for faster training or testing, useful for large datasets
    def subsample_pairs(self, sample_rate = 5, seed = 42): 
        np.random.seed(seed)
        mask = np.random.choice(len(self.temp_idx), size=len(self.temp_idx)//sample_rate, replace=False) #random indices
        subset = copy.deepcopy(self)
        subset.temp_idx = self.temp_idx[mask]
        print(f"Subsample {sample_rate}: {len(subset.temp_idx):,} pairs ({100/sample_rate}% orig)")
        return subset 

    # ----------------------------------------------------
    # function to build a list of temporal index pairs (i,j) for the windows, all possible pairs
    def _build_temp_idx(self):

        np.random.seed(self.seed)
        self.offsets = np.cumsum([0] + [self.samples_per_act] * self.n_acts)[:-1]
        n_win = self.n_win_per_activity

        temp_idx = []

        #ALL SIMILARS 
        for act_idx in range(self.n_acts):
            offset = self.offsets[act_idx]
            for k in range(n_win):
                i = k + offset
                j = k + offset
                temp_idx.append((i,j, 1))  #1 is the label for similar pairs

        
        #ALL DIFFERENT
        diversi = []
        #computing subsample for different pairs if balanced, otherwise leave all different pairs
        if self.balanced:
            subsample_per_pair = self.n_win_per_activity // (self.n_acts - 1)                         #11551/(n act -1) and find for every different pair how big it is
            print(f"Subsample per coppia diversa: {subsample_per_pair:,} / {self.n_win_per_activity:,}")
        else:
            subsample_per_pair = self.n_win_per_activity                                             #otherwise leave as is


        for act1 in range(self.n_acts):
            for act2 in range(self.n_acts):
                if act1 == act2:
                    continue  #skip similar pairs, already added
                offset_1 = self.offsets[act1]
                offset_2 = self.offsets[act2]

                #local list for this pair
                pair_k = []
                for k in range(n_win):
                    i = k + offset_1
                    j = k + offset_2
                    pair_k.append((i,j, 0))  #0 is the label for different pairs



                if self.balanced:
                    mask = np.random.choice(len(pair_k), size=subsample_per_pair, replace=False)  #pick indices 
                    pair_k = [pair_k[m] for m in mask]

                diversi.extend(pair_k)   #add different pairs of this activity pair to the global list of different pairs

     

        
        #different subsample, combines positive and negative
        temp_idx.extend(diversi)
        #save as numpy array for faster indexing
        self.temp_idx = np.array(temp_idx, dtype=np.int64)
        n_sim = self.n_acts * n_win #similar: for each activity 11551 similar pairs
        n_div = len(diversi)
        print(f"Pairs: {len(self.temp_idx):,} (sim:{n_sim:,} div:{n_div:,} | balanced:{self.balanced})")

    #---------------------------------------------
    def __len__(self):
        #one element for each pair (i,j)
        return len(self.temp_idx)
    #---------------------------------------------
    def __getitem__(self, idx):
        i, j, label = self.temp_idx[idx]
        
        start_i = int(i)
        end_i = start_i + self.window_size
        start_j = int(j)
        end_j = start_j + self.window_size
        
        w1_raw = self.file_1[start_i:end_i, :]
        w2_raw = self.file_2[start_j:end_j, :]
        
        w1 = torch.from_numpy(w1_raw).float()  # (450, 2048)
        w2 = torch.from_numpy(w2_raw).float()  # (450, 2048)
        
        #Z-SCORE for window: dim=[0,1] per 2D (time x subcarriers)
        w1_mean = w1.mean(dim=[0,1], keepdim=True)  # (1,1)
        w1_std = w1.std(dim=[0,1], keepdim=True)
        w1 = (w1 - w1_mean) / (w1_std + 1e-6)
        
        w2_mean = w2.mean(dim=[0,1], keepdim=True)
        w2_std = w2.std(dim=[0,1], keepdim=True)
        w2 = (w2 - w2_mean) / (w2_std + 1e-6)
        
        label = torch.tensor(label, dtype=torch.float32)
        act_idx1 = np.searchsorted(self.offsets, i, side='right') - 1  # ant1
        act_idx2 = np.searchsorted(self.offsets, j, side='right') - 1  # ant2
        
        return w1.unsqueeze(0), w2.unsqueeze(0), label, torch.tensor(act_idx1, dtype=torch.long), torch.tensor(act_idx2, dtype=torch.long) 


    #---------------------------------------------
    #to split dataset
    def split_dataset(self, train_ratio=0.7, gap=None, return_start_splits=False):
        
        gap = self.window_size if gap is None else int(gap)
        all_starts = np.arange(self.n_win_per_activity)

        train_end = int(len(all_starts) * train_ratio)
        test_start_idx = min(train_end + gap, len(all_starts))  # no overlap

        train_starts = all_starts[:train_end]
        test_starts = all_starts[test_start_idx:]

        pair_starts = self.temp_idx[:, 0] % self.samples_per_act

        splits = {
            'train': np.where(np.isin(pair_starts, train_starts))[0],
            'test': np.where(np.isin(pair_starts, test_starts))[0]
        }

        start_splits = {
            'train': train_starts,
            'test': test_starts
        }

        if return_start_splits:
            return splits, start_splits
        return splits
        
    #creates a subset of the dataset using only certain pair indices
    def create_subset(self, pair_indices): 
        subset = copy.copy(self)
        subset.temp_idx = self.temp_idx[pair_indices].copy()
        return subset


#single dataset for support set 
class SingleAntennaWindowDataset(Dataset):
    def __init__(self, activity_paths, antenna_idx, window_size=450, allowed_starts=None):
        self.samples = []
        self.window_size = window_size
        self.data = []
        for act_idx, path in enumerate(activity_paths):
            csi = np.abs(sio.loadmat(path)['csi'])[:, :, antenna_idx].astype(np.float32)
            self.data.append(csi)
            max_start = csi.shape[0] - window_size
            starts = np.arange(max_start + 1, dtype=np.int64)
            if allowed_starts is not None:
                starts = np.intersect1d(starts, np.asarray(allowed_starts, dtype=np.int64))
            for s in starts:
                self.samples.append((act_idx, int(s)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        act_idx, start = self.samples[idx]
        window = self.data[act_idx][start:start+self.window_size, :]
        window = torch.from_numpy(window).float()
        window = (window - window.mean()) / (window.std() + 1e-6)
        return window.unsqueeze(0), torch.tensor(act_idx, dtype=torch.long)

#----------------------------------------
#RETE EMBEDDING NUOVA CON 500 PARAMETRI
'''Input:  [B, 1, 450, 2048]  (batch x channels x time x frequenies)

1) x.squeeze(1) --> [B, 450, 2048] removes singleton channel (dataset adds it with unsqueeze)

2️) x.unfold(2, 16, 16).mean(-1) --> [B, 450, 128]
   • unfold(2,16,16): divides 2048 subcarriers into 128 chunks of 16 consecutive subcarriers each
   • stride=16 --> non-overlapping, covers exactly 2048
   • mean(-1): average each chunk --> compress 2048->128 freq 

3️) x.unfold(1, 15, 15).mean(-1)[:, :30, :] --> [B, 30, 128]
   • unfold(1,15,15):  divides 450 sample time into 30 chunks of 15 consecutive samples each 
   • 450/15=30 precise
   • [:, :30, :]: to be sure if >30, take only first 30 chunks

4️) reshape(B, -1) -> [B, 3840] (30×128)
   Flatten per FC layer
'''


class CSIPooling(nn.Module):
    def forward(self, x):
        B, C, T, F = x.shape
        x = x.squeeze(1)
        x = x.unfold(2, 16, 16).mean(dim=-1)
        x = x.unfold(1, 15, 15).mean(dim=-1)[:, :30, :] 
        
        return x.reshape(B, -1) 



class CSIEmbeddingNet(nn.Module):
    def __init__(self, emb_dim=8, n_acts=3): 
        super().__init__()
        self.pool = CSIPooling()


        self.fc1 = nn.Linear(3840, 128)  #now unfolds gives exactly 3840 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32) 
        self.fc_emb = nn.Linear(32, emb_dim)

        #classification branch: small hidden layer to introduce non-linearity, between 4 and 16 neurons, we chose 12
        self.fc_class_hidden = nn.Linear(emb_dim, 12) 
        self.fc_class = nn.Linear(12, n_acts)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.pool(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        #final embedding, h contains embeddings
        h = self.fc_emb(x) 

        h = F.normalize(h, p=2, dim=1)

        #classification branch that introduces non-linearity 
        #another small hidden layer to introduce non-linearity, between 4 and 16 neurons, we chose 12
        c = F.relu(self.fc_class_hidden(h))
        c = self.fc_class(c)
        return h, c



# SIAMESE NETWORK
class SiameseNetwork(nn.Module):
    def __init__(self, dataset, emb_dim=8):
        super().__init__()
        n_acts = dataset.n_acts
        self.embedding_net = CSIEmbeddingNet(emb_dim, n_acts) 
    
    def forward(self, x1, x2):
        emb1, logits1 = self.embedding_net(x1)
        emb2, logits2 = self.embedding_net(x2)
        dist = torch.norm(emb1 - emb2, dim=1) 
        return emb1, emb2, dist, logits1, logits2
     
#LOSS 
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.8):   #1 was the standard
        super().__init__()
        self.margin = margin     #margin is the distance at which I stop worrying about separating the (different) negatives: beyond that, the loss for the different ones is 0

    def forward(self, dist, labels):     # labels: 1 = similar, 0 = different
        labels = labels.float()          #float conversion
        y = 1.0 - labels

        # L = (1 - y)*D^2 + y*max(0, margin - D)^2
        loss_similar = (1 - y) * dist.pow(2)     #for similar pairs I minimize the distance
        loss_dissim  = y * torch.clamp(self.margin - dist, min=0.0).pow(2)     #for different pairs I penalize only if the distance is less than the margin

        loss = loss_similar + loss_dissim
        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, margin=0.8, cls_weight=0.5):  #cls_weight LOW VALUE = more weight to contrastive loss, HIGH VALUE = more weight to classification loss
        super().__init__()
        self.contrastive = ContrastiveLoss(margin)
        self.cls_loss = nn.CrossEntropyLoss()
        self.cls_weight = cls_weight
    
    def forward(self, dist, labels_sim, logits1, logits2, act_labels1, act_labels2):
        #contrastive
        loss_cont = self.contrastive(dist, labels_sim)
        
        #classification (stabile)
        loss_cls1 = self.cls_loss(logits1, act_labels1)
        loss_cls2 = self.cls_loss(logits2, act_labels2)
        loss_cls = 0.5 * (loss_cls1 + loss_cls2)
        
        #combined loss: weighted sum of contrastive and classification losses
        total_loss = (1 - self.cls_weight) * loss_cont + self.cls_weight * loss_cls
        
        return total_loss



##############
# TRAINING FUNCTION

#changing the seed for reproducibility
seed = 17
set_seed(seed)
data_seed = 42 
#parameters
batch_size = 128
epochs = 10
cls_ora = 0.05 #low cls_weight: classification stays but secondary role 
margine = 0.7
emb_dim = 8
prova = f"{combinazione}_{cls_ora}_{margine}_{emb_dim}_bs{batch_size}_ep{epochs}_{seed}_3"
print(f"Test: {prova} | with following parameters -> cls_weight: {cls_ora}, margin: {margine}")


#dataset, split
dataset = SiameseCSIDataset(paths,ant_idx_1=ant1,ant_idx_2=ant2,window_size=window_size,balanced=True,seed=data_seed)
print(f"dataset bilanciato: {len(dataset)} pairs")
#dataset = dataset.subsample_pairs(sample_rate=1, seed=seed)  #if subsampling is needed
#split based on start index (no overlap)
splits_base, start_splits = dataset.split_dataset(
    train_ratio=0.7,
    gap=window_size,
    return_start_splits=True
)

# train/val split inside initial train
train_starts_full = start_splits['train']
n_train_full = len(train_starts_full)
train_end = int(0.85 * n_train_full)

train_starts = train_starts_full[:train_end]
val_starts   = train_starts_full[train_end:]

pair_starts = dataset.temp_idx[:, 0] % dataset.samples_per_act

splits = {
    'train': np.where(np.isin(pair_starts, train_starts))[0],
    'val':   np.where(np.isin(pair_starts, val_starts))[0],
    'test':  splits_base['test']
}

train_ds = dataset.create_subset(splits['train'])
val_ds   = dataset.create_subset(splits['val'])
test_ds  = dataset.create_subset(splits['test'])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Ant{pair} | Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")
print(f"Similar/Total train: {int((train_ds.temp_idx[:, 2] == 1).sum())}/{len(train_ds)}")
print(f"Similar/Total val:   {int((val_ds.temp_idx[:, 2] == 1).sum())}/{len(val_ds)}")
print(f"Similar/Total test:  {int((test_ds.temp_idx[:, 2] == 1).sum())}/{len(test_ds)}")


#model, loss, optimizer, scheduler
model = SiameseNetwork(dataset).to(device)
criterion = CombinedLoss(margin=margine, cls_weight=cls_ora).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,factor=0.5)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")


#training 
loss_history = {'train': [], 'val': []}
cls_acc_history_ant1 = {'train': [], 'val': []}
cls_acc_history_ant2 = {'train': [], 'val': []}

best_val_loss = np.inf
best_model_path = f'{model_dir}/siamese_model_ant{ant1}-{ant2}_{prova}.pth'

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_cls_correct_ant1 = 0
    train_cls_correct_ant2 = 0
    train_cls_total = 0
    num_train_samples = 0

    for w1, w2, labels_sim, act_idx1, act_idx2 in train_loader:
        w1 = w1.to(device)
        w2 = w2.to(device)
        labels_sim = labels_sim.to(device)
        act_idx1 = act_idx1.to(device)
        act_idx2 = act_idx2.to(device)

        optimizer.zero_grad()

        _, _, dist, logits1, logits2 = model(w1, w2)
        loss = criterion(dist, labels_sim, logits1, logits2, act_idx1, act_idx2)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        batch_size_curr = w1.size(0)
        train_loss += loss.item() * batch_size_curr
        num_train_samples += batch_size_curr

        pred1 = torch.argmax(logits1, dim=1)
        pred2 = torch.argmax(logits2, dim=1)

        train_cls_correct_ant1 += (pred1 == act_idx1).sum().item()
        train_cls_correct_ant2 += (pred2 == act_idx2).sum().item()
        train_cls_total += act_idx1.size(0) + act_idx2.size(0)

    train_loss /= num_train_samples
    train_cls_acc_ant1 = train_cls_correct_ant1 / (train_cls_total / 2)
    train_cls_acc_ant2 = train_cls_correct_ant2 / (train_cls_total / 2)

    #validation
    model.eval()
    val_loss = 0.0
    val_cls_correct_ant1 = 0
    val_cls_correct_ant2 = 0
    val_cls_total = 0
    num_val_samples = 0

    with torch.no_grad():
        for w1, w2, labels_sim, act_idx1, act_idx2 in val_loader:
            w1 = w1.to(device)
            w2 = w2.to(device)
            labels_sim = labels_sim.to(device)
            act_idx1 = act_idx1.to(device)
            act_idx2 = act_idx2.to(device)

            _, _, dist, logits1, logits2 = model(w1, w2)
            loss = criterion(dist, labels_sim, logits1, logits2, act_idx1, act_idx2)

            batch_size_curr = w1.size(0)
            val_loss += loss.item() * batch_size_curr
            num_val_samples += batch_size_curr

            pred1 = torch.argmax(logits1, dim=1)
            pred2 = torch.argmax(logits2, dim=1)

            val_cls_correct_ant1 += (pred1 == act_idx1).sum().item()
            val_cls_correct_ant2 += (pred2 == act_idx2).sum().item()
            val_cls_total += act_idx1.size(0) + act_idx2.size(0)

    val_loss /= num_val_samples
    val_cls_acc_ant1 = val_cls_correct_ant1 / (val_cls_total / 2)
    val_cls_acc_ant2 = val_cls_correct_ant2 / (val_cls_total / 2)

    loss_history['train'].append(train_loss)
    loss_history['val'].append(val_loss)
    cls_acc_history_ant1['train'].append(train_cls_acc_ant1)
    cls_acc_history_ant1['val'].append(val_cls_acc_ant1)
    cls_acc_history_ant2['train'].append(train_cls_acc_ant2)
    cls_acc_history_ant2['val'].append(val_cls_acc_ant2)

    scheduler.step(val_loss)

    # best checkpoint on validation
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'cls_acc_history_ant1': cls_acc_history_ant1,
            'cls_acc_history_ant2': cls_acc_history_ant2,
            'ant1': ant1,
            'ant2': ant2,
            'window_size': window_size,
            'prova': prova,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, best_model_path)
        torch.save(model.embedding_net.state_dict(), f'{model_dir}/embedding_net_ant{ant1}-{ant2}_{prova}_BEST.pth')

    print(
        f"Ep {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Ant1 Train/Val: {train_cls_acc_ant1:.4f}/{val_cls_acc_ant1:.4f} | "
        f"Ant2 Train/Val: {train_cls_acc_ant2:.4f}/{val_cls_acc_ant2:.4f}"
    )

print("Training completed and best model saved.")


#save training history to CSV for analysis
with open(f'{model_dir}/train_history_ant{ant1}-{ant2}_{prova}.csv', 'w') as f:
    f.write("epoch,train_loss,val_loss,ant1_train_acc,ant1_val_acc,ant2_train_acc,ant2_val_acc\n")
    for i in range(epochs):
        f.write(
            f"{i+1},"
            f"{loss_history['train'][i]:.6f},"
            f"{loss_history['val'][i]:.6f},"
            f"{cls_acc_history_ant1['train'][i]:.6f},"
            f"{cls_acc_history_ant1['val'][i]:.6f},"
            f"{cls_acc_history_ant2['train'][i]:.6f},"
            f"{cls_acc_history_ant2['val'][i]:.6f}\n"
        )




#reload of the best model for evaluation
best_model_path = f'{model_dir}/siamese_model_ant{ant1}-{ant2}_{prova}.pth'
#loading checkpoint
checkpoint = torch.load(best_model_path, map_location=device)
#restore the model
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
#restore history
loss_history = checkpoint['loss_history']
cls_acc_history_ant1 = checkpoint['cls_acc_history_ant1']
cls_acc_history_ant2 = checkpoint['cls_acc_history_ant2']
#best_val_loss:
best_val_loss = checkpoint['best_val_loss']
start_epoch = checkpoint.get('epoch', 0)


# train/val visualization
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1) loss
axs[0].plot(loss_history['train'], marker='o', linewidth=2, label='Train', color='blue')
axs[0].plot(loss_history['val'], marker='s', linewidth=2, label='Val', color='orange')
axs[0].axhline(y=min(loss_history['val']), color='red', linestyle='--',
               label=f'Best Val: {min(loss_history['val']):.4f}')
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Average loss")
axs[0].set_title(f"Training Loss Ant{ant1}-{ant2}")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

#load besto model
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2-3) distances on validation set
all_indices, all_dists, all_labels_sim = [], [], []

with torch.no_grad():
    for batch_idx, (w1, w2, labels_sim, act_idx1, act_idx2) in enumerate(val_loader):
        batch_start = batch_idx * batch_size
        batch_indices = list(range(batch_start, batch_start + len(labels_sim)))

        w1 = w1.to(device)
        w2 = w2.to(device)
        labels_sim = labels_sim.to(device)

        _, _, dist, _, _ = model(w1, w2)

        all_indices.extend(batch_indices)
        all_dists.extend(dist.cpu().numpy())
        all_labels_sim.extend(labels_sim.cpu().numpy())

all_indices = np.array(all_indices)
all_dists = np.array(all_dists)
all_labels_sim = np.array(all_labels_sim)
mask_sim = all_labels_sim == 1
mask_dis = all_labels_sim == 0

# 2) scatter
axs[1].scatter(all_indices[mask_sim], all_dists[mask_sim], s=8, alpha=0.7,
                label="Same", color='green', edgecolors='darkgreen')
axs[1].scatter(all_indices[mask_dis], all_dists[mask_dis], s=8, alpha=0.7,
                label="Different", color='purple', edgecolors='darkviolet')
axs[1].set_xlabel("Pair index")
axs[1].set_ylabel("Euclidean Distance")
axs[1].set_title(f"Validation Distances Ant{ant1}-{ant2}")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# 3) histogram
axs[2].hist(all_dists[mask_sim], bins=30, alpha=0.7,
            label=f'Same (n={mask_sim.sum()})', color='green', density=True)
axs[2].hist(all_dists[mask_dis], bins=30, alpha=0.7,
            label=f'Different (n={mask_dis.sum()})', color='purple', density=True)
axs[2].axvline(all_dists[mask_sim].mean(), color='green', linestyle='--', linewidth=2,
               label=f'Average Same: {all_dists[mask_sim].mean():.3f}')
axs[2].axvline(all_dists[mask_dis].mean(), color='purple', linestyle='--', linewidth=2,
               label=f'Average Different: {all_dists[mask_dis].mean():.3f}')
axs[2].set_xlabel('Euclidean Distance')
axs[2].set_ylabel('Density')
axs[2].legend()
axs[2].set_ylim(0, 15)
axs[2].grid(True, alpha=0.3)
axs[2].set_title(f'History of Validation Distances, training on Ant{ant1}-{ant2}')

plt.tight_layout()
plt.savefig(f'{plots_dir}/dashboard_train_ant{ant1}-{ant2}_{prova}.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== Validation Distances ===")
print(f"Similar: average={all_dists[mask_sim].mean():.3f}, std={all_dists[mask_sim].std():.3f}, n={mask_sim.sum()}")
print(f"Different: average={all_dists[mask_dis].mean():.3f}, std={all_dists[mask_dis].std():.3f}, n={mask_dis.sum()}")
margin_achieved = (all_dists[mask_dis].mean() - all_dists[mask_sim].max()) / 2
print(f"Separation VAL: {margin_achieved:.3f}")


#save distances and labels to CSV for further analysis
df_plot = pd.DataFrame({
    'index': all_indices,
    'dist': all_dists,
    'label_sim': all_labels_sim
})
df_plot.to_csv(f'{plots_dir}/val_distances_ant{ant1}-{ant2}_{prova}.csv', index=False)

#loading csv
df_plot = pd.read_csv(f'{plots_dir}/val_distances_ant{ant1}-{ant2}_{prova}.csv')

all_indices = df_plot['index'].to_numpy()
all_dists = df_plot['dist'].to_numpy()
all_labels_sim = df_plot['label_sim'].to_numpy()
mask_sim = all_labels_sim == 1
mask_dis = all_labels_sim == 0


#TEST EVALUATION FUNCTION

def evaluate_test(test_loader, model, criterion, device, ant1, ant2, save_dir, prova, acts=None):
    if acts is None:
        acts = ['Walk', 'Jump', 'Empty']

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    test_loss = 0.0
    correct1, correct2, total = 0, 0, 0

    all_dists = []
    all_labels_sim = []
    all_scores = []
    all_true_act1, all_pred_act1 = [], []
    all_true_act2, all_pred_act2 = [], []

    with torch.no_grad():
        for w1, w2, labels_sim, act_idx1, act_idx2 in test_loader:
            w1 = w1.to(device)
            w2 = w2.to(device)
            labels_sim = labels_sim.to(device)
            act_idx1 = act_idx1.to(device)
            act_idx2 = act_idx2.to(device)

            emb1, emb2, dist, logits1, logits2 = model(w1, w2)

            loss = criterion(dist, labels_sim, logits1, logits2, act_idx1, act_idx2)
            test_loss += loss.item() * w1.size(0)

            pred1 = logits1.argmax(dim=1)
            pred2 = logits2.argmax(dim=1)

            correct1 += (pred1 == act_idx1).sum().item()
            correct2 += (pred2 == act_idx2).sum().item()
            total += act_idx1.size(0)

            # distanza piccola => più simili
            scores = -dist

            all_dists.extend(dist.detach().cpu().numpy())
            all_scores.extend(scores.detach().cpu().numpy())
            all_labels_sim.extend(labels_sim.detach().cpu().numpy())

            all_true_act1.extend(act_idx1.detach().cpu().numpy())
            all_pred_act1.extend(pred1.detach().cpu().numpy())
            all_true_act2.extend(act_idx2.detach().cpu().numpy())
            all_pred_act2.extend(pred2.detach().cpu().numpy())

    test_loss_mean = test_loss / total
    acc1 = correct1 / total
    acc2 = correct2 / total
    acc_avg = (acc1 + acc2) / 2



    all_dists = np.array(all_dists)
    all_scores = np.array(all_scores)
    all_labels_sim = np.array(all_labels_sim).astype(int)
    all_indices = np.arange(len(all_dists))


    all_true_act1 = np.array(all_true_act1).astype(int)
    all_pred_act1 = np.array(all_pred_act1).astype(int)
    all_true_act2 = np.array(all_true_act2).astype(int)
    all_pred_act2 = np.array(all_pred_act2).astype(int)

    df_test = pd.DataFrame({
    'index': all_indices,
    'dist': all_dists,
    'score': all_scores,
    'label_sim': all_labels_sim,
    'true_act1': all_true_act1,
    'pred_act1': all_pred_act1,
    'true_act2': all_true_act2,
    'pred_act2': all_pred_act2})

    df_test.to_csv(f"{save_dir}/test_results_{ant1}-{ant2}_{prova}.csv", index=False)

    # ROC
    fpr, tpr, thresholds = roc_curve(all_labels_sim, all_scores)
    roc_auc = auc(fpr, tpr)

    best_thresh = 0.0
    best_acc = accuracy_score(all_labels_sim, (all_scores >= best_thresh).astype(int))
    for thresh in thresholds:
        preds = (all_scores >= thresh).astype(int)
        acc = accuracy_score(all_labels_sim, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    pair_preds = (all_scores >= best_thresh).astype(int)

    cm_pair = confusion_matrix(all_labels_sim, pair_preds, labels=[0, 1])

    n_acts = len(acts)
    cm_act1 = confusion_matrix(all_true_act1, all_pred_act1, labels=np.arange(n_acts))
    cm_act2 = confusion_matrix(all_true_act2, all_pred_act2, labels=np.arange(n_acts))

    print("Distribuzione attività per antenna:")
    print("Ant1 True:", np.bincount(all_true_act1, minlength=n_acts))
    print("Ant1 Pred:", np.bincount(all_pred_act1, minlength=n_acts))
    print("Ant2 True:", np.bincount(all_true_act2, minlength=n_acts))
    print("Ant2 Pred:", np.bincount(all_pred_act2, minlength=n_acts))

    mask_sim = all_labels_sim == 1
    mask_dis = all_labels_sim == 0
    dists_sim = all_dists[mask_sim]
    dists_dis = all_dists[mask_dis]

    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))

    axs1[0].hist(dists_sim, bins=50, alpha=0.9, label=f"Same (n={len(dists_sim)})", color="green")
    axs1[0].hist(dists_dis, bins=50, alpha=0.9, label=f"Different (n={len(dists_dis)})", color="purple")
    axs1[0].axvline(dists_sim.mean(), color="green", linestyle="--", label=f"Mean same: {dists_sim.mean():.3f}")
    axs1[0].axvline(dists_dis.mean(), color="purple", linestyle="--", label=f"Mean different: {dists_dis.mean():.3f}")
    axs1[0].set_xlabel("Euclidean Distance")
    axs1[0].set_ylabel("Count")
    axs1[0].set_title("Distribution of Test Distances")
    axs1[0].legend()
    axs1[0].set_ylim(0, max(axs1[0].get_ylim()[1], 10))
    axs1[0].grid(True, alpha=0.4)

    axs1[1].scatter(all_indices[mask_sim], all_dists[mask_sim], s=8, alpha=0.7,
                    label="Same", color='green', edgecolors='darkgreen')
    axs1[1].scatter(all_indices[mask_dis], all_dists[mask_dis], s=8, alpha=0.7,
                    label="Different", color='purple', edgecolors='darkviolet')
    axs1[1].set_xlabel("Pair index")
    axs1[1].set_ylabel("Euclidean Distance")
    axs1[1].set_title(f"Test Distances Ant{ant1}-{ant2}")
    axs1[1].legend()
    axs1[1].grid(True, alpha=0.3)

    axs1[2].plot(fpr, tpr, linewidth=3, label=f"ROC (AUC={roc_auc:.3f})", color="darkblue")
    axs1[2].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axs1[2].set_xlabel("False Positive Rate")
    axs1[2].set_ylabel("True Positive Rate")
    axs1[2].set_title("Test ROC Curve")
    axs1[2].legend()
    axs1[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/dist_index_roc_{ant1}-{ant2}_{prova}.png", dpi=150, bbox_inches='tight')
    plt.show()


    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axs2[0].imshow(cm_pair, cmap='YlGn')
    axs2[0].set_title(f"Pair Confusion Matrix @ {best_thresh:.3f}\nAcc: {best_acc*100:.1f}%")
    axs2[0].set_xticks([0, 1])
    axs2[0].set_xticklabels(['Different', 'Same'])
    axs2[0].set_yticks([0, 1])
    axs2[0].set_yticklabels(['Different', 'Same'])
    axs2[0].set_xlabel("Pred")
    axs2[0].set_ylabel("True")
    for i in range(2):
        for j in range(2):
            axs2[0].text(j, i, f'{cm_pair[i, j]}', ha='center', va='center',
                        fontweight='bold',
                        color='white' if cm_pair[i, j] > cm_pair.max()/2 else 'black')

    im1 = axs2[1].imshow(cm_act1, cmap='YlGn')
    axs2[1].set_title(f"Ant{ant1} Confusion Matrix\nAcc: {acc1*100:.1f}%")
    axs2[1].set_xticks(range(n_acts))
    axs2[1].set_xticklabels(acts, rotation=45, ha='right')
    axs2[1].set_yticks(range(n_acts))
    axs2[1].set_yticklabels(acts)
    axs2[1].set_xlabel("Pred")
    axs2[1].set_ylabel("True")
    for i in range(n_acts):
        for j in range(n_acts):
            axs2[1].text(j, i, f'{cm_act1[i, j]}', ha='center', va='center',
                        fontweight='bold',
                        color='white' if cm_act1[i, j] > cm_act1.max()/2 else 'black')

    im2 = axs2[2].imshow(cm_act2, cmap='YlGn')
    axs2[2].set_title(f"Ant{ant2} Confusion Matrix\nAcc: {acc2*100:.1f}%")
    axs2[2].set_xticks(range(n_acts))
    axs2[2].set_xticklabels(acts, rotation=45, ha='right')
    axs2[2].set_yticks(range(n_acts))
    axs2[2].set_yticklabels(acts)
    axs2[2].set_xlabel("Pred")
    axs2[2].set_ylabel("True")
    for i in range(n_acts):
        for j in range(n_acts):
            axs2[2].text(j, i, f'{cm_act2[i, j]}', ha='center', va='center',
                        fontweight='bold',
                        color='white' if cm_act2[i, j] > cm_act2.max()/2 else 'black')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusions_{ant1}-{ant2}_{prova}.png", dpi=150, bbox_inches='tight')
    plt.show()


    report_path = f"{save_dir}/classification_report_{ant1}-{ant2}_{prova}.txt"
    with open(report_path, "w") as f:
        f.write("=== CLASSIFICATION REPORT ANT1 ===\n")
        f.write(classification_report(all_true_act1, all_pred_act1, target_names=acts, digits=4))
        f.write("\n\n=== CLASSIFICATION REPORT ANT2 ===\n")
        f.write(classification_report(all_true_act2, all_pred_act2, target_names=acts, digits=4))
        f.write("\n\n=== TEST SUMMARY ===\n")
        f.write(f"Loss: {test_loss_mean:.4f}\n")
        f.write(f"Acc1: {acc1*100:.1f}%\n")
        f.write(f"Acc2: {acc2*100:.1f}%\n")
        f.write(f"AccAvg: {acc_avg*100:.1f}%\n")
        f.write(f"Pair AUC: {roc_auc:.4f}\n")
        f.write(f"Best threshold: {best_thresh:.4f}\n")
        f.write(f"Best pair accuracy: {best_acc*100:.1f}%\n")

    print("\n=== CLASSIFICATION REPORT ===")
    print("Ant1:")
    print(classification_report(all_true_act1, all_pred_act1, target_names=acts, digits=4))
    print("Ant2:")
    print(classification_report(all_true_act2, all_pred_act2, target_names=acts, digits=4))

    print(f"\nTEST SUMMARY Ant{ant1}-{ant2}:")
    print(f"  Loss: {test_loss_mean:.4f}")
    print(f"  Acc1: {acc1*100:.1f}% | Acc2: {acc2*100:.1f}% | Avg: {acc_avg*100:.1f}%")
    print(f"  Pair AUC: {roc_auc:.3f} | Best Thresh: {best_thresh:.3f} | Best Pair Acc: {best_acc*100:.1f}%")

    return test_loss_mean, acc_avg, best_thresh, roc_auc


#load best model and test 
criterion = CombinedLoss(margin=margine, cls_weight=cls_ora).to(device)

checkpoint = torch.load(best_model_path, map_location=device) 
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

test_loss, test_acc, best_thresh, test_auc = evaluate_test(
    test_loader=test_loader,model=model,criterion=criterion,device=device,
    ant1=ant1,ant2=ant2,prova=prova,save_dir=plots_dir,acts=['Walk', 'Jump', 'Empty'])


#reload of csv if needed for other plots
df_test = pd.read_csv(f"{plots_dir}/test_results_{ant1}-{ant2}_{prova}.csv")

all_dists = df_test['dist'].to_numpy()
all_scores = df_test['score'].to_numpy()
all_labels_sim = df_test['label_sim'].to_numpy().astype(int)
all_indices = df_test['index'].to_numpy()

mask_sim = all_labels_sim == 1
mask_dis = all_labels_sim == 0


all_true_act1 = df_test['true_act1'].to_numpy().astype(int)
all_pred_act1 = df_test['pred_act1'].to_numpy().astype(int)
all_true_act2 = df_test['true_act2'].to_numpy().astype(int)
all_pred_act2 = df_test['pred_act2'].to_numpy().astype(int)

fpr, tpr, thresholds = roc_curve(all_labels_sim, all_scores)
roc_auc = auc(fpr, tpr)


n_acts = len(acts)

best_thresh = 0.0
best_acc = accuracy_score(all_labels_sim, (all_scores >= best_thresh).astype(int))

for thresh in thresholds:
    preds = (all_scores >= thresh).astype(int)
    acc = accuracy_score(all_labels_sim, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

acc1 = accuracy_score(all_true_act1, all_pred_act1)
acc2 = accuracy_score(all_true_act2, all_pred_act2)
pair_preds = (all_scores >= best_thresh).astype(int)
cm_pair = confusion_matrix(all_labels_sim, pair_preds, labels=[0, 1])
cm_act1 = confusion_matrix(all_true_act1, all_pred_act1, labels=np.arange(n_acts))
cm_act2 = confusion_matrix(all_true_act2, all_pred_act2, labels=np.arange(n_acts))


# ----------------------------------------------
# EXTRACTION EMBEDDINGS 

#loading model
model_path = f'{model_dir}/siamese_model_ant{ant1}-{ant2}_{prova}.pth'

model = SiameseNetwork(dataset).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

embedding_net = model.embedding_net.to(device)
embedding_net.eval()

for param in embedding_net.parameters():
    param.requires_grad = False

print("Model loaded and ready for embedding extraction")

def get_available_antennas(paths):
    sample = np.abs(sio.loadmat(paths[0])['csi'])
    return list(range(sample.shape[2]))


class SingleAntennaWindowDataset(Dataset):
    def __init__(self, activity_paths, antenna_idx, allowed_starts=None, window_size=450):
        self.activity_arrays = []
        self.indices = []
        self.window_size = window_size

        for act_idx, path in enumerate(activity_paths):
            csi = np.abs(sio.loadmat(path)['csi'])[:, :, antenna_idx].astype(np.float32)
            self.activity_arrays.append(csi)

            max_start = csi.shape[0] - window_size
            all_valid_starts = np.arange(max_start + 1, dtype=np.int64)

            if allowed_starts is None:
                valid_starts = all_valid_starts
            else:
                allowed_starts_arr = np.asarray(sorted(set(int(x) for x in allowed_starts)), dtype=np.int64)
                valid_starts = allowed_starts_arr[allowed_starts_arr <= max_start]

            for start in valid_starts:
                self.indices.append((act_idx, int(start)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        act_idx, start = self.indices[idx]
        window = self.activity_arrays[act_idx][start:start + self.window_size, :]
        window = torch.from_numpy(window).float()
        window = (window - window.mean()) / (window.std() + 1e-6)
        return window.unsqueeze(0), torch.tensor(act_idx, dtype=torch.long)


def extract_single_antenna_embeddings(embedding_net, paths, ant, allowed_starts=None, batch_size=128, window_size=450):
    ds = SingleAntennaWindowDataset(
        activity_paths=paths,
        antenna_idx=ant,
        allowed_starts=allowed_starts,
        window_size=window_size,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    embs_list, logits_list, labels_list = [], [], []
    with torch.no_grad():
        for windows, labels in loader:
            windows = windows.to(device).float()
            emb, logits = embedding_net(windows)
            emb = F.normalize(emb, p=2, dim=1)
            embs_list.append(emb.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    embs = np.vstack(embs_list)
    logits = np.vstack(logits_list)
    labels = np.concatenate(labels_list)

    print(f"Ant {ant}: embs={embs.shape}, logits={logits.shape}, labels={np.bincount(labels, minlength=len(acts))}")
    return embs, logits, labels

'''This function takes the target antenna and extracts a small support set, a few examples per class.
Then it passes these examples through the encoder to obtain embeddings and averages them per class: this average is the prototype of the class'''

def build_prototypes_from_support(embedding_net, support_paths, support_ant, n_shot=10, window_size=450):
    support_ds = SingleAntennaWindowDataset(activity_paths=support_paths,antenna_idx=support_ant,allowed_starts=None,window_size=window_size)

    selected = []
    counts = {c: 0 for c in range(len(acts))}
    for i, (act_idx, start) in enumerate(support_ds.indices):
        if counts[act_idx] < n_shot:
            selected.append(i)
            counts[act_idx] += 1
        if all(v >= n_shot for v in counts.values()):
            break

    support_ds.indices = [support_ds.indices[i] for i in selected]
    support_loader = DataLoader(support_ds, batch_size=30, shuffle=False, num_workers=2)

    support_embs = []
    support_labels = []
    with torch.no_grad():
        for x, y in support_loader:
            x = x.to(device).float()
            emb, _ = embedding_net(x)  #embedding net returns both embeddings and logits
            emb = F.normalize(emb, p=2, dim=1)
            support_embs.append(emb.cpu())
            support_labels.append(y.cpu())

    support_embs = torch.cat(support_embs, dim=0)
    support_labels = torch.cat(support_labels, dim=0)

    prototypes = []
    for c in range(len(acts)):
        proto = support_embs[support_labels == c].mean(dim=0)
        prototypes.append(proto)

    prototypes = torch.stack(prototypes, dim=0)  # [n_acts, emb_dim]
    return prototypes  

#This function takes the query windows, extracts the embeddings and compares them with the prototypes. The predicted class is the one of the closest prototype.
def classify_by_prototypes(embedding_net, query_paths, query_ant, prototypes, window_size=450, batch_size=128):
    #Dataset creation and dataloader creation for query antenna
    query_ds = SingleAntennaWindowDataset(
        activity_paths=query_paths,
        antenna_idx=query_ant,
        allowed_starts=None,
        window_size=window_size,
    )
    query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    #extraction of embeddings for all query windows and for each query window I get a normalized embedding, which becomes a point in the same latent space used for the prototypes
    all_embs, all_labels = [], []
    with torch.no_grad():
        for x, y in query_loader:
            x = x.to(device).float()
            emb, _ = embedding_net(x)
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu())
            all_labels.append(y.cpu())

    query_embs = torch.cat(all_embs, dim=0)
    query_labels = torch.cat(all_labels, dim=0)

    #computing distances between query embeddings and prototypes, predict class of the closest prototype
    dists = torch.cdist(query_embs, prototypes.cpu())
    #choose the class of the closest prototype for each query
    preds = dists.argmin(dim=1)


    #performance metrics
    acc = accuracy_score(query_labels.numpy(), preds.numpy())
    cm = confusion_matrix(query_labels.numpy(), preds.numpy(), labels=np.arange(len(acts)))
    report = classification_report(query_labels.numpy(), preds.numpy(), target_names=acts, zero_division=0)

    return acc, cm, report, query_labels.numpy(), preds.numpy(), query_embs.numpy()


# embedding extraction for all antennas
available_ants = get_available_antennas(paths)
train_ants = [ant1, ant2]
heldout_ants = [ant3, ant4] #or [ant for ant in available_ants if ant not in train_ants]

print(f'Available antennas: {available_ants}')
print(f'Train antennas: {train_ants} | Held-out antennas: {heldout_ants}')

data_per_ant = {}
for ant in available_ants:
    split_name = 'seen_train_ant' if ant in train_ants else 'unseen_test_ant'
    print(f"\n--- Extraction antenna {ant} ({split_name}) ---")

    embs, logits, labels = extract_single_antenna_embeddings(
        embedding_net=embedding_net,
        paths=paths,
        ant=ant,
        allowed_starts=None,
        batch_size=batch_size,
        window_size=window_size,
    )

    data_per_ant[ant] = {
        'embs': embs,
        'logits': logits,
        'labels': labels,
        'split': split_name,
    }


#save embeddings to pickle file for later use
pair = f"{ant1}-{ant2}"
payload = {
    'data_per_ant': data_per_ant,
    'train_ants': train_ants,
    'heldout_ants': heldout_ants,
    'activities': acts,
    'window_size': window_size,
}
filename = f'{emb_dir}/single_ant_embeddings_{pair}_{prova}.pkl'
with open(filename, 'wb') as f:
    pickle.dump(payload, f)

print('Single branch embeddings  saved!')


# TSNE
with open(filename, 'rb') as f:
    payload = pickle.load(f)

data_per_ant = payload['data_per_ant']
activity_names = payload['activities']
available_ants = sorted(data_per_ant.keys())

activity_colors = {0: "#edae49", 1: "#d1495b", 2: "#00798c"}

fig, axes = plt.subplots(1, len(available_ants), figsize=(6 * len(available_ants), 6))
if len(available_ants) == 1:
    axes = [axes]

fig.suptitle(f't-SNE embeddings single-branch - model {pair}', fontsize=16)

for col, ant in enumerate(available_ants):
    ax = axes[col]
    embs = data_per_ant[ant]['embs']
    labels = data_per_ant[ant]['labels'].astype(int)

    #add pca
    pca = PCA(n_components=min(8, embs.shape[1])) #uses all available dimensions
    embs_pca = pca.fit_transform(embs)

    #How much variance do the first two components explain
    print(f"Ant {ant} - Variance explained by the first 2 PCAs: {np.sum(pca.explained_variance_ratio_[:2]):.2f}")
    #add pca

    perplexity = min(50, max(5, len(embs) // 20))
    reducer = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto'
    )
    embs_2d = reducer.fit_transform(embs)

    for act_idx, act_name in enumerate(activity_names):
        mask = labels == act_idx
        ax.scatter(
            embs_2d[mask, 0],
            embs_2d[mask, 1],
            s=5,
            alpha=0.7,
            color=activity_colors.get(act_idx, 'gray'),
            label=act_name
        )

    ax.set_title(f"Antenna {ant} [{data_per_ant[ant]['split']}]", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

legend_elements = [Line2D([0], [0], marker='o', color='w', label=name,markerfacecolor=activity_colors[idx], markersize=8)
    for idx, name in enumerate(activity_names)
]
fig.legend(handles=legend_elements, loc='lower center', ncol=len(activity_names), frameon=False)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig(f'{emb_dir}/tsne_single_branch_{pair}_{prova}_pca.png', dpi=300, bbox_inches='tight')
plt.show()


#silhouette score for each antenna
print("\n=== ANTENNA CLASS SEPARABILITY (single-branch) ===")
for ant in sorted(data_per_ant.keys()):
    embs = data_per_ant[ant]['embs']
    labels = data_per_ant[ant]['labels'].astype(int)
    sil = silhouette_score(embs, labels)
    print(f"Ant {ant} [{data_per_ant[ant]['split']}]: silhouette={sil:.3f}")

''' 
# now if we want to reload and do other plots 
#reload if needed for further analysis
pair = f"{ant1}-{ant2}"
filename = f'{emb_dir}/single_ant_embeddings_{pair}_{prova}.pkl'

with open(filename, 'rb') as f:
    payload = pickle.load(f)

data_per_ant = payload['data_per_ant']
activity_names = payload['activities']
available_ants = sorted(data_per_ant.keys())

print("Antennas found in the file:", available_ants)
print("Classes:", activity_names)


rows = []

for ant in available_ants:
    embs = data_per_ant[ant]['embs']
    labels = data_per_ant[ant]['labels'].astype(int)
    split_name = data_per_ant[ant]['split']

    pca = PCA(n_components=min(8, embs.shape[1]))
    embs_pca = pca.fit_transform(embs)

    perplexity = min(50, max(5, len(embs) // 20))
    reducer = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto'
    )

    embs_2d = reducer.fit_transform(embs_pca)

    for i in range(len(labels)):
        rows.append({
            'ant': ant,
            'split': split_name,
            'label': int(labels[i]),
            'act_name': activity_names[int(labels[i])],
            'x': float(embs_2d[i, 0]),
            'y': float(embs_2d[i, 1]),
        })

df_tsne = pd.DataFrame(rows)

csv_path = f"{emb_dir}/tsne_coords_{pair}_{prova}.csv"
df_tsne.to_csv(csv_path, index=False)

print(f"Saved: {csv_path}")


#loading coordinates for further analysis
pair = f"{ant1}-{ant2}"
csv_path = f"{emb_dir}/tsne_coords_{pair}_{prova}.csv"

df_tsne = pd.read_csv(csv_path)
available_ants = sorted(df_tsne['ant'].unique())
activity_names = [df_tsne[df_tsne['label'] == i]['act_name'].iloc[0] for i in sorted(df_tsne['label'].unique())]

os.makedirs(emb_dir, exist_ok=True)

palettes = {
    "colors_1": {0: "#edae49", 1: "#d1495b", 2: "#00798c"},
    "colors_2": {0: 'seagreen', 1: 'teal', 2: 'darkorange'}}

fig_bg = 'white'
ax_bg = 'white'
grid_color = 'gray'
title_color = 'black'

x_min = df_tsne['x'].min()
x_max = df_tsne['x'].max()
y_min = df_tsne['y'].min()
y_max = df_tsne['y'].max()

pad_x = 0.05 * (x_max - x_min)
pad_y = 0.05 * (y_max - y_min)

x_limits = (x_min - pad_x, x_max + pad_x)
y_limits = (y_min - pad_y, y_max + pad_y)

for palette_name, activity_colors in palettes.items():

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    fig.patch.set_facecolor(fig_bg)
    fig.suptitle(f't-SNE embeddings - model trained on {pair} antennas', fontsize=16, color=title_color, fontweight='bold')

    for plot_idx, ant in enumerate(available_ants[:4]):
        ax = axes[plot_idx]
        ax.set_facecolor(ax_bg)

        df_ant = df_tsne[df_tsne['ant'] == ant]
        split_name = df_ant['split'].iloc[0]

        for act_idx in sorted(df_ant['label'].unique()):
            df_cls = df_ant[df_ant['label'] == act_idx]
            ax.scatter(
                df_cls['x'],
                df_cls['y'],
                s=7,
                alpha=0.75,
                color=activity_colors.get(act_idx, 'gray'),
                label=df_cls['act_name'].iloc[0]
            )

        ax.set_title(f"Antenna {ant} [{split_name}]", fontweight='bold', color=title_color)
        ax.grid(True, alpha=0.25, color=grid_color)
        #ax.set_aspect('equal')

        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_aspect('equal', adjustable='box')


    for k in range(len(available_ants), 4):
        axes[k].axis('off')

    activity_names = ['Walk', 'Jump', 'Empty']  # oppure payload['activities'] se vuoi
    legend_elements = [Line2D([0], [0], marker='o', color='w',label=name, markerfacecolor=activity_colors[idx], markersize=18)
        for idx, name in enumerate(activity_names)]

    fig.legend(handles=legend_elements, loc='lower center', ncol=len(activity_names), frameon=True, fontsize=16,  handletextpad=0.8, columnspacing=1.8, borderaxespad=1.2, facecolor=fig_bg)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    plt.savefig(f'{emb_dir_fast}/tsne_single_branch_2x2_{pair}_{prova}_fast_{palette_name}.png',
                dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

    
'''


# ZERO AND FEW SHOT 
#reload model 
model_path = f'{model_dir}/siamese_model_ant{ant1}-{ant2}_{prova}.pth'
model = SiameseNetwork(dataset).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

embedding_net = model.embedding_net.to(device)
embedding_net.eval()

for param in embedding_net.parameters():
    param.requires_grad = False

heldout_ants = [ant3, ant4]
print(f"Held-out antennas: {heldout_ants}")

print("Modello caricato e pronto per estrazione embeddings")


#utility functions 
def get_available_antennas(paths):
    sample = np.abs(sio.loadmat(paths[0])['csi'])
    return list(range(sample.shape[2]))


def normalize_embeddings_to_reference(query_embs, reference_embs):
    ref_mean = reference_embs.mean(dim=0)
    ref_std  = reference_embs.std(dim=0) + 1e-6
    q_mean   = query_embs.mean(dim=0)
    q_std    = query_embs.std(dim=0) + 1e-6
    normalized = (query_embs - q_mean) / q_std * ref_std + ref_mean
    return F.normalize(normalized, p=2, dim=1)


def get_num_valid_starts(activity_paths, antenna_idx, window_size):
    csi = np.abs(sio.loadmat(activity_paths[0])['csi'])[:, :, antenna_idx]
    max_start = csi.shape[0] - window_size
    return max_start + 1


def split_starts_into_4_subgroups(n_total_starts):
    all_starts = np.arange(n_total_starts, dtype=np.int64)
    groups = np.array_split(all_starts, 4)
    return groups


def select_support_starts_4x3(n_total_starts):
    groups = split_starts_into_4_subgroups(n_total_starts)
    support_starts = []
    for g_idx, group in enumerate(groups):
        if len(group) < 3:
            raise ValueError(f"Sottogruppo {g_idx} troppo piccolo: len={len(group)}")
        support_starts.extend(group[:3].tolist())
    support_starts = np.array(sorted(set(support_starts)), dtype=np.int64)
    return support_starts


def expand_exclusion_zone(support_starts, n_total_starts, window_size):
    excluded = np.zeros(n_total_starts, dtype=bool)
    for s in support_starts:
        left  = max(0, s - (window_size - 1))
        right = min(n_total_starts - 1, s + (window_size - 1))
        excluded[left:right + 1] = True
    return excluded


def build_support_query_starts_4x3_no_overlap(activity_paths, antenna_idx, window_size=450):
    n_total_starts = get_num_valid_starts(activity_paths, antenna_idx, window_size)
    support_starts = select_support_starts_4x3(n_total_starts)
    excluded_mask  = expand_exclusion_zone(support_starts, n_total_starts, window_size)
    all_starts     = np.arange(n_total_starts, dtype=np.int64)
    query_starts   = all_starts[~excluded_mask]
    excluded_starts = all_starts[excluded_mask]
    return support_starts, query_starts, excluded_starts


def build_support_and_query_datasets_4x3(activity_paths, antenna_idx, window_size=450):
    support_starts, query_starts, excluded_starts = build_support_query_starts_4x3_no_overlap(
        activity_paths=activity_paths,
        antenna_idx=antenna_idx,
        window_size=window_size
    )
    support_ds = SingleAntennaWindowDataset(
        activity_paths=activity_paths,
        antenna_idx=antenna_idx,
        allowed_starts=support_starts,
        window_size=window_size
    )
    query_ds = SingleAntennaWindowDataset(
        activity_paths=activity_paths,
        antenna_idx=antenna_idx,
        allowed_starts=query_starts,
        window_size=window_size
    )
    return support_ds, query_ds, support_starts, query_starts, excluded_starts


def build_prototypes_from_support_dataset(embedding_net, support_ds, batch_size=128, method='simple'):
    """
    method: 'simple', 'variance_weighted', 'attention', 'robust'
    """
    support_loader = DataLoader(support_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    support_embs   = []
    support_labels = []

    with torch.no_grad():
        for x, y in support_loader:
            x = x.to(device).float()
            emb, _ = embedding_net(x)
            emb = F.normalize(emb, p=2, dim=1)
            support_embs.append(emb.cpu())
            support_labels.append(y.cpu())

    support_embs   = torch.cat(support_embs, dim=0)
    support_labels = torch.cat(support_labels, dim=0)

    prototypes = []
    for c in range(len(acts)):
        class_embs = support_embs[support_labels == c]

        if method == 'simple':
            proto = class_embs.mean(dim=0)

        elif method == 'variance_weighted':
            class_mean = class_embs.mean(dim=0)
            vars_      = torch.norm(class_embs - class_mean, dim=1, p=2)
            weights    = 1.0 / (vars_ + 1e-6)
            weights    = weights / weights.sum()
            proto      = (class_embs * weights.unsqueeze(1)).sum(dim=0)

        elif method == 'attention':
            class_mean       = class_embs.mean(dim=0)
            dists_to_mean    = torch.norm(class_embs - class_mean, dim=1, p=2)
            attention_weights = torch.softmax(-dists_to_mean, dim=0)
            proto            = (class_embs * attention_weights.unsqueeze(1)).sum(dim=0)

        elif method == 'robust':
            class_mean  = class_embs.mean(dim=0)
            dists       = torch.norm(class_embs - class_mean, dim=1, p=2)
            threshold   = torch.quantile(dists, 0.8)
            clean_embs  = class_embs[dists <= threshold]
            proto       = clean_embs.mean(dim=0)

        else:
            raise ValueError(f"Method {method} non supportato")

        prototypes.append(proto)

    prototypes = torch.stack(prototypes, dim=0)
    return prototypes, support_embs.numpy(), support_labels.numpy()



def build_zeroshot_prototypes(embedding_net, activity_paths, seen_ants, window_size=450, batch_size=128, method='simple'):
    """
    Builds zero-shot prototypes by aggregating embeddings from seen antennas.

    Parameters
    ----------
    seen_ants : list[int]   e.g., [ant1, ant2]
    method    : same set as few-shot ('simple', 'variance_weighted', 'attention', 'robust')

    Returns
    -------
    prototypes : Tensor (n_classes, emb_dim)  — already normalized L2
    """
    #for every seen antenna, collect embeddings for each class
    per_ant_class_embs = {c: [] for c in range(len(acts))}

    for ant_idx in seen_ants:
        ds = SingleAntennaWindowDataset(
            activity_paths=activity_paths,
            antenna_idx=ant_idx,
            allowed_starts=None,   #uses all valid windows
            window_size=window_size
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device).float()
                emb, _ = embedding_net(x)
                emb = F.normalize(emb, p=2, dim=1).cpu()
                for c in range(len(acts)):
                    mask = (y == c)
                    if mask.any():
                        per_ant_class_embs[c].append(emb[mask])

    prototypes = []
    for c in range(len(acts)):
        #combines embeddings from all seen antennas for this class
        all_class_embs = torch.cat(per_ant_class_embs[c], dim=0)

        if method == 'simple':
            proto = all_class_embs.mean(dim=0)

        elif method == 'variance_weighted':
            class_mean = all_class_embs.mean(dim=0)
            vars_      = torch.norm(all_class_embs - class_mean, dim=1, p=2)
            weights    = 1.0 / (vars_ + 1e-6)
            weights    = weights / weights.sum()
            proto      = (all_class_embs * weights.unsqueeze(1)).sum(dim=0)

        elif method == 'attention':
            class_mean        = all_class_embs.mean(dim=0)
            dists_to_mean     = torch.norm(all_class_embs - class_mean, dim=1, p=2)
            attention_weights = torch.softmax(-dists_to_mean, dim=0)
            proto             = (all_class_embs * attention_weights.unsqueeze(1)).sum(dim=0)

        elif method == 'robust':
            class_mean = all_class_embs.mean(dim=0)
            dists      = torch.norm(all_class_embs - class_mean, dim=1, p=2)
            threshold  = torch.quantile(dists, 0.8)
            clean_embs = all_class_embs[dists <= threshold]
            proto      = clean_embs.mean(dim=0)

        else:
            raise ValueError(f"Method {method} not supported")

        prototypes.append(proto)

    prototypes = torch.stack(prototypes, dim=0)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes


def classify_query_dataset_by_prototypes(embedding_net, query_ds, prototypes, batch_size=128, reference_embs=None):
    query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_query_embs   = []
    all_query_labels = []

    with torch.no_grad():
        for x, y in query_loader:
            x = x.to(device).float()
            emb, _ = embedding_net(x)
            emb = F.normalize(emb, p=2, dim=1)
            all_query_embs.append(emb.cpu())
            all_query_labels.append(y.cpu())

    query_embs   = torch.cat(all_query_embs, dim=0)
    query_labels = torch.cat(all_query_labels, dim=0)

    if reference_embs is not None:
        query_embs = normalize_embeddings_to_reference(query_embs, reference_embs)
        prototypes = F.normalize(prototypes, p=2, dim=1)

    dists = torch.cdist(query_embs, prototypes.cpu())
    preds = dists.argmin(dim=1)

    acc    = accuracy_score(query_labels.numpy(), preds.numpy())
    cm     = confusion_matrix(query_labels.numpy(), preds.numpy(), labels=np.arange(len(acts)))
    report = classification_report(query_labels.numpy(), preds.numpy(), target_names=acts, zero_division=0)

    return acc, cm, report, query_labels.numpy(), preds.numpy(), query_embs.numpy()


METHODS = ['simple', 'variance_weighted', 'attention', 'robust']

results = []

def _tsne_2d(embs):
    perp = min(50, max(5, len(embs) // 20))
    return TSNE(n_components=2, random_state=42,
                perplexity=perp, init='pca',
                learning_rate='auto').fit_transform(embs)


# Pre-computes zero-shot prototypes for all 4 methods
# (identical for all held-out, depend only on ant1/ant2)
print("\n=== PRE-COMPUTES ZERO-SHOT PROTOTYPES (seen antennas) ===")
zs_protos_by_method = {}
for m in METHODS:
    zs_protos_by_method[m] = build_zeroshot_prototypes(
        embedding_net=embedding_net,
        activity_paths=paths,
        seen_ants=[ant1, ant2],
        window_size=window_size,
        batch_size=batch_size,
        method=m
    )
    print(f"  [{m}] zero-shot prototypes: {zs_protos_by_method[m].shape}")



# rows = held-out antenna, columns = metodo
n_rows = len(heldout_ants)
n_cols = len(METHODS)
fig_fs, axes_fs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
fig_zs, axes_zs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

#if I choose just one method
#fig_fs, axes_fs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
#fig_zs, axes_zs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)


fig_fs.suptitle('FEW-SHOT — Confusion matrices per metodo', fontsize=14, fontweight='bold')
fig_zs.suptitle('ZERO-SHOT — Confusion matrices per metodo', fontsize=14, fontweight='bold')

if n_rows == 1:
    axes_fs = np.array([axes_fs])
    axes_zs = np.array([axes_zs])


# ------------------------------------------------------------------
# Loop principale: antenna held-out × method
# ------------------------------------------------------------------
# memorizing embedding query for t-SNE (best method)
best_query_embs_fs = {}   # ant_test -> (embs, y_true)
best_query_embs_zs = {}

for row, ant_test in enumerate(heldout_ants):
    print(f"\n{'='*60}")
    print(f"ANTENNA HELD-OUT: {ant_test}")
    print(f"{'='*60}")

    # build support + query (identical for all methods for this antenna)
    support_ds, query_ds, support_starts, query_starts, excluded_starts = \
        build_support_and_query_datasets_4x3(
            activity_paths=paths,
            antenna_idx=ant_test,
            window_size=window_size
        )
    print(f"Support: {len(support_ds)} windows | Query: {len(query_ds)} windows")

    best_acc_fs = -1
    best_acc_zs = -1

    for col, method in enumerate(METHODS):
        # ── FEW-SHOT ──────────────────────────────────────────────
        print(f"\n  [FEW-SHOT | {method}]")
        fs_prototypes, _, _ = build_prototypes_from_support_dataset(
            embedding_net=embedding_net,
            support_ds=support_ds,
            batch_size=batch_size,
            method=method
        )
        acc_fs, cm_fs, report_fs, y_true_fs, y_pred_fs, query_embs_fs = \
            classify_query_dataset_by_prototypes(
                embedding_net=embedding_net,
                query_ds=query_ds,
                prototypes=fs_prototypes,
                batch_size=batch_size
            )
        print(f"    Accuracy: {acc_fs*100:.1f}%")
        print(report_fs)


        results.append({
        'test_ant': ant_test,
        'method': method,
        'mode': 'few-shot',
        'acc': acc_fs,
        'n_support_per_activity': 12,
        'n_support_total': len(support_ds),
        'n_query_total': len(query_ds),
        'cm': cm_fs,
        'y_true': y_true_fs,
        'y_pred': y_pred_fs,
        'query_embs': query_embs_fs
    })
        
        #save for t-SNE if it's the best so far
        if acc_fs > best_acc_fs:
            best_acc_fs = acc_fs
            best_query_embs_fs[ant_test] = (query_embs_fs, y_true_fs, method)

        # confusion matrix few-shot
        sns.heatmap(cm_fs, annot=True, fmt='d', cmap='Greens',
                    ax=axes_fs[row, col],
                    xticklabels=acts, yticklabels=acts)
        axes_fs[row, col].set_title(
            f'Ant {ant_test} | {method}\nAcc {acc_fs*100:.1f}%', fontsize=10)
        axes_fs[row, col].set_xlabel('Pred')
        axes_fs[row, col].set_ylabel('True')

        # ── ZERO-SHOT ─────────────────────────────────────────────
        print(f"\n  [ZERO-SHOT | {method}]")
        acc_zs, cm_zs, report_zs, y_true_zs, y_pred_zs, query_embs_zs = \
            classify_query_dataset_by_prototypes(
                embedding_net=embedding_net,
                query_ds=query_ds,
                prototypes=zs_protos_by_method[method],
                batch_size=batch_size
            )
        print(f"Accuracy: {acc_zs*100:.1f}%")
        print(report_zs)

        results.append({
        'test_ant': ant_test,
        'method': method,
        'mode': 'zero-shot',
        'acc': acc_zs,
        'n_support_per_activity': 0,
        'n_support_total': 0,
        'n_query_total': len(query_ds),
        'cm': cm_zs,
        'y_true': y_true_zs,
        'y_pred': y_pred_zs,
        'query_embs': query_embs_zs
    })
        

        if acc_zs > best_acc_zs:
            best_acc_zs = acc_zs
            best_query_embs_zs[ant_test] = (query_embs_zs, y_true_zs, method)

        # confusion matrix nella figura zero-shot
        sns.heatmap(cm_zs, annot=True, fmt='d', cmap='Oranges',
                    ax=axes_zs[row, col],
                    xticklabels=acts, yticklabels=acts)
        axes_zs[row, col].set_title(
            f'Ant {ant_test} | {method}\nAcc {acc_zs*100:.1f}%', fontsize=10)
        axes_zs[row, col].set_xlabel('Pred')
        axes_zs[row, col].set_ylabel('True')


#saves figures
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label=name,
               markerfacecolor=activity_colors[idx], markersize=8)
    for idx, name in enumerate(acts)
]

for fig_obj, tag, cmap_label in [
    (fig_fs, 'fewshot', 'Greens'),
    (fig_zs, 'zeroshot', 'Oranges')
]:
    fig_obj.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig_obj.savefig(
        f'{emb_dir}/cross_test_{tag}_all_methods_{pair}_{prova}.png',
        dpi=150, bbox_inches='tight'
    )
    fig_obj.show()


# Final results saving
os.makedirs(plots_dir, exist_ok=True)

results_df = pd.DataFrame(results)
results_df['acc_pct'] = results_df['acc'] * 100

print('\n=== Complete results ===')
print(  
    results_df[['test_ant', 'method', 'mode', 'acc_pct','n_support_per_activity', 'n_support_total', 'n_query_total']].to_string(index=False))

summary_df = results_df[
    ['test_ant', 'method', 'mode', 'acc', 'acc_pct',
     'n_support_per_activity', 'n_support_total', 'n_query_total']
].copy()

summary_df.to_csv(f'{plots_dir}/cross_test_results_{seed}.csv', index=False)

with open(f'{plots_dir}/cross_test_artifacts_{seed}.pkl', 'wb') as f:
    pickle.dump(results, f)

