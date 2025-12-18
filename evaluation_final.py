"""
FINAL EVALUATION SCRIPT FOR MODEL V3
Calculates all metrics for Classification, Trait ID, and Segmentation.
Uses CORRECT Mapping for Segmentation.
"""

import os
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, confusion_matrix
from collections import defaultdict

# Import Dataset & Model
from multitask_dataset_fishvista import FishVistaMultiTaskDataset
from multitask_model import MultiTaskSwinTransformer

# ================= CONFIGURATION =================
DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/home/ubuntu/23phuc.nh/fish-vista"
MASTER_CSV = os.path.join(DATA_ROOT, "master_dataset.csv")

# TRỎ VÀO MODEL V3
CHECKPOINT_PATH = "checkpoints_weighted_seg_v3_full/best_model_weighted_seg_v3_full.pth"
OUTPUT_DIR = "final_evaluation_v3_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_TRAITS = 4
NUM_SEG_CLASSES = 10

# Correct Mapping
SEG_CLASS_NAMES = {
    0: 'Background',
    1: 'Head',
    2: 'Eye',
    3: 'Dorsal Fin',
    4: 'Pectoral Fin',
    5: 'Pelvic Fin',
    6: 'Anal Fin',
    7: 'Caudal Fin',
    8: 'Adipose Fin',
    9: 'Barbel'
}

TRAIT_NAMES = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']

# CSV Files for Task Splits (Ensure these exist)
SPLIT_FILES = {
    "cls_test": "classification_test.csv",
    "trait_insp": "identification_test_insp.csv",
    "trait_ood": "identification_test_lvsp.csv",
    "trait_manual": "identification_test_manual_annot.csv",
    "seg_test": "segmentation_test.csv"
}

# ================= HELPER FUNCTIONS =================

def load_indices_from_csv(csv_filename, master_df):
    """Finds indices in master_df corresponding to filenames in csv_filename."""
    print(f"Loading indices from {csv_filename}...")
    if not os.path.exists(csv_filename):
        print(f"❌ Warning: {csv_filename} not found. Skipping.")
        return []
        
    target_df = pd.read_csv(csv_filename)
    # Create map for O(1) lookup
    filename_to_idx = {name: idx for idx, name in enumerate(master_df['filename'])}
    
    indices = []
    for fname in target_df['filename']:
        if fname in filename_to_idx:
            indices.append(filename_to_idx[fname])
            
    print(f"Found {len(indices)} / {len(target_df)} samples.")
    return indices

def get_species_groups(df, species_to_id):
    """Calculates species groups based on Train counts."""
    train_df = df[df['split'] == 'train']
    counts = train_df['standardized_species'].value_counts()
    
    groups = {}
    for name, count in counts.items():
        if name in species_to_id:
            sid = species_to_id[name]
            if count >= 500: groups[sid] = "Majority"
            elif count >= 100: groups[sid] = "Neutral"
            elif count >= 10: groups[sid] = "Minority"
            else: groups[sid] = "Ultra-Rare"
    return groups

# ================= EVALUATION LOGIC =================

def evaluate_classification(model, loader, species_groups):
    model.eval()
    all_preds, all_labels = [], []
    
    print("\n>>> Evaluating Classification...")
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image'].to(DEVICE)
            if batch['has_species_label'].any():
                mask = (batch['species_label'] != -1)
                if mask.any():
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                    preds = outputs['species'][mask].argmax(1).cpu().numpy()
                    labels = batch['species_label'][mask].cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)
    
    # Metrics
    if not all_labels: return {}
    
    acc_total = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for p, l in zip(all_preds, all_labels):
        grp = species_groups.get(l, "Unknown")
        group_stats[grp]["total"] += 1
        if p == l:
            group_stats[grp]["correct"] += 1
            
    group_acc = {g: s["correct"]/s["total"] for g, s in group_stats.items() if s["total"] > 0}
    
    return {
        "Accuracy Total": acc_total,
        "F1-Score Macro": f1,
        "Group Accuracy": group_acc
    }

def evaluate_traits(model, loader, split_name):
    model.eval()
    all_preds = {i: [] for i in range(NUM_TRAITS)}
    all_labels = {i: [] for i in range(NUM_TRAITS)}
    
    print(f"\n>>> Evaluating Traits ({split_name})...")
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image'].to(DEVICE)
            if batch['has_trait_labels'].any():
                mask = (batch['trait_labels'] != -1.0)
                if mask.any():
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                    probs = torch.sigmoid(outputs['traits'])
                    
                    for i in range(NUM_TRAITS):
                        mask_i = mask[:, i]
                        if mask_i.any():
                            all_preds[i].extend(probs[mask_i, i].cpu().numpy())
                            all_labels[i].extend(batch['trait_labels'][mask_i, i].cpu().numpy())
    
    # Metrics
    results = {}
    aps = []
    for i in range(NUM_TRAITS):
        if len(all_labels[i]) > 0 and sum(all_labels[i]) > 0:
            ap = average_precision_score(all_labels[i], all_preds[i])
            results[TRAIT_NAMES[i]] = ap
            aps.append(ap)
        else:
            results[TRAIT_NAMES[i]] = "N/A"
            
    if aps:
        results["mAP"] = np.mean(aps)
    return results

def evaluate_segmentation(model, loader):
    model.eval()
    total_cm = np.zeros((NUM_SEG_CLASSES, NUM_SEG_CLASSES), dtype=np.int64)
    
    print("\n>>> Evaluating Segmentation...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch['has_mask'].any():
                mask_flag = batch['has_mask'].to(DEVICE)
                images = batch['image'].to(DEVICE)[mask_flag]
                targets = batch['segmentation_mask'].to(DEVICE)[mask_flag]
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                preds = outputs['segmentation'].argmax(1)
                
                # CM calculation on CPU to save GPU mem
                preds_np = preds.cpu().numpy().flatten()
                targets_np = targets.cpu().numpy().flatten()
                
                total_cm += confusion_matrix(targets_np, preds_np, labels=range(NUM_SEG_CLASSES))
    
    # IoU
    results = {}
    ious = []
    for c in range(NUM_SEG_CLASSES):
        tp = total_cm[c, c]
        union = total_cm[c, :].sum() + total_cm[:, c].sum() - tp
        iou = tp / union if union > 0 else 0
        results[SEG_CLASS_NAMES[c]] = iou
        ious.append(iou)
        
    results["mIoU"] = np.mean(ious)
    return results

# ================= MAIN =================
def main():
    print(f"Loading Master Data: {MASTER_CSV}")
    master_df = pd.read_csv(MASTER_CSV, low_memory=False)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    full_dataset = FishVistaMultiTaskDataset(csv_file=MASTER_CSV, data_root=DATA_ROOT, transform=transform)
    num_species = len(full_dataset.species_to_id)
    
    # Groups
    species_groups = get_species_groups(master_df, full_dataset.species_to_id)
    
    # Load Model
    print(f"Loading Checkpoint: {CHECKPOINT_PATH}")
    model = MultiTaskSwinTransformer(num_species, NUM_TRAITS, NUM_SEG_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)['model_state_dict'])
    
    # --- RUN EVALS ---
    final_results = {}
    
    # 1. Classification (Test Split)
    indices = load_indices_from_csv(SPLIT_FILES["cls_test"], master_df)
    if indices:
        loader = DataLoader(Subset(full_dataset, indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        final_results["Classification"] = evaluate_classification(model, loader, species_groups)
        
    # 2. Traits (In-Species, OOD, Manual)
    for key, name in [("trait_insp", "In-Species"), ("trait_ood", "OOD"), ("trait_manual", "Manual")]:
        indices = load_indices_from_csv(SPLIT_FILES[key], master_df)
        if indices:
            loader = DataLoader(Subset(full_dataset, indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
            final_results[f"Trait ID ({name})"] = evaluate_traits(model, loader, name)
            
    # 3. Segmentation (Test Split)
    indices = load_indices_from_csv(SPLIT_FILES["seg_test"], master_df)
    if indices:
        loader = DataLoader(Subset(full_dataset, indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        final_results["Segmentation"] = evaluate_segmentation(model, loader)
        
    # Save
    json_path = os.path.join(OUTPUT_DIR, "all_metrics_v3.json")
    # Convert numpy types
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError
        
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=4, default=convert)
        
    print(f"\n✅ All Done! Results saved to {json_path}")
    print(json.dumps(final_results, indent=4, default=convert))

if __name__ == "__main__":
    main()