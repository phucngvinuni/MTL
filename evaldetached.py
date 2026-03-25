"""
FINAL EVALUATION SCRIPT FOR PROTO-SWIN MODEL (Non-parametric Prototype)
=======================================================================
Calculates all metrics:
- Classification: Accuracy, F1-Macro, Group Accuracy
- Trait ID: mAP, F1@0.5, F1@Optimal (requires Validation set)
- Segmentation: mIoU, IoU per class (Uses Prototype Head Logic)
"""

import os
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, confusion_matrix, precision_recall_curve
from collections import defaultdict

# ================= IMPORTS =================
from multitask_dataset_fishvista import get_datasets, FishVistaMultiTaskDataset
# CHANGE: Import ProtoSwin Model
from multitask_model_detached import AttributeAwareProtoSwin_Detached


# ================= CONFIGURATION =================
DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/home/ubuntu/23phuc.nh/fish-vista"
MASTER_CSV = os.path.join(DATA_ROOT, "master_dataset.csv")

# PATH TO YOUR BEST PROTO MODEL
CHECKPOINT_PATH = "/home/ubuntu/23phuc.nh/ATSIV/checkpoints_detached_compare/best_model_detached.pth"
OUTPUT_DIR = "final_results_atri_proto_detachedb64"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_TRAITS = 4
NUM_SEG_CLASSES = 10

# Helper Mappings
SEG_CLASS_NAMES = {
    0: 'Background', 1: 'Head', 2: 'Eye', 3: 'Dorsal Fin',
    4: 'Pectoral Fin', 5: 'Pelvic Fin', 6: 'Anal Fin',
    7: 'Caudal Fin', 8: 'Adipose Fin', 9: 'Barbel'
}
TRAIT_NAMES = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']

# CSV Files for Task Splits
SPLIT_FILES = {
    "cls_test": "classification_test.csv",
    "trait_val": "identification_val.csv", 
    "trait_insp": "identification_test_insp.csv",
    "trait_ood": "identification_test_lvsp.csv",
    "trait_manual": "identification_test_manual_annot.csv",
    "seg_test": "segmentation_test.csv"
}

# ================= HELPERS =================

def load_indices_from_csv(csv_filename, master_df):
    print(f"Loading indices from {csv_filename}...")
    full_path = os.path.join(DATA_ROOT, "splits", csv_filename) # Adjust path if needed
    if not os.path.exists(full_path):
        # Fallback to local dir
        full_path = csv_filename
        
    if not os.path.exists(full_path):
        print(f"❌ Warning: {csv_filename} not found. Skipping.")
        return []
        
    target_df = pd.read_csv(full_path)
    filename_to_idx = {name: idx for idx, name in enumerate(master_df['filename'])}
    
    indices = []
    for fname in target_df['filename']:
        if fname in filename_to_idx:
            indices.append(filename_to_idx[fname])
    print(f"Found {len(indices)} samples.")
    return indices

def get_species_groups(df, species_to_id):
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

# ================= METRIC FUNCTIONS =================

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
    
    if not all_labels: return {}
    acc_total = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for p, l in zip(all_preds, all_labels):
        grp = species_groups.get(l, "Unknown")
        group_stats[grp]["total"] += 1
        if p == l: group_stats[grp]["correct"] += 1
            
    group_acc = {g: s["correct"]/s["total"] for g, s in group_stats.items() if s["total"] > 0}
    return {"Accuracy Total": acc_total, "F1-Score Macro": f1, "Group Accuracy": group_acc}

def find_optimal_thresholds(model, val_loader):
    model.eval()
    all_probs, all_labels = [], []
    print("\n>>> Finding Optimal Thresholds...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch['image'].to(DEVICE)
            if batch['has_trait_labels'].any():
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                probs = torch.sigmoid(outputs['traits'])
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch['trait_labels'].cpu().numpy())

    if not all_probs: return [0.5] * NUM_TRAITS
    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    optimal_thresholds = []
    for i in range(NUM_TRAITS):
        valid_mask = y_true[:, i] != -1.0
        if not valid_mask.any():
            optimal_thresholds.append(0.5); continue
            
        precisions, recalls, thresholds = precision_recall_curve(y_true[valid_mask, i], y_probs[valid_mask, i])
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        best_idx = np.argmax(f1_scores)
        best_th = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        optimal_thresholds.append(best_th)
        print(f"   Trait {TRAIT_NAMES[i]}: Thresh={best_th:.4f}")
    return optimal_thresholds

def evaluate_traits(model, loader, split_name, thresholds=None):
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
    
    results = {}; aps = []
    use_thresholds = thresholds if thresholds else [0.5] * NUM_TRAITS
    
    for i in range(NUM_TRAITS):
        if len(all_labels[i]) > 0:
            y_true = np.array(all_labels[i])
            y_score = np.array(all_preds[i])
            
            ap = average_precision_score(y_true, y_score)
            results[f"AP_{TRAIT_NAMES[i]}"] = ap * 100
            aps.append(ap)
            
            f1_05 = f1_score(y_true, (y_score >= 0.5).astype(int))
            results[f"F1@0.5_{TRAIT_NAMES[i]}"] = f1_05 * 100
            
            f1_opt = f1_score(y_true, (y_score >= use_thresholds[i]).astype(int))
            results[f"F1@Opt_{TRAIT_NAMES[i]}"] = f1_opt * 100
        else:
            results[f"AP_{TRAIT_NAMES[i]}"] = "N/A"
            
    if aps: results["mAP"] = np.mean(aps) * 100
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
                
                # Prototype Head output is logits -> argmax gives class
                preds = outputs['segmentation'].argmax(1)
                
                preds_np = preds.cpu().numpy().flatten()
                targets_np = targets.cpu().numpy().flatten()
                total_cm += confusion_matrix(targets_np, preds_np, labels=range(NUM_SEG_CLASSES))
    
    results = {}; ious = []
    for c in range(NUM_SEG_CLASSES):
        tp = total_cm[c, c]
        union = total_cm[c, :].sum() + total_cm[:, c].sum() - tp
        iou = tp / union if union > 0 else 0
        results[SEG_CLASS_NAMES[c]] = iou * 100
        ious.append(iou)
    results["mIoU"] = np.mean(ious) * 100
    return results

# ================= MAIN =================
def main():
    print(f"Loading Master Data: {MASTER_CSV}")
    master_df = pd.read_csv(MASTER_CSV, low_memory=False)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    full_dataset = FishVistaMultiTaskDataset(csv_file=MASTER_CSV, data_root=DATA_ROOT, transform=transform)
    num_species = len(full_dataset.species_to_id)
    species_groups = get_species_groups(master_df, full_dataset.species_to_id)
    
    print(f"Loading Checkpoint: {CHECKPOINT_PATH}")
    # Initialize ProtoSwin Model
    model = AttributeAwareProtoSwin_Detached(
        num_species=num_species, 
        num_traits=NUM_TRAITS, 
        num_seg_classes=NUM_SEG_CLASSES,
        model_name='swin_base_patch4_window7_224.ms_in22k'
    ).to(DEVICE)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_results = {}
    
    # 1. Classification
    indices = load_indices_from_csv(SPLIT_FILES["cls_test"], master_df)
    if indices:
        loader = DataLoader(Subset(full_dataset, indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        final_results["Classification"] = evaluate_classification(model, loader, species_groups)
        
    # 2. Optimal Thresholds
    val_indices = load_indices_from_csv(SPLIT_FILES["trait_val"], master_df)
    optimal_thresholds = None
    if val_indices:
        val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        optimal_thresholds = find_optimal_thresholds(model, val_loader)

    # 3. Traits
    for key, name in [("trait_insp", "In-Species"), ("trait_ood", "OOD"), ("trait_manual", "Manual")]:
        indices = load_indices_from_csv(SPLIT_FILES[key], master_df)
        if indices:
            loader = DataLoader(Subset(full_dataset, indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
            final_results[f"Trait ID ({name})"] = evaluate_traits(model, loader, name, thresholds=optimal_thresholds)
            
    # 4. Segmentation
    indices = load_indices_from_csv(SPLIT_FILES["seg_test"], master_df)
    if indices:
        loader = DataLoader(Subset(full_dataset, indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        final_results["Segmentation"] = evaluate_segmentation(model, loader)
        
    # Save
    json_path = os.path.join(OUTPUT_DIR, "all_metrics_proto.json")
    def convert(o): return o.item() if isinstance(o, np.generic) else o
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=4, default=convert)
        
    print(f"\n✅ All Done! Results saved to {json_path}")

if __name__ == "__main__":
    main()