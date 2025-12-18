"""
TRAINING SCRIPT V3 - FULL DATASET (THE SOTA BEATER)
===================================================
Configuration:
1. Dataset: FULL Data (Fixed CSV), ~4.4x more segmentation data than V2.
2. Loss Strategy: Weighted CrossEntropy (Clamped @ 30.0) + Dice Loss.
3. Stability: LR Warmup (5 epochs) + Gradient Clipping (max_norm=1.0).
4. Augmentation: ColorJitter (Robustness) + Resize/Normalize.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics import JaccardIndex
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from multitask_dataset_fishvista import get_datasets
from multitask_model import MultiTaskSwinTransformer

# ================= CONFIGURATION =================
DATA_ROOT = "/home/ubuntu/23phuc.nh/fish-vista"
# Đảm bảo đây là file CSV đã fix (chứa 4312 ảnh train seg)
MASTER_CSV = os.path.join(DATA_ROOT, "master_dataset.csv") 

CHECKPOINT_DIR = "checkpoints_weighted_seg_v3_full"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Update device theo yêu cầu của bạn
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 5  # Giúp model làm quen với trọng số lớn
PATIENCE = 15      # Kiên nhẫn hơn vì dữ liệu nhiều hơn
NUM_WORKERS = 4
IMG_SIZE = 224

NUM_TRAITS = 4
NUM_SEG_CLASSES = 10

# ================= LOAD & CONFIG WEIGHTS =================
WEIGHTS_PATH = "segmentation_class_weights.json" 
with open(WEIGHTS_PATH, 'r') as f:
    weight_data = json.load(f)

print("="*80)
print("CONFIGURING CLASS WEIGHTS (V3 STRATEGY)")
# 1. Raw Inverse Frequency
raw_weights = torch.tensor(weight_data['weights']['inverse_frequency'], dtype=torch.float32)
# 2. Normalize (Mean = 1.0)
norm_weights = raw_weights / raw_weights.mean()
# 3. Clamp (Min 1.0, Max 30.0) - THE SECRET SAUCE
seg_class_weights = torch.clamp(norm_weights, min=1.0, max=30.0).to(DEVICE)

print("Weights applied:")
class_names = weight_data['class_names']
for i, w in enumerate(seg_class_weights):
    print(f"  Class {i} ({class_names[str(i)]:<20}): {w:.4f}x")
print("="*80)

# ================= DICE LOSS =================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        # Tính toán trên Float32 để tránh tràn số
        logits = logits.float()
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

# ================= TRAINING LOOP =================
def train_one_epoch(model, dataloader, species_criterion, trait_criterion, seg_ce_criterion, 
                     seg_dice_criterion, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0
    meters = {'spe': 0.0, 'tra': 0.0, 'seg_ce': 0.0, 'seg_dice': 0.0}
    count = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in pbar:
        images = batch['image'].to(DEVICE)
        species_labels = batch['species_label'].to(DEVICE)
        trait_labels = batch['trait_labels'].to(DEVICE)
        seg_labels = batch['segmentation_mask'].to(DEVICE)
        
        # Cờ báo hiệu dữ liệu hợp lệ
        has_species = batch['has_species_label'].to(DEVICE)
        has_traits = batch['has_trait_labels'].to(DEVICE)
        has_mask = batch['has_mask'].to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            
            # --- 1. Species Loss ---
            species_loss = torch.tensor(0.0, device=DEVICE)
            if has_species.any():
                species_loss = species_criterion(
                    outputs['species'][has_species], 
                    species_labels[has_species]
                )
            
            # --- 2. Trait Loss ---
            trait_loss = torch.tensor(0.0, device=DEVICE)
            if has_traits.any():
                valid_mask = (trait_labels != -1.0) # Bỏ qua nhãn thiếu (-1)
                raw_loss = trait_criterion(outputs['traits'], trait_labels)
                masked_loss = raw_loss * valid_mask.float()
                # Chỉ chia cho số lượng phần tử hợp lệ để tránh loss bị nhỏ đi
                if valid_mask.sum() > 0:
                    trait_loss = masked_loss.sum() / valid_mask.sum()
            
            # --- 3. Segmentation Loss (Weighted V2) ---
            seg_ce_loss = torch.tensor(0.0, device=DEVICE)
            seg_dice_loss = torch.tensor(0.0, device=DEVICE)
            
            if has_mask.any():
                # Lấy output của những ảnh có mask
                seg_logits = outputs['segmentation'][has_mask].float() # Force Float32
                seg_targets = seg_labels[has_mask]
                
                seg_ce_loss = seg_ce_criterion(seg_logits, seg_targets)
                seg_dice_loss = seg_dice_criterion(seg_logits, seg_targets)
            
            # Tổng hợp Loss
            total_loss = species_loss + trait_loss + seg_ce_loss + seg_dice_loss

        # Backpropagation
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Kẹp Gradient
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        bs = images.size(0)
        running_loss += total_loss.item() * bs
        meters['spe'] += species_loss.item() * bs
        meters['tra'] += trait_loss.item() * bs
        meters['seg_ce'] += seg_ce_loss.item() * bs
        count += bs
        
        # Show LR
        curr_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'seg': f'{(seg_ce_loss+seg_dice_loss).item():.4f}',
            'lr': f'{curr_lr:.2e}'
        })
    
    return running_loss / count

# ================= EVALUATION LOOP =================
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    count = 0
    
    # Metric chính để lưu model: mIoU
    jaccard = JaccardIndex(task='multiclass', num_classes=NUM_SEG_CLASSES, ignore_index=0).to(DEVICE)
    has_seg_val = False
    
    criterion_val = nn.CrossEntropyLoss(ignore_index=-1)
    
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch['image'].to(DEVICE)
        
        with autocast():
            outputs = model(images)
            
            # --- FIX: Ensure tensors are on same device for indexing ---
            if batch['has_species_label'].any():
                # Move everything to DEVICE first
                mask_species = batch['has_species_label'].to(DEVICE)
                target_species = batch['species_label'].to(DEVICE)
                
                l = criterion_val(
                    outputs['species'][mask_species], 
                    target_species[mask_species]
                )
                total_loss += l.item() * images.size(0)
            
            # --- Update metric segmentation ---
            if batch['has_mask'].any():
                # Move everything to DEVICE first
                mask_seg = batch['has_mask'].to(DEVICE)
                target_seg = batch['segmentation_mask'].to(DEVICE)
                
                preds = outputs['segmentation'][mask_seg].argmax(dim=1)
                jaccard.update(preds, target_seg[mask_seg])
                has_seg_val = True
                
        count += images.size(0)
    
    val_loss = total_loss / count if count > 0 else 0.0
    
    if has_seg_val:
        try:
            miou = jaccard.compute().item()
        except:
            miou = 0.0
    else:
        miou = 0.0
        
    return val_loss, miou

# ================= MAIN =================
def main():
    print(f"Device: {DEVICE}")
    print("Loading Datasets...")
    
    # Transform: V2 có ColorJitter để tăng tính tổng quát
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get Datasets
    train_subset, val_subset, _, species_to_id = get_datasets(MASTER_CSV, data_root=DATA_ROOT)
    
    # Áp dụng transform
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform
    
    print(f"Num Species: {len(species_to_id)}")
    print(f"Train Samples: {len(train_subset)}")
    print(f"Val Samples:   {len(val_subset)}")
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # Init Model
    model = MultiTaskSwinTransformer(
        num_species=len(species_to_id),
        num_traits=NUM_TRAITS,
        num_seg_classes=NUM_SEG_CLASSES
    ).to(DEVICE)
    
    # Optimizer & Scheduler (Warmup + Cosine)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    scheduler_main = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_main], milestones=[WARMUP_EPOCHS])
    
    scaler = GradScaler()
    
    # Losses
    species_criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
    trait_criterion = nn.BCEWithLogitsLoss(reduction='none')
    seg_ce_criterion = nn.CrossEntropyLoss(weight=seg_class_weights)
    seg_dice_criterion = DiceLoss()
    
    # Training Loop
    best_miou = 0.0
    patience_counter = 0
    
    print("\nSTARTING TRAINING LOOP")
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, species_criterion, trait_criterion, 
            seg_ce_criterion, seg_dice_criterion, optimizer, scaler, epoch
        )
        
        # Evaluate
        val_loss, val_miou = evaluate(model, val_loader)
        
        # Step LR
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val mIoU={val_miou:.4f}")
        
        # Checkpoint: Save based on mIoU (ưu tiên Segmentation)
        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            
            save_path = os.path.join(CHECKPOINT_DIR, "best_model_weighted_seg_v3_full.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
            }, save_path)
            print(f"  ✓ Saved BEST model to {save_path} (mIoU: {best_miou:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("TRAINING COMPLETE.")

if __name__ == "__main__":
    main()