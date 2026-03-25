"""
TRAINING SCRIPT: DETACHED (NO BACKPROP BETWEEN HEADS)
===============================================================================
- Model: AttributeAwareProtoSwin_Detached
- Logic: Seg updates SegHead. Traits updates TraitHead. Species updates SpeciesHead.
         (They only share the Backbone updates).
"""

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import os
import json
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics import JaccardIndex
import warnings
warnings.filterwarnings("ignore")

# ================= IMPORTS =================
try:
    from multitask_dataset_fishvista import get_datasets
    # IMPORT MODEL DETACHED
    from multitask_model_detached import AttributeAwareProtoSwin_Detached
except ImportError:
    print("ERROR: Could not import 'multitask_model_detached.py'")
    exit()

# ================= CONFIGURATION =================
DATA_ROOT = "/home/ubuntu/23phuc.nh/fish-vista"
MASTER_CSV = os.path.join(DATA_ROOT, "master_dataset.csv")

# Tên thư mục riêng cho bản so sánh này
CHECKPOINT_DIR = "checkpoints_detached_compare" 
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 5
PATIENCE = 15
NUM_WORKERS = 4
IMG_SIZE = 224

PROTO_TEMP = 0.1 
ORTHO_WEIGHT = 0.5

NUM_TRAITS = 4
NUM_SEG_CLASSES = 10

WEIGHTS_PATH = "segmentation_class_weights.json"

# Logging config
LOG_DIR = os.path.join(CHECKPOINT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"train_detached_{RUN_ID}.log")


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

LOGGER = setup_logger(LOG_FILE)

# ================= UTILS & LOSS =================
with open(WEIGHTS_PATH, "r") as f:
    weight_data = json.load(f)
raw_weights = torch.tensor(weight_data["weights"]["inverse_frequency"], dtype=torch.float32)
norm_weights = raw_weights / raw_weights.mean()
seg_class_weights = torch.clamp(norm_weights, min=1.0, max=30.0).to(DEVICE)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        logits = logits.float()
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

def prototype_orthogonality_loss(prototypes):
    p_norm = F.normalize(prototypes, p=2, dim=1)
    sim_matrix = torch.mm(p_norm, p_norm.t())
    identity = torch.eye(sim_matrix.shape[0], device=prototypes.device)
    loss = (sim_matrix - identity).pow(2).mean()
    return loss

class UncertaintyWeighting(nn.Module):
    def __init__(self, num_tasks=3, clamp_min=-5.0, clamp_max=5.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    def forward(self, losses, masks):
        total = 0.0
        for i, (loss_i, valid) in enumerate(zip(losses, masks)):
            if valid:
                lv = torch.clamp(self.log_vars[i], min=self.clamp_min, max=self.clamp_max)
                precision = torch.exp(-lv)
                total = total + precision * loss_i + lv
        return total

# ================= TRAINING LOOP =================
def train_one_epoch(
    model, dataloader, species_criterion, trait_criterion,
    seg_ce_criterion, seg_dice_criterion, uncertainty_weighting,
    optimizer, scaler, epoch, logger,
):
    model.train()
    running_total = 0.0
    running_ortho = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
    for batch in pbar:
        images = batch["image"].to(DEVICE)
        species_labels = batch["species_label"].to(DEVICE)
        trait_labels = batch["trait_labels"].to(DEVICE)
        seg_labels = batch["segmentation_mask"].to(DEVICE)

        has_species = batch["has_species_label"].to(DEVICE)
        has_traits = batch["has_trait_labels"].to(DEVICE)
        has_mask = batch["has_mask"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            # Forward (Đã có .detach() bên trong)
            outputs = model(images)

            species_loss = torch.tensor(0.0, device=DEVICE)
            if has_species.any():
                species_loss = species_criterion(outputs["species"][has_species], species_labels[has_species])

            trait_loss = torch.tensor(0.0, device=DEVICE)
            if has_traits.any():
                valid_mask = (trait_labels != -1.0)
                raw_loss = trait_criterion(outputs["traits"], trait_labels)
                masked_loss = raw_loss * valid_mask.float()
                if valid_mask.sum() > 0:
                    trait_loss = masked_loss.sum() / valid_mask.sum()

            seg_loss = torch.tensor(0.0, device=DEVICE)
            if has_mask.any():
                seg_logits = outputs["segmentation"][has_mask].float()
                seg_targets = seg_labels[has_mask]
                seg_loss = seg_ce_criterion(seg_logits, seg_targets) + seg_dice_criterion(seg_logits, seg_targets)

            ortho_loss = prototype_orthogonality_loss(model.proto_head.prototypes)

            # Weighting
            task_loss = uncertainty_weighting(
                losses=[species_loss, trait_loss, seg_loss],
                masks=[has_species.any(), has_traits.any(), has_mask.any()],
            )
            
            total_loss = task_loss + ORTHO_WEIGHT * ortho_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(uncertainty_weighting.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        running_total += total_loss.item() * bs
        running_ortho += ortho_loss.item() * bs
        
        curr_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "lr": f"{curr_lr:.2e}"})

    count = len(dataloader.dataset)
    avg_total = running_total / count
    log_vars = uncertainty_weighting.log_vars.detach().cpu().numpy().tolist()
    logger.info(
        f"Epoch {epoch+1} TRAIN | Total={avg_total:.4f} | Ortho={running_ortho/count:.4f} | "
        f"LogVars={log_vars}"
    )
    return avg_total

# ================= EVALUATION LOOP =================
@torch.no_grad()
def evaluate(model, dataloader, logger):
    model.eval()
    total_loss = 0.0
    count = 0
    jaccard = JaccardIndex(task="multiclass", num_classes=NUM_SEG_CLASSES, ignore_index=0).to(DEVICE)
    has_seg_val = False
    criterion_val = nn.CrossEntropyLoss(ignore_index=-1)

    for batch in tqdm(dataloader, desc="Validating", leave=True):
        images = batch["image"].to(DEVICE)
        with autocast():
            outputs = model(images)
            if batch["has_species_label"].any():
                mask_species = batch["has_species_label"].to(DEVICE)
                target_species = batch["species_label"].to(DEVICE)
                l = criterion_val(outputs["species"][mask_species], target_species[mask_species])
                total_loss += l.item() * images.size(0)
            if batch["has_mask"].any():
                mask_seg = batch["has_mask"].to(DEVICE)
                target_seg = batch["segmentation_mask"].to(DEVICE)
                preds = outputs["segmentation"][mask_seg].argmax(dim=1)
                jaccard.update(preds, target_seg[mask_seg])
                has_seg_val = True
        count += images.size(0)

    val_loss = total_loss / count if count > 0 else 0.0
    miou = 0.0
    if has_seg_val:
        try:
            miou = jaccard.compute().item()
        except Exception:
            miou = 0.0
    logger.info(f"VALID | Val Loss={val_loss:.4f} | Val mIoU={miou:.4f}")
    return val_loss, miou

# ================= MAIN =================
def main():
    LOGGER.info(f"Run ID: {RUN_ID}")
    LOGGER.info("Model: AttributeAwareProtoSwin_Detached (STOP GRADIENT)")
    
    # Init Data
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
    train_subset, val_subset, _, species_to_id = get_datasets(MASTER_CSV, data_root=DATA_ROOT)
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Init Model (Detached Version)
    model = AttributeAwareProtoSwin_Detached(
        num_species=len(species_to_id),
        num_traits=NUM_TRAITS,
        num_seg_classes=NUM_SEG_CLASSES,
        model_name='swin_base_patch4_window7_224.ms_in22k'
    ).to(DEVICE)

    if hasattr(model, 'proto_head'):
        model.proto_head.temp = PROTO_TEMP

    uncertainty_weighting = UncertaintyWeighting(num_tasks=3).to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(uncertainty_weighting.parameters()),
        lr=LEARNING_RATE, weight_decay=0.01,
    )
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    scheduler_main = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_main], milestones=[WARMUP_EPOCHS])
    scaler = GradScaler()

    # Criteria
    species_criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
    trait_criterion = nn.BCEWithLogitsLoss(reduction="none")
    seg_ce_criterion = nn.CrossEntropyLoss(weight=seg_class_weights, ignore_index=-1)
    seg_dice_criterion = DiceLoss()

    best_miou = 0.0
    patience_counter = 0

    LOGGER.info("STARTING TRAINING LOOP")
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, species_criterion, trait_criterion,
            seg_ce_criterion, seg_dice_criterion, uncertainty_weighting,
            optimizer, scaler, epoch, LOGGER
        )
        val_loss, val_miou = evaluate(model, val_loader, LOGGER)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        LOGGER.info(f"Epoch {epoch+1} finished in {epoch_time:.1f}s")

        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            save_path = os.path.join(CHECKPOINT_DIR, "best_model_detached.pth")
            torch.save({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_miou": best_miou,
            }, save_path)
            LOGGER.info(f"Saved BEST model (mIoU: {best_miou:.4f})")
        else:
            patience_counter += 1
            LOGGER.info(f"No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            break

if __name__ == "__main__":
    main()