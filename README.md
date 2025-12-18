
## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision pandas numpy tqdm scikit-learn torchmetrics opencv-python matplotlib
    ```

### 1. Training (Model

```bash
python trainwsegv3.py
```
*   **Config:** Batch size 16, LR 1e-4, 50 Epochs.
*   **Output:** Checkpoints will be saved to `checkpoints_weighted_seg_v3_full/`.

### 2. Evaluation
To evaluate the trained model on all tasks (Classification, Trait ID, Segmentation) and generate the final metrics JSON:

```bash
python evaluation_final.py
```
*   **Note:** Ensure you update the `CHECKPOINT_PATH` in `evaluation_final.py` to point to your best model (e.g., `best_model_weighted_seg_v3_full.pth`).



