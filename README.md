Here is a complete `README.md` file incorporating your installation and execution steps, along with a comprehensive overview and citation section based on the provided research paper. 

***

# AquaDeHi: Detached Hierarchical Multi-Task Learning for Fine-Grained Aquatic Species Recognition

[cite_start]This repository contains the official implementation of **AquaDeHi**, a novel framework designed for the fine-grained recognition of aquatic species, specifically addressing the challenges of long-tailed distributions where ultra-rare species are poorly represented[cite: 2, 3, 7]. 

[cite_start]AquaDeHi aligns network optimization with the natural biological hierarchy of aquatic species by breaking down recognition into three interrelated sub-tasks: semantic segmentation of anatomical parts, morphological trait prediction, and species classification[cite: 8, 33]. 

## ✨ Key Features
* [cite_start]**Detached Hierarchical Architecture:** Enforces a unidirectional information flow (Segmentation → Traits → Species) and applies gradient detachment at task boundaries to prevent high-level classification objectives from corrupting lower-level morphological representations[cite: 9, 55, 56].
* [cite_start]**Prototype-Based Segmentation:** Replaces conventional decoders with learnable prototypes and orthogonality regularization, enabling compact, disentangled, and interpretable delineations of subtle anatomical structures[cite: 57, 62, 175].
* [cite_start]**Dynamic Task Balancing:** Employs homoscedastic uncertainty weighting to dynamically balance the contributions of dense tasks (segmentation) with higher-level tasks during training[cite: 58].
* [cite_start]**High Performance:** Achieves state-of-the-art results on the large-scale Fish-Vista dataset, including 97.7% mAP for unseen trait identification and a 13.4% accuracy improvement for ultra-rare species compared to strong baselines[cite: 12, 13].

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/phucngvinuni/MTL
    cd MTL
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision pandas numpy tqdm scikit-learn torchmetrics opencv-python matplotlib
    ```

## 🚀 Usage

### 1. Training (Model)

To train the AquaDeHi model from scratch:

```bash
python train_detached.py
```
**Config:** Batch size 32 (default evaluated in paper), Learning Rate 1e-4 using AdamW, 50 Epochs with Cosine Annealing.
* **Output:** Checkpoints will be automatically saved to the `checkpoints_detached/` directory.

### 2. Evaluation

To evaluate the trained model across all three hierarchical tasks (Species Classification, Trait Identification, and Semantic Segmentation) and generate the final metrics JSON:

```bash
python evaluationdetached.py
```
* **Note:** Ensure you update the `CHECKPOINT_PATH` inside `evaluationdetached.py` to point to your best saved model (e.g., `best_model.pth`) before running.

## 💾 Checkpoints
Pre-trained model checkpoints can be found at: `[Insert Link to HuggingFace / Google Drive / Release Assets Here]`

## 📖 Citation
If you find this code or our research helpful in your work, please cite our paper:

```bibtex
abc
```

***
