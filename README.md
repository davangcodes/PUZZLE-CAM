🧩 Puzzle-CAM: Multi-Label Verb Classification on CholecT45
This project implements a Puzzle-CAM-enhanced ResNet50 model for multi-label verb classification in laparoscopic surgical frames using the CholecT45 dataset. The model aims to improve weakly-supervised attention by enforcing region-level consistency through Puzzle-CAM and data-driven augmentations.

🚀 Key Features
✅ Puzzle-CAM Loss: Encourages consistency between full image predictions and 2×2 patches using KL divergence.

✅ CLAHE Preprocessing: Enhances surgical visibility under varying lighting.

✅ W&B Integration: Tracks loss, mAP, and training curves live.

✅ Weighted BCE Loss: Handles imbalanced verb labels via class weights.

✅ OneCycleLR Scheduler: Ensures smooth convergence during training.

📁 Dataset Format (JSON)
The dataset is a .json file containing base64-encoded images and corresponding multi-label verb annotations.

Each item must include:

json
Copy
Edit
{
  "image": "<base64 string>",
  "verb_labels": [0, 1, 0, ..., 0]  // length = 10 (multi-label binary)
}
🏗️ Project Structure
bash
Copy
Edit
├── main.py                    # Training script (Puzzle-CAM + ResNet50)
├── weights/                  # Directory to save best model
├── instrument_verb_train.json
├── instrument_verb_val.json
└── README.md
📦 Setup Instructions
1. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
If using GPU: ensure CUDA is properly configured with PyTorch.

2. Login to Weights & Biases
bash
Copy
Edit
wandb login <your-api-key>
3. Run Training
bash
Copy
Edit
python main.py
📊 Evaluation Metrics
The model reports:

🔹 Binary Cross-Entropy Loss (BCEWithLogitsLoss)

🔹 Puzzle-CAM KL Divergence Loss

🔹 Validation Mean Average Precision (mAP) across all 10 verbs

🧠 Model Architecture
Backbone: ResNet50 (pretrained, excluding final FC layer)

Classifier Head:

Linear(2048 → 256)

ReLU + Dropout(0.3)

Linear(256 → 10)

Loss Function:

BCEWithLogitsLoss + 0.4 × PuzzleLoss

🧪 Evaluation Function
After each epoch:

Evaluate on validation set

Save best_model.pth if mAP improves

Log all metrics to W&B
