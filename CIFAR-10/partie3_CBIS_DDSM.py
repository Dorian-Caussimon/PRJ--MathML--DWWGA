"""
============================================================
  PROJET SM604 — PARTIE 3 : Application au diagnostic médical
  Détection de cancers du sein (CBIS-DDSM)
  Réseaux de Neurones Convolutifs — EFREI Paris, 2025-2026
============================================================

Ce fichier implémente intégralement la Partie 3 du projet :

  ── Section A : Chargement & prétraitement ────────────────
    A1. Lecture du CSV et extraction des étiquettes binaires
        (BENIGN / BENIGN_WITHOUT_CALLBACK  →  0
         MALIGNANT                          →  1)
    A2. Chargement des images DICOM ou PNG,
        redimensionnement en 128×128 (ou 224×224)
    A3. Normalisation des niveaux de gris
    A4. Gestion du déséquilibre des classes

  ── Section B : Modèles de classification ─────────────────
    B1. CNN PyTorch adapté à la classification binaire
        Architecture : Conv→Conv→Pool→Conv→Pool→Conv→Flatten→FC→Sigmoid
    B2. Entraînement avec pondération des classes (class_weight)
    B3. Évaluation : accuracy, matrice de confusion,
        sensibilité (recall), spécificité, AUC-ROC

  ── Section C : Analyse médicale ──────────────────────────
    C1. Matrice de confusion avec focus sur les faux négatifs
    C2. Courbe ROC et choix du seuil de décision
    C3. Discussion : importance clinique des faux négatifs

INSTRUCTIONS :
  pip install numpy matplotlib torch torchvision scikit-learn pillow pydicom
  python partie3_CBIS_DDSM.py --data_dir /chemin/vers/CBIS-DDSM

Structure attendue du répertoire CBIS-DDSM :
  CBIS-DDSM/
  ├── mass_case_description_train_set.csv
  ├── mass_case_description_test_set.csv
  └── jpeg/   (ou dicom/)
      ├── Mass-Training_.../
      │   └── *.jpg  (ou *.dcm)
      └── ...

Note : si les images DICOM ne sont pas disponibles, le script peut aussi
fonctionner avec des fichiers JPEG/PNG téléchargés depuis le TCIA.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import argparse
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

# ── PyTorch ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} disponible | Device : {device}")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠  PyTorch non installé → pip install torch torchvision")

# ── PIL (lecture images) ──────────────────────────────────────────────────────
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠  Pillow non installé → pip install pillow")

# ── pydicom (lecture DICOM) ───────────────────────────────────────────────────
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("ℹ  pydicom non installé (optionnel) → pip install pydicom")

# ── sklearn ───────────────────────────────────────────────────────────────────
try:
    from sklearn.metrics import (roc_auc_score, roc_curve,
                                  confusion_matrix, classification_report)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠  scikit-learn non installé → pip install scikit-learn")

import csv


# ═════════════════════════════════════════════════════════════════════════════
# A. CHARGEMENT ET PRÉTRAITEMENT
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# A1. Lecture du CSV et extraction des étiquettes
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_labels(csv_path):
    """
    Lit le fichier mass_case_description_train_set.csv (ou test_set.csv)
    et extrait les chemins d'images + étiquettes binaires.

    Règle de binarisation (colonne 'pathology') :
        BENIGN                 →  0  (bénin)
        BENIGN_WITHOUT_CALLBACK→  0  (bénin sans rappel)
        MALIGNANT              →  1  (malin / cancer)

    Le CSV contient typiquement les colonnes :
        patient_id, breast_density, side, image_view,
        abnormality_id, abnormality_type, mass shape,
        mass margins, assessment, pathology, subtlety,
        image_file_path, cropped_image_file_path, ROI_mask_file_path

    Retourne :
        records : liste de dict {'path': str, 'label': int, 'pathology': str}
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    records = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pathology = row.get('pathology', '').strip().upper()
            if pathology == '':
                continue   # ligne incomplète

            # Binarisation
            if 'MALIGNANT' in pathology:
                label = 1
            else:
                label = 0   # BENIGN ou BENIGN_WITHOUT_CALLBACK

            # Chemin de l'image (colonne 'image file path' ou 'image_file_path')
            img_path = (row.get('image file path') or
                        row.get('image_file_path') or
                        row.get('cropped image file path') or
                        row.get('cropped_image_file_path') or '')
            img_path = img_path.strip()

            records.append({
                'path'      : img_path,
                'label'     : label,
                'pathology' : pathology
            })

    n_total   = len(records)
    n_malin   = sum(r['label'] == 1 for r in records)
    n_benin   = n_total - n_malin
    print(f"  ✓ CSV chargé : {n_total} cas  |  Bénins : {n_benin}  |  Malins : {n_malin}")
    print(f"  Ratio malin/bénin : {n_malin/n_benin:.2f}  "
          f"(déséquilibre {'fort' if n_malin/n_benin < 0.5 else 'modéré'})")
    return records


def resolve_image_path(record_path, data_dir):
    """
    Cherche l'image sur le disque à partir du chemin indiqué dans le CSV.

    La structure des chemins CBIS-DDSM peut varier selon la source de
    téléchargement (TCIA, Kaggle, etc.). Cette fonction essaie plusieurs
    combinaisons pour maximiser les chances de trouver le fichier.

    Retourne le chemin absolu s'il existe, sinon None.
    """
    # 1) Chemin absolu tel quel
    if os.path.isfile(record_path):
        return record_path

    # 2) Concaténation avec data_dir
    candidate = os.path.join(data_dir, record_path)
    if os.path.isfile(candidate):
        return candidate

    # 3) Juste le nom de fichier dans data_dir (recherche récursive)
    basename = os.path.basename(record_path)
    for ext in ['', '.dcm', '.jpg', '.jpeg', '.png']:
        pattern = os.path.join(data_dir, '**', basename + ext)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]

    return None   # non trouvé


# ─────────────────────────────────────────────────────────────────────────────
# A2. Lecture et redimensionnement des images
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path, target_size=128):
    """
    Charge une image (JPEG, PNG ou DICOM) et la redimensionne en target_size×target_size.

    Pourquoi redimensionner ?
    - Les mammographies CBIS-DDSM font parfois 4000×3000 pixels.
    - Garder cette résolution exploirait une quantité de RAM démesurée
      (une seule image = 12 Mo en float32) et ralentirait l'entraînement.
    - 128×128 est un compromis acceptable entre détail clinique et efficacité.
    - 224×224 est utilisé pour le transfer learning (ResNet, EfficientNet…).

    Retourne :
        img_array : np.array de forme (target_size, target_size) float32 ∈ [0, 1]
        ou None en cas d'échec de lecture.
    """
    try:
        # ── DICOM ──────────────────────────────────────────────────────────────
        if path.lower().endswith('.dcm'):
            if not DICOM_AVAILABLE:
                raise RuntimeError("pydicom requis pour lire les DICOM")
            ds = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(np.float32)
            # Normalisation : certaines mammographies ont des valeurs > 255
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            img = Image.fromarray((arr * 255).astype(np.uint8), mode='L')

        # ── JPEG / PNG ────────────────────────────────────────────────────────
        else:
            img = Image.open(path).convert('L')   # niveaux de gris

        # Redimensionnement
        img = img.resize((target_size, target_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0   # ∈ [0, 1]
        return arr

    except Exception as e:
        print(f"    ⚠  Impossible de charger {path} : {e}")
        return None


def build_dataset(records, data_dir, target_size=128, max_samples=None):
    """
    Charge toutes les images et retourne des tableaux NumPy.

    Paramètres :
        records     : liste de dict issus de load_csv_labels()
        data_dir    : répertoire racine du dataset
        target_size : taille cible (pixels)
        max_samples : limite optionnelle pour les tests rapides

    Retourne :
        X : (N, target_size, target_size)  float32
        y : (N,)                            int  {0, 1}
    """
    images, labels = [], []
    n_missing = 0

    if max_samples is not None:
        records = records[:max_samples]

    for i, rec in enumerate(records):
        path = resolve_image_path(rec['path'], data_dir)
        if path is None:
            n_missing += 1
            continue

        img = load_image(path, target_size)
        if img is None:
            n_missing += 1
            continue

        images.append(img)
        labels.append(rec['label'])

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(records)} images chargées...", end='\r')

    print(f"\n  ✓ {len(images)} images chargées | {n_missing} non trouvées/invalides")

    if len(images) == 0:
        raise RuntimeError(
            "Aucune image chargée. Vérifiez data_dir et les chemins dans le CSV.\n"
            "Conseil : utilisez --demo pour tester avec des données synthétiques."
        )

    X = np.stack(images, axis=0)   # (N, H, W)
    y = np.array(labels, dtype=np.int64)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# A3. Données synthétiques (mode démo sans dataset réel)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_mammograms(n_benin=400, n_malin=200, target_size=128, seed=42):
    """
    Génère des mammographies synthétiques pour tester le pipeline sans les
    données CBIS-DDSM réelles.

    Simulation simplifiée :
    - Cas bénin  : fond texturé + masse circulaire lisse (bords réguliers)
    - Cas malin  : fond texturé + masse irrégulière (spicules, bords flous)

    ⚠  Ces données NE représentent PAS la réalité clinique.
       Elles servent uniquement à valider le code.
    """
    rng = np.random.RandomState(seed)
    H, W = target_size, target_size
    images, labels = [], []

    def add_texture(arr, rng, scale=0.05):
        """Bruit de fond simulant la densité mammaire."""
        return arr + rng.randn(*arr.shape).astype(np.float32) * scale

    def gaussian_blob(H, W, cy, cx, radius, rng, irregular=False):
        """Masse circulaire (bénigne) ou irrégulière (maligne)."""
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((y - cy)**2 + (x - cx)**2).astype(np.float32)
        if irregular:
            # Perturbation angulaire pour simuler des spicules
            angle = np.arctan2(y - cy, x - cx)
            radius_map = radius + 5 * np.sin(4 * angle + rng.rand() * np.pi)
            mask = (dist < radius_map).astype(np.float32)
        else:
            mask = np.exp(-dist**2 / (2 * (radius * 0.6)**2))
        return mask * 0.4

    for _ in range(n_benin):
        img = rng.uniform(0.1, 0.3, (H, W)).astype(np.float32)
        img = add_texture(img, rng, scale=0.04)
        cy, cx = rng.randint(H//4, 3*H//4), rng.randint(W//4, 3*W//4)
        r = rng.randint(8, 20)
        img += gaussian_blob(H, W, cy, cx, r, rng, irregular=False)
        img = np.clip(img, 0.0, 1.0)
        images.append(img)
        labels.append(0)

    for _ in range(n_malin):
        img = rng.uniform(0.1, 0.35, (H, W)).astype(np.float32)
        img = add_texture(img, rng, scale=0.06)
        cy, cx = rng.randint(H//4, 3*H//4), rng.randint(W//4, 3*W//4)
        r = rng.randint(10, 22)
        img += gaussian_blob(H, W, cy, cx, r, rng, irregular=True)
        img = np.clip(img, 0.0, 1.0)
        images.append(img)
        labels.append(1)

    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int64)

    # Mélange aléatoire
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def split_train_test(X, y, test_ratio=0.2, seed=42):
    """
    Séparation stratifiée train/test pour conserver la proportion
    bénin/malin dans les deux ensembles.
    """
    rng = np.random.RandomState(seed)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    n_test_0 = max(1, int(len(idx_0) * test_ratio))
    n_test_1 = max(1, int(len(idx_1) * test_ratio))

    rng.shuffle(idx_0)
    rng.shuffle(idx_1)

    test_idx  = np.concatenate([idx_0[:n_test_0],  idx_1[:n_test_1]])
    train_idx = np.concatenate([idx_0[n_test_0:],  idx_1[n_test_1:]])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ─────────────────────────────────────────────────────────────────────────────
# A4. Visualisation des données
# ─────────────────────────────────────────────────────────────────────────────

def visualize_samples_mammography(X, y, n_per_class=6):
    """
    Affiche n_per_class exemples de chaque classe (bénin / malin).

    Les mammographies réelles ont une texture dense et complexe.
    On utilise une colormap 'bone' qui rappelle l'aspect radiologique.
    """
    class_names = ['Bénin (0)', 'Malin (1)']
    fig, axes = plt.subplots(2, n_per_class, figsize=(n_per_class * 2.2, 5))
    fig.suptitle("CBIS-DDSM — Exemples de mammographies", fontsize=13,
                 fontweight='bold')

    for cls in range(2):
        idx = np.where(y == cls)[0][:n_per_class]
        for j, i in enumerate(idx):
            axes[cls, j].imshow(X[i], cmap='bone', vmin=0, vmax=1)
            axes[cls, j].axis('off')
        axes[cls, 0].set_ylabel(class_names[cls], fontsize=10,
                                rotation=0, labelpad=65, va='center')

    plt.tight_layout()
    plt.savefig("mammography_samples.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  ✓ Sauvegardé : mammography_samples.png")


def plot_class_distribution(y_train, y_test):
    """
    Affiche la distribution des classes dans les ensembles train et test.
    Essentiel pour visualiser le déséquilibre bénin/malin.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ['Bénin', 'Malin']
    colors = ['#3B82F6', '#EF4444']

    for ax, (split_y, title) in zip(axes, [(y_train, 'Train'), (y_test, 'Test')]):
        counts = [np.sum(split_y == 0), np.sum(split_y == 1)]
        bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=1.5)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        ax.set_title(f"Distribution des classes — {title}", fontweight='bold')
        ax.set_ylabel("Nombre d'images")
        ax.set_ylim(0, max(counts) * 1.15)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=100)
    plt.show()
    print("  ✓ Sauvegardé : class_distribution.png")


# ═════════════════════════════════════════════════════════════════════════════
# B. ARCHITECTURE CNN POUR CLASSIFICATION BINAIRE
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# B1. Dataset PyTorch
# ─────────────────────────────────────────────────────────────────────────────

class MammographyDataset(Dataset):
    """
    Dataset PyTorch pour les mammographies CBIS-DDSM.

    Gestion de l'augmentation de données à l'entraînement :
    - Flip horizontal et vertical (une mammographie peut être prise des deux côtés)
    - Rotation légère (±10°) : la lésion peut être orientée différemment
    - Ajout de bruit gaussien : robustesse aux artéfacts d'acquisition

    Ces transformations augmentent artificiellement la taille du dataset
    et réduisent le sur-apprentissage, surtout en présence de peu de données.
    """

    def __init__(self, X, y, augment=False, target_size=128):
        """
        Paramètres :
            X       : (N, H, W) float32 ∈ [0, 1]
            y       : (N,)      int64
            augment : True pour l'entraînement, False pour le test
        """
        self.X       = X
        self.y       = y
        self.augment = augment
        self.size    = target_size

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img   = self.X[idx].copy()   # (H, W) float32
        label = int(self.y[idx])

        # ── Augmentation (entraînement uniquement) ────────────────────────────
        if self.augment:
            rng = np.random.RandomState()

            # Flip horizontal (sein gauche / droit)
            if rng.rand() > 0.5:
                img = img[:, ::-1].copy()

            # Flip vertical
            if rng.rand() > 0.5:
                img = img[::-1, :].copy()

            # Rotation légère (±10°) via PIL
            if PIL_AVAILABLE:
                angle = rng.uniform(-10, 10)
                pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
                pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
                img = np.array(pil_img, dtype=np.float32) / 255.0

            # Bruit gaussien léger
            img = img + rng.randn(*img.shape).astype(np.float32) * 0.02
            img = np.clip(img, 0.0, 1.0)

        # Ajout de la dimension canal : (1, H, W) — image en niveaux de gris
        tensor_img = torch.tensor(img[np.newaxis, :, :], dtype=torch.float32)
        return tensor_img, label


# ─────────────────────────────────────────────────────────────────────────────
# B2. Architecture CNN
# ─────────────────────────────────────────────────────────────────────────────

class MammographyCNN(nn.Module):
    """
    CNN pour la détection binaire bénin/malin sur mammographies.

    Architecture inspirée du sujet (section 2.5) et adaptée à la classification
    binaire (sortie 1 neurone avec sigmoid au lieu de 10 avec softmax).

    Pour une entrée de taille (1, 128, 128) :

    Couche           | Sortie         | Description
    ─────────────────┼────────────────┼─────────────────────────────
    Conv2d(1, 32)    | (32,128,128)   | 32 filtres 3×3, détection bords
    BatchNorm + ReLU |                |
    Conv2d(32, 32)   | (32,128,128)   | 32 filtres 3D
    BatchNorm + ReLU |                |
    MaxPool2d(2,2)   | (32, 64, 64)   | Réduction spatiale ×2
    Conv2d(32, 64)   | (64, 64, 64)   | 64 filtres 3D
    BatchNorm + ReLU |                |
    MaxPool2d(2,2)   | (64, 32, 32)   | Réduction spatiale ×2
    Conv2d(64, 64)   | (64, 32, 32)   | 64 filtres 3D
    BatchNorm + ReLU |                |
    MaxPool2d(2,2)   | (64, 16, 16)   | Réduction spatiale ×2
    Flatten          | (16384,)       | Aplatissement
    Linear(16384,256)| (256,)         | Densification
    Dropout(0.5)     |                | Régularisation
    Linear(256, 1)   | (1,)           | Score binaire
    Sigmoid          |                | Probabilité P(malin)

    Pourquoi BatchNorm ?
    - Stabilise l'apprentissage en normalisant les activations couche par couche.
    - Réduit la sensibilité au learning rate.
    - Agit comme régularisateur (réduit le besoin de dropout).

    Pourquoi Dropout (p=0.5) ?
    - Avec peu de données (< 1000 images), le risque d'overfitting est élevé.
    - Dropout désactive aléatoirement 50% des neurones à chaque forward pass.
    """

    def __init__(self, target_size=128):
        super().__init__()

        # Bloc convolutif 1 : 1 → 32 canaux
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)         # → (32, H/2, W/2)
        )

        # Bloc convolutif 2 : 32 → 64 canaux
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)         # → (64, H/4, W/4)
        )

        # Bloc convolutif 3 : 64 → 64 canaux
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)         # → (64, H/8, W/8)
        )

        # Calcul dynamique de la taille après convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, target_size, target_size)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            dummy = self.block3(dummy)
            flat_size = dummy.numel()

        # Couche dense de classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1)
            # ⚠  Pas de Sigmoid ici : on utilise BCEWithLogitsLoss
            #    (numériquement plus stable que Sigmoid + BCELoss)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x   # logits — forme (batch, 1)

    def predict_proba(self, x):
        """Retourne la probabilité P(malin) ∈ [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(1)

    def predict(self, x, threshold=0.5):
        """Prédit la classe (0=bénin, 1=malin) selon un seuil."""
        proba = self.predict_proba(x)
        return (proba >= threshold).long()

    def count_parameters(self):
        """Compte le nombre total de paramètres apprenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# B3. Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y_train):
    """
    Calcule les poids inversement proportionnels à la fréquence de chaque classe.

    Formule : w_c = N_total / (N_classes × N_c)

    Pourquoi pondérer les classes ?
    Le dataset CBIS-DDSM présente un déséquilibre typique : les cas bénins sont
    plus nombreux que les cas malins. Sans pondération, le modèle apprend à
    prédire "bénin" la plupart du temps et atteint une accuracy trompeuse.

    Exemple :
      70% bénins, 30% malins → un modèle "toujours bénin" a 70% d'accuracy
      mais 0% de sensibilité (détecte 0 cancer) : inutile en clinique.

    La pondération force le modèle à accorder plus d'importance aux cas malins.
    """
    n_total  = len(y_train)
    n_malin  = np.sum(y_train == 1)
    n_benin  = np.sum(y_train == 0)
    w_malin  = n_total / (2.0 * n_malin)  if n_malin > 0 else 1.0
    w_benin  = n_total / (2.0 * n_benin)  if n_benin > 0 else 1.0

    print(f"  Poids des classes : bénin = {w_benin:.3f} | malin = {w_malin:.3f}")
    return float(w_benin), float(w_malin)


def train_cnn_mammography(X_train, y_train, X_test, y_test,
                          target_size=128, epochs=30, lr=1e-3, batch_size=32):
    """
    Entraîne le CNN de mammographie.

    Particularités par rapport à CIFAR-10 (Partie 2) :
    1. Classification BINAIRE → BCEWithLogitsLoss au lieu de CrossEntropyLoss
    2. Pondération des classes → pos_weight dans BCEWithLogitsLoss
    3. Augmentation de données → MammographyDataset(augment=True)
    4. Métriques médicales : sensibilité, spécificité en plus de l'accuracy

    Paramètres :
        X_train, y_train : données d'entraînement
        X_test,  y_test  : données de test
        target_size      : taille des images (doit correspondre à celle du CNN)
        epochs           : nombre d'époques
        lr               : taux d'apprentissage (Adam)
        batch_size       : taille des mini-lots

    Retourne :
        model   : MammographyCNN entraîné
        history : dict avec les métriques par époque
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch est requis pour l'entraînement.")

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    train_ds = MammographyDataset(X_train, y_train, augment=True,  target_size=target_size)
    test_ds  = MammographyDataset(X_test,  y_test,  augment=False, target_size=target_size)

    # WeightedRandomSampler : sur-échantillonne les malins à l'entraînement
    w_benin, w_malin = compute_class_weights(y_train)
    sample_weights = np.where(y_train == 1, w_malin, w_benin).astype(np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = MammographyCNN(target_size=target_size).to(device)
    print(f"\n  Architecture CNN Mammographie")
    print(f"  Paramètres apprenables : {model.count_parameters():,}")
    print(f"  Device : {device}")

    # ── Fonction de coût avec pos_weight ─────────────────────────────────────
    # pos_weight = w_malin / w_benin amplifie la pénalité sur les faux négatifs
    pos_weight = torch.tensor([w_malin / w_benin], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimiseur ────────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Réduction du LR si la loss de validation stagne (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc' : [], 'test_acc' : [],
        'sensitivity': [], 'specificity': []
    }

    best_auc = 0.0
    best_state = None

    print(f"\n  {'Époque':>6} | {'L train':>8} | {'L test':>8} | "
          f"{'Acc tr':>7} | {'Acc te':>7} | {'Sensib':>7} | {'Spécif':>7}")
    print("  " + "─" * 68)

    for epoch in range(1, epochs + 1):

        # ── Phase d'entraînement ──────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total   = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().to(device)

            optimizer.zero_grad()
            logits = model(X_batch).squeeze(1)          # (batch,)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            train_correct += (preds == y_batch.long()).sum().item()
            train_total   += len(y_batch)

        train_loss = train_loss_sum / train_total
        train_acc  = train_correct  / train_total

        # ── Phase de test ────────────────────────────────────────────────────
        model.eval()
        test_loss_sum = 0.0
        all_preds, all_labels, all_proba = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().to(device)

                logits = model(X_batch).squeeze(1)
                loss   = criterion(logits, y_batch)
                proba  = torch.sigmoid(logits)

                test_loss_sum += loss.item() * len(y_batch)
                preds = (proba >= 0.5).long()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.long().cpu().numpy())
                all_proba.append(proba.cpu().numpy())

        test_loss = test_loss_sum / len(test_ds)
        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_proba  = np.concatenate(all_proba)

        # Métriques médicales
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        tn = np.sum((all_preds == 0) & (all_labels == 0))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))

        sensitivity  = tp / (tp + fn + 1e-8)   # recall / taux de vrais positifs
        specificity  = tn / (tn + fp + 1e-8)   # taux de vrais négatifs
        test_acc     = (tp + tn) / len(all_labels)

        # AUC (si sklearn disponible)
        if SKLEARN_AVAILABLE and len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_proba)
            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
        else:
            auc = float('nan')

        scheduler.step(test_loss)

        # Enregistrement
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['sensitivity'].append(sensitivity)
        history['specificity'].append(specificity)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6} | {train_loss:>8.4f} | {test_loss:>8.4f} | "
                  f"{train_acc*100:>6.1f}% | {test_acc*100:>6.1f}% | "
                  f"{sensitivity*100:>6.1f}% | {specificity*100:>6.1f}%")

    # Rechargement du meilleur état (par AUC)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  ✓ Meilleur modèle rechargé (AUC = {best_auc:.4f})")

    return model, history


# ═════════════════════════════════════════════════════════════════════════════
# C. ÉVALUATION ET ANALYSE MÉDICALE
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, batch_size=32, target_size=128,
                   threshold=0.5):
    """
    Évaluation complète du modèle sur le jeu de test.

    Calcule les métriques standard en classification binaire :

    ┌───────────────────────────────────────────────────────────┐
    │  Réel \ Prédit │  Bénin (0)     │  Malin (1)             │
    │  ──────────────┼────────────────┼─────────────────────── │
    │  Bénin (0)     │  TN (vrai nég) │  FP (faux pos)         │
    │  Malin (1)     │  FN (faux nég) │  TP (vrai pos)         │
    └───────────────────────────────────────────────────────────┘

    Métriques clés en diagnostic médical :
    • Sensibilité (recall)  = TP / (TP + FN)  → détecte-t-on tous les cancers ?
    • Spécificité           = TN / (TN + FP)  → évite-t-on les faux alarmes ?
    • Précision             = TP / (TP + FP)
    • F1-score              = 2 × (Préc × Rappel) / (Préc + Rappel)
    • AUC-ROC               → performance globale indépendante du seuil

    ⚠  En diagnostic du cancer, les FAUX NÉGATIFS sont critiques :
       un cancer non détecté (FN) met la vie du patient en danger.
       → On cherche à maximiser la sensibilité, quitte à augmenter les FP.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch requis.")

    model.eval()
    test_ds = MammographyDataset(X_test, y_test, augment=False, target_size=target_size)
    loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds, all_labels, all_proba = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch).squeeze(1)
            proba   = torch.sigmoid(logits).cpu().numpy()
            preds   = (proba >= threshold).astype(int)
            all_proba.append(proba)
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())

    proba_arr  = np.concatenate(all_proba)
    preds_arr  = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)

    tp = np.sum((preds_arr == 1) & (labels_arr == 1))
    tn = np.sum((preds_arr == 0) & (labels_arr == 0))
    fp = np.sum((preds_arr == 1) & (labels_arr == 0))
    fn = np.sum((preds_arr == 0) & (labels_arr == 1))

    accuracy    = (tp + tn) / len(labels_arr)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision   = tp / (tp + fp + 1e-8)
    f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)

    auc = roc_auc_score(labels_arr, proba_arr) if SKLEARN_AVAILABLE else float('nan')

    metrics = {
        'accuracy'   : accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision'  : precision,
        'f1'         : f1,
        'auc'        : auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'preds' : preds_arr,
        'proba' : proba_arr,
        'labels': labels_arr
    }

    # Affichage
    print("\n" + "═" * 55)
    print("  RÉSULTATS D'ÉVALUATION")
    print("═" * 55)
    print(f"  Threshold de décision : {threshold}")
    print(f"  Accuracy             : {accuracy*100:.2f}%")
    print(f"  Sensibilité (recall) : {sensitivity*100:.2f}%  ← critique en clinique")
    print(f"  Spécificité          : {specificity*100:.2f}%")
    print(f"  Précision            : {precision*100:.2f}%")
    print(f"  F1-score             : {f1:.4f}")
    print(f"  AUC-ROC              : {auc:.4f}")
    print(f"\n  Matrice de confusion :")
    print(f"    VP (vrais positifs) : {tp:4d}  |  FN (faux négatifs) : {fn:4d}")
    print(f"    FP (faux positifs)  : {fp:4d}  |  VN (vrais négatifs): {tn:4d}")
    print("═" * 55)

    if fn > 0:
        print(f"\n  ⚠  {fn} cancer(s) NON DÉTECTÉ(S) (faux négatif).")
        print(f"     En clinique, cela pourrait retarder un diagnostic vital.")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations médicales
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix_medical(metrics, title="Matrice de confusion — Détection du cancer"):
    """
    Visualise la matrice de confusion avec un code couleur médical :
    - Vert  : bonnes prédictions (VP, VN)
    - Rouge : faux négatifs (FN)  → dangereux cliniquement
    - Orange: faux positifs (FP)  → angoisse inutile, examens supplémentaires
    """
    tp, tn, fp, fn = metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']
    cm = np.array([[tn, fp],
                   [fn, tp]])

    # Couleurs personnalisées
    color_matrix = np.array([
        [0.2, 0.7, 0.3],   # VN : vert foncé
        [0.95, 0.6, 0.1],  # FP : orange
        [0.9, 0.1, 0.1],   # FN : rouge vif (CRITIQUE)
        [0.2, 0.8, 0.4],   # VP : vert clair
    ]).reshape(2, 2, 3)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(color_matrix, interpolation='nearest', aspect='auto')

    labels = ['Bénin (0)', 'Malin (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Prédit : Bénin', 'Prédit : Malin'], fontsize=12)
    ax.set_yticklabels(['Réel : Bénin', 'Réel : Malin'], fontsize=12)
    ax.set_xlabel("Prédiction du modèle", fontsize=12, fontweight='bold')
    ax.set_ylabel("Vérité terrain", fontsize=12, fontweight='bold')

    annotations = [
        (0, 0, f"VN\n{tn}", "Vrais négatifs\n(bénins corrects)"),
        (1, 0, f"FP\n{fp}", "Faux positifs\n(surdiagnostic)"),
        (0, 1, f"FN\n{fn}", "⚠ FAUX NÉGATIFS\n(cancers manqués)"),
        (1, 1, f"VP\n{tp}", "Vrais positifs\n(cancers détectés)"),
    ]
    for (j, i, main_text, sub_text) in annotations:
        ax.text(j, i, main_text, ha='center', va='center',
                fontsize=15, fontweight='bold', color='white')
        ax.text(j, i + 0.35, sub_text, ha='center', va='center',
                fontsize=7, color='white', alpha=0.9)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Légende des métriques
    sens = metrics['sensitivity'] * 100
    spec = metrics['specificity'] * 100
    ax.text(1.02, 0.5,
            f"Sensibilité : {sens:.1f}%\nSpécificité : {spec:.1f}%\n"
            f"AUC-ROC     : {metrics['auc']:.3f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#1e293b', alpha=0.8, edgecolor='gray'),
            color='white')

    plt.tight_layout()
    plt.savefig("confusion_matrix_medical.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  ✓ Sauvegardé : confusion_matrix_medical.png")


def plot_roc_curve(metrics, title="Courbe ROC — Détection du cancer du sein"):
    """
    Trace la courbe ROC (Receiver Operating Characteristic).

    La courbe ROC représente le compromis sensibilité / (1 - spécificité)
    pour tous les seuils de décision possibles.

    AUC (Area Under the Curve) :
    - AUC = 1.0 : modèle parfait
    - AUC = 0.5 : modèle aléatoire
    - AUC > 0.85 : acceptable en clinique (règle empirique)

    Le point optimal (maximise sensibilité + spécificité) est le point de
    la courbe le plus proche du coin supérieur gauche (0, 1).

    En oncologie, on peut choisir un seuil bas pour maximiser la sensibilité
    (détecter tous les cancers) quitte à avoir plus de FP.
    """
    if not SKLEARN_AVAILABLE:
        print("sklearn requis pour tracer la courbe ROC.")
        return

    fpr, tpr, thresholds = roc_curve(metrics['labels'], metrics['proba'])
    auc_val = metrics['auc']

    # Point optimal (distance minimale au coin parfait (0,1))
    dist_to_optimal = np.sqrt(fpr**2 + (1 - tpr)**2)
    opt_idx = np.argmin(dist_to_optimal)
    opt_threshold = thresholds[opt_idx]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Courbe ROC
    ax.plot(fpr, tpr, color='#2563EB', linewidth=2.5,
            label=f'CNN (AUC = {auc_val:.3f})')

    # Point optimal
    ax.scatter(fpr[opt_idx], tpr[opt_idx], color='#DC2626', s=120, zorder=5,
               label=f'Seuil optimal = {opt_threshold:.2f}\n'
                     f'(Sensib. = {tpr[opt_idx]*100:.1f}%, '
                     f'Spécif. = {(1-fpr[opt_idx])*100:.1f}%)')

    # Diagonale aléatoire
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Aléatoire (AUC = 0.5)')

    # Zone remplie sous la courbe
    ax.fill_between(fpr, tpr, alpha=0.08, color='#2563EB')

    ax.set_xlabel("Taux de faux positifs (1 − Spécificité)", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs (Sensibilité)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.3)

    # Annotation du seuil optimal
    ax.annotate(f"Seuil = {opt_threshold:.2f}",
                xy=(fpr[opt_idx], tpr[opt_idx]),
                xytext=(fpr[opt_idx] + 0.1, tpr[opt_idx] - 0.1),
                arrowprops=dict(arrowstyle='->', color='#DC2626'),
                fontsize=9, color='#DC2626')

    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=100)
    plt.show()
    print(f"  ✓ Sauvegardé : roc_curve.png")
    print(f"  Seuil optimal : {opt_threshold:.3f}  "
          f"(Sensibilité : {tpr[opt_idx]*100:.1f}%, "
          f"Spécificité : {(1-fpr[opt_idx])*100:.1f}%)")

    return opt_threshold


def plot_training_curves_medical(history, title="Courbes d'entraînement — CBIS-DDSM"):
    """
    Affiche les courbes d'entraînement :
    - Loss train / test
    - Accuracy train / test
    - Sensibilité et spécificité (métriques médicales)
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Courbe de loss
    axes[0].plot(epochs, history['train_loss'], '#2563EB', linewidth=2, label='Train')
    axes[0].plot(epochs, history['test_loss'],  '#DC2626', linewidth=2,
                 linestyle='--', label='Test')
    axes[0].set_title("Loss (BCEWithLogitsLoss)")
    axes[0].set_xlabel("Époques")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Courbe d'accuracy
    axes[1].plot(epochs, [a * 100 for a in history['train_acc']],
                 '#2563EB', linewidth=2, label='Train')
    axes[1].plot(epochs, [a * 100 for a in history['test_acc']],
                 '#DC2626', linewidth=2, linestyle='--', label='Test')
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Époques")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 105)

    # Sensibilité & Spécificité
    axes[2].plot(epochs, [s * 100 for s in history['sensitivity']],
                 '#16A34A', linewidth=2, label='Sensibilité (recall)')
    axes[2].plot(epochs, [s * 100 for s in history['specificity']],
                 '#9333EA', linewidth=2, linestyle='--', label='Spécificité')
    axes[2].set_title("Métriques médicales (%)")
    axes[2].set_xlabel("Époques")
    axes[2].set_ylabel("%")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig("training_curves_medical.png", dpi=100)
    plt.show()
    print("  ✓ Sauvegardé : training_curves_medical.png")


def plot_false_negatives(X_test, y_test, metrics, n_max=8):
    """
    Affiche les faux négatifs : les mammographies malignes que le modèle
    a incorrectement classées comme bénignes.

    Cette analyse est cruciale en clinique : comprendre POURQUOI le modèle
    rate certains cancers aide à améliorer l'architecture ou le preprocessing.
    """
    preds  = metrics['preds']
    proba  = metrics['proba']
    labels = metrics['labels']

    fn_idx = np.where((preds == 0) & (labels == 1))[0]

    if len(fn_idx) == 0:
        print("  ✓ Aucun faux négatif détecté sur l'ensemble de test !")
        return

    n_show = min(n_max, len(fn_idx))
    fig, axes = plt.subplots(2, n_show // 2 + n_show % 2, figsize=(n_show * 2.2, 5))
    axes = axes.flatten()
    fig.suptitle(f"⚠  Faux négatifs : {len(fn_idx)} cancers non détectés\n"
                 "(malin classé bénin par le modèle)", fontsize=12,
                 fontweight='bold', color='#DC2626')

    for i, idx in enumerate(fn_idx[:n_show]):
        axes[i].imshow(X_test[idx], cmap='bone', vmin=0, vmax=1)
        axes[i].set_title(f"P(malin) = {proba[idx]*100:.1f}%\n(réel : MALIN)",
                          fontsize=8, color='#DC2626')
        axes[i].axis('off')

    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("false_negatives.png", dpi=100, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Sauvegardé : false_negatives.png")
    print(f"  Analyse : {len(fn_idx)} faux négatif(s) — probabilité malin moyenne : "
          f"{proba[fn_idx].mean()*100:.1f}%")


def analyze_threshold_sensitivity(metrics):
    """
    Étudie l'effet du seuil de décision sur les métriques médicales.

    En ajustant le seuil (par défaut 0.5), on peut :
    - Baisser le seuil → plus de sensibilité, moins de spécificité (moins de FN)
    - Hausser le seuil → moins de FP, risque de rater des cancers (plus de FN)

    Cette analyse permet au médecin (ou à l'équipe) de choisir un seuil
    adapté au compromis risque/bénéfice clinique.
    """
    labels = metrics['labels']
    proba  = metrics['proba']

    thresholds = np.linspace(0.1, 0.9, 81)
    sens_list, spec_list, f1_list = [], [], []

    for t in thresholds:
        preds = (proba >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        f1   = 2 * prec * sens / (prec + sens + 1e-8)

        sens_list.append(sens)
        spec_list.append(spec)
        f1_list.append(f1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Impact du seuil de décision sur les performances",
                 fontsize=13, fontweight='bold')

    # Sensibilité & Spécificité vs. seuil
    axes[0].plot(thresholds, sens_list, '#DC2626', linewidth=2, label='Sensibilité')
    axes[0].plot(thresholds, spec_list, '#2563EB', linewidth=2, label='Spécificité')
    axes[0].axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Seuil = 0.5')
    axes[0].set_xlabel("Seuil de décision", fontsize=11)
    axes[0].set_ylabel("Valeur", fontsize=11)
    axes[0].set_title("Sensibilité / Spécificité", fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].fill_between(thresholds, sens_list, spec_list, alpha=0.05, color='gray')

    # F1-score vs. seuil
    axes[1].plot(thresholds, f1_list, '#16A34A', linewidth=2.5, label='F1-score')
    best_t_idx = np.argmax(f1_list)
    axes[1].axvline(thresholds[best_t_idx], color='#16A34A', linestyle='--',
                    label=f'Seuil F1 max = {thresholds[best_t_idx]:.2f}')
    axes[1].axvline(0.5, color='gray', linestyle=':', alpha=0.7)
    axes[1].set_xlabel("Seuil de décision", fontsize=11)
    axes[1].set_ylabel("F1-score", fontsize=11)
    axes[1].set_title("F1-score", fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("threshold_analysis.png", dpi=100)
    plt.show()
    print("  ✓ Sauvegardé : threshold_analysis.png")
    print(f"  Seuil maximisant le F1-score : {thresholds[best_t_idx]:.2f}")
    print(f"  → Sensibilité correspondante : {sens_list[best_t_idx]*100:.1f}%")
    print(f"  → Spécificité correspondante : {spec_list[best_t_idx]*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# D. PROGRAMME PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Partie 3 — Détection de cancers du sein (CBIS-DDSM)"
    )
    parser.add_argument(
        '--data_dir', type=str, default='./CBIS-DDSM',
        help='Répertoire racine du dataset CBIS-DDSM'
    )
    parser.add_argument(
        '--train_csv', type=str,
        default='mass_case_description_train_set.csv',
        help='Nom du CSV d\'entraînement (dans data_dir)'
    )
    parser.add_argument(
        '--test_csv', type=str,
        default='mass_case_description_test_set.csv',
        help='Nom du CSV de test (dans data_dir)'
    )
    parser.add_argument(
        '--target_size', type=int, default=128,
        help='Taille cible de redimensionnement (128 ou 224)'
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
        help='Nombre d\'époques d\'entraînement'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Taux d\'apprentissage (Adam)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Taille des mini-lots'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Mode démo : données synthétiques (sans CBIS-DDSM réel)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Seuil de décision pour la classification (défaut : 0.5)'
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print("\n" + "═" * 65)
    print("  PARTIE 3 — Détection de cancers du sein (CBIS-DDSM)")
    print("═" * 65)

    # ── A1 & A2 : Chargement des données ──────────────────────────────────────
    print("\n[1/5] Chargement et prétraitement des données...")

    if args.demo:
        print("\n  ★ MODE DÉMO : génération de mammographies synthétiques ★")
        print("  (utilisez --data_dir pour pointer vers les données CBIS-DDSM réelles)")
        X, y = generate_synthetic_mammograms(
            n_benin=500, n_malin=250,
            target_size=args.target_size
        )
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_ratio=0.2)

    else:
        # Données réelles CBIS-DDSM
        train_csv_path = os.path.join(args.data_dir, args.train_csv)
        test_csv_path  = os.path.join(args.data_dir, args.test_csv)

        print(f"  CSV train : {train_csv_path}")
        records_train = load_csv_labels(train_csv_path)
        X_train, y_train = build_dataset(records_train, args.data_dir,
                                         target_size=args.target_size)

        if os.path.isfile(test_csv_path):
            print(f"  CSV test  : {test_csv_path}")
            records_test = load_csv_labels(test_csv_path)
            X_test, y_test = build_dataset(records_test, args.data_dir,
                                           target_size=args.target_size)
        else:
            print("  CSV test non trouvé → split 80/20 depuis les données train")
            X_train, X_test, y_train, y_test = split_train_test(
                X_train, y_train, test_ratio=0.2
            )

    print(f"\n  Train : {len(y_train)} images "
          f"(bénins : {np.sum(y_train==0)}, malins : {np.sum(y_train==1)})")
    print(f"  Test  : {len(y_test)}  images "
          f"(bénins : {np.sum(y_test==0)}, malins : {np.sum(y_test==1)})")

    # ── A3 & A4 : Visualisation ───────────────────────────────────────────────
    print("\n[2/5] Visualisation des données...")
    visualize_samples_mammography(X_train, y_train, n_per_class=6)
    plot_class_distribution(y_train, y_test)

    # ── B : Entraînement du CNN ────────────────────────────────────────────────
    if not TORCH_AVAILABLE:
        print("\n⚠  PyTorch non disponible. Arrêt après la visualisation.")
        print("   Installez PyTorch : pip install torch torchvision")
        sys.exit(0)

    print(f"\n[3/5] Entraînement du CNN...")
    print(f"  Architecture : Conv(32)×2 → Pool → Conv(64) → Pool → Conv(64)"
          f" → Flatten → Dense(256) → Dense(1)")
    print(f"  Cible       : {args.target_size}×{args.target_size} pixels")
    print(f"  Époques     : {args.epochs} | LR : {args.lr} | Batch : {args.batch_size}")

    model, history = train_cnn_mammography(
        X_train, y_train, X_test, y_test,
        target_size = args.target_size,
        epochs      = args.epochs,
        lr          = args.lr,
        batch_size  = args.batch_size
    )

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "mammography_cnn.pth")
    print("  ✓ Modèle sauvegardé : mammography_cnn.pth")

    # ── C : Évaluation ────────────────────────────────────────────────────────
    print(f"\n[4/5] Évaluation du modèle (seuil = {args.threshold})...")
    metrics = evaluate_model(model, X_test, y_test,
                             target_size=args.target_size,
                             threshold=args.threshold)

    # ── Visualisations médicales ───────────────────────────────────────────────
    print("\n[5/5] Visualisations médicales...")
    plot_training_curves_medical(history)
    plot_confusion_matrix_medical(metrics)
    opt_threshold = plot_roc_curve(metrics)
    plot_false_negatives(X_test, y_test, metrics)
    analyze_threshold_sensitivity(metrics)

    # Réévaluation avec le seuil optimal (ROC)
    if opt_threshold is not None and abs(opt_threshold - args.threshold) > 0.05:
        print(f"\n  Réévaluation avec le seuil optimal (ROC) = {opt_threshold:.2f}...")
        metrics_opt = evaluate_model(model, X_test, y_test,
                                     target_size=args.target_size,
                                     threshold=opt_threshold)

    print("""
═══════════════════════════════════════════════════════════════════════
  ANALYSE & DISCUSSION — PARTIE 3
───────────────────────────────────────────────────────────────────────

  1. DÉSÉQUILIBRE DES CLASSES
     CBIS-DDSM contient plus de cas bénins que malins.
     Sans correction, le modèle favorise la classe majoritaire.
     Solutions appliquées :
       • WeightedRandomSampler : sur-échantillonnage des malins
       • pos_weight dans BCEWithLogitsLoss : pénalise davantage les FN
       • Métriques adaptées : sensibilité/spécificité au lieu de l'accuracy

  2. IMPORTANCE DES FAUX NÉGATIFS (FN) EN CLINIQUE
     Un faux négatif = un cancer diagnostiqué "bénin" = retard de traitement.
     En oncologie mammaire, un retard de 3-6 mois peut changer le pronostic.
     → Maximiser la sensibilité est prioritaire, même si cela augmente les FP.
     → Les FP se vérifient via d'autres examens (biopsie, IRM).

  3. SEUIL DE DÉCISION
     Le seuil par défaut (0.5) n'est pas toujours optimal cliniquement.
     La courbe ROC montre le compromis sensibilité/spécificité pour chaque seuil.
     En pratique, les radiologues calibrent le seuil selon le contexte :
       • Dépistage large (population générale) → seuil bas → haute sensibilité
       • Confirmation avant biopsie → seuil plus élevé → haute spécificité

  4. LIMITES DU MODÈLE
     • Données limitées (~1000 images) : sur-apprentissage probable
     • Redimensionnement à 128×128 : perte d'information sub-millimétrique
     • Pas d'interprétabilité (boîte noire) : difficile à valider cliniquement
     Solutions avancées : transfer learning (ResNet-50), Grad-CAM,
     augmentation agressive, validation croisée.

  5. COMPARAISON AVEC LES PARTIES 1 ET 2
     MNIST  : images simples, centrées, fond uniforme → modèle linéaire suffit
     CIFAR-10 : images couleur complexes → CNN nécessaire (~25% erreur)
     CBIS-DDSM : images médicales très complexes, déséquilibre de classes
                 → CNN + stratégies médicales spécifiques

  6. AUC-ROC
     L'AUC mesure la capacité à ordonner les cas malins avant les bénins.
     AUC > 0.85 est généralement considéré comme acceptable en radiologie.
     Les systèmes CAD (Computer-Aided Detection) commerciaux visent AUC > 0.90.
═══════════════════════════════════════════════════════════════════════
    """)
