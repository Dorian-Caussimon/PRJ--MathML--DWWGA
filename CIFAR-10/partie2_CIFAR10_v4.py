"""
============================================================
  PROJET SM604 — PARTIE 2 : Classification CIFAR-10
  Réseaux de Neurones Convolutifs — EFREI Paris, 2025-2026
============================================================

Ce fichier implémente intégralement la Partie 2 du projet :

  ── Section A : Travail préliminaire ──────────────────────
    A1. Chargement & visualisation de CIFAR-10
    A2. Conversion niveaux de gris (formule pondérée)
    A3. MLP linéaire & couches cachées sur images grises (1024 entrées)
    A4. MLP sur images couleur aplaties (3072 entrées)
    A5. Tableau comparatif avec la littérature scientifique

  ── Section B : Filtres de convolution (2.3.2) ───────────
    B1. Implémentation manuelle de la convolution 2D (zero-padding)
    B2. Application des 6 filtres K1…K6 sur image en N&B
    B3. Visualisation des feature maps

  ── Section C : CNN (Option B — PyTorch) ─────────────────
    C1. Architecture complète : Conv→Conv→Pool→Conv→Pool→Conv→Flatten→Dense
    C2. Entraînement par rétropropagation automatique (Autograd)
    C3. Évaluation & comparaison avec les résultats précédents

INSTRUCTIONS :
  pip install numpy matplotlib torch torchvision scikit-learn pillow
  python partie2_CIFAR10.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─── Vérification PyTorch ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torchvision
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} disponible | Device : {device}")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠  PyTorch non installé → Section C (CNN) désactivée.")
    print("   pip install torch torchvision")


# ═════════════════════════════════════════════════════════════════════════════
#  PARTIE A — TRAVAIL PRÉLIMINAIRE : MLP SUR CIFAR-10
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# A1. Chargement de CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "avion", "automobile", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]

def load_cifar10_numpy():
    """
    Charge CIFAR-10 depuis les fichiers binaires locaux.

    Structure attendue (dossier du script comme point de départ) :
      CIFAR-10/
        cifar-10-batches-py/
          data_batch_1 … data_batch_5   ← train
          test_batch                     ← test
          batches.meta

    Retourne :
      X_train : (50000, 32, 32, 3)  float32  ∈ [0, 1]
      y_train : (50000,)            int64
      X_test  : (10000, 32, 32, 3)  float32  ∈ [0, 1]
      y_test  : (10000,)            int64
    """
    import pickle, os

    print("Chargement de CIFAR-10... (lecture locale)")

    # ── Résolution du chemin ─────────────────────────────────────────────────
    # On cherche le dossier "CIFAR-10/cifar-10-batches-py" à partir du répertoire
    # du script, puis du répertoire de travail courant (cwd), en fallback.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "CIFAR-10", "cifar-10-batches-py"),
        os.path.join(os.getcwd(),  "CIFAR-10", "cifar-10-batches-py"),
    ]
    base = None
    for c in candidates:
        if os.path.isdir(c):
            base = c
            break

    if base is None:
        raise FileNotFoundError(
            "\n\n"
            "════════════════════════════════════════════════════════════\n"
            "  ERREUR : Dossier CIFAR-10 introuvable.\n\n"
            "  ► Structure attendue :\n"
            "      <dossier du script>/\n"
            "        CIFAR-10/\n"
            "          cifar-10-batches-py/\n"
            "            data_batch_1 … data_batch_5\n"
            "            test_batch\n"
            "            batches.meta\n\n"
            "  Chemins vérifiés :\n"
            + "\n".join(f"    • {c}" for c in candidates) + "\n"
            "════════════════════════════════════════════════════════════\n"
        )

    print(f"  → Dossier trouvé : {base}")

    # ── Lecture des fichiers pickle ──────────────────────────────────────────
    def unpickle(path):
        with open(path, "rb") as f:
            return pickle.load(f, encoding="bytes")

    # Train : 5 batches × 10 000 images
    X_parts, y_parts = [], []
    for i in range(1, 6):
        fpath = os.path.join(base, f"data_batch_{i}")
        d = unpickle(fpath)
        X_parts.append(d[b"data"])          # (10000, 3072) uint8
        y_parts.extend(d[b"labels"])
        print(f"    data_batch_{i} chargé ✓")

    X_train_raw = np.concatenate(X_parts, axis=0)   # (50000, 3072)
    y_train     = np.array(y_parts, dtype=np.int64)

    # Test
    d_test     = unpickle(os.path.join(base, "test_batch"))
    X_test_raw = d_test[b"data"]                    # (10000, 3072)
    y_test     = np.array(d_test[b"labels"], dtype=np.int64)
    print("    test_batch chargé ✓")

    # ── Reshape & normalisation ──────────────────────────────────────────────
    # (N, 3072) uint8  →  (N, 3, 32, 32)  →  (N, 32, 32, 3)  float32 ∈ [0,1]
    def reshape(X):
        return (X.reshape(-1, 3, 32, 32)
                  .transpose(0, 2, 3, 1)
                  .astype(np.float32) / 255.0)

    X_train = reshape(X_train_raw)
    X_test  = reshape(X_test_raw)

    print(f"  ✓ Train : {X_train.shape}  | Test : {X_test.shape}")
    return X_train, y_train, X_test, y_test


def visualize_cifar10(X, y, n_per_class=8):
    """Affiche n_per_class exemples pour chacune des 10 classes."""
    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 1.3, 14))
    for cls in range(10):
        idx = np.where(y == cls)[0][:n_per_class]
        for j, i in enumerate(idx):
            axes[cls, j].imshow(X[i])
            axes[cls, j].axis("off")
        axes[cls, 0].set_ylabel(CIFAR10_CLASSES[cls], fontsize=9,
                                rotation=0, labelpad=55, va="center")
    fig.suptitle("CIFAR-10 — Exemples par classe", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("cifar10_samples.png", dpi=100, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# A2. Conversion niveaux de gris & aplatissement
# ─────────────────────────────────────────────────────────────────────────────

def rgb_to_grayscale(X_rgb):
    """
    Convertit des images RGB en niveaux de gris selon la formule standard :

        x_j = 0.299·R_j + 0.587·G_j + 0.114·B_j

    Ces coefficients sont issus de la norme ITU-R BT.601.
    Ils pondèrent la sensibilité de l'œil humain aux trois couleurs :
    l'œil est plus sensible au vert (0.587) qu'au rouge (0.299) et encore
    moins au bleu (0.114).

    Paramètres :
      X_rgb : (N, 32, 32, 3)
    Retourne :
      X_gray : (N, 32, 32)   float32
    """
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (X_rgb * w).sum(axis=-1)   # (N, 32, 32)


def flatten_images(X):
    """
    Aplatit chaque image en vecteur ligne.
    (N, H, W)    → (N, H*W)
    (N, H, W, C) → (N, H*W*C)
    """
    return X.reshape(X.shape[0], -1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# A3 & A4. Réutilisation du MLP de la Partie 1
#          (copié ici pour rendre ce fichier autonome)
# ─────────────────────────────────────────────────────────────────────────────

def softmax(O):
    """Softmax numériquement stable (soustraction du max ligne à ligne)."""
    O_s = O - O.max(axis=1, keepdims=True)
    E = np.exp(O_s)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(P, Y, eps=1e-12):
    return -np.mean(np.sum(Y * np.log(P + eps), axis=1))


def one_hot(y, K=10):
    Y = np.zeros((len(y), K), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def relu(z):    return np.maximum(0.0, z)
def relu_d(z):  return (z > 0).astype(np.float32)


class MLP:
    """
    Perceptron multi-couches générique (identique à la Partie 1).
    Paramètres :
      layer_sizes : ex. [1024, 256, 10] pour H=1 couche cachée sur images grises
                        [3072, 256, 10] pour H=1 couche cachée sur images couleur
    """
    def __init__(self, layer_sizes):
        self.sizes = layer_sizes
        self.L     = len(layer_sizes) - 1
        # Initialisation He : std = sqrt(2/fan_in)
        self.W = [np.random.randn(layer_sizes[l+1], layer_sizes[l]).astype(np.float32)
                  * np.sqrt(2.0 / layer_sizes[l])
                  for l in range(self.L)]
        self.b = [np.zeros(layer_sizes[l+1], dtype=np.float32) for l in range(self.L)]

    def forward(self, X):
        """Forward pass avec mise en cache (nécessaire pour backprop)."""
        cache, z = [], X
        for l in range(self.L):
            zp = z
            o  = zp @ self.W[l].T + self.b[l]
            z  = relu(o) if l < self.L - 1 else softmax(o)
            cache.append((zp, o, z))
        return cache[-1][2], cache

    def predict(self, X):
        P, _ = self.forward(X)
        return np.argmax(P, axis=1)

    def backward(self, cache, Y):
        """
        Rétropropagation du gradient.

        Couche de sortie (softmax + cross-entropy) :
          δ_out = (P - Y) / n

        Couche l (ReLU) :
          δ_l = (δ_{l+1} @ W_{l+1}) ⊙ ReLU'(o_l)

        Gradients :
          ∂L/∂W_l = δ_l^T @ z_{l-1}
          ∂L/∂b_l = mean(δ_l, axis=0)
        """
        n = Y.shape[0]
        gW = [None] * self.L
        gb = [None] * self.L
        zp, o, P = cache[-1]
        delta = (P - Y) / n
        gW[-1] = delta.T @ zp
        gb[-1] = delta.mean(0)
        for l in range(self.L - 2, -1, -1):
            zp, o, z = cache[l]
            delta = (delta @ self.W[l+1]) * relu_d(o)
            gW[l] = delta.T @ zp
            gb[l] = delta.mean(0)
        return gW, gb

    def train(self, X_tr, Y_tr, X_te, y_te,
              lr=0.05, epochs=30, batch=256, verbose=True):
        """Mini-batch SGD."""
        n = X_tr.shape[0]
        y_tr_lbl = np.argmax(Y_tr, axis=1)
        hist = {"loss": [], "train_err": [], "test_err": []}
        for ep in range(epochs):
            perm = np.random.permutation(n)
            Xs, Ys = X_tr[perm], Y_tr[perm]
            ep_loss, nb = 0.0, 0
            for s in range(0, n, batch):
                Xb, Yb = Xs[s:s+batch], Ys[s:s+batch]
                P, cache = self.forward(Xb)
                ep_loss += cross_entropy(P, Yb); nb += 1
                gW, gb  = self.backward(cache, Yb)
                for l in range(self.L):
                    self.W[l] -= lr * gW[l]
                    self.b[l]  -= lr * gb[l]
            avg  = ep_loss / nb
            te   = 1 - np.mean(self.predict(X_tr) == y_tr_lbl)
            vae  = 1 - np.mean(self.predict(X_te) == y_te)
            hist["loss"].append(avg)
            hist["train_err"].append(te)
            hist["test_err"].append(vae)
            if verbose and ((ep+1) % 10 == 0 or ep == 0):
                print(f"  Ép {ep+1:3d}/{epochs} | Loss={avg:.4f} | "
                      f"Err Train={te*100:.1f}% | Err Test={vae*100:.1f}%")
        return hist


# ─────────────────────────────────────────────────────────────────────────────
# A5. Tableau de comparaison avec la littérature
# ─────────────────────────────────────────────────────────────────────────────

STATE_OF_ART = [
    ("Conv. Deep Belief Nets",            21.1, "août 2010"),
    ("Maxout Networks",                    9.38, "févr. 2013"),
    ("Fractional Max-Pooling",             3.47, "déc. 2014"),
    ("Densely Connected Conv. Nets",       3.46, "août 2016"),
    ("Coupled Ensembles",                  2.68, "sept. 2017"),
    ("ViT (16×16 Words)",                  0.50, "juin 2021"),
]

def print_full_comparison(our_results):
    """
    Affiche un tableau récapitulatif complet :
    nos résultats + état de l'art de la littérature.
    """
    print("\n" + "═"*62)
    print("  COMPARAISON AVEC LA LITTÉRATURE — CIFAR-10")
    print("═"*62)
    print(f"  {'Modèle':<38} {'Err Test (%)':>10}")
    print("─"*62)
    for name, err_tr, err_te in our_results:
        marker = "◀ nous"
        print(f"  {name:<38} {err_te*100:>9.2f}%  {marker}")
    print("─"*62)
    for name, err, date in STATE_OF_ART:
        print(f"  {name:<38} {err:>9.2f}%  ({date})")
    print("═"*62)
    print("  ↑ Les architectures profondes utilisent des techniques")
    print("    avancées (data augmentation, batch norm, dropout…)")
    print("    que nos modèles basiques n'implémentent pas.")
    print("═"*62)


# ═════════════════════════════════════════════════════════════════════════════
#  PARTIE B — FILTRES DE CONVOLUTION (implémentation manuelle, section 2.3)
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# B1. Convolution 2D avec zero-padding (image N&B)
# ─────────────────────────────────────────────────────────────────────────────

def convolve2d(image, K, bias=0.0):
    """
    Convolution 2D avec zero-padding, taille de sortie identique à l'entrée.

    Formule du sujet (section 2.3.1) :
        m'_{u,v} = Σ_{u'=1}^{3} Σ_{v'=1}^{3}  K_{u',v'} · m_{u'+u-2, v'+v-2}  + l

    Implémentation :
      1. On ajoute une bordure de 1 zéro tout autour de l'image (zero-padding).
         Cela permet de calculer m'_{u,v} même pour les pixels de bord,
         car les pixels hors bord valent 0.
      2. Pour chaque pixel (u, v) de l'image de sortie, on extrait le patch
         3×3 centré sur (u, v) dans l'image paddée et on fait le produit
         scalaire avec le filtre K (+ biais).

    Complexité : O(H × W × 9) — linéaire en nombre de pixels.

    Paramètres :
      image : (H, W)  numpy array (niveaux de gris, float32)
      K     : (3, 3)  filtre
      bias  : scalaire (l dans la formule)
    Retourne :
      out   : (H, W)  image filtrée (feature map)
    """
    H, W   = image.shape
    kH, kW = K.shape          # 3, 3
    pH, pW = kH // 2, kW // 2  # padding = 1

    # Zero-padding : ajoute une bordure de 0 (1 pixel tout autour)
    padded = np.pad(image, ((pH, pH), (pW, pW)), mode="constant", constant_values=0)

    # Convolution : produit scalaire filtre × patch pour chaque pixel
    out = np.zeros((H, W), dtype=np.float32)
    for u in range(H):
        for v in range(W):
            patch = padded[u:u+kH, v:v+kW]    # patch 3×3
            out[u, v] = np.sum(K * patch) + bias

    return out


def convolve2d_fast(image, K, bias=0.0):
    """
    Version vectorisée de la convolution 2D (bien plus rapide grâce à NumPy).

    Au lieu de boucler pixel par pixel, on extrait toutes les colonnes du
    patch simultanément en utilisant la technique des "stride tricks" ou
    une simple boucle sur les 9 positions du filtre.
    Ici on utilise une boucle sur les positions (u', v') du filtre
    (9 itérations seulement au lieu de H×W).
    """
    H, W   = image.shape
    kH, kW = K.shape
    pH     = kH // 2
    padded = np.pad(image, pH, mode="constant", constant_values=0)
    out    = np.zeros((H, W), dtype=np.float32)
    for du in range(kH):
        for dv in range(kW):
            # Contribution du coefficient K[du, dv] à tous les pixels en parallèle
            out += K[du, dv] * padded[du:du+H, dv:dv+W]
    return out + bias


# Définition des 6 filtres du sujet (section 2.3.2)
FILTERS = {
    "K1 — Moyenne (lissage)": np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.float32) / 9.0,

    "K2 — Sharpen (netteté)": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),

    "K3 — Contours verticaux": np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], dtype=np.float32),

    "K4 — Contours horizontaux": np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32),

    "K5 — Sobel (gradients horiz.)": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32),

    "K6 — Emboss (relief diagonal)": np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32),
}

FILTER_DESC = {
    "K1 — Moyenne (lissage)":
        "Flou par moyenne : chaque pixel prend la\nmoyenne de ses 9 voisins → supprime le bruit.",
    "K2 — Sharpen (netteté)":
        "Accentue les contrastes locaux.\nLe centre ×5 amplifie les détails fins.",
    "K3 — Contours verticaux":
        "Détecte les transitions horizontales\n(colonnes de pixels : différence haut/bas).",
    "K4 — Contours horizontaux":
        "Détecte les transitions verticales\n(lignes de pixels : différence gauche/droite).",
    "K5 — Sobel (gradients horiz.)":
        "Estimateur de gradient horizontal\nplus robuste que K4 (pondéré au centre).",
    "K6 — Emboss (relief diagonal)":
        "Effet d'embossage : simule un relief\néclairé depuis le coin supérieur gauche.",
}


def apply_and_visualize_filters(image_gray, clip=True):
    """
    Applique les 6 filtres du sujet sur une image en N&B et affiche les résultats.

    Chaque feature map est renormalisée dans [0, 1] pour l'affichage
    (clip + normalisation min-max).
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Image originale
    axes[0].imshow(image_gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Image originale", fontweight="bold", fontsize=11)
    axes[0].axis("off")

    for i, (name, K) in enumerate(FILTERS.items()):
        fm = convolve2d_fast(image_gray, K, bias=0.0)
        if clip:
            fm = np.clip(fm, 0, 1)
        # Normalisation min-max pour l'affichage
        fmin, fmax = fm.min(), fm.max()
        if fmax > fmin:
            fm_disp = (fm - fmin) / (fmax - fmin)
        else:
            fm_disp = fm
        axes[i+1].imshow(fm_disp, cmap="gray")
        axes[i+1].set_title(name, fontsize=9, fontweight="bold")
        axes[i+1].set_xlabel(FILTER_DESC[name], fontsize=7, style="italic",
                              ha="left", x=0.0)
        axes[i+1].axis("off")

    # Masquer la 8e case (on a 6 filtres + 1 originale = 7)
    axes[7].axis("off")

    fig.suptitle("Effets des 6 filtres de convolution (section 2.3.2)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("convolution_filters.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  ✓ Visualisation sauvegardée : convolution_filters.png")


# ─────────────────────────────────────────────────────────────────────────────
# B2. Convolution sur image couleur (section 2.3.3)
# ─────────────────────────────────────────────────────────────────────────────

def convolve_color(image_rgb, K_R, K_G, K_B, bias=0.0):
    """
    Convolution sur une image couleur RGB avec trois filtres séparés.

    Formule du sujet (section 2.3.3) :
      m'_{u,v} = Σ K^(R)_{u',v'} · m^(R) + Σ K^(G)_{u',v'} · m^(G)
               + Σ K^(B)_{u',v'} · m^(B) + l

    Les trois canaux sont traités séparément puis sommés →
    la sortie est une seule feature map 2D (N&B).

    Paramètres :
      image_rgb : (H, W, 3)  image couleur
      K_R, K_G, K_B : (3, 3) filtres pour chaque canal
    Retourne :
      out : (H, W) feature map
    """
    R = convolve2d_fast(image_rgb[:, :, 0], K_R)
    G = convolve2d_fast(image_rgb[:, :, 1], K_G)
    B = convolve2d_fast(image_rgb[:, :, 2], K_B)
    return R + G + B + bias


# ═════════════════════════════════════════════════════════════════════════════
#  PARTIE C — CNN (PyTorch, Option B, section 2.6.2)
# ═════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    # ─────────────────────────────────────────────────────────────────────────
    # C1. Architecture CNN conforme au sujet (section 2.5)
    # ─────────────────────────────────────────────────────────────────────────

    class CIFAR10_CNN(nn.Module):
        """
        Architecture convolutive exacte décrite dans la section 2.5 du sujet.

        Flux des données :
        ──────────────────────────────────────────────────────────────────
        Entrée    : (N, 3, 32, 32)   image couleur normalisée

        Conv1     : 64 filtres 3×3, padding=1 → (N, 64, 32, 32)
          Paramètres : 64 × (3×3×3 + 1) = 1 792
          → Chaque filtre est un parallélépipède 3×3×3 (un canal par couleur)
          → 64 feature maps de même taille que l'entrée (grâce au padding)

        Conv2     : 64 filtres 3D 3×3×64, padding=1 → (N, 64, 32, 32)
          Paramètres : 64 × (3×3×64 + 1) = 36 928
          → Chaque filtre combine les 64 feature maps de Conv1

        MaxPool1  : 2×2 stride=2 → (N, 64, 16, 16)
          Pas de paramètre. Réduit la résolution spatiale ÷ 2.

        Conv3     : 64 filtres 3D 3×3×64, padding=1 → (N, 64, 16, 16)
          Paramètres : 64 × (3×3×64 + 1) = 36 928

        MaxPool2  : 2×2 stride=2 → (N, 64, 8, 8)

        Conv4     : 64 filtres 3D 3×3×64, padding=1 → (N, 64, 8, 8)
          Paramètres : 64 × (3×3×64 + 1) = 36 928

        Flatten   : (N, 64×8×8) = (N, 4096)

        FC (Dense): 4096 → 10 (Softmax implicite dans CrossEntropyLoss)
          Paramètres : 4096 × 10 + 10 = 40 970

        Total paramètres entraînables : ~153 546
        ──────────────────────────────────────────────────────────────────

        Remarque sur ReLU :
          On applique ReLU après chaque convolution (et après FC).
          ReLU introduit la non-linéarité indispensable pour que le réseau
          apprenne des représentations complexes.
          Sans activation, empiler des convolutions resterait linéaire.

        Remarque sur le padding :
          padding=1 avec kernel_size=3 conserve la taille spatiale de l'image,
          conformément à la formule du sujet (les images restent 32×32 après
          chaque convolution, avant le pooling).
        """

        def __init__(self, num_classes=10):
            super().__init__()

            # ── Couche 1 : Convolution couleur (section 2.5.2) ──────────────
            # in_channels=3 (RGB) | out_channels=64 filtres | kernel=3×3
            self.conv1 = nn.Conv2d(3,  64, kernel_size=3, padding=1)

            # ── Couche 2 : Convolution 3D (section 2.5.3) ───────────────────
            # in_channels=64 (profondeur de l'entrée = 64 feature maps)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            # ── MaxPooling 1 (section 2.5.4) ─────────────────────────────────
            # 32×32 → 16×16
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            # ── Couche 3 : Convolution 3D ────────────────────────────────────
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            # ── MaxPooling 2 (section 2.5.5) ──────────────────────────────────
            # 16×16 → 8×8
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            # ── Couche 4 : Convolution 3D finale ─────────────────────────────
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            # ── Aplatissement + Densification (section 2.4.3 & 2.5.5) ────────
            # 8×8×64 = 4096 → 10 classes
            self.fc = nn.Linear(64 * 8 * 8, num_classes)

            self.relu = nn.ReLU()

        def forward(self, x):
            """
            Passe avant (forward pass).

            À chaque étape, on peut lire les dimensions entre parenthèses
            pour une image seule (batch_size=1) ou un batch de N images.

            x : (N, 3, 32, 32)
            """
            # Conv1 + ReLU : (N, 3, 32, 32) → (N, 64, 32, 32)
            x = self.relu(self.conv1(x))

            # Conv2 + ReLU : (N, 64, 32, 32) → (N, 64, 32, 32)
            x = self.relu(self.conv2(x))

            # MaxPool1 : (N, 64, 32, 32) → (N, 64, 16, 16)
            x = self.pool1(x)

            # Conv3 + ReLU : (N, 64, 16, 16) → (N, 64, 16, 16)
            x = self.relu(self.conv3(x))

            # MaxPool2 : (N, 64, 16, 16) → (N, 64, 8, 8)
            x = self.pool2(x)

            # Conv4 + ReLU : (N, 64, 8, 8) → (N, 64, 8, 8)
            x = self.relu(self.conv4(x))

            # Aplatissement : (N, 64, 8, 8) → (N, 4096)
            x = torch.flatten(x, 1)

            # Dense : (N, 4096) → (N, 10)  [logits, softmax dans la loss]
            x = self.fc(x)
            return x

    # ─────────────────────────────────────────────────────────────────────────
    # C2. Entraînement du CNN
    # ─────────────────────────────────────────────────────────────────────────

    def train_cnn(epochs=20, lr=0.001, batch_size=128):
        """
        Entraînement du CNN sur CIFAR-10 avec PyTorch (Option B du sujet).

        Chargement via torchvision avec normalisation standard :
          mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
          → soustrait la moyenne et divise par l'écart-type canal par canal
          → les activations restent dans une plage numériquement favorable

        Optimiseur : Adam (adaptatif, convergence plus rapide que SGD)
        Loss       : CrossEntropyLoss = LogSoftmax + NLLLoss
                     (softmax intégré → pas besoin de l'appliquer dans forward)

        Rétropropagation automatique (Autograd) :
          loss.backward() → calcule ∂L/∂θ pour tous les paramètres θ
          optimizer.step() → applique la mise à jour θ ← θ - lr·∂L/∂θ

        Retourne :
          model   : modèle entraîné
          history : dict avec loss, train_err, test_err
        """
        print("\nChargement CIFAR-10 (format PyTorch avec normalisation)...")

        # Normalisation : µ et σ calculés sur le jeu d'entraînement CIFAR-10
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),      # data augmentation légère
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # ── Chargement local (CIFAR-10/cifar-10-batches-py) ──────────────────
        import pickle as _pickle, os as _os
        from torch.utils.data import Dataset as _Dataset
        from PIL import Image as _Image

        def _unpickle_cnn(path):
            with open(path, "rb") as _f:
                return _pickle.load(_f, encoding="bytes")

        _script_dir = _os.path.dirname(_os.path.abspath(__file__))
        _cnn_candidates = [
            _os.path.join(_script_dir, "CIFAR-10", "cifar-10-batches-py"),
            _os.path.join(_os.getcwd(), "CIFAR-10", "cifar-10-batches-py"),
        ]
        _base_cnn = next((c for c in _cnn_candidates if _os.path.isdir(c)), None)
        if _base_cnn is None:
            raise FileNotFoundError(
                "Dossier CIFAR-10/cifar-10-batches-py introuvable. "
                "Placez-le au même niveau que le script.")

        class _LocalCIFAR10(_Dataset):
            def __init__(self, base, train=True, transform=None):
                self.transform = transform
                if train:
                    parts = [_unpickle_cnn(_os.path.join(base, f"data_batch_{i}"))
                             for i in range(1, 6)]
                    self.data   = np.concatenate([p[b"data"] for p in parts])
                    self.labels = sum([p[b"labels"] for p in parts], [])
                else:
                    d = _unpickle_cnn(_os.path.join(base, "test_batch"))
                    self.data, self.labels = d[b"data"], d[b"labels"]
                self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            def __len__(self): return len(self.labels)
            def __getitem__(self, idx):
                img = _Image.fromarray(self.data[idx])
                if self.transform: img = self.transform(img)
                return img, self.labels[idx]

        train_set = _LocalCIFAR10(_base_cnn, train=True,  transform=transform_train)
        test_set  = _LocalCIFAR10(_base_cnn, train=False, transform=transform_test)

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_set,  batch_size=512,
                                  shuffle=False, num_workers=0)

        model     = CIFAR10_CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Compte les paramètres
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Nombre de paramètres entraînables : {total_params:,}")

        history = {"loss": [], "train_err": [], "test_err": []}

        for epoch in range(epochs):
            # ── Phase d'entraînement ──────────────────────────────────────
            model.train()
            ep_loss, correct_tr, total_tr = 0.0, 0, 0

            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)

                optimizer.zero_grad()       # réinitialise les gradients accumulés
                logits = model(Xb)          # forward
                loss   = criterion(logits, yb)  # cross-entropy
                loss.backward()             # rétropropagation (Autograd)
                optimizer.step()            # mise à jour des poids

                ep_loss   += loss.item()
                preds      = logits.argmax(dim=1)
                correct_tr += (preds == yb).sum().item()
                total_tr   += yb.size(0)

            # ── Phase d'évaluation ────────────────────────────────────────
            model.eval()
            correct_te, total_te = 0, 0
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    preds   = model(Xb).argmax(dim=1)
                    correct_te += (preds == yb).sum().item()
                    total_te   += yb.size(0)

            avg_loss  = ep_loss / len(train_loader)
            train_err = 1 - correct_tr / total_tr
            test_err  = 1 - correct_te / total_te

            history["loss"].append(avg_loss)
            history["train_err"].append(train_err)
            history["test_err"].append(test_err)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Ép {epoch+1:3d}/{epochs} | Loss={avg_loss:.4f} | "
                      f"Err Train={train_err*100:.1f}% | Err Test={test_err*100:.1f}%")

        return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations communes
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(histories, labels, title="Courbes d'entraînement — CIFAR-10"):
    """Compare les courbes de perte et d'erreur de plusieurs modèles."""
    colors = ["#1D4ED8", "#B91C1C", "#15803D", "#7C3AED", "#D97706"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for h, lbl, c in zip(histories, labels, colors):
        ax1.plot(h["loss"], label=lbl, color=c, lw=2)
        ax2.plot([e*100 for e in h["test_err"]],  color=c, lw=2,  label=lbl+" (test)")
        ax2.plot([e*100 for e in h["train_err"]], color=c, lw=1.2,
                 linestyle=":", alpha=0.55, label=lbl+" (train)")
    ax1.set_title("Évolution de la loss", fontweight="bold")
    ax1.set_xlabel("Époques"); ax1.set_ylabel("Cross-Entropy")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.set_title("Taux d'erreur (— test  ··· train)", fontweight="bold")
    ax2.set_xlabel("Époques"); ax2.set_ylabel("Erreur (%)")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cifar10_training_curves.png", dpi=100)
    plt.show()


def visualize_cnn_feature_maps(model, X_sample, device):
    """
    Visualise les feature maps des 4 couches de convolution pour une image.

    Permet de comprendre ce que "voit" le réseau à chaque niveau :
    - Conv1 : détecteurs de bords, textures bas niveau
    - Conv2 : combinaisons de bords → formes simples
    - Conv3 : formes intermédiaires
    - Conv4 : représentations sémantiques plus abstraites
    """
    model.eval()
    x = torch.tensor(X_sample).unsqueeze(0).to(device)  # (1, 3, 32, 32)
    activations = []
    hooks = []

    def hook_fn(m, inp, out):
        activations.append(out.detach().cpu().numpy())

    for layer in [model.conv1, model.conv2, model.conv3, model.conv4]:
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    layer_names = ["Conv1 (32×32×64)", "Conv2 (32×32×64)",
                   "Conv3 (16×16×64)", "Conv4 (8×8×64)"]

    for l, (act, name) in enumerate(zip(activations, layer_names)):
        for f in range(8):
            ax = axes[l, f]
            fm = act[0, f]  # f-ième feature map
            ax.imshow(fm, cmap="viridis", aspect="auto")
            ax.axis("off")
        axes[l, 0].set_ylabel(name, fontsize=8, rotation=0, labelpad=80, va="center")

    fig.suptitle("Feature maps du CNN — 8 filtres par couche",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cnn_feature_maps.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("  ✓ Feature maps sauvegardées : cnn_feature_maps.png")


# ═════════════════════════════════════════════════════════════════════════════
#  PROGRAMME PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "═"*60)
    print("  PARTIE 2 — Classification CIFAR-10 + CNN")
    print("═"*60)

    # ── Chargement ────────────────────────────────────────────────────────────
    print("\n[1] Chargement de CIFAR-10...")
    X_train_rgb, y_train, X_test_rgb, y_test = load_cifar10_numpy()
    visualize_cifar10(X_train_rgb, y_train)

    all_results = []   # pour le tableau final

    # ─────────────────────────────────────────────────────────────────────────
    # ── Section A2 : Conversion niveaux de gris ───────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2] Conversion en niveaux de gris (formule ITU-R)...")
    X_train_gray = rgb_to_grayscale(X_train_rgb)  # (50000, 32, 32)
    X_test_gray  = rgb_to_grayscale(X_test_rgb)

    # Aplatissement → vecteurs 1024D
    X_tr_flat_g  = flatten_images(X_train_gray)   # (50000, 1024)
    X_te_flat_g  = flatten_images(X_test_gray)
    Y_tr         = one_hot(y_train)

    # ─────────────────────────────────────────────────────────────────────────
    # ── Section A3 : MLP sur images grises ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3] MLP linéaire sur images grises (1024 → 10)...")
    print(f"    Paramètres : {1024*10 + 10:,}")
    lin_gray = MLP([1024, 10])
    hist_lin_gray = lin_gray.train(X_tr_flat_g, Y_tr, X_te_flat_g, y_test,
                                   lr=0.05, epochs=30, batch=256)
    err_lin_gray_tr = 1 - np.mean(lin_gray.predict(X_tr_flat_g) == y_train)
    err_lin_gray_te = 1 - np.mean(lin_gray.predict(X_te_flat_g) == y_test)
    all_results.append(("Linéaire (niveaux de gris)", err_lin_gray_tr, err_lin_gray_te))
    print(f"    → Train : {err_lin_gray_tr*100:.1f}%  |  Test : {err_lin_gray_te*100:.1f}%")

    print("\n    MLP H=1 sur images grises (1024 → 256 → 10)...")
    print(f"    Paramètres : {1024*256+256 + 256*10+10:,}")
    mlp1_gray = MLP([1024, 256, 10])
    hist_mlp1_gray = mlp1_gray.train(X_tr_flat_g, Y_tr, X_te_flat_g, y_test,
                                     lr=0.05, epochs=30, batch=256)
    err1_tr = 1 - np.mean(mlp1_gray.predict(X_tr_flat_g) == y_train)
    err1_te = 1 - np.mean(mlp1_gray.predict(X_te_flat_g) == y_test)
    all_results.append(("MLP H=1 niveaux de gris (→256→10)", err1_tr, err1_te))
    print(f"    → Train : {err1_tr*100:.1f}%  |  Test : {err1_te*100:.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # ── Section A4 : MLP sur images couleur ──────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4] MLP sur images couleur (3072 → 10)...")
    print(f"    Paramètres : {3072*10 + 10:,}")
    X_tr_flat_c = flatten_images(X_train_rgb)   # (50000, 3072)
    X_te_flat_c = flatten_images(X_test_rgb)

    lin_color = MLP([3072, 10])
    hist_lin_color = lin_color.train(X_tr_flat_c, Y_tr, X_te_flat_c, y_test,
                                     lr=0.05, epochs=30, batch=256)
    err_lc_tr = 1 - np.mean(lin_color.predict(X_tr_flat_c) == y_train)
    err_lc_te = 1 - np.mean(lin_color.predict(X_te_flat_c) == y_test)
    all_results.append(("Linéaire couleur (3072 → 10)", err_lc_tr, err_lc_te))
    print(f"    → Train : {err_lc_tr*100:.1f}%  |  Test : {err_lc_te*100:.1f}%")

    print("\n    MLP H=1 couleur (3072 → 512 → 10)...")
    print(f"    Paramètres : {3072*512+512 + 512*10+10:,}")
    mlp1_color = MLP([3072, 512, 10])
    hist_mlp1_c = mlp1_color.train(X_tr_flat_c, Y_tr, X_te_flat_c, y_test,
                                   lr=0.03, epochs=30, batch=256)
    err_mc_tr = 1 - np.mean(mlp1_color.predict(X_tr_flat_c) == y_train)
    err_mc_te = 1 - np.mean(mlp1_color.predict(X_te_flat_c) == y_test)
    all_results.append(("MLP H=1 couleur (→512→10)", err_mc_tr, err_mc_te))
    print(f"    → Train : {err_mc_tr*100:.1f}%  |  Test : {err_mc_te*100:.1f}%")

    # Courbes comparatives MLP
    plot_curves(
        [hist_lin_gray, hist_mlp1_gray, hist_lin_color, hist_mlp1_c],
        ["Linéaire gris", "MLP H=1 gris", "Linéaire couleur", "MLP H=1 couleur"],
        title="Comparaison MLP — CIFAR-10"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ── Section B : Filtres de convolution ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[5] Application des filtres de convolution...")

    # On prend une image de test (ex. un chat)
    cat_idx = np.where(y_test == 3)[0][0]  # classe 3 = chat
    img_rgb  = X_test_rgb[cat_idx]         # (32, 32, 3)
    img_gray = X_test_gray[cat_idx]        # (32, 32)

    apply_and_visualize_filters(img_gray, clip=True)

    # Vérification de la convolution (exemple K2 manuellement)
    fm_naive = convolve2d(img_gray, FILTERS["K2 — Sharpen (netteté)"])
    fm_fast  = convolve2d_fast(img_gray, FILTERS["K2 — Sharpen (netteté)"])
    max_diff = np.abs(fm_naive - fm_fast).max()
    print(f"  ✓ Vérification : diff max naïf vs vectorisé = {max_diff:.2e} (doit être ≈ 0)")

    # ─────────────────────────────────────────────────────────────────────────
    # ── Section C : CNN PyTorch ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        print("\n[6] Entraînement du CNN (Option B — PyTorch)...")
        print("  Architecture : Conv(64)→Conv(64)→Pool→Conv(64)→Pool→Conv(64)→FC(10)")
        cnn_model, hist_cnn = train_cnn(epochs=20, lr=0.001, batch_size=128)

        # Erreur finale CNN
        cnn_model.eval()
        correct, total = 0, 0
        X_te_t = torch.tensor(
            X_test_rgb.transpose(0,3,1,2),   # (N,H,W,C) → (N,C,H,W)
            dtype=torch.float32
        ).to(device)
        # Normalisation identique à l'entraînement
        mean_t = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
        std_t  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1).to(device)
        X_te_norm = (X_te_t - mean_t) / std_t
        with torch.no_grad():
            for i in range(0, len(X_te_norm), 256):
                xb = X_te_norm[i:i+256]
                yb = torch.tensor(y_test[i:i+256]).to(device)
                p  = cnn_model(xb).argmax(1)
                correct += (p == yb).sum().item()
                total   += yb.size(0)
        err_cnn_te = 1 - correct / total
        all_results.append(("CNN (Conv×4 + Pool×2 + FC)", None, err_cnn_te))
        print(f"  → Err Test CNN : {err_cnn_te*100:.1f}%")

        # Visualisation feature maps
        visualize_cnn_feature_maps(
            cnn_model,
            X_te_norm[cat_idx].cpu().numpy(),   # (3, 32, 32) normalisé
            device
        )

        # Courbes CNN
        plot_curves([hist_cnn], ["CNN PyTorch"],
                    title="CNN — Courbes d'entraînement CIFAR-10")

    # ─────────────────────────────────────────────────────────────────────────
    # ── Tableau final ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    print_full_comparison(all_results)

    print("""
═══════════════════════════════════════════════════════════════════
  ANALYSE & DISCUSSION — PARTIE 2
───────────────────────────────────────────────────────────────────
  1. POURQUOI LES MLP SONT LIMITÉS SUR CIFAR-10 ?
     - Traiter les 3072 pixels comme un vecteur plat ignore la structure
       spatiale locale (voisinage, bords, textures).
     - Les même poids sont appris pour chaque position de pixel :
       pas d'invariance aux translations.
     - Erreur test typique : ~45-55% (proche d'un classifieur aléatoire !)

  2. RÔLE DES FILTRES DE CONVOLUTION
     K1 (lissage) → supprime le bruit → utile en pré-traitement
     K2 (sharpen) → accentue les contours fins
     K3/K4 → détection directionnelle de bords (horizontal / vertical)
     K5 (Sobel) → estimateur de gradient plus robuste que K4
     K6 (emboss) → effet de relief, combinaison diagonal
     → En CNN, ces filtres sont APPRIS automatiquement, pas choisis à la main.

  3. APPORT DU CNN
     - Partage de poids (même filtre glissé sur toute l'image) :
       beaucoup moins de paramètres qu'un MLP équivalent
     - Invariance locale aux translations (grâce au pooling)
     - Hiérarchie de représentations : bords → formes → objets
     - Erreur test typique : ~25-35% (bien meilleur que MLP !)

  4. COMPARAISON AVEC LA LITTÉRATURE
     Nos résultats restent bien au-dessus des 3% des meilleurs modèles,
     car ceux-ci utilisent :
     • Data augmentation intensive
     • Batch Normalization (stabilise l'apprentissage)
     • Dropout (régularisation)
     • Architectures plus profondes (ResNet, DenseNet, ViT…)

  5. OVERFITTING
     Si err_train << err_test → sur-apprentissage.
     Solutions : dropout, L2 régularisation, data augmentation.
═══════════════════════════════════════════════════════════════════
    """)
