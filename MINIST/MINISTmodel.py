import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ── 1. Charger une image groupe (grille de chiffres 28×28) ──────────────────
def charger_grille(chemin_image, n_cols, n_rows, taille_pixel=28):
    """
    Découpe une image grille en chiffres individuels 28×28.
    Retourne un tableau numpy (N, 784).
    """
    img = Image.open(chemin_image).convert('L')  # niveaux de gris
    img_array = np.array(img)

    H, W = img_array.shape
    h_chiffre = H // n_rows
    w_chiffre = W // n_cols

    images = []
    for row in range(n_rows):
        for col in range(n_cols):
            y0 = row * h_chiffre
            y1 = y0 + h_chiffre
            x0 = col * w_chiffre
            x1 = x0 + w_chiffre

            chiffre = img_array[y0:y1, x0:x1]
            # Redimensionner à 28×28 si nécessaire
            chiffre_pil = Image.fromarray(chiffre).resize((28, 28))
            chiffre_vec = np.array(chiffre_pil).flatten()  # vecteur ℝ^784
            images.append(chiffre_vec)

    return np.array(images)  # shape (N, 784)


# ── 2. Charger directement depuis keras (recommandé) ────────────────────────
from tensorflow.keras.datasets import mnist  # ou : sklearn, torchvision

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train : (60000, 28, 28) → on vectorise en (60000, 784)
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# ── 3. Normalisation ─────────────────────────────────────────────────────────
# Pixels entre 0 et 255 → on ramène entre 0 et 1
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"X_train : {X_train.shape}, valeurs : [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"X_test  : {X_test.shape}")
print(f"Classes présentes : {np.unique(y_train)}")


# ── 4. Visualiser quelques chiffres ─────────────────────────────────────────
def afficher_grille(X, y, n=10):
    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(X[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"y={y[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


'''afficher_grille(X_train, y_train)

# ── 5. Si vous utilisez vos propres fichiers PNG (comme le vôtre) ───────────
chemin = "GROUPS\mnist_v5_MNIST-1_00001-01000_25x40.png"
X_groupe = charger_grille(chemin, n_cols=40, n_rows=25)
y_groupe = np.full(len(X_groupe), fill_value=9)  # tous des "9"

X_groupe = X_groupe / 255.0
print(f"Groupe chargé : {X_groupe.shape}")  # (1000, 784)'''