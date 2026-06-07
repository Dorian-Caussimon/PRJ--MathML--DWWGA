import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

#PyTorch
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

#MLP SUR CIFAR-10
#Chargement CIFAR-10
CIFAR10_CLASSES = [
    "avion", "automobile", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]

def load_cifar10_numpy():
    print("Chargement de CIFAR-10... (téléchargement automatique si absent)")
    transform = transforms.ToTensor()

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)

    def dataset_to_numpy(ds):
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
        X_t, y_t = next(iter(loader))
        X = X_t.numpy().transpose(0, 2, 3, 1).astype(np.float32)
        y = y_t.numpy()
        return X, y

    X_train, y_train = dataset_to_numpy(train_set)
    X_test,  y_test  = dataset_to_numpy(test_set)

    print(f"  ✓ Train : {X_train.shape}  | Test : {X_test.shape}")
    return X_train, y_train, X_test, y_test

def visualize_cifar10(X, y, n_per_class=8):
    """Affiche n_per_class exemples pour 10 classes"""
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

def rgb_to_grayscale(X_rgb):
    """Convertit des images RGB en niveaux de gris (formule standard)"""
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (X_rgb * w).sum(axis=-1)

def flatten_images(X):
    """Aplatit chaque image en vecteur ligne."""
    return X.reshape(X.shape[0], -1).astype(np.float32)

def softmax(O):
    """Softmax (soustraction du max ligne à ligne)"""
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
    """Perceptron multi-couches
      layer_sizes un truc genre [1024, 256, 10] pour H=1 couche cachée sur images grises
      [3072, 256, 10] pour H=1 couche cachée sur images couleur"""
    def __init__(self, layer_sizes):
        self.sizes = layer_sizes
        self.L     = len(layer_sizes) - 1
        #Initialisation He
        self.W = [np.random.randn(layer_sizes[l+1], layer_sizes[l]).astype(np.float32)
                  * np.sqrt(2.0 / layer_sizes[l])
                  for l in range(self.L)]
        self.b = [np.zeros(layer_sizes[l+1], dtype=np.float32) for l in range(self.L)]

    def forward(self, X):
        """Forward pass avec mise en cache (nécessaire pour backprop)"""
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
        """Rétropropagation du gradient"""
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
        """Mini-batch SGD"""
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

STATE_OF_ART = [
    ("Conv. Deep Belief Nets",            21.1, "août 2010"),
    ("Maxout Networks",                    9.38, "févr. 2013"),
    ("Fractional Max-Pooling",             3.47, "déc. 2014"),
    ("Densely Connected Conv. Nets",       3.46, "août 2016"),
    ("Coupled Ensembles",                  2.68, "sept. 2017"),
    ("ViT (16×16 Words)",                  0.50, "juin 2021"),
]

def print_full_comparison(our_results):
    """Affiche un tableau récapitulatif complet
    nos résultats + état"""
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

def convolve2d(image, K, bias=0.0):
    H, W   = image.shape
    kH, kW = K.shape
    pH, pW = kH // 2, kW // 2 #padding=1

    #Ajoute une bordure de 0 (1 pixel tout autour)
    padded = np.pad(image, ((pH, pH), (pW, pW)), mode="constant", constant_values=0)

    #Convolution (produit scalaire filtre*patch pour chaque pixel)
    out = np.zeros((H, W), dtype=np.float32)
    for u in range(H):
        for v in range(W):
            patch = padded[u:u+kH, v:v+kW] #patch=3×3
            out[u, v] = np.sum(K * patch) + bias
    return out

def convolve2d_fast(image, K, bias=0.0):
    """Version vectorisée de la convolution 2D"""
    H, W   = image.shape
    kH, kW = K.shape
    pH     = kH // 2
    padded = np.pad(image, pH, mode="constant", constant_values=0)
    out    = np.zeros((H, W), dtype=np.float32)
    for du in range(kH):
        for dv in range(kW):
            out += K[du, dv] * padded[du:du+H, dv:dv+W]
    return out + bias

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

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    #Image originale
    axes[0].imshow(image_gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Image originale", fontweight="bold", fontsize=11)
    axes[0].axis("off")

    for i, (name, K) in enumerate(FILTERS.items()):
        fm = convolve2d_fast(image_gray, K, bias=0.0)
        if clip:
            fm = np.clip(fm, 0, 1)
        #Normalisation min-max
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

    axes[7].axis("off")

    fig.suptitle("Effets des 6 filtres de convolution (section 2.3.2)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("convolution_filters.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("OK Visualisation sauvegardée : convolution_filters.png")

def convolve_color(image_rgb, K_R, K_G, K_B, bias=0.0):
    """Convolution sur une image couleur RGB avec trois filtres séparés"""
    R = convolve2d_fast(image_rgb[:, :, 0], K_R)
    G = convolve2d_fast(image_rgb[:, :, 1], K_G)
    B = convolve2d_fast(image_rgb[:, :, 2], K_B)
    return R + G + B + bias

if True:
    class CIFAR10_CNN(nn.Module):
        """Architecture convolutive"""
        def __init__(self, num_classes=10):
            super().__init__()

            #Couche 1 Convolution couleur
            self.conv1 = nn.Conv2d(3,  64, kernel_size=3, padding=1)

            #Couche 2 Convolution 3D
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            #MaxPooling 1
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            #Couche 3 Convolution 3D_2
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            #MaxPooling 2
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            #Couche 4 Convolution 3D finale
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            #Aplatissement + Densification
            self.fc = nn.Linear(64 * 8 * 8, num_classes)

            self.relu = nn.ReLU()

        def forward(self, x):
            """forward pass"""
            #Conv1 + ReLU
            x = self.relu(self.conv1(x))

            #Conv2 + ReLU
            x = self.relu(self.conv2(x))

            #MaxPool1
            x = self.pool1(x)

            #Conv3 + ReLU
            x = self.relu(self.conv3(x))

            #MaxPool2
            x = self.pool2(x)

            #Conv4 + ReLU
            x = self.relu(self.conv4(x))

            #Aplatissement
            x = torch.flatten(x, 1)

            #Dense (logits, softmax dans la loss)
            x = self.fc(x)
            return x
    # Train
    def train_cnn(epochs=20, lr=0.001, batch_size=128):

        print("\nChargement CIFAR-10 (format PyTorch avec normalisation)...")

        #Normalisation (calculés sur le jeu d'entraînement CIFAR-10)
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(), #data augmentation légère
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True,  transform=transform_train, download=True)
        test_set  = torchvision.datasets.CIFAR10(
            root="./data", train=False, transform=transform_test,  download=True)

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_set,  batch_size=512,
                                  shuffle=False, num_workers=0)

        model     = CIFAR10_CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        #Compte les paramètres
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Nombre de paramètres entraînables : {total_params:,}")

        history = {"loss": [], "train_err": [], "test_err": []}

        for epoch in range(epochs):
            #Phase train
            model.train()
            ep_loss, correct_tr, total_tr = 0.0, 0, 0

            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)

                optimizer.zero_grad()
                logits = model(Xb)
                loss   = criterion(logits, yb) #cross-entropy
                loss.backward()
                optimizer.step()

                ep_loss   += loss.item()
                preds      = logits.argmax(dim=1)
                correct_tr += (preds == yb).sum().item()
                total_tr   += yb.size(0)

            #Phase évaluation
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

#Visualisations communes

def plot_curves(histories, labels, title="Courbes d'entraînement — CIFAR-10"):
    """Compare les courbes de perte et d'erreur"""
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

    model.eval()
    x = torch.tensor(X_sample).unsqueeze(0).to(device)
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
            fm = act[0, f]
            ax.imshow(fm, cmap="viridis", aspect="auto")
            ax.axis("off")
        axes[l, 0].set_ylabel(name, fontsize=8, rotation=0, labelpad=80, va="center")

    fig.suptitle("Feature maps du CNN — 8 filtres par couche",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cnn_feature_maps.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("  ✓ Feature maps sauvegardées : cnn_feature_maps.png")

#Prinp.
if True:

    print("\n" + "═"*60)
    print("  PARTIE 2 — Classification CIFAR-10 + CNN")
    print("═"*60)

    print("\n[1] Chargement de CIFAR-10...")
    X_train_rgb, y_train, X_test_rgb, y_test = load_cifar10_numpy()
    visualize_cifar10(X_train_rgb, y_train)

    all_results = []

    #Conversion niveaux de gris
    print("\n[2] Conversion en niveaux de gris (formule ITU-R)...")
    X_train_gray = rgb_to_grayscale(X_train_rgb)
    X_test_gray  = rgb_to_grayscale(X_test_rgb)

    #Aplatissement en vecteurs
    X_tr_flat_g  = flatten_images(X_train_gray)
    X_te_flat_g  = flatten_images(X_test_gray)
    Y_tr         = one_hot(y_train)

    #MLP sur images grises
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

    #MLP sur images couleur
    print("\n[4] MLP sur images couleur (3072 → 10)...")
    print(f"    Paramètres : {3072*10 + 10:,}")
    X_tr_flat_c = flatten_images(X_train_rgb)
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

    #Courbes comparatives MLP
    plot_curves(
        [hist_lin_gray, hist_mlp1_gray, hist_lin_color, hist_mlp1_c],
        ["Linéaire gris", "MLP H=1 gris", "Linéaire couleur", "MLP H=1 couleur"],
        title="Comparaison MLP — CIFAR-10"
    )

    #Filtres de convolution
    print("\n[5] Application des filtres de convolution...")
    #Test
    cat_idx = np.where(y_test == 3)[0][0]
    img_rgb  = X_test_rgb[cat_idx]
    img_gray = X_test_gray[cat_idx]

    apply_and_visualize_filters(img_gray, clip=True)

    fm_naive = convolve2d(img_gray, FILTERS["K2 — Sharpen (netteté)"])
    fm_fast  = convolve2d_fast(img_gray, FILTERS["K2 — Sharpen (netteté)"])
    max_diff = np.abs(fm_naive - fm_fast).max()
    print(f"  ✓ Vérification : diff max naïf vs vectorisé = {max_diff:.2e} (doit être ≈ 0)")

    #CNN PyTorch
    if TORCH_AVAILABLE:
        print("\n[6] Entraînement du CNN (Option B — PyTorch)...")
        print("  Architecture : Conv(64)→Conv(64)→Pool→Conv(64)→Pool→Conv(64)→FC(10)")
        cnn_model, hist_cnn = train_cnn(epochs=20, lr=0.001, batch_size=128)

        #Erreur finale CNN
        cnn_model.eval()
        correct, total = 0, 0
        X_te_t = torch.tensor(
            X_test_rgb.transpose(0,3,1,2),
            dtype=torch.float32
        ).to(device)
        #Normalisation
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

        #Visualisation feature maps
        visualize_cnn_feature_maps(
            cnn_model,
            X_te_norm[cat_idx].cpu().numpy(),
            device
        )

        #Courbes CNN
        plot_curves([hist_cnn], ["CNN PyTorch"],
                    title="CNN — Courbes d'entraînement CIFAR-10")

    #
    #Tableau final
    print_full_comparison(all_results)

    print("""
  ANALYSE & DISCUSSION - 2

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
    """)
