import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

#Pt1

def load_mnist():
    """Chargement dataset MNIST"""
    print("Chargement de MNIST (peut prendre ~30s la première fois)")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)
    print(f" X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y

def normalize(X):
    """Normalisation Min-Max"""
    return X / 255.0

def one_hot_encode(y, num_classes=10):
    """Encode les étiquettes y (scalaires) en matrice one-hot"""
    n = len(y)
    Y = np.zeros((n, num_classes), dtype=np.float32)
    Y[np.arange(n), y] = 1.0
    return Y

def split_data(X, y, test_size=10000):
    """Sépare les données en training set et test set)"""
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    print(f"  Train : {X_train.shape[0]} exemples | Test : {X_test.shape[0]} exemples")
    return X_train, X_test, y_train, y_test

def visualize_samples(X, y, n=10):
    """Affiche n exemples de chaque chiffre (0 à 9)"""
    fig, axes = plt.subplots(10, n, figsize=(n * 1.2, 13))
    fig.suptitle("Exemples de chiffres manuscrits (MNIST)", fontsize=14, y=1.01)
    for digit in range(10):
        idx = np.where(y == digit)[0][:n]
        for j, i in enumerate(idx):
            ax = axes[digit, j]
            ax.imshow(X[i].reshape(28, 28), cmap="gray_r", interpolation="nearest")
            ax.axis("off")
        axes[digit, 0].set_ylabel(str(digit), fontsize=11, rotation=0, labelpad=15)
    plt.tight_layout()
    plt.savefig("mnist_samples.png", dpi=100, bbox_inches="tight")
    plt.show()
    print(" Visualisation sauvegardée : mnist_samples.png")

#Pt2

def softmax(O):
    """Fonction softmax appliquée ligne par ligne sur la matrice des scores O"""
    O_stable = O - O.max(axis=1, keepdims=True) #en gros pour la stabilité numérique
    E = np.exp(O_stable)
    return E / E.sum(axis=1, keepdims=True)

def cross_entropy_loss(P, Y):
    """Fonction coût entropie croisée (log loss) multi-classe"""
    eps = 1e-12 #évite log(0)
    return -np.mean(np.sum(Y * np.log(P + eps), axis=1))

def accuracy(y_pred, y_true):
    """Accuracy"""
    return np.mean(y_pred == y_true)

def error_rate(y_pred, y_true):
    """Error = 1 - accuracy"""
    return 1.0 - accuracy(y_pred, y_true)

#Pt3

class LinearModel:
    """Modèle linéaire"""
    def __init__(self, input_dim=784, num_classes=10):
        #Initialisation avec une loi normale
        self.A = np.random.randn(num_classes, input_dim).astype(np.float32) * 0.01
        self.b = np.zeros(num_classes, dtype=np.float32)

    def forward(self, X):
        """Calcule les probabilités pour chaque exemple de X"""
        O = X @ self.A.T + self.b
        P = softmax(O)
        return P

    def predict(self, X):
        """Prédit la classe (0-9) pour chaque exemple"""
        P = self.forward(X)
        return np.argmax(P, axis=1)

    @staticmethod
    def compute_gradient(X, Y, P):
        """Gradient de la cross-entropy"""
        n = X.shape[0]
        delta = (P - Y) / n
        dA = delta.T @ X
        db = delta.mean(axis=0)
        return dA, db

    def train(self, X_train, Y_train, X_test, y_test,
              lr=0.1, epochs=50, batch_size=256):
        """Descente de gradient"""
        n = X_train.shape[0]
        history = {"loss": [], "train_err": [], "test_err": []}
        for epoch in range(epochs):
            #Mélange aléatoire
            perm = np.random.permutation(n)
            X_sh, Y_sh = X_train[perm], Y_train[perm]
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, batch_size):
                Xb = X_sh[start:start + batch_size]
                Yb = Y_sh[start:start + batch_size]

                #Forward
                P = self.forward(Xb)

                #Loss
                epoch_loss += cross_entropy_loss(P, Yb)
                num_batches += 1

                #Backward
                dA, db = self.compute_gradient(Xb, Yb, P)

                #Màj paramètres
                self.A -= lr * dA
                self.b -= lr * db

            #Test fin
            avg_loss = epoch_loss / num_batches
            train_err = error_rate(self.predict(X_train), np.argmax(Y_train, axis=1))
            test_err  = error_rate(self.predict(X_test),  y_test)
            history["loss"].append(avg_loss)
            history["train_err"].append(train_err)
            history["test_err"].append(test_err)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Époque {epoch+1:3d}/{epochs} | "
                      f"Loss={avg_loss:.4f} | "
                      f"Err Train={train_err*100:.2f}% | "
                      f"Err Test={test_err*100:.2f}%")
        return history

#Pt4

def relu(z):
    """ReLU (Rectified Linear Unit. ReLU(z) = max(0, z))"""
    return np.maximum(0.0, z)

def relu_deriv(z):
    """Dérivée de ReLU"""
    return (z > 0).astype(np.float32)

class MLP:
    """Réseau de neurones multi-couches (Multi-Layer Perceptron)"""
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases  = []

        for l in range(self.num_layers):
            fan_in  = layer_sizes[l]
            fan_out = layer_sizes[l + 1]
            std = np.sqrt(2.0 / fan_in) #initialisation He
            W = np.random.randn(fan_out, fan_in).astype(np.float32) * std
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """Propagation avant (forward pass)"""
        cache = []
        z = X #entrée initiale
        for l in range(self.num_layers):
            z_prev = z
            o = z_prev @ self.weights[l].T + self.biases[l]
            #ReLU
            if l < self.num_layers - 1:
                z = relu(o)
            else:
                z = softmax(o)
            cache.append((z_prev, o, z))
        P = cache[-1][2] #prob softmax
        return P, cache

    def predict(self, X):
        """Prédit la classe pour chaque exemple (argmax des probs)"""
        P, _ = self.forward(X)
        return np.argmax(P, axis=1)

    def backward(self, cache, Y):
        """Rétropropagation du gradient (Backpropagation)"""
        n = Y.shape[0]
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers
        #Couche de sortie
        z_prev, o_out, P = cache[-1]
        delta = (P - Y) / n
        grads_W[-1] = delta.T @ z_prev
        grads_b[-1] = delta.mean(axis=0)

        #Couches cachées
        for l in range(self.num_layers - 2, -1, -1):
            z_prev, o, z = cache[l]

            delta = (delta @ self.weights[l + 1]) * relu_deriv(o)

            grads_W[l] = delta.T @ z_prev
            grads_b[l] = delta.mean(axis=0)
        return grads_W, grads_b

    def train(self, X_train, Y_train, X_test, y_test,
              lr=0.05, epochs=50, batch_size=256):
        """Training par mini-batch SGD (+rétropropagation)"""
        n = X_train.shape[0]
        y_train_labels = np.argmax(Y_train, axis=1)
        history = {"loss": [], "train_err": [], "test_err": []}

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_sh, Y_sh = X_train[perm], Y_train[perm]
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, batch_size):
                Xb = X_sh[start:start + batch_size]
                Yb = Y_sh[start:start + batch_size]

                #Forward
                P, cache = self.forward(Xb)

                #Loss
                epoch_loss += cross_entropy_loss(P, Yb)
                num_batches += 1

                #Backward
                grads_W, grads_b = self.backward(cache, Yb)

                #Màj descente de gradient
                for l in range(self.num_layers):
                    self.weights[l] -= lr * grads_W[l]
                    self.biases[l]  -= lr * grads_b[l]
            avg_loss = epoch_loss / num_batches
            train_err = error_rate(self.predict(X_train), y_train_labels)
            test_err  = error_rate(self.predict(X_test),  y_test)
            history["loss"].append(avg_loss)
            history["train_err"].append(train_err)
            history["test_err"].append(test_err)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Époque {epoch+1:3d}/{epochs} | "
                      f"Loss={avg_loss:.4f} | "
                      f"Err Train={train_err*100:.2f}% | "
                      f"Err Test={test_err*100:.2f}%")
        return history

#Pt5

def confusion_matrix_custom(y_pred, y_true, num_classes=10):
    """Matrice de confusion K×K"""
    C = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    return C

def plot_confusion_matrix(C, title="Matrice de confusion"):
    """Affiche la matrice de confusion avec heatmap"""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Classe prédite", fontsize=11)
    ax.set_ylabel("Classe réelle", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for i in range(10):
        for j in range(10):
            color = "white" if C[i, j] > C.max() * 0.6 else "black"
            ax.text(j, i, str(C[i, j]), ha="center", va="center",
                    fontsize=8, color=color)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    plt.show()

def plot_misclassified(X_test, y_test, y_pred, n=20, title="Exemples mal classés"):
    """Affiche n exemples mal classifiés avec vraie et prédite étiquette"""
    wrong_idx = np.where(y_pred != y_test)[0][:n]
    cols = 10
    rows = max(1, (len(wrong_idx) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes = axes.flatten() if rows > 1 else [axes] if len(wrong_idx) == 1 else axes.flatten()
    for i, idx in enumerate(wrong_idx):
        axes[i].imshow(X_test[idx].reshape(28, 28), cmap="gray_r")
        axes[i].set_title(f"V:{y_test[idx]}\nP:{y_pred[idx]}", fontsize=8,
                          color="red" if y_pred[idx] != y_test[idx] else "green")
        axes[i].axis("off")
    for i in range(len(wrong_idx), len(axes)):
        axes[i].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("misclassified.png", dpi=100)
    plt.show()

def plot_training_curves(histories, labels):
    """Comparaison des courbes d'entraînement des diff modèles"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#2563EB", "#DC2626", "#16A34A"]
    for (h, label, c) in zip(histories, labels, colors):
        axes[0].plot(h["loss"], label=label, color=c, linewidth=2)
        axes[1].plot([e * 100 for e in h["test_err"]], label=label,
                     color=c, linewidth=2, linestyle="--")
        axes[1].plot([e * 100 for e in h["train_err"]], color=c,
                     linewidth=1, linestyle=":", alpha=0.6)
    axes[0].set_title("Évolution de la loss (Cross-Entropy)", fontweight="bold")
    axes[0].set_xlabel("Époques")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Taux d'erreur (— test  ··· train)", fontweight="bold")
    axes[1].set_xlabel("Époques")
    axes[1].set_ylabel("Erreur (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=100)
    plt.show()

def visualize_pca_2d(X_test, y_test, n_samples=3000):
    """Projection en 2D par ACP (PCA) pour visualiser la séparabilité des classes"""
    idx = np.random.choice(len(X_test), n_samples, replace=False)
    X_sub, y_sub = X_test[idx], y_test[idx]
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_sub)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors_map = plt.cm.get_cmap("tab10", 10)
    for digit in range(10):
        mask = (y_sub == digit)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors_map(digit)],
                   label=str(digit), s=8, alpha=0.6)
    ax.set_title(f"Projection PCA 2D de MNIST ({n_samples} exemples)",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend(title="Chiffre", markerscale=3, loc="upper right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("pca_2d.png", dpi=100)
    plt.show()
    print(f"  Variance expliquée (PC1+PC2) : "
          f"{sum(pca.explained_variance_ratio_[:2])*100:.1f}%")

def print_comparison_table(results):
    """Affiche un tableau récapitulatif des perfs"""
    print("\n" + "═" * 60)
    print("  TABLEAU COMPARATIF DES PERFORMANCES")
    print("═" * 60)
    print(f"  {'Modèle':<30} {'Err Train':>10} {'Err Test':>10}")
    print("─" * 60)
    for name, train_err, test_err in results:
        print(f"  {name:<30} {train_err*100:>9.2f}% {test_err*100:>9.2f}%")
    print("═" * 60)

#Pt6

if True :
    print("\n" + "═"*60)
    print("  PARTIE 1 — Classification MNIST")
    print("═"*60)
    #Chargement et prétraitement
    print("\n[1/5] Chargement des données MNIST...")
    X, y = load_mnist()
    X = normalize(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    Y_train = one_hot_encode(y_train)   # (60000, 10)
    print("\n[2/5] Visualisation des échantillons...")
    visualize_samples(X_train, y_train)

    results = [] #stockage TE

    #Modèle linéaire
    print("\n" + "─"*50)
    print("[3/5] Entraînement — Modèle linéaire (sans couche cachée)")
    print("  Paramètres : A ∈ ℝ^(10×784), b ∈ ℝ^10")
    print(f"  Nombre total de paramètres : {10*784 + 10:,}")
    print("─"*50)
    linear_model = LinearModel(input_dim=784, num_classes=10)
    hist_linear = linear_model.train(
        X_train, Y_train, X_test, y_test,
        lr=0.1, epochs=50, batch_size=256
    )
    err_lin_train = error_rate(linear_model.predict(X_train), y_train)
    err_lin_test  = error_rate(linear_model.predict(X_test),  y_test)
    results.append(("Linéaire", err_lin_train, err_lin_test))
    print(f"\n  → Taux d'erreur final | Train : {err_lin_train*100:.2f}% | Test : {err_lin_test*100:.2f}%")

    #MLP couche cachée 1
    print("\n" + "─"*50)
    print("[4/5] Entraînement — MLP avec 1 couche cachée (H=1)")
    print("  Architecture : 784 → 256 → 10")
    print("  Activation : ReLU (couche cachée) + Softmax (sortie)")
    print(f"  Nombre de paramètres : {784*256+256 + 256*10+10:,}")
    print("─"*50)
    mlp1 = MLP(layer_sizes=[784, 256, 10])
    hist_mlp1 = mlp1.train(
        X_train, Y_train, X_test, y_test,
        lr=0.05, epochs=50, batch_size=256
    )
    err_mlp1_train = error_rate(mlp1.predict(X_train), y_train)
    err_mlp1_test  = error_rate(mlp1.predict(X_test),  y_test)
    results.append(("MLP H=1 (784→256→10)", err_mlp1_train, err_mlp1_test))
    print(f"\n  → Taux d'erreur final | Train : {err_mlp1_train*100:.2f}% | Test : {err_mlp1_test*100:.2f}%")

    # '' 2
    print("\n" + "─"*50)
    print("[5/5] Entraînement — MLP avec 2 couches cachées (H=2)")
    print("  Architecture : 784 → 256 → 128 → 10")
    print("  Activation : ReLU (couches cachées) + Softmax (sortie)")
    print(f"  Nombre de paramètres : {784*256+256 + 256*128+128 + 128*10+10:,}")
    print("─"*50)
    mlp2 = MLP(layer_sizes=[784, 256, 128, 10])
    hist_mlp2 = mlp2.train(
        X_train, Y_train, X_test, y_test,
        lr=0.05, epochs=50, batch_size=256
    )
    err_mlp2_train = error_rate(mlp2.predict(X_train), y_train)
    err_mlp2_test  = error_rate(mlp2.predict(X_test),  y_test)
    results.append(("MLP H=2 (784→256→128→10)", err_mlp2_train, err_mlp2_test))
    print(f"\n  → Taux d'erreur final | Train : {err_mlp2_train*100:.2f}% | Test : {err_mlp2_test*100:.2f}%")

    #Visualisations et analyse
    print("\n[6/6] Visualisations & analyse des erreurs...")
    #Courbes d'entraînement
    plot_training_curves(
        [hist_linear, hist_mlp1, hist_mlp2],
        ["Linéaire", "MLP H=1", "MLP H=2"]
    )
    #Matrice de confusion (modèle MLP H=2)
    y_pred_mlp2 = mlp2.predict(X_test)
    C = confusion_matrix_custom(y_pred_mlp2, y_test)
    plot_confusion_matrix(C, title="Matrice de confusion — MLP H=2 (sur test)")
    #Mal classés
    plot_misclassified(X_test, y_test, y_pred_mlp2,
                       title="Exemples mal classés — MLP H=2")
    # PCA 2D
    visualize_pca_2d(X_test, y_test)
    # Tableau récap
    print_comparison_table(results)
    print("""
═══════════════════════════════════════════════════════════════
  ANALYSE & DISCUSSION
───────────────────────────────────────────────────────────────
  1. MODÈLE LINÉAIRE
     - Erreur test typique : ~8% (bon point de départ)
     - Limitation : séparabilité linéaire seulement.
       Le modèle apprend un "prototype" par classe mais ne peut
       pas capturer les transformations non-linéaires de l'écriture.

  2. MLP H=1 (256 neurones cachés)
     - Erreur test typique : ~2-3%
     - La couche cachée avec ReLU permet d'apprendre des
       combinaisons non-linéaires de pixels → bien meilleur.

  3. MLP H=2 (256 + 128 neurones)
     - Erreur test typique : ~1.5-2%
     - Légère amélioration, mais risque de sur-apprentissage
       si le modèle est trop grand sans régularisation.

  4. CHIFFRES AMBIGUS (cf. matrice de confusion)
     - 4 ↔ 9 : formes similaires dans le haut
     - 3 ↔ 5 : courbes similaires
     - 1 ↔ 7 : selon l'inclinaison

  5. POURQUOI MNIST RESTE SIMPLE vs CIFAR-10 ?
     - Images centrées, fond uniforme, une seule classe par image
     - Pas de bruit de fond ni de variabilité de point de vue
     - 28×28 = 784 entrées seulement (vs 3072 pour CIFAR-10)
═══════════════════════════════════════════════════════════════
    """)
