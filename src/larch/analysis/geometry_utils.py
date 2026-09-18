import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def preprocess_embeddings(
        X,
        pca=None,
        center=False,
        normalize_before_pca=True,
        normalize_after_pca=False,
        drop_first_pca=False,
        random_state=0,
        whiten=False,
):

    X = X.astype(np.float32)

    ## Initial L2 normalization
    if normalize_before_pca: X = normalize(X, norm="l2", axis=1)
    
    # Center
    if center: X = X - X.mean(axis=0)
    #print("Centered embeddings:", X.shape)

    # PCA
    if pca is not None:
        pca_model = PCA(n_components=pca,
                        whiten=whiten,
                        svd_solver="randomized",
                        random_state=random_state
                        )
        X = pca_model.fit_transform(X)
        print(f"PCA reduced to {pca} components")

        # Drop first PC
        if drop_first_pca:
            X = X[:, 1:]
            print(f"Dropped first PCA component, new shape {X.shape}")

    ## Optionally normalize at the end
    if normalize_after_pca: X = normalize(X, norm="l2", axis=1)

    return X.astype(np.float32) #, pca_model

def cosine_similarity_distribution(A, B, centering=True):
    A = A.copy()
    B = B.copy()

    # normalize A
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    if centering:
        A = A - A.mean(axis=0)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)

    # normalize B
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    if centering:
        B = B - B.mean(axis=0)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)

    cos_sim = np.sum(A * B, axis=1)

    return cos_sim


def plot_similarity_distributions(nom, aug1, aug2, centering=True, bins=100, save_name=None):

    cos1 = cosine_similarity_distribution(nom, aug1, centering)
    cos2 = cosine_similarity_distribution(aug1, aug2, centering)
    aug2_shuffled = aug2[np.random.permutation(aug2.shape[0])]    
    cos3 = cosine_similarity_distribution(aug1, aug2_shuffled, centering)

    distributions = [
        ("sim(x, aug(x))", cos1),
        (r"sim(aug$_1$(x), aug$_2$(x))", cos2),
        (r"sim(aug(x$_1$), aug(x$_2$))", cos3),
    ]
    
    plt.figure()

    for label, cos_sim in distributions:
        hist, edges = np.histogram(cos_sim, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        mean = cos_sim.mean()
        std = cos_sim.std()

        legend_label = f"{label}"# (μ={mean:.3f}, σ={std:.3f})"

        plt.bar(
            centers,
            hist,
            width=width,
            alpha=0.4,
            edgecolor=None,
            linewidth=0,
            label=legend_label
        )

    plt.yscale("log")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name,
                    dpi=300,
                    bbox_inches='tight')
    plt.show()
    plt.close()


def cosine_spectrum(z):

    if isinstance(z, torch.Tensor): z = z.cpu().numpy()

    ## Normalize
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / (norm + 1e-10)

    ## Center
    z = z - z.mean(axis=0)       
    
    ## Cosine similarity matrix
    G = z.T @ z

    ## Top eigenvalues of the matrix
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.flip(eigvals)
    
    return eigvals


def pca_spectrum(z):
    
    if isinstance(z, torch.Tensor):
        z = z.cpu().numpy()

    ## Normalize
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / (norms + 1e-10)
    
    ## Center
    z = z - z.mean(axis=0)

    s = np.linalg.svd(z, compute_uv=False)
    eigvals = (s ** 2) / (z.shape[0] - 1)    
        
    return eigvals


def plot_spectrum(eigvals, xlim=None,
                  log_scale=True,
                  save_name=None):

    ## Calculate effective number of dimensions
    d_eff = (eigvals.sum() ** 2) / (np.sum(eigvals ** 2))    
    if xlim is not None:
        eigvals = eigvals[:xlim]
    
    plt.figure(figsize=(6,4))

    ndim = eigvals.shape[0]
    ## Optionally show a histogram if we're limiting the range
    edgecolor=None
    if ndim < 100:
        edgecolor='k'

    plt.bar(range(eigvals.shape[0]), eigvals, width=1.0, edgecolor=edgecolor, align='edge')
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Variance")

    ## Somewhat unusually for me, add a title:
    plt.title(rf"d$_{{\mathrm{{eff}}}} = {d_eff:.2f}$")
    if log_scale: plt.yscale('log')
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name,
                    dpi=200,
                    bbox_inches='tight')
    plt.show()
    plt.close()
    return d_eff



def plot_cumulative_variance(eigvals,
                             xlim=None,
                             save_name=None):

    eigvals = np.asarray(eigvals)

    total_var = eigvals.sum()
    cumulative = np.cumsum(eigvals) / total_var

    if xlim is not None:
        eigvals = eigvals[:xlim]
        cumulative = cumulative[:xlim]

    plt.figure(figsize=(6,4))

    x = np.arange(1, len(cumulative)+1)

    plt.plot(x, cumulative)
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.ylim(0, 1.01)

    # Add reference lines
    for frac in [0.9, 0.95, 0.99]:
        plt.axhline(frac, linestyle='--', linewidth=0.8)

    plt.tight_layout()

    if save_name:
        plt.savefig(save_name,
                    dpi=200,
                    bbox_inches='tight')

    plt.show()
    plt.close()
    return
