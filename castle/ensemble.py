
def optimize_n_clusters(X):
    S = []
    for i in np.arange(2, min(len(X)//10, 10)):
        gmm = GaussianMixture(n_components=i, n_init=3, reg_covar=1e-2)
        labels = gmm.fit_predict(X)
        S.append(silhouette_score(X, labels, metric='euclidean'))
    nopt = np.argmax(S*np.arange(1, 1+len(S))**0.5)
    gmm = GaussianMixture(n_components=nopt, n_init=5, reg_covar=1e-2).fit(X)
    print("Using %i clusters" %(nopt))
    return gmm


def cluster_gvect(X, e, n_clusters='auto', clustering='e_gmm'):
    """Auxiliary function that calls the correct clustering
        algorithm. Options are: kmeans clustering and advanced
        density peaks clustering. If the latter is chosen, the
        adp python package must be installed first.
    Args:
        X (np.array): Descriptor vector for each atom in each structure
        e (np.array): Per-atom energy of each structure
        n_clusters (float): Number of clusters
        clustering (str): Clustering algorithm to use

    Returns:
        labels (np.array): cluster (int) assigned
                           to each atom in each structure

    """
    if clustering == 'e_gmm':
        # Resize X based only on global std and energy std, separately
        # So that e is comparable to X but we do not lose information
        # on the magnitude of each component of X.
        mean = np.mean(X, axis=0)
        std = np.std(X)
        X = (X - mean[None, :]) / std[None, None]
        e = (e - np.mean(e)) / np.std(e)
        X = np.concatenate((X, e[:, None]), axis=1)
        if n_clusters == 'auto':
            gmm = optimize_n_clusters(X)
        else:
            gmm = GaussianMixture(n_components=n_clusters, n_init=5, reg_covar=1e-2).fit(X)
        labels = gmm.predict(X)
        weights = gmm.weights_
        centers = gmm.means_[:, :-1] * std + mean[None, :]
        precisions = gmm.precisions_[:, :-1, :-1] / std / 20

    elif clustering == 'gmm':
        if n_clusters == 'auto':
            gmm = optimize_n_clusters(X)
        else:
            gmm = GaussianMixture(n_components=n_clusters, n_init=5, reg_covar=1e-2).fit(X)
        labels = gmm.predict(X)
        weights = gmm.weights_
        centers = gmm.means_ 
        precisions = gmm.precisions_  / 20

    elif clustering == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        precisions = 1/np.array([np.std(X[labels == i], axis = 0) for i in range(n_clusters)])
        weights = np.array([len(labels == i) for i in range(n_clusters)])

    return labels, centers, precisions, weights