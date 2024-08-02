import sklearn.decomposition
import sklearn.preprocessing

class KpcaEmbedder:
    # def __init__(self):
    #     pass

    def _preprocess(self, data):
        """
        Divide each feature in the data by its sample standard deviation.

        Args:
            data (sparray): (t, d) where t is the number of time points and
                d is the dimensionality of the sparse array. Each row 
                represents a full flattened representation of the data
                at each timestamp.
        Returns:
            (t, d) scipy scaled scipy sparse array .
        """
        return sklearn.preprocessing.scale(data, with_mean=False)

    def embed(self, data, n_components=2, gamma=None, preprocess=True):
        """
        Args:
            data (sparray): (t, d) where t is the number of time points and
                d is the dimensionality of the sparse array. Each row 
                represents a full flattened representation of the data
                at each timestamp.
            n_components (int): dimensionality of embedding space.
            gamma (float): hyperparameter for rbf kernel. A good setting
                is 1/(num_ips**2).
        Returns:
            (t, n_components) np_array representing the embedding for each
                timestamp in the input data.
        """
        if gamma is None:
            # TODO: can we find out how many unique IPs are in the data?
            n_features = data.shape[0]
            gamma = 1./n_features

        if preprocess:
            data = self._preprocess(data)

        kpca = sklearn.decomposition.KernelPCA(n_components=n_components,
                kernel = 'rbf', gamma=gamma)
        transformed = kpca.fit_transform(data)

        return transformed

