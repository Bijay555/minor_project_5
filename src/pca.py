from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAEmbeddings():
    """
    A class used to apply PCA on dataset.

    ...

    Attributes
    ----------
    feature_vectors : list
        List of feature vectors

    n_components : int
        Number of components to use in PCA

    pca : sklearn.decomposition._pca.PCA
        A pca object 

    scaled_vectors : numpy.ndarray
        A scaled vectors with mean zero and standard deviation one

    feature_vectors_pca : numpy.ndarray
        Feature vectors after PCA decomposition


    Methods
    -------
    standardize() 
        Creates scaled vectors with mean zero and standard deviation one

    transform()
        Uses PCA to transform the feature vectors
    """

    def __init__(self, feature_vectors, n_components):
        self.feature_vectors = feature_vectors
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.standardize()

    def standardize(self):
        """
        Computes scaled vectors with mean zero and standard deviation one

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        scalar = StandardScaler()
        scalar.fit(self.feature_vectors)
        self.scaled_vectors = scalar.transform(self.feature_vectors)

    def transform(self):
        """
        Uses PCA to transform the feature vectors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.pca.fit(self.scaled_vectors)
        self.feature_vectors_pca = self.pca.transform(self.scaled_vectors)
        return self.feature_vectors_pca
