import numpy as np

class k_means():
    def __init__(self, K=3, method='kmeans++',max_iter = 1000,tol = 1e-4):
        self.K = K 
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.method = method
        self.labels = None

    def _kmeans_plus_plus(self, X):
        N, D = X.shape
        first_index = np.random.choice(N)
        self.centroids = np.zeros((self.K, D))
        self.centroids[0] = X[first_index].copy()

        min_distance = np.full(N, np.inf)
        for i in range(1, self.K):
            current_centroid = self.centroids[i-1]
            distance = np.sum((X-current_centroid)**2, axis =1)
            min_distance = np.minimum(min_distance, distance)
            tot_distance = np.sum(min_distance)
            if tot_distance==0:
                prob = np.full(N, 1.0/N)
            else:
                prob = min_distance/tot_distance

            chosen_idx = np.random.choice(N, p = prob)
            self.centroids[i] = X[chosen_idx]

        return self


    def fit(self, X):
        N, D = X.shape

        if self.method=='kmeans++':
            self._kmeans_plus_plus(X)
        else:

            idx = np.random.choice(N, self.K, replace=False)
            self.centroids = X[idx].copy()


        for iteration in range(self.max_iter):
            prev_centroids = self.centroids.copy()
            # distance = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis = 2, keepdims=False)

            X_sqr = np.sum(X**2, axis =1)
            Centroid_sqr = np.sum(self.centroids**2, axis=1)
            distance = np.maximum(X_sqr[:, None] + Centroid_sqr[None, :] - 2*(X@self.centroids.T), 0.0)

            self.labels=np.argmin(distance, axis=1)

            one_hot = np.zeros((N, self.K))

            one_hot[np.arange(N), self.labels] = 1.0
            counts = np.sum(one_hot, axis =0)

            non_empty_clusters = (counts>0)
            self.centroids[non_empty_clusters] = (one_hot.T[non_empty_clusters]@X)/counts[non_empty_clusters, None]

            if np.any(counts == 0):
                empty_indices = np.where(~non_empty_clusters)[0]
                num_empty = len(empty_indices)

                # Find points farthest from their assigned centroid
                min_distance_to_centroid = np.min(distance, axis=1)
                farthest_indices = np.argsort(min_distance_to_centroid)[
                    -num_empty:
                ]

                self.centroids[empty_indices] = X[farthest_indices]


            if iteration>0:
                if np.sum((self.centroids-prev_centroids)**2)<self.tol:
                    break

        return self


    def predict(self, X):
        X_sqr = np.sum(X**2, axis=1, keepdims=True)
        Centroid_sqr = np.sum(self.centroids**2, axis=1, keepdims=True).T
        distance = np.maximum(
            0.0, X_sqr + Centroid_sqr - 2 * (X @ self.centroids.T)
        )
        return np.argmin(distance, axis=1)

    
class GaussianMixture():
    def __init__(self, K ):
        self.K = K
        self.means = None
        self.covariance = None
        self.pi = None
        self.gamma = None
        self.regularizer = 1e-6




    def _log_density(self, X, l):

        N, D = X.shape

        cov_matrix = self.covariance[l] + self.regularizer*np.eye(D)
        L = np.linalg.cholesky(cov_matrix)
        exp_vec  = np.linalg.solve(L, (X-self.means[l]).T)
        log_det = np.sum(np.log(np.diag(L)))

        return -0.5*(D*np.log(2*np.pi) + 2*log_det + np.sum(exp_vec**2, axis =0))
        

    
    def _e_step(self, X, N, D):

        self.gamma = np.zeros((N,self.K))

        for l in range(self.K):

            log_dens = self._log_density(X, l) 
            self.gamma[:, l] = np.log(self.pi[l] + 1e-8) + log_dens
        
        max_value = np.max(self.gamma, axis =1, keepdims=True)

        log_sum = max_value + np.log(np.sum(np.exp(self.gamma - max_value), axis=1, keepdims=True))

        self.gamma = np.exp(self.gamma - log_sum)

        return self.gamma

    def _m_step(self, X, N, D):
        self.pi    =  (1.0/N)*np.sum(self.gamma, axis=0)
        self.means  =  self.gamma.T@X/(N*self.pi[:, None])

        for l in range(self.K):
            diff = X - self.means[l] 
            self.covariance[l] = (diff.T @ (diff * self.gamma[:, l][:, None])) / (N * self.pi[l])


        return self
        

    def fit(self, X, max_iter=100, tol=1e-4):
        N, D = X.shape


        self.pi = np.full(self.K, fill_value=(1.0 / self.K))
        random_selection= np.random.choice(N, self.K, replace=False)

        km_init = k_means(K=self.K, method="kmeans++", max_iter=20).fit(X)
        self.means = km_init.centroids.copy()
        # self.means = X[random_selection].copy()
        self.covariance = np.array([np.eye(D) for _ in range(self.K)])

        for iteration in range(max_iter):
            prev_means = self.means.copy()
            self._e_step(X, N, D)
            self._m_step(X, N, D)

            if iteration > 0:
                mean_change = np.linalg.norm(self.means - prev_means)
                if mean_change < tol:
                    break

        return self

    def predict(self, X):
        N, D = X.shape
        gamma = np.zeros((N, self.K))

        for l in range(self.K):
            log_dens = self._log_density(X, l)
            gamma[:, l] = np.log(self.pi[l] + 1e-8) + log_dens

        max_value = np.max(gamma, axis=1, keepdims=True)
        log_sum = max_value + np.log(np.sum(np.exp(gamma - max_value), axis=1, keepdims=True))
        gamma = np.exp(gamma - log_sum)

        return np.argmax(gamma, axis=1)