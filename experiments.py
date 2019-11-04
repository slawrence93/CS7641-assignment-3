import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
import collections
import sklearn.metrics as metrics
from sklearn import metrics
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curve(classifier, X, y, cv=3, scoring='accuracy',
                        n_jobs=-1, training_sizes=np.linspace(0.01, 1.0, 50)):
    train_sizes, train_scores, test_scores, _, _ = learning_curve(classifier,
                                                                  X=X,
                                                                  y=y,
                                                                  # Number of folds in cross-validation
                                                                  cv=cv,
                                                                  # Evaluation metric
                                                                  scoring=scoring,
                                                                  # Use all computer cores
                                                                  n_jobs=n_jobs,
                                                                  # 50 different sizes of the training set
                                                                  train_sizes=training_sizes,
                                                                  return_times=True)

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    learning_curve_dict = {'training sizes': train_sizes, 'training mean': train_mean, 'test mean': test_mean}
    df = pd.DataFrame(data=learning_curve_dict)
    df.to_csv('learning_curve.csv')

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    #     plt.savefig("learning_curve.png")
    plt.show()

pen_digits_df = pd.read_csv("pen_digits.csv")
default_of_credit_card_clients_df = pd.read_excel('default_of_credit_card_clients.xls', header=1, index_col=0)
default_of_credit_card_clients_df.columns = map(str.lower, default_of_credit_card_clients_df.columns)
default_of_credit_card_clients_df.rename(columns={'default payment next month': 'default_payment'}, inplace=True)

feature_cols = pen_digits_df.columns.values[:-1]
X_digits = pen_digits_df[feature_cols]
y_digits = pen_digits_df["8"]
X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(X_digits, y_digits, test_size=.3, random_state=42)
default_of_credit_card_clients_df.head()

feature_cols = default_of_credit_card_clients_df.columns.values[:-1]
X_default = default_of_credit_card_clients_df[feature_cols]
y_default = default_of_credit_card_clients_df.default_payment
X_default_train, X_default_test, y_default_train, y_default_test = train_test_split(X_default, y_default, test_size=.35, random_state=42)

kmeans = KMeans(random_state=45)
gaussian_mixture = GaussianMixture(random_state=45)
clustering_stats = collections.defaultdict(list)
for k in range(5, 40, 5):
    clustering_stats['k'].append(k)
    kmeans.set_params(n_clusters=k)
    gaussian_mixture.set_params(n_components=k)
    kmeans.fit(X_digits_train)
    gaussian_mixture.fit(X_digits_train)
    kmeans_labels = kmeans.predict(X_digits_train)
    gaussian_mixture_labels = gaussian_mixture.predict(X_digits_train)
    kmeans_silhouette_score = silhouette_score(X_digits_train, kmeans_labels)
    clustering_stats['kmeans silhouette score'].append(kmeans_silhouette_score)
    gaussian_mixture_silhouette_score = silhouette_score(X_digits_train, gaussian_mixture_labels)
    clustering_stats['gmm silhouette score'].append(gaussian_mixture_silhouette_score)
    kmeans_score = kmeans.score(X_digits_train)
    clustering_stats['kmeans score'].append(kmeans_score)
    clustering_stats['kmeans accuracy score'].append(metrics.accuracy_score(kmeans_labels, y_digits_train))
    gaussian_mixture_score = gaussian_mixture.score(X_digits_train)
    clustering_stats['gmm score'].append(gaussian_mixture_score)
    clustering_stats['gmm accuracy score'].append(metrics.accuracy_score(gaussian_mixture_labels, y_digits_train))
    bayesian_info_score = gaussian_mixture.bic(X_digits_train)
    clustering_stats['gmm bayesian info score'].append(bayesian_info_score)
    kmeans_adjusted_mutual_info_score = adjusted_mutual_info_score(y_digits_train, kmeans_labels)
    clustering_stats['kmeans adjusted mutual info score'].append(kmeans_adjusted_mutual_info_score)
    gaussian_mixture_adjusted_mutual_info_score = adjusted_mutual_info_score(y_digits_train, gaussian_mixture_labels)
    clustering_stats['gmm adjusted mutual info score'].append(gaussian_mixture_adjusted_mutual_info_score)
digits_clustering_df = pd.DataFrame(data=clustering_stats)
digits_clustering_df.to_csv('digits_clustering_data.csv')

clustering_stats = collections.defaultdict(list)
for k in range(2, 16, 2):
    clustering_stats['k'].append(k)
    kmeans.set_params(n_clusters=k)
    gaussian_mixture.set_params(n_components=k)
    kmeans.fit(X_default_train)
    gaussian_mixture.fit(X_default_train)
    kmeans_labels = kmeans.predict(X_default_train)
    gaussian_mixture_labels = gaussian_mixture.predict(X_default_train)
    kmeans_silhouette_score = silhouette_score(X_default_train, kmeans_labels)
    clustering_stats['kmeans silhouette score'].append(kmeans_silhouette_score)
    gaussian_mixture_silhouette_score = silhouette_score(X_default_train, gaussian_mixture_labels)
    clustering_stats['gmm silhouette score'].append(gaussian_mixture_silhouette_score)
    kmeans_score = kmeans.score(X_default_train)
    clustering_stats['kmeans accuracy score'].append(metrics.accuracy_score(kmeans_labels, y_default_train))
    clustering_stats['kmeans score'].append(kmeans_score)
    gaussian_mixture_score = gaussian_mixture.score(X_default_train)
    clustering_stats['gmm score'].append(gaussian_mixture_score)
    clustering_stats['gmm accuracy score'].append(metrics.accuracy_score(gaussian_mixture_labels, y_default_train))
    bayesian_info_score = gaussian_mixture.bic(X_default_train)
    clustering_stats['gmm bayesian info score'].append(bayesian_info_score)
    kmeans_adjusted_mutual_info_score = adjusted_mutual_info_score(y_default_train, kmeans_labels)
    clustering_stats['kmeans adjusted mutual info score'].append(kmeans_adjusted_mutual_info_score)
    gaussian_mixture_adjusted_mutual_info_score = adjusted_mutual_info_score(y_default_train, gaussian_mixture_labels)
    clustering_stats['gmm adjusted mutual info score'].append(gaussian_mixture_adjusted_mutual_info_score)
default_clustering_df = pd.DataFrame(data=clustering_stats)
default_clustering_df.to_csv('default_clustering_data.csv')

num_components = 16
pca_stats = collections.defaultdict(list)
pca = PCA(n_components=num_components)
pca.fit(X_digits_test)
for i in range(1, num_components + 1):
    pca_stats['num_components'].append(i)
pca_stats['variance'].extend(pca.explained_variance_ratio_)
pca_df = pd.DataFrame(data=pca_stats)
pca_df.to_csv('digits_pca_variance.csv')

num_components = 23
pca_stats = collections.defaultdict(list)
pca = PCA(n_components=num_components)
pca.fit(X_default_test)
for i in range(1, num_components + 1):
    pca_stats['num_components'].append(i)
pca_stats['variance'].extend(pca.explained_variance_ratio_)
pca_df = pd.DataFrame(data=pca_stats)
pca_df.to_csv('default_pca_variance.csv')

num_components = 16
kurtosis = collections.defaultdict(list)
for i in range(1, num_components + 1):
    kurtosis['num components'].append(i)
    ica = FastICA(n_components=i)
    ica_transformed_data = ica.fit_transform(X_default_train)
    kurtosis['avg kurtosis'].append(pd.DataFrame(data=ica_transformed_data).kurt(axis=0).abs().mean())
kurtosis_df = pd.DataFrame(data=kurtosis)
kurtosis_df.to_csv('digits_avg_kurtosis.csv')

kurtosis = collections.defaultdict(list)
for i in range(1, num_components + 1):
    kurtosis['num components'].append(i)
    ica = FastICA(n_components=i)
    ica_transformed_data = ica.fit_transform(X_default_train)
    kurtosis['avg kurtosis'].append(pd.DataFrame(data=ica_transformed_data).kurt(axis=0).abs().mean())
kurtosis_df = pd.DataFrame(data=kurtosis)
kurtosis_df.to_csv('default_avg_kurtosis.csv')

num_components = 16
rp_stats = collections.defaultdict(list)
for i in range(1, num_components):
    rp_stats['num components'].append(i)
    rp = SparseRandomProjection(n_components=i)
    nnm = MLPClassifier()
    rp_nnm = Pipeline([('rp', rp), ('nnm', nnm)])
    rp_nnm.fit(X_digits_train, y_digits_train)
    accuracy_score = metrics.accuracy_score(rp_nnm.predict(X_digits_test), y_digits_test)
    rp_stats['accuracy score'].append(accuracy_score)
rp_df = pd.DataFrame(data=rp_stats)
rp_df.to_csv('digits_rp_data.csv')

num_components = 23
rp_stats = collections.defaultdict(list)
for i in range(1, num_components):
    rp_stats['num components'].append(i)
    rp = SparseRandomProjection(n_components=i)
    nnm = MLPClassifier()
    rp_nnm = Pipeline([('rp', rp), ('nnm', nnm)])
    rp_nnm.fit(X_default_train, y_default_train)
    accuracy_score = metrics.accuracy_score(rp_nnm.predict(X_default_test), y_default_test)
    rp_stats['accuracy score'].append(accuracy_score)
rp_df = pd.DataFrame(data=rp_stats)
rp_df.to_csv('default_rp_data.csv')

num_components = 16
lda_stats = collections.defaultdict(list)
for i in range(1, num_components):
    lda_stats['num components'].append(i)
    lda = LDA(n_components=i)
    nnm = MLPClassifier()
    lda_nnm = Pipeline([('lda', lda), ('nnm', nnm)])
    lda_nnm.fit(X_digits_train, y_digits_train)
    accuracy_score = metrics.accuracy_score(lda_nnm.predict(X_digits_test), y_digits_test)
    lda_stats['accuracy score'].append(accuracy_score)
lda_df = pd.DataFrame(data=lda_stats)
lda_df.to_csv('digits_lda_data.csv')

num_components = 23
lda_stats = collections.defaultdict(list)
for i in range(1, num_components):
    lda_stats['num components'].append(i)
    lda = LDA(n_components=i)
    nnm = MLPClassifier()
    lda_nnm = Pipeline([('lda', lda), ('nnm', nnm)])
    lda_nnm.fit(X_default_train, y_default_train)
    accuracy_score = metrics.accuracy_score(lda_nnm.predict(X_default_test), y_default_test)
    lda_stats['lda score'].append(accuracy_score)
lda_df = pd.DataFrame(data=lda_stats)
lda_df.to_csv('default_lda_data.csv')

ica = FastICA(n_components=6)
nnm = MLPClassifier()
ica_nnm = Pipeline([('ica', ica), ('nnm', nnm)])
plot_learning_curve(ica_nnm, X_digits_train, y_digits_train)

ica = FastICA(n_components=6)
nnm = MLPClassifier()
ica_nnm = Pipeline([('ica', ica), ('nnm', nnm)])
plot_learning_curve(ica_nnm, X_default_train, y_default_train)

rp = SparseRandomProjection(n_components=9)
nnm = MLPClassifier()
ica_nnm = Pipeline([('rp', rp), ('nnm', nnm)])
plot_learning_curve(ica_nnm, X_digits_train, y_digits_train)

rp = SparseRandomProjection(n_components=7)
nnm = MLPClassifier()
ica_nnm = Pipeline([('rp', rp), ('nnm', nnm)])
plot_learning_curve(ica_nnm, X_default_train, y_default_train)

lda = LDA(n_components=9)
nnm = MLPClassifier()
lda_nnm = Pipeline([('lda', lda), ('nnm', nnm)])
plot_learning_curve(lda_nnm, X_digits_train, y_digits_train)

lda = LDA(n_components=9)
nnm = MLPClassifier()
lda_nnm = Pipeline([('lda', lda), ('nnm', nnm)])
plot_learning_curve(lda_nnm, X_default_train, y_default_train)