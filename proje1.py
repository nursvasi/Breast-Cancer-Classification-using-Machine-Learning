import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA


#warning library
import warnings
warnings.filterwarnings("ignore") # Uyarı mesajlarını görmezden gelmek için bir filtre uygulanır.

data = pd.read_csv("cancer.csv") # veri seti okunur
data.drop(['Unnamed: 32', 'id'], inplace=True, axis=1) # Gereksiz sütunlar olan "Unnamed: 32" ve "id" düşürülür.

data.rename(columns={"diagnosis": "target"}, inplace=True) #"diagnosis" sütunu "target" olarak yeniden adlandırılır.

data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]

sns.countplot(x="target", data=data) #"target" sütununa göre hedef sınıfların sayısını gösteren bir çubuk grafiği çizilir.
print(data.target.value_counts())

print(len(data))

print(data.head())

print("Data shape", data.shape)

data.info() #missing value kolayca bulunabilir

describe = data.describe()

"""
standardization
missing value: none
"""
# gerekli kütüphaneler oluşturulur, dataset okunur ve dataset sınıflarının bilgileri grafikte gösterilir.
# %% EDA # keşifsel-açıklayıcı veri analizi

# Correlation # Korelasyon matrisi hesaplanır ve bu matrisin ısı haritası çizilir.
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()

#
threshold = 0.50
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Threshold 0.50")

#

"""
there some correlated features
"""

# box plot # Kutu grafiği ile özelliklerin sınıflar arasındaki dağılımları karşılaştırılır.
data_melted = pd.melt(data, id_vars = "target",
                      var_name = "features",
                      value_name = "value")

plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()
"""

buradan anlam çıkartabilmek için 
standardization-normalization yapmak lazım çünkü scale çok geniş
"""

# pair plot # İki boyutlu dağılım grafiği çizilir, bu grafikteki çizgiler sınıfları temsil eder. Bu sayede sınıflar arasındaki özellik dağılımları hakkında bilgi edinilebilir.
sns.pairplot(data[corr_features], diag_kind = "kde", markers = "+",hue = "target")
plt.show()

"""
normalization ve standiration yap
skewness
"""

# %% outlier # veri setindeki aykırı değerleri tespit edip kaldırmak için Local Outlier Factor (LOF) algoritmasını kullanılır
y = data.target # Hedef değişkeni "target" olarak ayarlanır.
x = data.drop(["target"],axis = 1) # Hedef değişken hariç diğer tüm özellikleri içeren "x" DataFrame'i oluşturulur.
columns = x.columns.tolist() #Özellik isimlerini içeren bir liste oluşturulur.

clf = LocalOutlierFactor() #Local Outlier Factor (LOF) algoritması kullanılarak bir LOF sınıflandırıcısı oluşturulur.
y_pred = clf.fit_predict(x) 
X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = X_score

# threshold
threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()


plt.figure()
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1],color = "blue", s = 50, label = "Outliers")
plt.scatter(x.iloc[:,0], x.iloc[:,1], color = "k", s = 3, label = "Data Points")

radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0], x.iloc[:,1], s = 1000*radius, edgecolors = "r",facecolors = "none", label = "Outlier Scores")
plt.legend()
plt.show()

# drop outliers #Belirlenen aykırı değerler, veri setinden çıkarılır.
x = x.drop(outlier_index)
y = y.drop(outlier_index).values

# %% Train test split #veri setini eğitim ve test alt kümelerine bölmek için yapılan işlemleri belirtir.
test_size = 0.3 #Test alt kümesinin oranını belirler.
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = 42)

# %% #özellikleri standartlaştırmak (StandardScaler kullanarak) ve ardından bu özelliklerin dağılımını görselleştirmek için kullanılır.
#Bu algoritma, özelliklerin ortalamasını sıfıra ve standart sapmasını bir birime dönüştürerek veriyi ölçekler. 
#Bu sayede, farklı özellikler arasındaki ölçek farklılıkları ortadan kaldırılır ve makine öğrenimi modellerinin daha iyi performans göstermesi sağlanır.
scaler = StandardScaler() #Veri setini standartlaştırmak için StandardScaler sınıfından bir örnek oluşturulur.
X_train = scaler.fit_transform(X_train) #Eğitim veri seti standartlaştırılır. 
X_test = scaler.transform(X_test) #Test veri seti, eğitim veri seti üzerinde öğrenilen parametrelerle standartlaştırılır.

X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe() #Standartlaştırılmış eğitim veri setinin temel istatistiksel özellikleri hesaplanır.
X_train_df["target"] = Y_train
# box plot 
data_melted = pd.melt(X_train_df, id_vars = "target",
                      var_name = "features",
                      value_name = "value")

plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()


# pair plot 
sns.pairplot(X_train_df[corr_features], diag_kind = "kde", markers = "+",hue = "target")
plt.show()


# %% Basic KNN Method


 from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

def KNN_Best_Params(x_train, x_test, y_train, y_test, cv=10):
    knn = KNeighborsClassifier()
    
    # Define the parameter grid for KNN with only odd values for n_neighbors
    param_grid = {
        'n_neighbors': np.arange(1, 31, 2),  # only odd numbers from 1 to 30
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    grid = GridSearchCV(knn, param_grid, cv=cv, scoring="accuracy")
    grid.fit(x_train, y_train)
    
    print("Best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print()
    
    # Fit the model with the best parameters
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)
    
    # Predict and evaluate the model
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
    print()
    print("CM Test: ", cm_test)
    print("CM Train: ", cm_train)
    
    # Cross-validation scores
    cv_scores = cross_val_score(knn, np.vstack((x_train, x_test)), np.hstack((y_train, y_test)), cv=cv)
    print("Cross-validation scores: ", cv_scores)
    print("Mean cross-validation score: ", np.mean(cv_scores))
    
    return grid

# Assuming X_train, X_test, Y_train, and Y_test are already defined
grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test)


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

def SVM_Best_Params(x_train, x_test, y_train, y_test, cv=10):
    
    # Define the parameter grid for SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC()
    grid = GridSearchCV(svm, param_grid, cv=cv, scoring="accuracy")
    grid.fit(x_train, y_train)
    
    print("Best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print()
    
    # Fit the model with the best parameters
    svm = SVC(**grid.best_params_)
    svm.fit(x_train, y_train)
    
    # Predict and evaluate the model
    y_pred_test = svm.predict(x_test)
    y_pred_train = svm.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
    print()
    print("CM Test: ", cm_test)
    print("CM Train: ", cm_train)
    
    # Cross-validation scores on the entire dataset
    cv_scores = cross_val_score(svm, np.vstack((x_train, x_test)), np.hstack((y_train, y_test)), cv=cv)
    print("Cross-validation scores: ", cv_scores)
    print("Mean cross-validation score: ", np.mean(cv_scores))
    
    return grid

# Assuming X_train, X_test, Y_train, and Y_test are already defined
grid = SVM_Best_Params(X_train, X_test, Y_train, Y_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

def DecisionTree_Best_Params(x_train, x_test, y_train, y_test, cv=10):
    
    # Define the parameter grid for Decision Tree
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    dt = DecisionTreeClassifier()
    grid = GridSearchCV(dt, param_grid, cv=cv, scoring="accuracy")
    grid.fit(x_train, y_train)
    
    print("Best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print()
    
    # Fit the model with the best parameters
    dt = DecisionTreeClassifier(**grid.best_params_)
    dt.fit(x_train, y_train)
    
    # Predict and evaluate the model
    y_pred_test = dt.predict(x_test)
    y_pred_train = dt.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
    print()
    print("CM Test: ", cm_test)
    print("CM Train: ", cm_train)
    
    # Cross-validation scores on the entire dataset
    cv_scores = cross_val_score(dt, np.vstack((x_train, x_test)), np.hstack((y_train, y_test)), cv=cv)
    print("Cross-validation scores: ", cv_scores)
    print("Mean cross-validation score: ", np.mean(cv_scores))
    
    return grid

# Assuming X_train, X_test, Y_train, and Y_test are already defined
grid = DecisionTree_Best_Params(X_train, X_test, Y_train, Y_test)




# %% PCA
#Principal Component Analysis (PCA) kullanarak veriyi daha düşük boyutlu bir uzaya dönüştürür ve 
#ardından bu düşük boyutlu uzayda K-En Yakın Komşu (KNN) modelini eğitir.
#Temel amacı veri setindeki değişkenliği koruyarak yeni bir özellik uzayı oluşturmaktır. PCA, veri setini birbirinden bağımsız olan ana bileşenlere dönüştürmeye odaklanır.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap

# Örnek veri oluşturma (Bu kısmı kendi veri setinizle değiştirin)
# x = np.array([...])
# y = np.array([...])

# `x` değişkenini yeniden şekillendirme
if x.ndim == 1:
    x = x.reshape(-1, 1)

# Veriyi standartlaştırma
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# PCA ile veri boyutunu düşürme
pca = PCA(n_components=2)
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca, columns=["p1", "p2"])
pca_data["target"] = y

# PCA sonuçlarını görselleştirme
sns.scatterplot(x="p1", y="p2", hue="target", data=pca_data)
plt.title("PCA: p1 vs p2")
plt.show()

# Eğitim ve test verilerini ayırma
X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_reduced_pca, y, test_size=0.3, random_state=42)

# KNN, SVM, Karar Ağacı ve Naive Bayes modelleri için parametre arama
def best_params(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train_pca, Y_train_pca)
    return grid_search

# KNN
knn_param_grid = {'n_neighbors': np.arange(1, 31), 'weights': ['uniform', 'distance']}
grid_knn = best_params(KNeighborsClassifier(), knn_param_grid)
knn_train_score = grid_knn.best_score_
knn_best_params = grid_knn.best_params_
knn_test_score = grid_knn.score(X_test_pca, Y_test_pca)
knn_train_predictions = grid_knn.predict(X_train_pca)
knn_test_predictions = grid_knn.predict(X_test_pca)
knn_cm_train = confusion_matrix(Y_train_pca, knn_train_predictions)
knn_cm_test = confusion_matrix(Y_test_pca, knn_test_predictions)

print("KNN:")
print("Best training score:", knn_train_score, "with parameters:", knn_best_params)
print("Test Score:", knn_test_score, ", Train Score:", knn_train_score)
print("CM Test:", knn_cm_test)
print("CM Train:", knn_cm_train)

# SVM
svm_param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
grid_svm = best_params(SVC(), svm_param_grid)
svm_train_score = grid_svm.best_score_
svm_best_params = grid_svm.best_params_
svm_test_score = grid_svm.score(X_test_pca, Y_test_pca)
svm_train_predictions = grid_svm.predict(X_train_pca)
svm_test_predictions = grid_svm.predict(X_test_pca)
svm_cm_train = confusion_matrix(Y_train_pca, svm_train_predictions)
svm_cm_test = confusion_matrix(Y_test_pca, svm_test_predictions)

print("\nSVM:")
print("Best training score:", svm_train_score, "with parameters:", svm_best_params)
print("Test Score:", svm_test_score, ", Train Score:", svm_train_score)
print("CM Test:", svm_cm_test)
print("CM Train:", svm_cm_train)

# Decision Tree
tree_param_grid = {'max_depth': np.arange(1, 21)}
grid_tree = best_params(DecisionTreeClassifier(), tree_param_grid)
tree_train_score = grid_tree.best_score_
tree_best_params = grid_tree.best_params_
tree_test_score = grid_tree.score(X_test_pca, Y_test_pca)
tree_train_predictions = grid_tree.predict(X_train_pca)
tree_test_predictions = grid_tree.predict(X_test_pca)
tree_cm_train = confusion_matrix(Y_train_pca, tree_train_predictions)
tree_cm_test = confusion_matrix(Y_test_pca, tree_test_predictions)

print("\nDecision Tree:")
print("Best training score:", tree_train_score, "with parameters:", tree_best_params)
print("Test Score:", tree_test_score, ", Train Score:", tree_train_score)
print("CM Test:", tree_cm_test)
print("CM Train:", tree_cm_train)

# Naive Bayes
nb_param_grid = {}  # Naive Bayes için parametre gerekmez
grid_nb = best_params(GaussianNB(), nb_param_grid)
nb_train_score = grid_nb.best_score_
nb_best_params = "No parameters"  # NB'de parametre yok
nb_test_score = grid_nb.score(X_test_pca, Y_test_pca)
nb_train_predictions = grid_nb.predict(X_train_pca)
nb_test_predictions = grid_nb.predict(X_test_pca)
nb_cm_train = confusion_matrix(Y_train_pca, nb_train_predictions)
nb_cm_test = confusion_matrix(Y_test_pca, nb_test_predictions)

print("\nNaive Bayes:")
print("Best training score:", nb_train_score, "with parameters:", nb_best_params)
print("Test Score:", nb_test_score, ", Train Score:", nb_train_score)
print("CM Test:", nb_cm_test)
print("CM Train:", nb_cm_train)

# KNN
print("\nKNN Cross-Validation Results:")
print("Best training score with CV:", grid_knn.best_score_)
print("Mean cross-validation score:", grid_knn.cv_results_['mean_test_score'].mean())
print("Best parameters with CV:", grid_knn.best_params_)

# SVM
print("\nSVM Cross-Validation Results:")
print("Best training score with CV:", grid_svm.best_score_)
print("Mean cross-validation score:", grid_svm.cv_results_['mean_test_score'].mean())
print("Best parameters with CV:", grid_svm.best_params_)

# Decision Tree
print("\nDecision Tree Cross-Validation Results:")
print("Best training score with CV:", grid_tree.best_score_)
print("Mean cross-validation score:", grid_tree.cv_results_['mean_test_score'].mean())
print("Best parameters with CV:", grid_tree.best_params_)

# Naive Bayes
print("\nNaive Bayes Cross-Validation Results:")
print("Best training score with CV:", grid_nb.best_score_)
print("Mean cross-validation score:", grid_nb.cv_results_['mean_test_score'].mean())

import pandas as pd

# Boş bir liste oluşturma
results = []

# KNN sonuçları ekleme
results.append({'Model': 'KNN', 
                'Best Training Score': knn_train_score, 
                'Test Score': knn_test_score, 
                'Best Parameters': knn_best_params, 
                'CM Test': knn_cm_test})

# SVM sonuçları ekleme
results.append({'Model': 'SVM', 
                'Best Training Score': svm_train_score, 
                'Test Score': svm_test_score, 
                'Best Parameters': svm_best_params, 
                'CM Test': svm_cm_test})

# Decision Tree sonuçları ekleme
results.append({'Model': 'Decision Tree', 
                'Best Training Score': tree_train_score, 
                'Test Score': tree_test_score, 
                'Best Parameters': tree_best_params, 
                'CM Test': tree_cm_test})

# Naive Bayes sonuçları ekleme
results.append({'Model': 'Naive Bayes', 
                'Best Training Score': nb_train_score, 
                'Test Score': nb_test_score, 
                'Best Parameters': nb_best_params, 
                'CM Test': nb_cm_test})

# Listeyi DataFrame'e dönüştürme
results_df = pd.DataFrame(results)

# DataFrame'i gösterme
print(results_df)

import matplotlib.pyplot as plt
import numpy as np

# Test ve eğitim skorları
test_scores = [0.9239766081871345, 0.9473684210526315, 0.9298245614035088, 0.8947368421052632]
train_scores = [0.9369303797468355, 0.9495569620253164, 0.9293670886075949, 0.9192405063291138]

# Karmaşıklık matrisleri
cm_test = [[103, 6], [105, 4], [100, 9], [101, 8]], [[7, 55], [5, 57], [3, 59], [10, 52]]
cm_train = [[241, 7], [241, 7], [237, 11], [237, 11]], [[14, 135], [14, 135], [7, 142], [20, 129]]

# Algoritma isimleri
algorithms = ['KNN', 'SVM', 'Decision Tree', 'Naive Bayes']

import matplotlib.pyplot as plt

# Veriler
models = ['KNN', 'SVM', 'Decision Tree', 'Naive Bayes']
best_train_scores = [0.9369, 0.9496, 0.9344, 0.9192]
test_scores = [0.924, 0.947, 0.93, 0.895]
train_scores = [0.9369, 0.9496, 0.9344, 0.9192]
mean_cv_scores = [0.9277, 0.9278, 0.9125, 0.9192]

# Histogramı çizdirme
barWidth = 0.15
r1 = range(len(models))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.figure(figsize=(12, 6))
plt.bar(r1, best_train_scores, color='skyblue', width=barWidth, edgecolor='grey', label='Best Train Score')
plt.bar(r2, test_scores, color='lightgreen', width=barWidth, edgecolor='grey', label='Test Score')
plt.bar(r3, train_scores, color='salmon', width=barWidth, edgecolor='grey', label='Train Score')
plt.bar(r4, mean_cv_scores, color='orange', width=barWidth, edgecolor='grey', label='Mean CV Score')

plt.xlabel('Model', fontweight='bold')
plt.ylabel('Scores', fontweight='bold')
plt.xticks([r + 1.5 * barWidth for r in range(len(models))], models)
plt.title('Scores of Different Models')
plt.legend()
plt.ylim(0.8, 1.0)  # Y ekseninin sınırlarını belirleme
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



#%% NCA 
#Neighborhood Components Analysis (NCA) kullanarak veriyi daha düşük boyutlu bir uzaya dönüştürür ve 
#ardından bu düşük boyutlu uzayda K-En Yakın Komşu (KNN) modelini eğitir.
#Amacı sınıflandırma performansını artırmaktır. NCA, örneklerin sınıflarına göre gruplandığı bir özellik uzayı oluşturarak sınıflandırma doğruluğunu artırmaya çalışır.
from sklearn.svm import SVC

# Create an SVM classifier
svm = SVC(kernel='linear')

# Fit the SVM classifier on the reduced data
svm.fit(X_train_nca, Y_train_nca)

# Plot decision boundary
Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z_svm = Z_svm.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z_svm, cmap=cmap_light)

# Plot the training points
plt.scatter(X_train_nca[:, 0], X_train_nca[:, 1], c=Y_train_nca, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("SVM Classification")
plt.show()



# %% find wrong decision
#NCA ile dönüştürülmüş veri üzerinde eğitilen K-En Yakın Komşu (KNN) modelinin performansını değerlendirmek ve yanlış sınıflandırılmış örnekleri görselleştirmek amacıyla kullanılır.
knn = KNeighborsClassifier(**grid_nca.best_params_)
knn.fit(X_train_nca,Y_train_nca)
y_pred_nca = knn.predict(X_test_nca)
acc_test_nca = accuracy_score(y_pred_nca,Y_test_nca)
knn.score(X_test_nca,Y_test_nca)

test_data = pd.DataFrame()
test_data["X_test_nca_p1"] = X_test_nca[:,0]
test_data["X_test_nca_p2"] = X_test_nca[:,1]
test_data["y_pred_nca"] = y_pred_nca
test_data["Y_test_nca"] = Y_test_nca

plt.figure()
sns.scatterplot(x="X_test_nca_p1", y="X_test_nca_p2", hue="Y_test_nca",data=test_data)

diff = np.where(y_pred_nca!=Y_test_nca)[0]
plt.scatter(test_data.iloc[diff,0],test_data.iloc[diff,1],label = "Wrong Classified",alpha = 0.2,color = "red",s = 1000)






