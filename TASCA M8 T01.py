#!/usr/bin/env python
# coding: utf-8

# # Tasca M8 T01

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from kneed import KneeLocator


# In[3]:


df = pd.read_csv("C:/publicacions_facebook_thailandia.csv")
df.head(5)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df1 = df.drop(['Column1','Column2','Column3','Column4','status_id','status_published'], axis = 1)
df1.head()


# In[7]:


df1.shape


# In[8]:


df1.isnull()


# In[9]:


df1_clean = df1.drop_duplicates()
if len(df1_clean) == len(df1):
    print("No hi ha elements duplicats")
else:
    print("Hi ha elements duplicats")


# In[10]:


X = np.array(df1_clean[['num_reactions','num_comments','num_shares','num_likes','num_loves','num_wows','num_hahas','num_sads','num_angrys']])
X_standard = StandardScaler().fit_transform(X)
df_X = pd.DataFrame(X_standard, columns = ['num_reactions','num_comments','num_shares','num_likes','num_loves','num_wows','num_hahas','num_sads','num_angrys'])
df_X['index'] = df_X.index
df_X.head()


# In[11]:


df_X.shape


# In[12]:


dummy = pd.get_dummies(df1_clean['status_type'])
dummy_df = pd.DataFrame(dummy, columns = ['link','photo','status','video'])
dummy_df['index'] = dummy_df.index
dummy_df.head()


# In[13]:


dummy_df.shape


# In[14]:


df2 = df_X.merge(dummy_df, on = 'index', how = 'inner')
df3 = df2.drop(['index'], axis = 1)
df3.head()


# In[15]:


df3.shape


# # K-Means

# In[16]:


# per poder aplicar l'algoritme k-means em de saber cuants clusters em d'emprar
# faig la curva de l'elbow
kmeans_kwargs = {"init":"k-means++",
                "n_init" : 10,
                "max_iter" : 400,
                "random_state" : 42,}
# una llista dels SEE valors per cada k
sse = []
for k in range(1,11):
    kmeans = KMeans(n_clusters = k, **kmeans_kwargs)
    kmeans.fit(df3)
    sse.append(kmeans.inertia_)


# In[17]:


plt.style.use("ggplot")
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel("number of clusters")
plt.ylabel("SSE")
plt.show()


# In[18]:


# vaig a utilitzar kneed
kl = KneeLocator(range(1,11), sse, curve = "convex", direction = "decreasing")
kl.elbow


# In[19]:


pca = PCA().fit(df3)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('number of cumulative explained variance')


# ## aplicando el método means a la base de datos

# In[20]:


clustering = KMeans(n_clusters = 3, max_iter = 300) # creamos el modelo
clustering.fit(df3)


# ## agregando la clasificación al modelo original

# In[21]:


df3['KMeans_Clusters'] = clustering.labels_
df3.head()


# ## visualizando los clusters que se han formado

# Per poder fer la PCA primer em de saber quants components hem d'emprar, veiem que per explicar el 100% de la variança són necessaris 8 elements, com que 8 elements noo els puc representar gràficament triaré el màxim que són 3.

# In[22]:


pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(df3)
principalDf = pd.DataFrame(data = principalComponents, columns = ['Componente_1','Componente_2'])
pca_nombres_data = pd.concat([principalDf, df3['KMeans_Clusters']], axis = 1)
pca_nombres_data.head()


# In[23]:


fig = plt.figure(figsize = (6,6))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize = 15)
ax.set_ylabel('Componente 2', fontsize = 15)
ax.set_title('Componentes principales', fontsize = 20)

color_theme = np.array(['blue','green','red'])
ax.scatter(x = pca_nombres_data.Componente_1, 
           y = pca_nombres_data.Componente_2,
           c = color_theme[pca_nombres_data.KMeans_Clusters], s = 50)
plt.show()


# In[24]:


pca.explained_variance_ratio_


# In[25]:


# si faig el coeficient de silhouette
silhouette_coefficients = []
for k in range (2,11):
    kmeans = KMeans(n_clusters = k, **kmeans_kwargs)
    kmeans.fit(pca_nombres_data)
    score = silhouette_score(pca_nombres_data, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[26]:


plt.style.use("fivethirtyeight")
plt.plot(range(2,11), silhouette_coefficients)
plt.xticks(range(2,11))
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette coefficient")
plt.show()


# In[27]:


kmeans = KMeans(init = "random",
               n_clusters = 3,
               n_init = 10,
               max_iter = 300,
               random_state = 42)


# In[28]:


clustering.inertia_


# In[29]:


clustering.n_iter_


# In[30]:


clustering.cluster_centers_


# ## Exercici 2: Classifica els diferents registres utilitzant l'algoritme de clustering jeràrquic.

# In[ ]:


plt.figure(figsize=(10,7))
dendogram = sch.dendrogram(sch.linkage(df3, method = 'ward'))


# In[ ]:


clusteringjerarquic = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean',
                               linkage = 'ward')
clusteringjerarquic.fit(df3)
clustering_jerarquic_labels = clusteringjerarquic.labels_
labels


# ## Exercici 3: Calcula el rendiment del clustering mitjançant un paràmetre com pot ser silhouette.

# In[ ]:


kmeans_silhouette = silhouette_score(df3, clustering.labels_).round(2)
print('kmeans:', kmeans_silhouette)


# In[ ]:


jerarquic_silhouette = silhouette_score(df3, clustering_jerarquic_labels).round(2)
print('clustering jerarquic', jerarquic_silhouette)


# ## otro

# In[ ]:




