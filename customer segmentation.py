#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[37]:


df = pd.read_csv(r'C:\Users\hp\Downloads\ifood_df.csv')


# In[38]:


df.head()


# In[39]:


#checking null values


# In[40]:


df.isnull().sum()


# In[41]:


#checking missing values


# In[42]:


df.isna().sum()
sns.heatmap(df.isna())


# In[43]:


#checking duplicated
df.duplicated().sum()


# In[44]:


df.drop_duplicates()


# In[45]:


#checking unique values
df.nunique()


# In[46]:


# descriptive statistics
stats = ['Income','Age','Recency','NumDealsPurchases','MntTotal','MntRegularProds','AcceptedCmpOverall']
df1 = df[stats].copy()


# In[47]:


df1.describe()


# In[48]:


# correlation for stats


# In[49]:


sns.heatmap(df1.corr(), annot = True)


# In[50]:


# visualising some columns
# total amount in purchase


# In[51]:


plt.figure(figsize=(6, 4))  
sns.boxplot(data=df, y='MntTotal')
plt.title('Box Plot for MntTotal')
plt.ylabel('MntTotal')
plt.show()


# In[52]:


# Income
plt.figure(figsize=(6, 4))  
sns.boxplot(data=df, y='Income', color='red')
plt.title('Box Plot for Income')
plt.ylabel('MntTotal')
plt.show()


# In[53]:


# income plot
plt.figure(figsize=(8, 6))  
sns.histplot(data=df, x='Income', bins=30, kde=True)
plt.title('Histogram for Income')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()


# In[54]:


# age plot
plt.figure(figsize=(8, 6))  
sns.histplot(data=df, x='Age', bins=30, kde=True, color = 'green')
plt.title('Histogram for Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[55]:


# in realtionsip for married and couples only
def get_relationship(row):
    if row['marital_Married'] ==1:
        return 1
    elif row['marital_Together'] == 1:
        return 1
    else:
        return 0
df['In_relationship'] = df.apply(get_relationship, axis=1)
df.head()


# In[61]:


def get_marital_status(row):
    if row['marital_Divorced'] == 1:
        return 'Divorced'
    elif row['marital_Married'] == 1:
        return 'Married'
    elif row['marital_Single'] == 1:
        return 'Single'
    elif row['marital_Together'] == 1:
        return 'Together'
    elif row['marital_Widow'] == 1:
        return 'Widow'
    else:
        return 'Unknown'
df['Marital'] = df.apply(get_marital_status, axis=1)


# In[62]:


# plotting for maritial and mnt total
plt.figure(figsize=(8, 6))
sns.barplot(x='Marital', y='MntTotal', data=df, palette='viridis')
plt.title('MntTotal by marital status')
plt.xlabel('Marital status')
plt.ylabel('MntTotal')


# In[63]:


# cluster 1
# in_realtionship and accepted_overall clusterings
x1 = df.loc[:, ['AcceptedCmpOverall', 'In_relationship']].values
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10)
    kmeans.fit(x1)
    wcss.append(kmeans.inertia_)

#plotting for age and income
plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss, marker = '*')
plt.title('Elbow graph')
plt.xlabel('AcceptedCmpOverall')
plt.ylabel('In_relationship')
plt.show()


# In[64]:


# cluster 1 silhouette score
from sklearn.metrics import silhouette_score
silhouette_list = []
for K in range(2,10):
    model = KMeans(n_clusters = K, n_init = 10)
    clusters = model.fit_predict(x1)
    s_avg = silhouette_score(x1, clusters)
    silhouette_list.append(s_avg)

plt.figure(figsize=[7,5])
plt.plot(range(2,10), silhouette_list, color='b', marker = '*')
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()


# In[65]:


# cluster 1 k-means prediction
kmeans = KMeans(n_clusters = 4)
labels = kmeans.fit_predict(x1)
print(labels)


# In[66]:


# centroids 
print(kmeans.cluster_centers_)


# In[67]:


#plotting for age and income clusters
plt.figure(figsize = (14, 8))

plt.scatter(x1[:, 0], x1[:, 1], c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red')
plt.title('Clusters of Customers\n', fontsize = 20)
plt.xlabel('aceepted overall campaign')
plt.ylabel('In_relationship')
plt.show()


# In[68]:


#cluster 1 report
import pandas as pd

# Assuming labels contains the cluster labels assigned by KMeans
# Create a DataFrame with original data and cluster labels
cluster_df = pd.DataFrame(x1, columns=['AcceptedCmpOverall', 'In_relationship'])
cluster_df['Cluster-1'] = labels
cluster_percentage = cluster_df['Cluster-1'].value_counts(normalize=True) * 100
cluster_percentage = cluster_percentage.rename('Percentage').reset_index()


# Display the DataFrame with cluster percentages
print(cluster_percentage)


# In[69]:


# cluster 2
x2 = df.loc[:, ['NumDealsPurchases', 'MntRegularProds']].values
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init = 10)
    kmeans.fit(x2)
    wcss.append(kmeans.inertia_)
    
#plotting for age and products
plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss, marker = 'o')
plt.title('elbow graph')
plt.xlabel('NumDealsPurchases')
plt.ylabel('MntRegularProds')
plt.show()


# In[70]:


# k means prediction for cluster 2
kmeans = KMeans(n_clusters = 4)
labels = kmeans.fit_predict(x2)
print(labels)


# In[71]:


print(kmeans.cluster_centers_)


# In[72]:


#plotting for cluster 2
plt.figure(figsize = (14,8))

plt.scatter(x2[:, 0], x2[:, 1], c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red')
plt.title('Clusters of Customers\n', fontsize = 20)
plt.xlabel('deals purchase')
plt.ylabel('regular products')
plt.show()


# In[73]:


#cluster 2 report
import pandas as pd

# Assuming labels contains the cluster labels assigned by KMeans
# Create a DataFrame with original data and cluster labels
cluster_df = pd.DataFrame(x2, columns=['NumDealsPurchases', 'MntRegularProds'])
cluster_df['Cluster-2'] = labels
cluster_percentage = cluster_df['Cluster-2'].value_counts(normalize=True) * 100
cluster_percentage = cluster_percentage.rename('Percentage').reset_index()


# Display the DataFrame with cluster percentages
print(cluster_percentage)


# In[74]:


# cluster 3
x3 = df.loc[:, ['Income', 'MntTotal']].values
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init = 10)
    kmeans.fit(x3)
    wcss.append(kmeans.inertia_)
    
#plotting for age and products
plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss, marker = 'o')
plt.title('elbow graph')
plt.xlabel('Income')
plt.ylabel('MntTotal')
plt.show()


# In[75]:


# k means prediction for cluster 3
kmeans = KMeans(n_clusters = 4)
labels = kmeans.fit_predict(x3)
print(labels)


# In[76]:


print(kmeans.cluster_centers_)


# In[77]:


#plotting for cluster 3
plt.figure(figsize = (14, 8))

plt.scatter(x3[:, 0], x3[:, 1], c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red')
plt.title('Clusters of Customers\n', fontsize = 20)
plt.xlabel('Income')
plt.ylabel('MntTotal')
plt.show()


# In[78]:


import pandas as pd

# Assuming labels contains the cluster labels assigned by KMeans
# Create a DataFrame with original data and cluster labels
cluster_df = pd.DataFrame(x3, columns=['Income', 'MntTotal'])
cluster_df['Cluster-3'] = labels
cluster_percentage = cluster_df['Cluster-3'].value_counts(normalize=True) * 100
cluster_percentage = cluster_percentage.rename('Percentage').reset_index()


# Display the DataFrame with cluster percentages
print(cluster_percentage)


# In[ ]:


# Number of clusters = 4
Optimal number of clusters = 4 :

The Elbow Method and Silhouette Analysis suggested 4 clusters (k=4). 
The elbow method highlighted the number of 4 or 5 clusters as a reasonable number of clusters. 
The silhouette score analysis revealed a peak silhouette score for k=4.


# In[ ]:


#Conclusion


# In[ ]:


# cluster characteristics for cluster 1:

Results

"This section contains the results of the K-means clustering analysis, which aimed to identify"
"distinct customer segments based on different types of features."
 
feature 1 :
    
    Relationship and Accepted overall campaign offers.

1th Segmentation :
    
    Cluster 1: 
    
        High value customers in relationship (married or together).
        This cluster represents 51% of the customer base.
        These customers have high income and they are in a relationship.
    
    Cluster 2: 
        
        Low value customers in realtionship (married or together).
        This cluster represents 6% of the customer base.
        These customers have low income and they are (married or together).
        
    Cluster o: 
        
        Mid range value customers in realtionship (married or together).
        This cluster represents 33% of the customer base.
        These customers have Mid range of income and they are (married or together).


# In[ ]:


feture 2:
    
    Regular products purchase and Deals purchase in ( store or website ).
    
2st Segmentation :
    
    cluster 0 :
        
        High values customers based on their Income and Relationships.
        This cluster represents 51.60% of the customer base.
        These customers have high income and they Purchase Regular Products while Deals.
        
    Cluster 1 :
        
        Mid Level values customers based on their Income and Relationships.
        This cluster represents 20% of the customer base.
        These customers have Mid-level of income and they Purchase Usual Regular Products while the Deals.
        
    Cluster 2 :
        
        Low Level values customers based on their Income and Relationships.
        This cluster represents 10% of the customer base.
        These customers have Low-level of income and they Purchase less Regular Products while the Deals.
        


# In[ ]:


feture 3:
    
    products purchase Total amount and Income.
    
3st Segmentation :
    
    cluster 1 :
        
        High values customers based on their products purchase Total amount and Income
        This cluster represents 30% of the customer base.
        These customers have high income and products purchase Total amount
        
    Cluster 0 :
        
        Mid Level values customers based on their products purchase Total amount and Income
        This cluster represents 28% of the customer base.
        These customers have Mid-level of income and products purchase Total amount.
        
    Cluster 3 :
        
        Low Level values customers based on their products purchase Total amount and Income
        This cluster represents 20% of the customer base.
        These customers have Low-level of income and products purchase Total amount.


# In[ ]:


Recommendations:
    http://localhost:8888/notebooks/customer%20segmentation.ipynb#
    Based on the clusters, tailored marketing strategies can be created. 
    Customers from these segments will have different interests and product preferences.


# In[ ]:


Marketing Strategies for Each Cluster or features :
    
    feature 1: 
        
        High value customers in relationship (either married or together).
        Preliminary analysis showed that high income customers buy more wines and fruits.
        A tailored campaign to promote high quality wines may bring good results.
        This cluster contains customers in relationship, family-oriented promo-images should be quite effective for this audience.
        
    feature 2: 
        
        Low value customers in relationship (either married or together).
        Promos with discounts and coupons may bring good results for this targeted group.
        Loyalty program may stimulate these customers to purchase more often.
        
    feature 3: 
        
        High value customers in relationship (either married or together).
        Similar to the Cluster 1, these customers buy a lot of wines and fruits.
        This cluster contains single customers. Promo images with friends, parties or single trips may be more efficient for in relationship (either married or together).
        

