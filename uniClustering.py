import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('College_Data', index_col=0)

df.head()

sns.lmplot(data=df, x='Room.Board', y='Grad.Rate', hue='Private', fit_reg=False, palette='coolwarm', size=6, aspect=1)

sns.lmplot(data=df, x='Outstate', y='F.Undergrad', hue='Private', fit_reg=False, size=6, aspect=1)

g = sns.FacetGrid(df, hue='Private', palette='coolwarm', size=6, aspect=2)
g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)

g = sns.FacetGrid(df, hue='Private', palette='coolwarm', size=6, aspect=2)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)

# To check the value of graduation higher than 100 and set it to 100
df[df['Grad.Rate'] > 100]

df['Grad.Rate']['Cazenovia College'] = 100

# Applying the K means clustering algorithm

kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private'), axis=1)

kmeans.cluster_centers_


# Converter function

def converter(private):
    if private == 'Yes':
        return 1
    else:
        return 0


df['Cluster'] = df['Private'].apply(converter)

df.head()

print(confusion_matrix(df['Cluster'], kmeans.labels_))
print(classification_report(df['Cluster'], kmeans.labels_))




























