import pandas as pd
from sklearn.cluster import KMeans

# Load the data
data = pd.read_csv("customers.csv")

# Train the model
kmeans = KMeans(n_clusters=3).fit(data)

# Get the cluster labels for each customer
labels = kmeans.labels_

# Assign the cluster labels back to the data
data["cluster"] = labels

# Analyze the results
print(data.groupby("cluster").mean())
