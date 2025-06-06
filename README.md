Unsupervised machine learning models are created from the wine data for clustering from Kaggle.com. First by reducing dimensionality with PCA at 80% variance, then carrying out K-means with 3 clusters using the elbow and silhouette methods and finally by carrying out hierarchical clustering for comparison to K-means.
Instructions: Download the  Wine Data for Clustering available on Kaggle.com. Conduct PCA to collapse correlated variables into a subset that includes 80% of the variance of the entirety of the data.  Then conduct k-means to identify clusters, and evaluate different values for k (e.g., 3, 4,…).  Finally, conduct hierarchical clustering.Investigate assumptions.  Interpret all of your findings.

1. Interpretation of PCA Results: 
https://github.com/gmurage/Unsupervised-Machine-Learning/blob/main/PCA_80__Variance.png
The plots is a visualization of Principal Component Analysis (PCA) results, specifically showing the explained variance ratio for each principal component and the cumulative explained variance. The interpretation of the PCA plot is as follows"
Key Elements:
Y-axis (0.0 to 1.0):
Represents the explained variance ratio which is the proportion of variance explained by each principal component or cumulatively.
X-axis:
This represents the principal components PC1, PC2 and so forth.
Individual explained variance bars:
Each bar shows the variance explained by a single principal component. For example PC1 explains the most variance, followed by PC2 and so forth.
Cumulative explained variance line:
The line shows how much variance is explained as more principal components are added. The first few components help explain most of the variance.
Observations:
1. The first few principal components explain the majority of the variance as shown by a steep initial slope in the cumulative line.
2. Around 80% of the total variance is explained by the first few components as indicated by the label "0.8" at the corresponding 4th principal component bar.
3. The remaining components contribute minimally to the explained variance as shown by flattening of the cumulative line.
Implications:
1. Dimensionality reduction is effective since only a small number of principal components 1-5 retain most of the information which is 80% variance.
2. This means that further components may be dropped to simplify the model without significant loss of information.

2. Interpretation of K Means Clustering
These two graphs illustrate methods for determining the optimal number of clusters in the wine dataset:

A) Elbow Method (left graph):https://github.com/gmurage/Unsupervised-Machine-Learning/blob/main/Elbow_method.png
It plots the Within-Cluster Sum of Squares (WCSS) against the number of clusters. The WCSS decreases sharply from 2 to 3 clusters and then levels off more gradually, forming an “elbow” at 3 clusters. This suggests that 3 clusters provide a good balance between reducing intra-cluster variance without overcomplicating the model.

B) Silhouette Scores (right graph): https://github.com/gmurage/Unsupervised-Machine-Learning/blob/main/Silhouette_scores.png
This method evaluates the quality of clustering using the Silhouette Score, which measures how well-separated the clusters are. The score peaks at around 0.42 when the number of clusters is 3, indicating that this configuration offers the best separation.

Rationale:
Both methods independently suggest that 3 clusters might be the ideal number for structuring the wine dataset.
The plot visualizes the results of a K-means clustering algorithm applied to the wine dataset using Principal Component Analysis (PCA).

The key interpretation are as follows:
Clusters Identified =3 : https://github.com/gmurage/Unsupervised-Machine-Learning/blob/main/K-means_clustering_results.png

The wine dataset has been partitioned into three distinct clusters, each represented by different colors :purple, teal, and yellow. These clusters indicate groups of data points that are most similar to each other based on their features.
Axes (PC1 & PC2):These represent the first two principal components from PCA, reducing the dimensionality of the wine dataset while retaining most of its variance at 80%. This transformation allows for a clear visualization of cluster separation.
Cluster Structure:
The clusters appear well-separated, suggesting that K-means successfully identified meaningful groupings. However, some slight overlap may indicate areas where clustering boundaries are not perfectly distinct.
Conclusions
This plot of clusters aligns with the Elbow Method and Silhouette Score findings, confirming that 3 clusters is an optimal choice. It reinforces that the approach is well-tuned and appropriate.
Further implications¶To have additional validations, further clustering with hierarchical clustering will be performed for comparison.

3. Hierarchical Clustering Results interpreted
This is a Hierarchical Clustering Dendrogram using the Ward method, which is useful for determining the natural grouping of data points. The interpretations are as follows:
Dendrogram Structure:https://github.com/gmurage/Unsupervised-Machine-Learning/blob/main/Hierarchical_Clustering_Dendrogram.png
The branching hierarchy illustrates how clusters are merged as the algorithm progresses. The x-axis represents individual sample indices, while the y-axis represents the distance at which clusters are joined. The higher the connection point, the more dissimilar the merged clusters are.

Three Main Clusters:
The wine dataset has been divided into three groups, marked by orange, green, and red sections, reinforcing previous findings from K-means clustering and silhouette analysis.
Ward Method:This technique minimizes the variance within clusters, making it particularly useful for structured datasets.
Conclusions:Since previously above the K-means clustering also suggested three clusters, this dendrogram validates that three is a meaningful choice in the wine data clustering.

The scatter plot visualizes the Hierarchical Clustering Results using Principal Component Analysis (PCA): https://github.com/gmurage/Unsupervised-Machine-Learning/blob/main/Hierarchical_clustering_results.png
Cluster Separation:
The data has been grouped into three distinct clusters, represented by purple, teal, and yellow dots. This aligns with the findings from K-means clustering and the dendrogram analysis, reinforcing that three clusters provide an optimal structure.
Axes Representation:
The x-axis (PC1) and y-axis (PC2) correspond to the first two principal components, reducing the dataset’s dimensionality while preserving variance at 80%. This allows for a clear visualization of cluster separation.
Hierarchical Clustering Performance:
The clusters are well-defined, with minimal overlap, suggesting that hierarchical clustering effectively captured meaningful relationships between data points. Since this supports the earlier insights from the dendrogram analysis, it's a strong validation of the clustering approach of using 3 K-means clusters versus 4! 

4. Compare Clustering Results between K-Means and Hierarchical clustering
Interpretation of the cluster assignment comparison
This cross-tabulation compares the K-means clustering results (rows) with the hierarchical clustering results (columns). Each cell shows how many observations were assigned to a given cluster in K-means (0, 1, 2) vs. hierarchical clustering (1, 2, 3).

[(https://github.com/gmurage/Unsupervised-Machine-Learning/blob/main/Cross%20Validation%20of%20K-means%20versus%20Hierarchical%20Clusters.docx)](https://github.com/gmurage/Unsupervised-Machine-Learning/blob/main/Cross%20Validation%20of%20K-means%20versus%20Hierarchical%20Clusters.docx)

Key observations:

K-means Cluster 0 vs. Hierarchical Clusters:
1. Mostly aligns with Hierarchical Cluster 3 (55 out of 60).
2. 5 observations were assigned to Hierarchical Cluster 1 instead.
Interpretation:K-means Cluster 0 strongly corresponds to Hierarchical Cluster 3, with minor mismatch of approximately 8% .

K-means Cluster 1 vs. Hierarchical Clusters:
1. Mostly aligns with Hierarchical Cluster 1 (50 out of 52).
2. 2 observations were assigned to Hierarchical Cluster 3.
Interpretation:K-means Cluster 1 strongly corresponds to Hierarchical Cluster 1, with very little mismatch of approximately 4%.

K-means Cluster 2 vs. Hierarchical Clusters:
1. Almost perfectly aligns with Hierarchical Cluster 2 (65 out of 66).
2. Only 1 observation was assigned to Hierarchical Cluster 3.
Interpretation:K-means Cluster 2 is nearly identical to Hierarchical Cluster 2 with an approximate 1.5% mismatch.

Overall comparison between methods( K-means cluster versus Hierarchical Clusters):
1.  High Consistency:Both methods largely are in agreement on cluster assignments.
2.  Discrepancies:
    5 cases where K-means Cluster 0 was assigned to Hierarchical Cluster 1.
    2 cases where K-means Cluster 1 was assigned to Hierarchical Cluster 3.
    1 case where K-means Cluster 2 was assigned to Hierarchical Cluster 3.

Possible Reasons for Mismatches:
1. Different Algorithms:
K-means is centroid-based, while hierarchical clustering is linkage-based and in ithis case used Ward’s method.
2. Boundary Cases:
Some observations may lie near cluster boundaries, leading to different assignments.

Conclusions:
1.  Both methods identify similar structures in the wine data, suggesting robust clustering.
2.  K-means Cluster 0 = Hierarchical Cluster 3
3.  K-means Cluster 1 = Hierarchical Cluster 1
4.  K-means Cluster 2 = Hierarchical Cluster 2
5.  Minor disagreements likely represent ambiguous cases that could be further analyzed.
   
Recommendation:
If using these clusters for decision-making, it warrants furthers checks to establish if mismatched observations have unique characteristics in the wine data. Consider visualizing these observations in PCA space to see if they fall between clusters.

Next Steps for Deeper Analysis:
Examine misclassified observations:

Are they outliers or borderline cases?
Check feature distributions for each cluster to understand their differences.
Validate with domain knowledge

For example do clusters correspond to known wine types?.
This comparison shows that both clustering methods largely agree, supporting the validity of the identified clusters. The small discrepancies could be due to algorithmic differences or natural overlaps in the data.

5. Investigate Assumptions
1. This is a QQ (Quantile-Quantile) plot for PC1 (Principal Component 1). https://github.com/gmurage/Unsupervised-Machine-Learning/blob/main/QQ_Plot_for_PC1.png
It helps assess whether the data follows a normal distribution by comparing theoretical quantiles (x-axis) with observed values (y-axis).
Results:
1. Blue dots represent the actual data points.
2. The red line (y = x) is the reference for a perfectly normal distribution.
3. If the data follows a normal distribution, the points should closely follow the red line.
4. The deviations at the tails suggest that PC1 may not fully adhere to normality.

Conclusions:
Given that normality is often an assumption in statistical analyses and modeling, this plot is useful for determining whether transformations or alternative approaches might be needed. Nevertheless PCA is most useful for dimensionality reduction, normality is less critical for PCA if the goal is dimensionality reduction versus statistical inference.

2. PC1 distribution by cluster results on assumptions

This plot helps evaluate how well PC1 differentiates the clusters and whether any extreme values exist. The purpose is to assess the distribution of the first principal component (PC1) across clusters from K-means. This is a box plot visualizing the distribution of PC1 (Principal Component 1) values across three different KMeans clusters (0, 1, and 2). It provides insights into how PC1 varies within each cluster.
Results

    The y-axis represents the PC1 values, ranging from approximately -4 to 4.
    The x-axis represents the three clusters assigned by KMeans.
    Each box captures the interquartile range (IQR) which is the he middle 50% of the data.
    The whiskers show the range of most data points, while dots outside them indicate outliers.

Key observations:

    Cluster 0 has a median PC1 value near 0, with a tight distribution, meaning the data is centered around a narrow range.
    Cluster 1 has a slightly lower median than Cluster 0 and shows two significant outliers (one below -4 and one above -2).
    Cluster 2 is visibly shifted higher in PC1, suggesting that this cluster is distinctly separated along this principal component.



    
 Results show better separation of clusters with 3 versus 4 clusters. Finally hierarchical clustering using the ward method is carried out as a third unsupervised machine learning method. Hierarchical clustering also achieves 3 main clusters just like in K-means using PCA with 80% variance. 
According to Kannan and Menaga (2025), in hierarchical clustering, similar objects are clustered into groups using either the distance matrix or raw data. The algorithm works by first detecting the two clusters closest together. Secondly, the hierarchical clustering algorithm fuses the two most analogous groups and continues this process until all the clusters are combined. The output, which is a dendrogram, shows the categorized relationship between clusters and the dimension of the straight line between two different groups indicates the Euclidian  distance between the groups. 
Sung (2025) argues that dimensionality reduction helps mitigate potential challenges caused by high dimensional features. Examples provided are overfitting, computational inefficiencies and noisy patterns. The author argues that Principal Component Analysis (PCA) is employed as an unsupervised linear transformation method that identifies orthogonal directions in the data. Known as principal components. There are several benefits of PCA that are achieved: firstly, is variance retention whereby the value of k is chosen based on the cumulative explained variance ratio that ensures that 90% of the original variance is preserved . This approach apparently strikes a good balance between complexity reduction and information retention. This compares well with the 80% variance on PCA using the wine data research. The second benefit of PCA is noise reduction, this is because principal  components with negligible variance are eliminated. In so doing, PCA acts as a form of noise filtering which can improve the subsequent model performance.
In another study, Shang (2025) defines  K-means algorithm as a distance based clustering algorithm, which divides the samples in the dataset into k clusters. The algorithm makes the samples in the same cluster more similar, while the samples between different clusters are less similar. K-means clustering helps to find potential structures and patterns in the data and thus provides valuable information for sports injury prediction. There, however, exists disadvantages to K-means algorithm in that it is vulnerable to the problem of data imbalance. This occurs  when the number of samples of one class in the data set is far more than that of other classes. This has an effect in  the clustering. The LOF algorithm can identify outliers in the dataset by calculating the local outlier factor (LOF) of each data point. The LOF algorithm can evaluate the degree of anomaly of the point relative to its local neighborhood class and if the LOF value of a point is far greater than 1, it is considered as an outlier. Removal of these outliers before clustering K-means clustering process, thus improving the accuracy of clustering.
References:
Kannan, K., & Menaga, A. (2025). An efficient approach on risk factor prediction related to cardiovascular disease around Kumbakonam, Tamil Nadu, India, using unsupervised machine learning techniques. Scientific Reports, 15(1), 5369. https://doi.org/10.1038/s41598-025-89403-4
Shang, T. (2025). Improvement of key feature mining algorithm for sports injury data based on LOF enhanced k-means and sparse PCA. Informatica, 49(8). https://doi.org/10.31449/inf.v49i8.7230
Sung, P. H. (2025). Advanced machine learning for housing market analysis: Predicting property prices in Washington, DC. Authorea Preprints. 1-8. https://doi.org/10.36227/techrxiv.173603502.26172421/v2



