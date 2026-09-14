"""
model.py - K-Means Clustering Model and Tier Prediction

This module handles:
1. Training K-Means (k=3) on scaled tourism features (Density & Spend Per Visitor)
2. Mapping cluster numbers to semantic tiers: 'Overcrowded', 'Balanced', 'Under-visited'
   based on the average density of each cluster
3. Predicting the tier for new hypothetical states/areas
4. Computing cluster-level summary statistics
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def train_kmeans_model(X_scaled: np.ndarray, df: pd.DataFrame, random_state: int = 42):
    """
    Train a K-Means model (k=3) and map clusters to domain labels based on Density.
    
    Parameters:
        X_scaled (np.ndarray): 2D array of scaled features (Density, Spend_Per_Visitor_RM).
        df (pd.DataFrame): Original DataFrame with unscaled 'Density' column.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (kmeans_model, df_results, label_map, cluster_summary)
    """
    # 1. Initialize and fit KMeans with k=3
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    raw_clusters = kmeans.fit_predict(X_scaled)

    # 2. Attach raw cluster IDs to dataframe temporarily to compute cluster average densities
    df_temp = df.copy()
    df_temp["_cluster_id"] = raw_clusters

    # Calculate average unscaled density for each cluster
    cluster_densities = df_temp.groupby("_cluster_id")["Density"].mean().to_dict()

    # Sort cluster IDs by average density in ascending order
    # Lowest density -> Under-visited
    # Medium density -> Balanced
    # Highest density -> Overcrowded
    sorted_clusters_by_density = sorted(cluster_densities.keys(), key=lambda c: cluster_densities[c])

    label_map = {
        sorted_clusters_by_density[0]: "Under-visited",
        sorted_clusters_by_density[1]: "Balanced",
        sorted_clusters_by_density[2]: "Overcrowded"
    }

    # 3. Assign semantic labels to the dataset
    df_results = df.copy()
    df_results["KMeans_Cluster_ID"] = raw_clusters
    df_results["KMeans_Predicted_Tier"] = df_results["KMeans_Cluster_ID"].map(label_map)

    # Identify whether KMeans agrees with existing rule-based classification
    if "Rule_Based_Tier" in df_results.columns:
        df_results["Agreement"] = df_results["Rule_Based_Tier"] == df_results["KMeans_Predicted_Tier"]

    # 4. Compute cluster statistics for comparison charts
    cluster_summary = df_results.groupby("KMeans_Predicted_Tier").agg(
        Count=("State", "count"),
        Avg_Density=("Density", "mean"),
        Avg_Spend=("Spend_Per_Visitor_RM", "mean"),
        Avg_Visitors=("Domestic_Visitors", "mean"),
        Avg_Population=("Population", "mean")
    ).reset_index()

    # Reorder summary rows logically: Overcrowded, Balanced, Under-visited
    tier_order = {"Overcrowded": 0, "Balanced": 1, "Under-visited": 2}
    cluster_summary["_order"] = cluster_summary["KMeans_Predicted_Tier"].map(tier_order)
    cluster_summary = cluster_summary.sort_values("_order").drop(columns=["_order"])

    return kmeans, df_results, label_map, cluster_summary


def predict_tier(density: float, spend_per_visitor: float, scaler, kmeans_model, label_map: dict):
    """
    Predict the overtourism risk tier for a new user input.
    
    Parameters:
        density (float): Calculated visitor density (visitors per resident).
        spend_per_visitor (float): Calculated spend per visitor in RM.
        scaler: The fitted StandardScaler/MinMaxScaler instance.
        kmeans_model: The fitted KMeans model instance.
        label_map (dict): Mapping from cluster ID to tier name.
        
    Returns:
        dict: Result containing raw cluster ID, predicted tier, and formatted metrics.
    """
    # Create input DataFrame with the exact feature column names
    input_df = pd.DataFrame([[density, spend_per_visitor]], columns=["Density", "Spend_Per_Visitor_RM"])

    # Scale the input using the same scaler fitted on training data
    input_scaled = scaler.transform(input_df)

    # Predict cluster index
    cluster_id = int(kmeans_model.predict(input_scaled)[0])
    predicted_tier = label_map.get(cluster_id, "Unknown")

    return {
        "cluster_id": cluster_id,
        "predicted_tier": predicted_tier,
        "scaled_density": float(input_scaled[0, 0]),
        "scaled_spend": float(input_scaled[0, 1])
    }
