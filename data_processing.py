"""
data_processing.py - Data Loading and Preprocessing for Malaysia Overtourism Classifier

This module handles:
1. Loading the Excel sheet 'Cleaned_Data_V3' from Sustainable_Tourism_V1.xlsx
2. Cleaning and standardizing column names
3. Feature extraction and scaling (StandardScaler / MinMaxScaler)
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_tourism_data(file_path: str = "Sustainable_Tourism_V1.xlsx", sheet_name: str = "Cleaned_Data_V3") -> pd.DataFrame:
    """
    Load and clean tourism data from the Excel spreadsheet.
    
    Parameters:
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the target worksheet.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with stripped column headers and proper data types.
    """
    # Try resolving path relative to script directory if not found directly
    resolved_path = file_path
    if not os.path.exists(resolved_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, file_path)
        if os.path.exists(alt_path):
            resolved_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset file '{file_path}' not found at '{resolved_path}' or '{alt_path}'.")

    # Read the target worksheet using openpyxl engine
    df = pd.read_excel(resolved_path, sheet_name=sheet_name, engine="openpyxl")

    # Strip any leading/trailing whitespaces in column headers
    df.columns = [str(col).strip() for col in df.columns]

    # Standardize column naming if variations exist
    column_mapping = {
        "Domestic_Visitors_2023": "Domestic_Visitors",
        "Total_Receipts_2023 (RM)": "Total_Receipts_RM",
        "Population_2023": "Population",
        "Density (visitors per resident)": "Density",
        "Spend_Per_Visitor(RM)": "Spend_Per_Visitor_RM",
        "Tier": "Rule_Based_Tier"
    }
    
    # Rename columns matching the mapping keys
    for orig_col, new_col in column_mapping.items():
        for c in df.columns:
            if c.lower().startswith(orig_col.lower()[:12]):
                df.rename(columns={c: new_col}, inplace=True)
                break

    # Ensure numeric types
    numeric_cols = ["Domestic_Visitors", "Total_Receipts_RM", "Population", "Density", "Spend_Per_Visitor_RM"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def prepare_features(df: pd.DataFrame, scaler_type: str = "standard"):
    """
    Extract and scale features ('Density' and 'Spend_Per_Visitor_RM') for K-Means clustering.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'Density' and 'Spend_Per_Visitor_RM'.
        scaler_type (str): 'standard' for StandardScaler or 'minmax' for MinMaxScaler.
        
    Returns:
        tuple: (X_scaled, scaler, feature_cols)
    """
    feature_cols = ["Density", "Spend_Per_Visitor_RM"]
    X = df[feature_cols].copy()

    # Initialize scaler
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    # Fit and transform features
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, feature_cols
