"""
socio_data.py - Data Processing for Socio-Economic Metrics

This module loads, cleans, and merges three datasets:
1. Population_by_State_2019_2025.xlsx
2. Labour_Force_2019_2023.xlsx
3. HIES_State_2022&2024.xlsx

It outputs a unified long-format DataFrame with interpolated missing values,
suitable for ARIMA forecasting.
"""

import os
import pandas as pd
import numpy as np

def load_and_clean_socio_data(
    pop_file="Population_by_State_2019_2025.xlsx",
    lf_file="Labour_Force_2019_2023.xlsx",
    hies_file="HIES_State_2022&2024.xlsx"
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    def resolve_path(file_path):
        if os.path.exists(file_path):
            return file_path
        alt_path = os.path.join(script_dir, file_path)
        if os.path.exists(alt_path):
            return alt_path
        raise FileNotFoundError(f"Dataset '{file_path}' not found.")

    pop_path = resolve_path(pop_file)
    lf_path = resolve_path(lf_file)
    hies_path = resolve_path(hies_file)

    # 1. Process Population Data
    df_pop_raw = pd.read_excel(pop_path, skiprows=2)
    # Ensure columns are strings so we can melt correctly
    df_pop_raw.columns = [str(c).split(".")[0] for c in df_pop_raw.columns]
    
    # Filter out empty rows if any
    df_pop_raw = df_pop_raw.dropna(subset=["State"])
    
    df_pop = df_pop_raw.melt(
        id_vars=["State"], 
        var_name="Year", 
        value_name="Population"
    )
    df_pop["Year"] = pd.to_numeric(df_pop["Year"], errors="coerce")
    df_pop = df_pop.dropna(subset=["Year"])
    df_pop["Year"] = df_pop["Year"].astype(int)

    # 2. Process Labour Force Data
    lf_xls = pd.ExcelFile(lf_path)
    lf_frames = []
    for sheet in lf_xls.sheet_names:
        df_sheet = pd.read_excel(lf_xls, sheet_name=sheet, skiprows=2)
        df_sheet = df_sheet.dropna(subset=["State"])
        df_sheet["Year"] = int(sheet)
        
        # Standardize column names
        rename_map = {
            "Labour Force Size": "Labour_Force_Size",
            "Employed Persons": "Employed",
            "Unemployed\nPersons": "Unemployed",
            "Unemployment Rate\n(%)": "Unemployment_Rate",
            "Employment Population\nRatio (%)": "Employment_Population_Ratio"
        }
        
        for col in rename_map.keys():
            if col not in df_sheet.columns:
                # Fallback fuzzy match
                for c in df_sheet.columns:
                    if col.replace("\n", "").lower() in str(c).replace("\n", "").lower():
                        df_sheet.rename(columns={c: rename_map[col]}, inplace=True)
                        break
            else:
                df_sheet.rename(columns={col: rename_map[col]}, inplace=True)
                
        # Keep only the standardized columns we need + State and Year
        keep_cols = ["State", "Year"] + list(rename_map.values())
        existing_cols = [c for c in keep_cols if c in df_sheet.columns]
        lf_frames.append(df_sheet[existing_cols])
        
    df_lf = pd.concat(lf_frames, ignore_index=True)

    # 3. Process HIES Data
    hies_xls = pd.ExcelFile(hies_path)
    hies_frames = []
    for sheet in hies_xls.sheet_names:
        df_sheet = pd.read_excel(hies_xls, sheet_name=sheet, skiprows=2)
        df_sheet = df_sheet.dropna(subset=["State"])
        # Some sheets might have weird names, but we expect "2022" and "2024"
        try:
            year_val = int(sheet)
        except ValueError:
            # Fallback if sheet name is not just the year
            year_val = 2022 if "2022" in sheet else 2024
        df_sheet["Year"] = year_val
        
        # Standardize column names
        rename_map = {
            "Mean Income\n(RM)": "Mean_Income",
            "Gini": "Gini",
            "Poverty Rate\n(%)": "Poverty_Rate"
        }
        
        for col in rename_map.keys():
            if col not in df_sheet.columns:
                # Fallback fuzzy match
                for c in df_sheet.columns:
                    if col.replace("\n", "").lower() in str(c).replace("\n", "").lower():
                        df_sheet.rename(columns={c: rename_map[col]}, inplace=True)
                        break
            else:
                df_sheet.rename(columns={col: rename_map[col]}, inplace=True)
                
        keep_cols = ["State", "Year"] + list(rename_map.values())
        existing_cols = [c for c in keep_cols if c in df_sheet.columns]
        hies_frames.append(df_sheet[existing_cols])
        
    df_hies = pd.concat(hies_frames, ignore_index=True)

    # Normalize State names
    def normalize_state(s):
        s = str(s).strip()
        if s == "Kuala Lumpur": return "W.P. Kuala Lumpur"
        if s == "Labuan": return "W.P. Labuan"
        if s == "Putrajaya": return "W.P. Putrajaya"
        if s == "Pinang": return "Pulau Pinang"
        return s

    df_pop["State"] = df_pop["State"].apply(normalize_state)
    df_lf["State"] = df_lf["State"].apply(normalize_state)
    df_hies["State"] = df_hies["State"].apply(normalize_state)

    # 4. Merge all three dataframes
    # Merge Population and Labour Force
    df_merged = pd.merge(df_pop, df_lf, on=["State", "Year"], how="outer")
    # Merge with HIES
    df_merged = pd.merge(df_merged, df_hies, on=["State", "Year"], how="outer")
    
    # Sort logically
    df_merged = df_merged.sort_values(["State", "Year"]).reset_index(drop=True)

    # 5. Interpolate missing HIES values (and others) linearly by State
    # This addresses the gaps in 2019-2021 and 2023 for HIES data
    metrics = [
        "Population", "Labour_Force_Size", "Employed", "Unemployed",
        "Unemployment_Rate", "Employment_Population_Ratio",
        "Mean_Income", "Gini", "Poverty_Rate"
    ]
    
    # Ensure numeric types
    for m in metrics:
        if m in df_merged.columns:
            df_merged[m] = pd.to_numeric(df_merged[m], errors="coerce")

    # Apply interpolation per state
    def interpolate_state(group):
        state_name = group.name
        if "State" in group.columns:
            group = group.drop(columns=["State"])
        group = group.set_index("Year")
        # Generate full range of years from min to max for this state
        full_years = range(int(group.index.min()), int(group.index.max()) + 1)
        group = group.reindex(full_years)
        
        # Interpolate linearly, then backfill/forward-fill remaining NAs
        group[metrics] = group[metrics].interpolate(method="linear").bfill().ffill()
        group["State"] = state_name  # Ensure State is filled for reindexed rows
        group = group.reset_index()
        # Ensure Year is named correctly and is int
        group = group.rename(columns={"index": "Year"})
        return group

    df_merged = df_merged.groupby("State", group_keys=False).apply(interpolate_state)
    # Just in case 'State' got pushed to index
    if "State" not in df_merged.columns and df_merged.index.name == "State":
        df_merged = df_merged.reset_index()
    elif "State" not in df_merged.columns and "State" in df_merged.index.names:
        df_merged = df_merged.reset_index(level="State")

    df_merged = df_merged.sort_values(["State", "Year"]).reset_index(drop=True)
    
    return df_merged

if __name__ == "__main__":
    df = load_and_clean_socio_data()
    print("Merged and Interpolated DataFrame:")
    print(df.head(10))
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
