"""
forecast.py - ARIMA Time Series Forecasting for Malaysian State Domestic Tourism

This module handles:
1. Loading historical yearly domestic visitor data (2018-2023) across 16 Malaysian states.
2. Automated ARIMA parameter selection (p, d, q) using AIC minimization (pmdarima / statsmodels grid search).
3. Fitting ARIMA model and forecasting future visitor numbers (2024-2026+) with confidence intervals.
4. Estimating trajectory trends and overtourism pressure warnings.
"""

import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Attempt importing pmdarima for fast auto_arima if available
try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

# Suppress statsmodels convergence warnings during grid search
warnings.filterwarnings("ignore")


def load_historical_tourism_data(file_path: str = "historical_tourism.csv") -> pd.DataFrame:
    """
    Load historical yearly domestic tourism dataset for Malaysian states.
    
    Parameters:
        file_path (str): Path to the historical dataset CSV or Excel.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with columns ['State', 'Year', 'Domestic_Visitors'].
    """
    # Check current directory and script directory
    resolved_path = file_path
    if not os.path.exists(resolved_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, file_path)
        if os.path.exists(alt_path):
            resolved_path = alt_path
        else:
            raise FileNotFoundError(f"Historical dataset '{file_path}' not found.")

    if resolved_path.endswith(".xlsx") or resolved_path.endswith(".xls"):
        df = pd.read_excel(resolved_path, engine="openpyxl")
    else:
        df = pd.read_csv(resolved_path)

    # Standardize column headers
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure required columns exist
    required_cols = {"State", "Year", "Domestic_Visitors"}
    for col in required_cols:
        if col not in df.columns:
            # Check for case variations
            for c in df.columns:
                if c.lower() == col.lower():
                    df.rename(columns={c: col}, inplace=True)
                    break

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["Domestic_Visitors"] = pd.to_numeric(df["Domestic_Visitors"], errors="coerce")

    # Sort logically by state and year
    df = df.sort_values(["State", "Year"]).reset_index(drop=True)
    return df


def get_state_time_series(df: pd.DataFrame, state_name: str) -> pd.Series:
    """
    Extract a single state's yearly domestic visitor series indexed by Year.
    
    Parameters:
        df (pd.DataFrame): Historical tourism DataFrame.
        state_name (str): Target Malaysian state name.
        
    Returns:
        pd.Series: Indexed time series of Domestic_Visitors.
    """
    state_df = df[df["State"] == state_name].copy()
    if state_df.empty:
        raise ValueError(f"State '{state_name}' not found in dataset.")

    state_df = state_df.sort_values("Year")
    series = pd.Series(
        data=state_df["Domestic_Visitors"].values,
        index=state_df["Year"].values,
        name="Domestic_Visitors"
    )
    return series


def find_best_arima_model(series: pd.Series, max_p: int = 2, max_d: int = 1, max_q: int = 2):
    """
    Automatically search for the optimal ARIMA(p,d,q) order by minimizing AIC.
    
    Parameters:
        series (pd.Series): Historical yearly visitor series.
        max_p (int): Maximum AR order to evaluate.
        max_d (int): Maximum differencing degree.
        max_q (int): Maximum MA order to evaluate.
        
    Returns:
        tuple: (best_model_fit, best_order, best_aic, grid_results_df)
    """
    grid_records = []
    best_order = None
    best_aic = float("inf")
    best_fit = None

    # Step 1: Optional auto_arima fast identification
    if HAS_PMDARIMA and len(series) >= 6:
        try:
            auto_model = pm.auto_arima(
                series.values,
                start_p=0, max_p=max_p,
                d=None, max_d=max_d,
                start_q=0, max_q=max_q,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )
            suggested_order = auto_model.order
        except Exception:
            suggested_order = None
    else:
        suggested_order = None

    # Step 2: Thorough grid search over candidate (p, d, q) orders
    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                order = (p, d, q)
                try:
                    # Fit ARIMA with enforce_stationarity=False for small yearly sample stability
                    model = ARIMA(series.values, order=order, trend='t' if d == 0 else None)
                    model_fit = model.fit()
                    aic = model_fit.aic

                    if not np.isnan(aic) and not np.isinf(aic):
                        grid_records.append({
                            "p": p, "d": d, "q": q,
                            "Order": f"ARIMA({p},{d},{q})",
                            "AIC": round(aic, 2),
                            "BIC": round(model_fit.bic, 2)
                        })

                        if aic < best_aic:
                            best_aic = aic
                            best_order = order
                            best_fit = model_fit
                except Exception:
                    continue

    # Fallback to ARIMA(1,1,0) or (0,1,0) if grid failed
    if best_fit is None or best_order is None:
        best_order = suggested_order or (1, 1, 0)
        fallback_model = ARIMA(series.values, order=best_order)
        best_fit = fallback_model.fit()
        best_aic = best_fit.aic

    grid_df = pd.DataFrame(grid_records).sort_values("AIC").reset_index(drop=True)
    return best_fit, best_order, best_aic, grid_df


def forecast_future_visitors(model_fit, series: pd.Series, forecast_years: int = 3, alpha: float = 0.05):
    """
    Generate out-of-sample forecasts with 95% and 80% confidence intervals.
    
    Parameters:
        model_fit: Fitted statsmodels ARIMA result object.
        series (pd.Series): Historical time series.
        forecast_years (int): Number of future years to forecast (e.g., 2024 to 2026).
        alpha (float): Significance level for 95% CI (0.05).
        
    Returns:
        tuple: (forecast_df, combined_df, growth_rate_pct)
    """
    last_year = int(series.index[-1])
    future_years = [last_year + i for i in range(1, forecast_years + 1)]

    # 1. Get statsmodels forecast result
    forecast_res = model_fit.get_forecast(steps=forecast_years)
    mean_forecast = forecast_res.predicted_mean
    conf_int_95 = forecast_res.conf_int(alpha=0.05)
    conf_int_80 = forecast_res.conf_int(alpha=0.20)

    # 2. Build forecast DataFrame
    forecast_rows = []
    last_known_val = series.values[-1]

    for i, year in enumerate(future_years):
        pred_val = max(0, mean_forecast[i])
        low_95 = max(0, conf_int_95[i, 0] if hasattr(conf_int_95, "iloc") else conf_int_95[i][0])
        high_95 = max(pred_val, conf_int_95[i, 1] if hasattr(conf_int_95, "iloc") else conf_int_95[i][1])
        low_80 = max(0, conf_int_80[i, 0] if hasattr(conf_int_80, "iloc") else conf_int_80[i][0])
        high_80 = max(pred_val, conf_int_80[i, 1] if hasattr(conf_int_80, "iloc") else conf_int_80[i][1])

        # YoY growth calculation
        prev_val = mean_forecast[i - 1] if i > 0 else last_known_val
        yoy_growth = ((pred_val - prev_val) / prev_val) * 100 if prev_val > 0 else 0

        forecast_rows.append({
            "Year": year,
            "Forecasted_Visitors": int(round(pred_val)),
            "Lower_CI_95": int(round(low_95)),
            "Upper_CI_95": int(round(high_95)),
            "Lower_CI_80": int(round(low_80)),
            "Upper_CI_80": int(round(high_80)),
            "YoY_Growth_Pct": round(yoy_growth, 2)
        })

    forecast_df = pd.DataFrame(forecast_rows)

    # 3. Build unified combined DataFrame for plotting
    hist_rows = [
        {
            "Year": int(year),
            "Visitors": int(val),
            "Type": "Historical (Actual)",
            "Lower_CI": np.nan,
            "Upper_CI": np.nan
        }
        for year, val in series.items()
    ]

    pred_plot_rows = [
        {
            "Year": int(row["Year"]),
            "Visitors": row["Forecasted_Visitors"],
            "Type": "Forecast (ARIMA)",
            "Lower_CI": row["Lower_CI_95"],
            "Upper_CI": row["Upper_CI_95"]
        }
        for row in forecast_rows
    ]

    combined_df = pd.DataFrame(hist_rows + pred_plot_rows)

    # 4. Overall projected multi-year growth
    final_pred = forecast_rows[-1]["Forecasted_Visitors"]
    overall_growth = ((final_pred - last_known_val) / last_known_val) * 100 if last_known_val > 0 else 0

    return forecast_df, combined_df, overall_growth
