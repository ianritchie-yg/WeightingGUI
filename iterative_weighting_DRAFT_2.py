from typing import Any, Dict, Tuple, Callable, Union
import pandas as pd
import streamlit as st
import numpy as np
import logging
import plotly.express as px
from functools import partial
from io import StringIO

# ---------------------- Configuration ----------------------
st.set_page_config(page_title="Survey Weighting Suite", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ---------------------- Helper Functions ----------------------
def standardize_cell_key(cell: Any, grouping: Tuple[Any, ...]) -> Union[Any, Tuple[Any, ...]]:
    """
    Standardizes a cell key based on the grouping.
    For multi-column groupings, ensures the key is a tuple.
    """
    if not isinstance(grouping, tuple) or not grouping:
        raise ValueError("`grouping` must be a non-empty tuple")
    
    if len(grouping) > 1:
        return cell if isinstance(cell, tuple) else (cell,)
    else:
        return cell

def precompute_group_indices(df: pd.DataFrame, keys: list) -> Dict[Any, Any]:
    return {grouping: df.groupby(list(grouping)).indices for grouping in keys}

class WeightingEngine:
    def __init__(self, df: pd.DataFrame, targets: Dict[Tuple[str, ...], Dict[Any, float]], 
                 min_weight: float, max_weight: float) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a DataFrame")
        if not isinstance(targets, dict):
            raise TypeError("`targets` must be a dictionary")
        if not isinstance(min_weight, (int, float)):
            raise TypeError("`min_weight` must be numeric")
        if not isinstance(max_weight, (int, float)):
            raise TypeError("`max_weight` must be numeric")
        self.df = df
        self.targets = self._normalize_targets(targets)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.group_indices = precompute_group_indices(df, list(self.targets.keys()))
        self._validate_targets()

    def optimize(self, weights: pd.Series, max_iter: int, threshold: float, 
                 progress_callback: Callable[[int, int], None] = None) -> None:
        convergence_data = []
        for it in range(max_iter):
            if progress_callback:
                progress_callback(it + 1, max_iter)
            max_rel_diff, worst_group = self._evaluate_groups(weights)
            convergence_data.append((it + 1, max_rel_diff))
            if max_rel_diff <= threshold:
                break
            weights = self._adjust_weights(weights, worst_group)
            
            # Log convergence trend for tracking non-convergence issues
            logging.info(f"Iteration {it + 1}: Max Relative Difference = {max_rel_diff}")
        
        self.convergence_data = convergence_data

    def _calculate_observed(self, weights: pd.Series, grouping: Tuple[Any, ...]) -> Dict[Any, float]:
        weighted_df = self.df.assign(weight=weights)
        group_totals = weighted_df.groupby(list(grouping))['weight'].sum()
        observed = {standardize_cell_key(cell, grouping): group_totals.get(cell, 0)
                    for cell in self.group_indices[grouping].keys()}
        return observed

    def _adjust_weights(self, weights: pd.Series, worst_group: Any) -> pd.Series:
        observed_totals = self._calculate_observed(weights, worst_group)
        target_totals = self.targets[worst_group]
        adjustment_factors = {
            cell: (target_totals[cell] / observed_totals[cell]) if observed_totals.get(cell, 0) > 0 else 1.0
            for cell in target_totals.keys()
        }
        new_weights = weights.copy()
        for cell, indices in self.group_indices[worst_group].items():
            factor = adjustment_factors.get(cell, 1.0)
            new_weights.loc[indices] *= factor
        
        return new_weights.clip(lower=self.min_weight, upper=self.max_weight)

def effective_sample_size(weights):
    if (weights <= 0).any():
        raise ValueError("All weights must be positive and non-zero.")
    return (weights.sum() ** 2) / (weights ** 2).sum()

def plot_convergence(convergence_data):
    df = pd.DataFrame(convergence_data, columns=["Iteration", "Max Relative Difference"])
    fig = px.line(df, x="Iteration", y="Max Relative Difference", title="Convergence Trend")
    st.plotly_chart(fig, use_container_width=True)

def handle_missing_values(df):
    for col in df.select_dtypes(include=['category', 'object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def main():
    st.title("📊 Survey Weighting Suite")
    uploaded_file = st.sidebar.file_uploader("Upload survey data", type=["csv"])
    if not uploaded_file:
        st.info("Upload a CSV file to begin")
        return
    df = pd.read_csv(uploaded_file)
    df = handle_missing_values(df)
    min_weight = st.sidebar.number_input("Minimum weight", min_value=0.0, value=0.1, step=0.01)
    max_weight = st.sidebar.number_input("Maximum weight", min_value=0.1, value=5.0, step=0.1)
    threshold = st.sidebar.number_input("Convergence threshold", min_value=0.0001, value=0.001, step=0.0001)
    max_iter = st.sidebar.number_input("Maximum iterations", min_value=1, value=50, step=1)
    cat_cols = st.multiselect("Select categorical columns", options=df.columns, 
                              default=list(df.select_dtypes('object').columns))
    for col in cat_cols:
        df[col] = df[col].astype('category')
    st.sidebar.button("Run Weighting Analysis", on_click=lambda: st.info("Weighting not yet implemented."))

if __name__ == "__main__":
    main()
