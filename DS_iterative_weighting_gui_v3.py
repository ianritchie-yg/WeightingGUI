from typing import Any, Dict, Tuple, Callable, Union
import pandas as pd
import numpy as np
import streamlit as st
import logging
import plotly.express as px
from functools import partial
from io import StringIO

# ---------------------- Configuration ----------------------
st.set_page_config(page_title="Survey Weighting Suite", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ---------------------- Helper Functions ----------------------
@st.cache_data
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in categorical columns"""
    for col in df.select_dtypes(include=['category', 'object']).columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logging.info(f"Filled missing values in {col} with mode: {mode_val}")
    return df

@st.cache_data
def standardize_cell_key(cell: Any, grouping: Tuple[Any, ...]) -> Union[Any, Tuple[Any, ...]]:
    """Standardizes cell key based on grouping structure"""
    if not isinstance(grouping, tuple) or not grouping:
        raise ValueError("`grouping` must be a non-empty tuple")
    
    return cell if (len(grouping) == 1 or isinstance(cell, tuple)) else (cell,)

@st.cache_data
def precompute_group_indices(df: pd.DataFrame, groupings: list) -> Dict[Any, Any]:
    """Precompute group indices with type validation"""
    if not isinstance(groupings, list):
        raise TypeError("Groupings must be a list of tuples")
    
    return {
        grouping: df.groupby(list(grouping)).indices
        for grouping in groupings
    }

def effective_sample_size(weights: pd.Series) -> float:
    """Calculate effective sample size with validation"""
    if (weights <= 0).any():
        raise ValueError("Weights must be positive for ESS calculation")
    return (weights.sum() ** 2) / (weights ** 2).sum()

# ---------------------- Enhanced Weighting Core ----------------------
class WeightingEngine:
    def __init__(self, df: pd.DataFrame, targets: Dict[Tuple[str, ...], Dict[Any, float]], 
                 min_weight: float, max_weight: float, **params) -> None:
        """Initialize with comprehensive validation"""
        self._validate_inputs(df, targets, min_weight, max_weight)
        
        self.df = df.copy()
        self.targets = self._normalize_targets(targets)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.params = params
        
        np.random.seed(self.params.get('random_seed', 42))
        self.group_indices = precompute_group_indices(df, list(self.targets.keys()))
        self._validate_targets()
        self.convergence_data = []

    def _validate_inputs(self, df, targets, min_weight, max_weight):
        """Centralized input validation"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if min_weight >= max_weight:
            raise ValueError("Minimum weight must be less than maximum weight")
        if any(len(group) == 0 for group in targets.keys()):
            raise ValueError("Target groupings cannot be empty")

    def run(self, threshold: float = 0.001, max_iter: int = 50, 
           progress_callback: Callable[[int, int], None] = None) -> Tuple[pd.Series, list]:
        """Optimized weighting algorithm with convergence tracking"""
        weights = pd.Series(np.ones(len(self.df)), index=self.df.index)
        
        for iteration in range(1, max_iter + 1):
            self._update_progress(progress_callback, iteration, max_iter)
            
            max_rel_diff, worst_group = self._evaluate_convergence(weights)
            self._log_iteration(iteration, max_rel_diff)
            
            if max_rel_diff <= threshold:
                logging.info(f"Convergence achieved at iteration {iteration}")
                break
                
            weights = self._adjust_weights(weights, worst_group)
        
        return weights, self.convergence_data

    def _evaluate_convergence(self, weights: pd.Series) -> Tuple[float, Any]:
        """Evaluate convergence across all target groups"""
        max_rel_diff = 0.0
        worst_group = None
        
        for grouping, tdict in self.targets.items():
            observed = self._calculate_observed(weights, grouping)
            for cell, target in tdict.items():
                rel_diff = self._calculate_relative_diff(observed, cell, target)
                if rel_diff > max_rel_diff:
                    max_rel_diff, worst_group = rel_diff, grouping
        
        return max_rel_diff, worst_group

    def _calculate_relative_diff(self, observed: Dict, cell: Any, target: float) -> float:
        """Calculate relative difference with error handling"""
        try:
            return abs(observed.get(cell, 0) - target) / target
        except ZeroDivisionError:
            return float('inf')

# ---------------------- Streamlit Interface Improvements ----------------------
def main():
    st.title("📊 Survey Weighting Suite")
    
    # Data processing with missing value handling
    df = load_and_preprocess_data()
    if df is None: return
    
    # Parameter configuration
    targets = configure_targets(df)
    params = get_weighting_parameters()
    
    # Analysis execution
    if st.sidebar.button("Run Weighting Analysis"):
        with st.spinner("Optimizing weights..."):
            results = execute_weighting(df, targets, params)
            display_results(results, params)

def load_and_preprocess_data() -> Union[pd.DataFrame, None]:
    """Enhanced data loading with preprocessing"""
    uploaded_file = st.sidebar.file_uploader("Upload survey data", type=["csv"])
    if not uploaded_file:
        st.info("Upload a CSV file to begin")
        return None
    
    df = pd.read_csv(uploaded_file)
    df = handle_missing_values(df)
    
    # Smart categorical detection
    default_cat_cols = list(df.select_dtypes(include=['object']).columns)
    cat_cols = st.sidebar.multiselect(
        "Select categorical columns", 
        options=df.columns,
        default=default_cat_cols
    )
    
    return df.assign(**{col: df[col].astype('category') for col in cat_cols})

def display_results(results: dict, params: dict) -> None:
    """Structured results display"""
    st.header("Analysis Results")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            plot_weight_distribution(results['weights'], params['hist_bins'])
        with col2:
            plot_convergence_trend(results['convergence'])
            
    show_effective_sample_size(results['weights'])
    
    if params.get('compute_variance'):
        st.subheader("Weight Variance")
        st.metric("Variance", f"{results['weights'].var():.4f}")

def plot_convergence_trend(convergence: list) -> None:
    """Enhanced convergence plotting"""
    df = pd.DataFrame(convergence, columns=["Iteration", "Max Relative Difference"])
    fig = px.line(df, x="Iteration", y="Max Relative Difference", 
                 title="Convergence Trend", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Execution ----------------------
if __name__ == "__main__":
    main()
