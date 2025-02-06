from typing import Any, Dict, Tuple, Callable, Union
import pandas as pd
import numpy as np
import streamlit as st
import logging
import plotly.express as px
from functools import partial
from io import StringIO

# Explanation of Key Updates
	# 1.	Additional Algorithm Parameters:
	# •	max_adj_factor & smoothing_factor: These parameters control the magnitude and damping of weight adjustments in _adjust_weights.
	# •	zero_cell_strategy: Specifies how to handle cells with zero observed respondents when a nonzero target is present (options: "error", "impute", "skip").
	# •	verbose & reporting_frequency: Control detailed iteration logging.
	# •	weighting_method: Provides a (currently nominal) option to select different weighting approaches.
	# •	random_seed: Ensures reproducibility.
	# •	compute_variance: Toggles variance estimation on the final weights.
	# 2.	GUI Enhancements:
	# •	The get_weighting_params function now collects these additional parameters via Streamlit’s sidebar controls.
	# •	A new parameter for histogram bin count (hist_bins) has been added and used in the weight distribution plot.
	# 3.	Algorithm Modifications:
	# •	In _adjust_weights, the computed adjustment factor is modified by the smoothing factor and capped by the maximum adjustment factor.
	# •	The code now checks the zero_cell_strategy and handles zero-observed cells accordingly.
	# •	Verbose logging and reporting frequency are used in the run loop to provide detailed iteration feedback.
	# 4.	Variance Estimation:
	# •	If compute_variance is enabled, the variance of the final weights is calculated and displayed in the results.

# ---------------------- Configuration ----------------------
st.set_page_config(page_title="Survey Weighting Suite", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ---------------------- Helper Functions ----------------------
@st.cache_data
def standardize_cell_key(cell: Any, grouping: Tuple[Any, ...]) -> Union[Any, Tuple[Any, ...]]:
    """
    Standardizes a cell key based on the grouping.
    For multi-column groupings, ensures the key is a tuple.
    """
    # Validate that grouping is a non-empty tuple
    if not isinstance(grouping, tuple) or not grouping:
        raise ValueError("`grouping` must be a non-empty tuple")
    if len(grouping) > 1:
        return cell if isinstance(cell, tuple) else (cell,)
    else:
        return cell

@st.cache_data
def precompute_group_indices(df: pd.DataFrame, groupings: list) -> Dict[Any, Any]:
    """Precompute group indices for all target groupings"""
    return {
        grouping: df.groupby(list(grouping)).indices
        for grouping in groupings
    }

def effective_sample_size(weights: pd.Series) -> float:
    """Calculate effective sample size"""
    return (weights.sum() ** 2) / (weights ** 2).sum()

def plot_distribution_comparison(df: pd.DataFrame, col: str, original_weights: np.ndarray, adjusted_weights: np.ndarray):
    """Plot pre/post distribution comparison using Plotly"""
    # Prepare two datasets: one with original uniform weights and one with adjusted weights.
    df_orig = df.copy()
    df_orig['Weight'] = original_weights
    df_orig['Dataset'] = 'Original'
    df_adj = df.copy()
    df_adj['Weight'] = adjusted_weights
    df_adj['Dataset'] = 'Adjusted'
    combined = pd.concat([df_orig, df_adj])
    
    fig = px.histogram(
        combined, x=col, color='Dataset', barmode='group',
        histnorm='percent', title=f'Distribution Comparison: {col}',
        labels={col: col}, nbins=20
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Enhanced Weighting Core ----------------------
class WeightingEngine:
    def __init__(self, 
                 df: pd.DataFrame, 
                 targets: Dict[Tuple[str, ...], Dict[Any, float]], 
                 min_weight: float, 
                 max_weight: float,
                 max_adj_factor: float = 2.0,
                 smoothing_factor: float = 1.0,
                 zero_cell_strategy: str = "error",
                 verbose: bool = False,
                 reporting_frequency: int = 1,
                 weighting_method: str = "Raking",
                 random_seed: int = 42,
                 compute_variance: bool = False
                 ) -> None:
        # Validate input types
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a DataFrame")
        if not isinstance(targets, dict):
            raise TypeError("`targets` must be a dictionary")
        if not isinstance(min_weight, (int, float)):
            raise TypeError("`min_weight` must be numeric")
        if not isinstance(max_weight, (int, float)):
            raise TypeError("`max_weight` must be numeric")
        
        # Set attributes
        self.df = df
        self.targets = self._normalize_targets(targets)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_adj_factor = max_adj_factor
        self.smoothing_factor = smoothing_factor
        self.zero_cell_strategy = zero_cell_strategy  # "error", "impute", or "skip"
        self.verbose = verbose
        self.reporting_frequency = reporting_frequency
        self.weighting_method = weighting_method  # Currently only "Raking" is implemented
        self.random_seed = random_seed
        self.compute_variance = compute_variance
        
        np.random.seed(self.random_seed)
        self.group_indices = precompute_group_indices(df, list(self.targets.keys()))
        self._validate_targets()
        self.convergence_data = []

    def _normalize_targets(self, targets: Dict) -> Dict:
        """Ensure all grouping keys are tuples"""
        normalized = {}
        for key, target_dict in targets.items():
            if isinstance(key, str):
                normalized[(key,)] = target_dict
            else:
                normalized[key] = target_dict
        return normalized

    def _validate_targets(self) -> None:
        """Comprehensive target validation"""
        for grouping, tdict in self.targets.items():
            for col in grouping:
                if col not in self.df.columns:
                    raise ValueError(f"Column '{col}' not found in data")
                # Additional validations (e.g., ensuring targets are non-negative) can be added here.

    def run(self, threshold: float = 0.001, max_iter: int = 50, progress_callback: Callable[[int, int], None] = None) -> Tuple[pd.Series, list]:
        """Enhanced iterative weighting algorithm with progress updates"""
        weights = pd.Series(np.ones(len(self.df)), index=self.df.index)
        convergence_data = []
        
        for it in range(max_iter):
            if progress_callback:
                progress_callback(it + 1, max_iter)
                
            max_rel_diff, worst_group = self._evaluate_groups(weights)
            convergence_data.append((it + 1, max_rel_diff))
            
            if self.verbose and ((it + 1) % self.reporting_frequency == 0):
                logging.info(f"Iteration {it+1}: max relative difference = {max_rel_diff:.4f}")
            
            if max_rel_diff <= threshold:
                break
                
            weights = self._adjust_weights(weights, worst_group)
        
        self.convergence_data = convergence_data
        
        if self.compute_variance:
            self.variance = weights.var()
        else:
            self.variance = None
            
        return weights, convergence_data

    def _calculate_observed(self, weights: pd.Series, grouping: Tuple[str, ...], group_indices: Dict[Any, Any]) -> Dict[Any, float]:
        """Calculate observed weighted counts for a given grouping"""
        weighted_df = self.df.assign(weight=weights)
        group_totals = weighted_df.groupby(list(grouping))['weight'].sum()
        observed = {
            standardize_cell_key(cell, grouping): group_totals.get(cell, 0)
            for cell in group_indices.keys()
        }
        return observed

    def _adjust_weights(self, weights: pd.Series, grouping: Any) -> pd.Series:
        """Adjust weights for a particular grouping using additional parameters"""
        group_indices = self.group_indices[grouping]
        tdict = self.targets[grouping]
        adjustment_factors = {}
        
        for cell, target in tdict.items():
            standardized_cell = standardize_cell_key(cell, grouping)
            indices = group_indices.get(standardized_cell, [])
            observed = weights.loc[indices].sum() if len(indices) > 0 else 0
            
            if observed == 0:
                # Handle zero observed counts according to the selected strategy
                if self.zero_cell_strategy == "error":
                    raise ValueError(f"Zero respondents in cell {standardized_cell} for grouping {grouping} with nonzero target {target}.")
                elif self.zero_cell_strategy == "impute":
                    # Impute a very small positive value to allow adjustment
                    observed = 1e-6
                elif self.zero_cell_strategy == "skip":
                    adjustment_factors[standardized_cell] = 1.0
                    continue
            
            raw_factor = target / observed
            # Apply smoothing/dampening: new_factor = 1 + smoothing*(raw_factor - 1)
            new_factor = 1 + self.smoothing_factor * (raw_factor - 1)
            # Cap the adjustment factor to avoid extreme changes
            if new_factor > self.max_adj_factor:
                new_factor = self.max_adj_factor
            adjustment_factors[standardized_cell] = new_factor

        new_weights = weights.copy()
        for cell, indices in group_indices.items():
            standardized_cell = standardize_cell_key(cell, grouping)
            factor = adjustment_factors.get(standardized_cell, 1.0)
            new_weights.loc[indices] = new_weights.loc[indices] * factor
        
        # Ensure weights remain within specified bounds
        new_weights = new_weights.clip(lower=self.min_weight, upper=self.max_weight)
        return new_weights

    def _evaluate_groups(self, weights: pd.Series) -> Tuple[float, Any]:
        """Evaluate all target groups and return worst performer based on max relative difference"""
        max_rel_diff = 0.0
        worst_group = None
        
        for grouping, tdict in self.targets.items():
            group_indices = self.group_indices[grouping]
            observed = self._calculate_observed(weights, grouping, group_indices)
            for cell, target in tdict.items():
                standardized_cell = standardize_cell_key(cell, grouping)
                obs_val = observed.get(standardized_cell, 0)
                try:
                    rel_diff = abs(obs_val - target) / target
                except ZeroDivisionError:
                    rel_diff = float('inf')
                if rel_diff > max_rel_diff:
                    max_rel_diff = rel_diff
                    worst_group = grouping
                    
        return max_rel_diff, worst_group

# ---------------------- Streamlit Interface ----------------------
def main():
    st.title("📊 Survey Weighting Suite")
    
    # Data Upload Section
    uploaded_file = st.sidebar.file_uploader("Upload survey data", type=["csv"])
    if not uploaded_file:
        st.info("Upload a CSV file to begin")
        return
    
    # Data Processing
    df = load_and_preprocess(uploaded_file)
    
    # Target Configuration
    targets = configure_targets(df)
    
    # Weighting Parameters
    params = get_weighting_params()
    
    # Warn if an unimplemented convergence metric was chosen
    if params['convergence_metric'] != "Max Relative Difference":
        st.warning("Chi-Square convergence metric not implemented; defaulting to Max Relative Difference.")
    
    # Run Engine with progress bar
    if st.sidebar.button("Run Weighting Analysis"):
        progress_bar = st.progress(0)
        def progress_callback(current, total):
            progress_bar.progress(current / total)
        
        with st.spinner("Running iterative weighting..."):
            engine = WeightingEngine(
                df, targets, params['min_weight'], params['max_weight'],
                max_adj_factor=params['max_adj_factor'],
                smoothing_factor=params['smoothing_factor'],
                zero_cell_strategy=params['zero_cell_strategy'],
                verbose=params['verbose'],
                reporting_frequency=params['reporting_frequency'],
                weighting_method=params['weighting_method'],
                random_seed=params['random_seed'],
                compute_variance=params['compute_variance']
            )
            weights, convergence = engine.run(params['threshold'], params['max_iter'], progress_callback)
        
        show_results(df, weights, convergence, params['hist_bins'], params['compute_variance'])

def load_and_preprocess(uploaded_file) -> pd.DataFrame:
    """Handle data loading and preprocessing"""
    df = pd.read_csv(uploaded_file)
    st.sidebar.markdown("### Select Categorical Columns")
    cat_cols = st.sidebar.multiselect(
        "Choose categorical columns", 
        options=df.columns,
        default=[]  # Start with no preselected columns
    )
    for col in cat_cols:
        df[col] = df[col].astype('category')
    return df

def configure_targets(df: pd.DataFrame) -> Dict:
    """Interactive target configuration with CSV support"""
    st.sidebar.header("Target Configuration")
    target_file = st.sidebar.file_uploader("Upload targets CSV", type=["csv"], key="target_csv")
    if target_file:
        return parse_target_csv(target_file)
    else:
        return configure_interactive_targets(df)

def parse_target_csv(uploaded_file) -> Dict:
    """Parse CSV target configuration
    Expected CSV columns: grouping, cell, value
    Example row: age_group,25-34,300
    For interlocked groupings, separate cell values with a pipe character, e.g. "age_group,gender", "18-24|Male",120
    """
    try:
        target_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error parsing target CSV: {e}")
        return {}
    
    targets = {}
    for _, row in target_df.iterrows():
        grouping_str = str(row.get('grouping', '')).strip()
        if grouping_str:
            grouping = tuple([x.strip() for x in grouping_str.split(',') if x.strip()])
        else:
            grouping = tuple()  # Should not happen ideally.
        cell_val = str(row.get('cell', '')).strip()
        if '|' in cell_val:
            cell = tuple([x.strip() for x in cell_val.split('|')])
        else:
            cell = cell_val
        try:
            value = float(row.get('value', 0))
        except ValueError:
            st.error(f"Invalid target value in row: {row}")
            continue
        if grouping in targets:
            targets[grouping][cell] = value
        else:
            targets[grouping] = {cell: value}
    return targets

def configure_interactive_targets(df: pd.DataFrame) -> Dict:
    """Interactive target configuration UI"""
    st.sidebar.markdown("### Interactive Target Setup")
    targets = {}
    columns = list(df.columns)
    selected_groupings = st.sidebar.multiselect("Select grouping columns for targets", options=columns)
    if not selected_groupings:
        st.sidebar.info("No grouping columns selected. Using default sample targets.")
        return {
            'gender': {'Male': 480, 'Female': 520},
            ('age_group',): {'18-24': 250, '25-34': 300, '35-44': 300, '45+': 150}
        }
    else:
        for col in selected_groupings:
            st.sidebar.markdown(f"#### Targets for {col}")
            unique_vals = sorted(df[col].dropna().unique())
            target_dict = {}
            for val in unique_vals:
                default_target = int(df.shape[0] / len(unique_vals))
                target_val = st.sidebar.number_input(f"Target for {val}", min_value=0, value=default_target, key=f"{col}_{val}")
                target_dict[val] = target_val
            targets[col] = target_dict
    return targets

def get_weighting_params() -> Dict:
    """Collect weighting algorithm parameters from user"""
    st.sidebar.header("Algorithm Parameters")
    threshold = st.sidebar.slider(
        "Convergence Threshold (relative difference)",
        min_value=0.0001, max_value=0.01, value=0.001, step=0.0001,
        help="Lower values require closer matching to targets."
    )
    max_iter = st.sidebar.number_input(
        "Maximum Iterations", min_value=1, value=50, step=1,
        help="Maximum number of iterations before stopping."
    )
    min_weight = st.sidebar.number_input(
        "Minimum Weight", min_value=0.0, value=0.1, step=0.1,
        help="Weights will not go below this value."
    )
    max_weight = st.sidebar.number_input(
        "Maximum Weight", min_value=0.0, value=5.0, step=0.1,
        help="Weights will not exceed this value."
    )
    max_adj_factor = st.sidebar.number_input(
        "Maximum Adjustment Factor", min_value=1.0, value=2.0, step=0.1,
        help="Cap on the adjustment factor applied per iteration."
    )
    smoothing_factor = st.sidebar.slider(
        "Smoothing/Dampening Factor", min_value=0.0, max_value=1.0, value=1.0, step=0.1,
        help="Factor to dampen adjustments (0 = no adjustment, 1 = full adjustment)."
    )
    zero_cell_strategy = st.sidebar.selectbox(
        "Zero-Cell Handling Strategy", 
        options=["error", "impute", "skip"], index=0,
        help="How to handle cells with zero observed respondents but nonzero target."
    )
    verbose = st.sidebar.checkbox("Enable Detailed Iteration Logs", value=False)
    hist_bins = st.sidebar.number_input(
        "Number of Histogram Bins", min_value=10, value=50, step=1,
        help="Number of bins for weight distribution histograms."
    )
    convergence_metric = st.sidebar.selectbox(
        "Convergence Metric", 
        options=["Max Relative Difference", "Chi-Square"], index=0,
        help="Metric used for evaluating convergence (only Max Relative Difference is implemented)."
    )
    random_seed = st.sidebar.number_input(
        "Random Seed", min_value=0, value=42, step=1,
        help="Random seed for reproducibility."
    )
    reporting_frequency = st.sidebar.number_input(
        "Reporting Frequency", min_value=1, value=1, step=1,
        help="How often to log iteration details (every nth iteration)."
    )
    weighting_method = st.sidebar.selectbox(
        "Weighting Method", options=["Raking", "Calibration"], index=0,
        help="Choice of weighting algorithm (currently, only Raking is implemented)."
    )
    compute_variance = st.sidebar.checkbox("Compute Variance Estimates", value=False, help="Compute variance of final weights.")
    
    return {
        'threshold': threshold,
        'max_iter': max_iter,
        'min_weight': min_weight,
        'max_weight': max_weight,
        'max_adj_factor': max_adj_factor,
        'smoothing_factor': smoothing_factor,
        'zero_cell_strategy': zero_cell_strategy,
        'verbose': verbose,
        'hist_bins': hist_bins,
        'convergence_metric': convergence_metric,
        'random_seed': random_seed,
        'reporting_frequency': reporting_frequency,
        'weighting_method': weighting_method,
        'compute_variance': compute_variance
    }

def plot_weight_distribution(weights: pd.Series, nbins: int):
    """Plot the distribution of the weights"""
    weights_df = pd.DataFrame({'Weight': weights})
    fig = px.histogram(weights_df, x="Weight", nbins=nbins, title="Weight Distribution")
    st.plotly_chart(fig, use_container_width=True)

def plot_convergence(convergence: list):
    """Plot convergence of the algorithm"""
    iterations = [x[0] for x in convergence]
    rel_diffs = [x[1] for x in convergence]
    fig = px.line(x=iterations, y=rel_diffs, labels={'x':'Iteration', 'y':'Max Relative Diff'}, title="Convergence Plot")
    st.plotly_chart(fig, use_container_width=True)

def export_data(df: pd.DataFrame):
    """Provide an option to download the weighted data"""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Weighted Data", data=csv, file_name="weighted_data.csv", mime="text/csv")

def show_results(df: pd.DataFrame, weights: pd.Series, convergence: list, hist_bins: int, compute_variance: bool):
    """Visualization and analysis of results"""
    st.header("Results Analysis")
    
    # Use tabs to organize results.
    tab1, tab2, tab3 = st.tabs(["Weight Distribution", "Convergence", "Effective Sample Size"])
    
    with tab1:
        plot_weight_distribution(weights, hist_bins)
    with tab2:
        plot_convergence(convergence)
    with tab3:
        ess = effective_sample_size(weights)
        st.metric("Effective Sample Size", f"{ess:.1f}", help="Measures weighting efficiency (higher is better)")
    
    if compute_variance:
        st.subheader("Weight Variance")
        st.write(f"Variance of final weights: {np.var(weights):.4f}")
    
    st.subheader("Distribution Comparison")
    # For each categorical variable, compare pre/post distributions.
    cat_columns = df.select_dtypes(['category']).columns.tolist()
    for col in cat_columns:
        st.markdown(f"**{col}**")
        # Assume original weights are uniform (i.e. ones).
        plot_distribution_comparison(df, col, np.ones(len(df)), weights)
    
    export_data(df.assign(weight=weights))

# ---------------------- Execution ----------------------
if __name__ == "__main__":
    main()