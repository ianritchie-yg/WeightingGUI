from typing import Any, Dict, Tuple, Callable, Union, Optional
import pandas as pd
import numpy as np
import streamlit as st
import logging
import plotly.express as px
from functools import partial
from io import StringIO
from scipy.stats import chisquare, ks_2samp
import importlib


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

def plot_distribution_comparison(df: pd.DataFrame, col: str,
                                 original_weights: np.ndarray,
                                 adjusted_weights: np.ndarray):
    """Plot pre/post distribution comparison using Plotly"""
    # Prepare two datasets: one with original weights and one with adjusted weights.
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
    
    # Optionally, return the figure for further testing or manipulation
    return fig

# Add new trimming functions
def trim_weights(weights: pd.Series, pct: float = 0.02) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Trim extreme weights and return trimming statistics.
    
    Args:
        weights: Weight series to trim
        pct: Percentage of weights to trim (e.g., 0.02 for top 2%)
    
    Returns:
        Tuple of (trimmed weights, trimming statistics)
    """
    if pct <= 0:
        return weights, {'trimmed_count': 0, 'trim_threshold': float('inf')}
    
    upper_bound = weights.quantile(1 - pct)
    trimmed_weights = weights.clip(upper=upper_bound)

    # Re-normalize trimmed weights to preserve weighted sum
    trimmed_weights *= weights.sum() / trimmed_weights.sum()
    
    stats = {
        'trimmed_count': (weights > upper_bound).sum(),
        'trim_threshold': upper_bound,
        'max_weight_pre_trim': weights.max(),
        'max_weight_post_trim': trimmed_weights.max(),
    }
    
    return trimmed_weights, stats

def validate_targets(df: pd.DataFrame, targets: Dict):
    """Validate that all target cells are present in the data"""
    if not targets:
        raise ValueError("No targets provided")
        
    for grouping, target_dict in targets.items():
        # Convert single string to tuple if necessary
        if isinstance(grouping, str):
            grouping = (grouping,)
            
        # Check if all columns in grouping exist in dataframe
        missing_cols = [col for col in grouping if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in dataframe")
            
        # For single column grouping
        if len(grouping) == 1:
            unique_values = set(df[grouping[0]].dropna().unique())
        # For multiple column grouping
        else:
            unique_values = set(tuple(row) for row in df[list(grouping)].dropna().itertuples(index=False))
            
        # Check if all target cells exist in data
        for cell in target_dict:
            if isinstance(cell, tuple):
                if cell not in unique_values:
                    raise ValueError(f"Target cell {cell} not present in data for grouping {grouping}")
            else:
                if cell not in unique_values:
                    raise ValueError(f"Target cell {cell} not present in data for grouping {grouping[0]}")

def impute_data(df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
    """Impute missing data in the DataFrame"""
    if method == 'mean':
        return df.fillna(df.mean())
    elif method == 'median':
        return df.fillna(df.median())
    elif method == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError(f"Unknown imputation method: {method}")

def load_plugin(plugin_name: str):
    """Load a custom plugin"""
    try:
        plugin = importlib.import_module(plugin_name)
        return plugin
    except ImportError as e:
        st.error(f"Error loading plugin {plugin_name}: {e}")
        return None

def apply_custom_plugin(weights: pd.Series, plugin_name: str) -> pd.Series:
    """Apply a custom plugin to adjust weights"""
    plugin = load_plugin(plugin_name)
    if plugin and hasattr(plugin, 'custom_adjust_weights'):
        return plugin.custom_adjust_weights(weights)
    return weights

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
                 compute_variance: bool = False,
                 trim_percentage: float = 0.0  # Add trimming parameter
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
        self.trim_percentage = trim_percentage
        self.trimming_stats = None
        
        np.random.seed(self.random_seed)
        self.group_indices = precompute_group_indices(df, list(self.targets.keys()))
        self._validate_targets()
        self.convergence_data = []
        self.hierarchical_levels = list(self._normalize_targets(targets).keys())

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
        """Enhanced iterative weighting algorithm with dual convergence criteria"""
        weights = pd.Series(np.ones(len(self.df)), index=self.df.index)
        prev_weights = weights.copy()
        convergence_data = []
        
        for it in range(max_iter):
            if progress_callback:
                progress_callback(it + 1, max_iter)
            
            # Store previous weights for stability check
            prev_weights = weights.copy()
            
            max_rel_diff, worst_group = self._evaluate_groups(weights)
            weights = self._adjust_weights(weights, worst_group)
            
            # Calculate both convergence metrics
            weight_stability = calculate_weight_stability(weights, prev_weights)
            weight_stability_threshold = np.maximum(0.0005, threshold * weights.std())  # Dynamic threshold
            
            # Store convergence metrics
            convergence_data.append({
                'iteration': it + 1,
                'max_rel_diff': max_rel_diff,
                'weight_stability': weight_stability
            })
            
            if self.verbose and ((it + 1) % self.reporting_frequency == 0):
                logging.info(
                    f"Iteration {it+1}: max diff = {max_rel_diff:.4f}, "
                    f"weight stability = {weight_stability:.6f}"
                )
            
            # Dual convergence criteria
            if it > 10:  # Allow some initial iterations
                target_met = max_rel_diff <= threshold
                weights_stable = weight_stability <= weight_stability_threshold
                
                if target_met and weights_stable:
                    if self.verbose:
                        logging.info("Convergence achieved: both targets and weights are stable")
                    break
            
            if it == max_iter - 1 and self.verbose:
                logging.warning("Maximum iterations reached without convergence")
        
        self.convergence_data = convergence_data
        
        if self.compute_variance:
            self.variance = weights.var()
        
        # Apply trimming before returning if enabled
        if self.trim_percentage > 0:
            weights, self.trimming_stats = trim_weights(weights, self.trim_percentage)
            
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

# Add new convergence helper
def calculate_weight_stability(weights: pd.Series, prev_weights: pd.Series) -> float:
    """Calculate the mean absolute change in weights between iterations"""
    return np.abs(weights - prev_weights).mean()

def calculate_design_effect(weights: pd.Series, subgroup_col: Optional[str] = None) -> float:
    """Calculate design effect, optionally stratified by a subgroup"""
    if subgroup_col and subgroup_col in weights.index:
        deffs = weights.groupby(subgroup_col).apply(lambda w: 1 + (w.std() / w.mean())**2)
        return np.average(deffs, weights=weights.groupby(subgroup_col).count())  # Weighted average DEFF
    return 1 + (weights.std() / weights.mean())**2

def check_target_accuracy(df: pd.DataFrame, weights: pd.Series, targets: Dict) -> Tuple[pd.DataFrame, float, float]:
    """Compare achieved vs target distributions and perform chi-square test"""
    results = []
    chi2_values = []

    for grouping, target_dict in targets.items():
        achieved = df.groupby(list(grouping))['weight'].sum()
        total_weighted = achieved.sum()

        for category, target in target_dict.items():
            achieved_pct = (achieved.get(category, 0) / total_weighted) * 100
            diff = achieved_pct - target
            results.append({
                "Category": category,
                "Target %": target,
                "Achieved %": achieved_pct,
                "Difference": diff
            })
            chi2_values.append((achieved.get(category, 0), target / 100 * total_weighted))

    # Perform Chi-Square Goodness of Fit Test
    observed, expected = zip(*chi2_values)
    chi2_stat, p_value = chisquare(observed, expected)
    
    return pd.DataFrame(results), chi2_stat, p_value

def compute_goodness_of_fit(observed: np.ndarray, expected: np.ndarray):
    """Compute goodness-of-fit metrics"""
    chi2_stat, chi2_p = chisquare(observed, expected)
    ks_stat, ks_p = ks_2samp(observed, expected)
    return {
        'chi2_stat': chi2_stat,
        'chi2_p': chi2_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p
    }

def check_weight_instability(weights: pd.Series) -> bool:
    """Check for weight instability based on coefficient of variation"""
    cv = weights.std() / weights.mean()
    threshold = 0.5  # Define an appropriate threshold for instability
    return cv > threshold

def analyze_weight_distribution(weights: pd.Series) -> Dict[str, float]:
    """Analyze weight distribution statistics"""
    return {
        'mean': weights.mean(),
        'median': weights.median(),
        'std': weights.std(),
        'min': weights.min(),
        'max': weights.max(),
        'skewness': weights.skew(),
        'kurtosis': weights.kurtosis(),
        'cv': weights.std() / weights.mean()
    }

def plot_weight_cdf(weights: pd.Series):
    """Plot the cumulative distribution of weights"""
    weights_sorted = np.sort(weights)
    cdf = np.arange(1, len(weights)+1) / len(weights)
    plt.figure()
    plt.step(weights_sorted, cdf, where='post')
    plt.title('Cumulative Distribution of Weights')
    plt.xlabel('Weight')
    plt.ylabel('CDF')
    st.pyplot(plt.gcf())

# Modify show_results to use enhanced convergence plotting
def show_results(df: pd.DataFrame, weights: pd.Series, convergence: list, hist_bins: int, compute_variance: bool, trimming_stats: Optional[Dict] = None):
    """Enhanced visualization and analysis of results"""
    st.header("Results Analysis")
    
    # Create tabs for different analysis aspects
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Weight Distribution", 
        "Convergence", 
        "Sample Efficiency",
        "Target Accuracy",
        "Detailed Diagnostics",
        "Additional Visuals"
    ])
    with tab1:
        plot_weight_distribution(weights, hist_bins)
        
    with tab2:
        st.write("### Convergence Diagnostics")
        convergence_df = pd.DataFrame(convergence)
        
        # Plot both metrics
        fig = px.line(convergence_df, x='iteration',
                     y=['max_rel_diff', 'weight_stability'],
                     title='Convergence Metrics',
                     labels={
                         'max_rel_diff': 'Target Difference',
                         'weight_stability': 'Weight Change'
                     })
        st.plotly_chart(fig)
        
        # Add convergence summary
        final_metrics = convergence_df.iloc[-1]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Target Difference", 
                     f"{final_metrics['max_rel_diff']:.6f}")
        with col2:
            st.metric("Final Weight Stability",
                     f"{final_metrics['weight_stability']:.6f}")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            ess = effective_sample_size(weights)
            st.metric("Effective Sample Size", f"{ess:.1f}", 
                     help="Measures weighting efficiency (higher is better)")
        with col2:
            deff = calculate_design_effect(weights)
            st.metric("Design Effect", f"{deff:.2f}", 
                     help="Inflation in variance due to weighting (closer to 1 is better)")
    
    with tab4:
        st.write("### Target Accuracy")
        accuracy_df, chi2_stat, p_value = check_target_accuracy(df, weights, targets)
        st.dataframe(accuracy_df)
        
        # Visualize largest discrepancies
        fig = px.bar(
            accuracy_df.nlargest(5, 'Difference').abs(),
            x='Category',
            y='Difference',
            title='Largest Target vs. Achieved Differences'
        )
        st.plotly_chart(fig)
        
        # Display chi-square test result
        st.metric("Chi-Square Goodness of Fit", f"p = {p_value:.4f}", help="p < 0.05 suggests a mismatch between achieved and target.")
    
    with tab5:
        # Detailed weight statistics
        stats = analyze_weight_distribution(weights)
        st.write("### Weight Distribution Statistics")
        stats_df = pd.DataFrame(stats.items(), columns=['Metric', 'Value'])
        st.table(stats_df)
        
        # Add trimming statistics if available
        if trimming_stats:
            st.write("### Weight Trimming Summary")
            st.write(f"- Number of weights trimmed: {trimming_stats['trimmed_count']}")
            st.write(f"- Trim threshold: {trimming_stats['trim_threshold']:.3f}")
            st.write(f"- Maximum weight before trimming: {trimming_stats['max_weight_pre_trim']:.3f}")
            st.write(f"- Maximum weight after trimming: {trimming_stats['max_weight_post_trim']:.3f}")
        
        # Weight correlation analysis
        if len(df.select_dtypes(['number']).columns) > 0:
            st.write("### Weight Correlations with Numeric Variables")
            corr = df.select_dtypes(['number']).corrwith(weights)
            fig = px.bar(
                x=corr.index,
                y=corr.values,
                title='Weight Correlations with Numeric Variables'
            )
            st.plotly_chart(fig)
    
    with tab6:
        plot_weight_cdf(weights)
    
    if compute_variance:
        st.subheader("Variance Estimates")
        st.write(f"Variance of weights: {np.var(weights):.4f}")
        st.write(f"Coefficient of variation: {stats['cv']:.4f}")
    
    # Export option
    export_data(df.assign(weight=weights))

def export_table_to_excel(df: pd.DataFrame, filename: str = "diagnostics.xlsx"):
    """Export tables to Excel"""
    df.to_excel(filename, index=False)
    st.download_button("Download Diagnostics Excel", data=open(filename, 'rb').read(), file_name=filename, mime='application/vnd.ms-excel')

# ---------------------- Streamlit Interface ----------------------
# Modify main() to accept a config parameter
def main(config: Optional[Dict[str, Any]] = None):
    # ...existing code...
    
    # Add debug prints
    st.write("DataFrame columns:", df.columns.tolist())
    st.write("Target groupings:", list(targets.keys()))
    
    # Then validate
    try:
        validate_targets(df, targets)
    except ValueError as ve:
        st.error(f"Target validation error: {ve}")
        return
    
    # ...existing code...

def load_and_preprocess(uploaded_file) -> pd.DataFrame:
    """Handle data loading and preprocessing"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension == 'sav':
        import pyreadstat
        df, meta = pyreadstat.read_sav(uploaded_file)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(uploaded_file)
    elif file_extension == 'txt':
        df = pd.read_csv(uploaded_file, delimiter='\t')
    else:
        st.error("Unsupported file format")
        return pd.DataFrame()
    
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
    """Interactive target configuration with percentage support"""
    st.sidebar.header("Target Configuration")
    
    # Add input mode selection
    input_mode = st.sidebar.radio(
        "Target Input Mode",
        ["Counts", "Percentages"],
        help="Choose whether to input target values as counts or percentages"
    )
    
    # Allow users to select or enter column headers
    st.sidebar.markdown("### Select or Enter Column Headers")
    columns = list(df.columns)
    selected_groupings = st.sidebar.multiselect("Select grouping columns for targets", options=columns)
    
    if not selected_groupings:
        st.sidebar.info("No grouping columns selected. Using default sample targets.")
        return {
            'gender': {'Male': 48.0, 'Female': 52.0} if input_mode == "Percentages" else {'Male': 480, 'Female': 520},
            ('age_group',): {'18-24': 25.0, '25-34': 30.0, '35-44': 30.0, '45+': 15.0} if input_mode == "Percentages" 
                           else {'18-24': 250, '25-34': 300, '35-44': 300, '45+': 150}
        }
    
    targets = {}
    total_sample = len(df)
    
    for col in selected_groupings:
        st.sidebar.markdown(f"#### Targets for {col}")
        unique_vals = sorted(df[col].dropna().unique())
        target_dict = {}
        running_total = 0
        
        for val in unique_vals:
            if input_mode == "Percentages":
                default_target = 100.0 / len(unique_vals)
                target_val = st.sidebar.number_input(
                    f"Target % for {val}", 
                    min_value=0.0, 
                    max_value=100.0,
                    value=default_target,
                    step=0.1,
                    key=f"{col}_{val}_pct"
                )
                target_dict[val] = target_val
                running_total += target_val
            else:
                target_val = st.sidebar.number_input(
                    f"Target count for {val}", 
                    min_value=0, 
                    value=int(total_sample / len(unique_vals)),
                    step=1,
                    key=f"{col}_{val}_count"
                )
                target_dict[val] = target_val
                running_total += target_val
        
        if input_mode == "Percentages" and abs(running_total - 100.0) > 0.01:
            st.warning(f"⚠️ Percentages for {col} sum to {running_total:.1f}%. They should sum to 100%.")
        
        targets[col] = target_dict
    
    return targets

def configure_interactive_targets(df: pd.DataFrame, use_percentages: bool = False) -> Dict:
    """Interactive target configuration UI with percentage support"""
    st.sidebar.markdown("### Interactive Target Setup")
    targets = {}
    columns = list(df.columns)
    selected_groupings = st.sidebar.multiselect("Select grouping columns for targets", options=columns)
    total_sample = len(df)
    
    if not selected_groupings:
        st.sidebar.info("No grouping columns selected. Using default sample targets.")
        return {
            'gender': {'Male': 48.0, 'Female': 52.0} if use_percentages else {'Male': 480, 'Female': 520},
            ('age_group',): {'18-24': 25.0, '25-34': 30.0, '35-44': 30.0, '45+': 15.0} if use_percentages 
                           else {'18-24': 250, '25-34': 300, '35-44': 300, '45+': 150}
        }
    
    for col in selected_groupings:
        st.sidebar.markdown(f"#### Targets for {col}")
        unique_vals = sorted(df[col].dropna().unique())
        target_dict = {}
        running_total = 0
        
        for val in unique_vals:
            if use_percentages:
                default_target = 100.0 / len(unique_vals)
                target_val = st.sidebar.number_input(
                    f"Target % for {val}", 
                    min_value=0.0, 
                    max_value=100.0,
                    value=default_target,
                    step=0.1,
                    key=f"{col}_{val}_pct"
                )
                target_dict[val] = target_val
                running_total += target_val
            else:
                target_val = st.sidebar.number_input(
                    f"Target count for {val}", 
                    min_value=0, 
                    value=int(total_sample / len(unique_vals)),
                    step=1,
                    key=f"{col}_{val}_count"
                )
                target_dict[val] = target_val
                running_total += target_val
        
        if use_percentages and abs(running_total - 100.0) > 0.01:
            st.warning(f"⚠️ Percentages for {col} sum to {running_total:.1f}%. They should sum to 100%.")
        
        targets[col] = target_dict
    
    return targets

def parse_target_csv(uploaded_file, use_percentages: bool = False) -> Dict:
    """Parse CSV target configuration with percentage support"""
    try:
        target_df = pd.read_csv(uploaded_file)
        if 'is_percentage' not in target_df.columns:
            target_df['is_percentage'] = use_percentages
    except Exception as e:
        st.error(f"Error parsing target CSV: {e}")
        return {}
    
    targets = {}
    for grouping in target_df['grouping'].unique():
        group_data = target_df[target_df['grouping'] == grouping]
        total = group_data['value'].sum()
        
        if use_percentages and abs(total - 100.0) > 0.01:
            st.warning(f"⚠️ Percentages for {grouping} sum to {total:.1f}%. They should sum to 100%.")
    
    # Continue with existing CSV parsing logic
    # ...existing parsing code...
    
    return targets

# Add trimming parameter to get_weighting_params
def get_weighting_params() -> Dict:
    """Collect weighting algorithm parameters from user"""
    st.sidebar.header("Algorithm Parameters")
    
    # Convergence Parameters
    threshold = st.sidebar.slider(
        "Convergence Threshold",
        min_value=0.0001, max_value=0.01, value=0.001, step=0.0001,
        help="Lower values require closer matching to targets."
    )
    
    max_iter = st.sidebar.number_input(
        "Maximum Iterations", min_value=1, value=50, step=1,
        help="Maximum number of iterations before stopping."
    )
    
    # Weight Bounds
    min_weight = st.sidebar.number_input(
        "Minimum Weight", min_value=0.0, value=0.1, step=0.1,
        help="Weights will not go below this value."
    )
    
    max_weight = st.sidebar.number_input(
        "Maximum Weight", min_value=0.0, value=5.0, step=0.1,
        help="Weights will not exceed this value."
    )
    
    # Adjustment Parameters
    max_adj_factor = st.sidebar.number_input(
        "Maximum Adjustment Factor", min_value=1.0, value=2.0, step=0.1,
        help="Cap on the adjustment factor applied per iteration."
    )
    
    smoothing_factor = st.sidebar.slider(
        "Smoothing/Dampening Factor", min_value=0.0, max_value=1.0, value=1.0, step=0.1,
        help="Factor to dampen adjustments (0 = no adjustment, 1 = full adjustment)."
    )
    
    # Strategy Parameters
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
        "Weighting Method", 
        options=["Raking", "Geometric Mean", "Entropy"],
        index=0,
        help="Choice of weighting algorithm (Raking = linear, Geometric = sqrt, Entropy = log)"
    )
    
    compute_variance = st.sidebar.checkbox(
        "Compute Variance Estimates", 
        value=False,
        help="Compute variance of final weights."
    )
    
    # Trimming Parameters
    enable_trimming = st.sidebar.checkbox(
        "Enable Weight Trimming",
        value=False,
        help="Trim extreme weights to improve variance efficiency"
    )
    
    trim_percentage = 0.0
    if enable_trimming:
        trim_percentage = st.sidebar.slider(
            "Trimming Percentage",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Percentage of weights to trim from the top (e.g., 2.0 = top 2%)"
        ) / 100.0
    
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
        'compute_variance': compute_variance,
        'trim_percentage': trim_percentage,
    }

def plot_weight_distribution(weights: pd.Series, nbins: int):
    """Plot the distribution of the weights"""
    weights_df = pd.DataFrame({'Weight': weights})
    fig = px.histogram(weights_df, x="Weight", nbins=nbins, title="Weight Distribution")
    st.plotly_chart(fig, use_container_width=True)
    """Plot convergence of the algorithm"""

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

# ---------------------- Execution ----------------------
if __name__ == "__main__":
    config = get_weighting_params()
    main(config)
