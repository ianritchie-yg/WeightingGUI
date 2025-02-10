from typing import Any, Dict, Tuple, Callable, Union
import pandas as pd
import numpy as np
import streamlit as st
import logging
import plotly.express as px
import plotly.graph_objects as go
from functools import partial
from io import StringIO
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV

# ---------------------- Configuration ----------------------
st.set_page_config(page_title="Survey Weighting Suite", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ---------------------- Helper Functions ----------------------
@st.cache_data
def standardize_cell_key(cell: Any, grouping: Tuple[Any, ...]) -> Union[Any, Tuple[Any, ...]]:
    """Standardize cell key format based on grouping structure."""
    if not isinstance(grouping, tuple) or not grouping:
        raise ValueError("Invalid grouping format")
    return cell if (len(grouping) == 1 or isinstance(cell, tuple)) else (cell,)

def handle_missing_values(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """Handle missing values in categorical columns."""
    df_clean = df.copy()
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode().iloc[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            logging.warning(f"Imputed missing values in {col} with {mode_val}")
    return df_clean

def trim_weights(weights: pd.Series, lower_percentile: float = 5, upper_percentile: float = 95) -> pd.Series:
    """Trim extreme weights based on percentiles."""
    lower_bound = np.percentile(weights, lower_percentile)
    upper_bound = np.percentile(weights, upper_percentile)
    return weights.clip(lower_bound, upper_bound)

def plot_distribution_comparison(df: pd.DataFrame, col: str, 
                                 original_weights: np.ndarray, 
                                 adjusted_weights: np.ndarray,
                                 nbins: int = 20):
    """Plot pre/post distribution comparison for a categorical variable."""
    df_orig = df.assign(Weight=original_weights, Dataset='Original')
    df_adj = df.assign(Weight=adjusted_weights, Dataset='Adjusted')
    combined = pd.concat([df_orig, df_adj])
    
    fig = px.histogram(
        combined, x=col, color='Dataset', barmode='group',
        histnorm='percent', title=f'Distribution Comparison: {col}',
        labels={col: col}, nbins=nbins
    )
    st.plotly_chart(fig, use_container_width=True)

def bootstrap_variance_estimation(df: pd.DataFrame, weighting_func: Callable, n_bootstrap: int = 100) -> float:
    """
    Estimate the variance of the weights (or a downstream statistic) using bootstrap.
    """
    estimates = []
    for i in range(n_bootstrap):
        sample = df.sample(frac=1, replace=True)
        weights = weighting_func(sample)
        estimates.append(np.var(weights))
    return np.mean(estimates)

def calculate_asmd(df: pd.DataFrame, weights: pd.Series, var: str) -> float:
    """
    Calculate the Absolute Standardized Mean Difference (ASMD) for a continuous variable.
    """
    unweighted_mean = df[var].mean()
    unweighted_std = df[var].std()
    weighted_mean = np.average(df[var], weights=weights)
    return abs(weighted_mean - unweighted_mean) / unweighted_std

def calculate_ks_statistic(df: pd.DataFrame, weights: pd.Series, var: str) -> Dict[str, float]:
    """
    Calculate the KS statistic for a given variable before and after weighting.
    """
    from scipy.stats import ks_2samp
    raw_values = df[var].dropna().values
    # Create a weighted sample by repeating each value proportional to its weight
    # (Here we multiply weights by 100 and cast to int for simplicity.)
    weighted_sample = np.repeat(df[var].values, (weights * 100).astype(int))
    stat, p_value = ks_2samp(raw_values, weighted_sample)
    return {"statistic": stat, "p_value": p_value}

def check_propensity_overlap(propensity_scores: np.ndarray, threshold: float = 0.05) -> float:
    """
    Check overlap by computing the proportion of propensity scores near 0 or 1.
    Returns the combined proportion of scores below threshold or above 1 - threshold.
    """
    low_overlap = np.mean(propensity_scores < threshold)
    high_overlap = np.mean(propensity_scores > (1 - threshold))
    return low_overlap + high_overlap

def calibration_residuals(X: np.ndarray, weights: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
    """
    Compute residuals between the weighted moments of X and the target vector.
    """
    current = X.T.dot(weights)
    return current - target_vec

def taylor_series_variance(weights: pd.Series) -> float:
    """
    Estimate variance using a simple Taylor series approximation.
    """
    n = len(weights)
    return np.sum(weights**2) / (np.sum(weights)**2)

# ---------------------- Weighting Engine ----------------------
class WeightingEngine:
    def __init__(self, df: pd.DataFrame, 
                 min_weight: float = 0.1,
                 max_weight: float = 5.0,
                 max_adj_factor: float = 2.0,
                 smoothing_factor: float = 1.0,
                 zero_cell_strategy: str = "default",
                 convergence_metric: str = "max_rel_diff",
                 verbose: bool = False,
                 random_seed: int = 42):
        self.df = df.copy()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_adj_factor = max_adj_factor
        self.smoothing_factor = smoothing_factor
        self.zero_cell_strategy = zero_cell_strategy
        self.convergence_metric = convergence_metric
        self.verbose = verbose
        self.random_seed = random_seed
        self.convergence_data = []
        np.random.seed(self.random_seed)

    def _calculate_observed(self, weights: pd.Series, grouping: Tuple) -> Dict:
        """Calculate observed weighted counts for a grouping."""
        return weights.groupby([self.df[col] for col in grouping]).sum().to_dict()

    def post_stratification(self, targets: Dict, max_iter: int = 50, threshold: float = 0.001) -> pd.Series:
        """Iterative post-stratification with convergence tracking."""
        weights = pd.Series(np.ones(len(self.df)), index=self.df.index)
        group_indices = {g: self.df.groupby(list(g)).indices for g in targets.keys()}
        for iteration in range(max_iter):
            max_diff = 0.0
            worst_group = None
            for grouping, tdict in targets.items():
                for cell, target in tdict.items():
                    standardized_cell = standardize_cell_key(cell, grouping)
                    indices = group_indices[grouping].get(standardized_cell, [])
                    observed = weights.iloc[indices].sum()
                    if observed <= 0:
                        if self.zero_cell_strategy == "smooth":
                            observed = 1e-6
                        elif self.zero_cell_strategy == "ignore":
                            continue
                        else:
                            raise ValueError(f"Zero observed in cell {cell}")
                    ratio = target / observed
                    adj_factor = 1 + self.smoothing_factor * (ratio - 1)
                    adj_factor = np.clip(adj_factor, 1/self.max_adj_factor, self.max_adj_factor)
                    weights.iloc[indices] *= adj_factor
                    rel_diff = abs(observed - target) / target
                    if rel_diff > max_diff:
                        max_diff = rel_diff
                        worst_group = grouping
            self.convergence_data.append((iteration+1, max_diff))
            if self.verbose:
                logging.info(f"Post-Stratification Iteration {iteration+1}: max relative diff = {max_diff:.6f}")
            if max_diff <= threshold:
                break
        return weights.clip(self.min_weight, self.max_weight)

    def inverse_probability_weighting(self, probabilities: pd.Series) -> pd.Series:
        """Inverse probability weighting with validation."""
        if (probabilities <= 0).any() or (probabilities > 1).any():
            raise ValueError("Probabilities must be between 0 and 1")
        return (1 / probabilities).clip(self.min_weight, self.max_weight)

    def propensity_score_weighting(self, outcome: str, predictors: list) -> pd.Series:
        """Propensity score weighting using calibrated classifier."""
        X = self.df[predictors]
        y = self.df[outcome]
        cat_cols = X.select_dtypes(include=['category', 'object']).columns
        if not cat_cols.empty:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        base_model = LogisticRegression(max_iter=1000)
        model = CalibratedClassifierCV(base_model, cv=5)
        model.fit(X, y)
        propensity_scores = model.predict_proba(X)[:, 1]
        return (1 / propensity_scores).clip(self.min_weight, self.max_weight)

    def calibration_weighting(self, targets: Dict, predictors: list, max_iter: int = 100, reg_const: float = 0.01) -> pd.Series:
        """Calibration weighting using optimization."""
        X = pd.get_dummies(self.df[predictors], drop_first=True)
        X = X.values  # convert to numpy array
        sorted_keys = sorted(targets.keys())
        target_vec = np.array([targets[k] for k in sorted_keys])
        def loss(weights):
            current = X.T.dot(weights)
            return np.sum((current - target_vec)**2) + reg_const * np.sum(weights**2)
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - len(w)}
        bounds = [(self.min_weight, self.max_weight)] * len(self.df)
        result = minimize(loss, x0=np.ones(len(self.df)),
                          method='SLSQP', bounds=bounds,
                          constraints=cons, options={'maxiter': max_iter})
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        return pd.Series(result.x, index=self.df.index).clip(self.min_weight, self.max_weight)

    def rake_weighting(self, margin_targets: Dict[str, Dict[Any, float]], max_iter: int = 50, threshold: float = 0.001) -> pd.Series:
        """
        Perform raking (rim weighting) to adjust weights so that the marginal distributions match target values.
        """
        weights = pd.Series(np.ones(len(self.df)), index=self.df.index)
        for iteration in range(max_iter):
            max_diff = 0.0
            for var, target_dict in margin_targets.items():
                groups = self.df.groupby(var).groups
                for category, indices in groups.items():
                    if category not in target_dict:
                        continue
                    target = target_dict[category]
                    observed = weights.loc[indices].sum()
                    if observed <= 0:
                        if self.zero_cell_strategy == "smooth":
                            observed = 1e-6
                        elif self.zero_cell_strategy == "ignore":
                            continue
                        else:
                            raise ValueError(f"Zero observed for category '{category}' in variable '{var}'")
                    ratio = target / observed
                    adj_factor = 1 + self.smoothing_factor * (ratio - 1)
                    adj_factor = np.clip(adj_factor, 1 / self.max_adj_factor, self.max_adj_factor)
                    weights.loc[indices] *= adj_factor
                    rel_diff = abs(observed - target) / target if target > 0 else 0
                    max_diff = max(max_diff, rel_diff)
            self.convergence_data.append((iteration + 1, max_diff))
            if self.verbose:
                logging.info(f"Raking Iteration {iteration+1}: max relative diff = {max_diff:.6f}")
            if max_diff <= threshold:
                break
        return weights.clip(self.min_weight, self.max_weight)

    def mrp_weighting(self, outcome: str, group_vars: list, predictors: list) -> pd.Series:
        """
        Stub for Multilevel Regression Post-Stratification (MRP) weighting.
        This method should fit a multilevel model and post-stratify the estimates.
        """
        raise NotImplementedError("MRP weighting has not been implemented yet.")

    def entropy_balancing(self, targets: Dict, predictors: list, max_iter: int = 100) -> pd.Series:
        """
        Implement entropy balancing to match weighted moments to target values using convex optimization.
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("Please install cvxpy to use entropy balancing.")
        X = pd.get_dummies(self.df[predictors], drop_first=True).values
        n = len(self.df)
        w = cp.Variable(n)
        epsilon = 1e-8
        objective = cp.Minimize(cp.sum(cp.entr(w + epsilon)))
        sorted_keys = sorted(targets.keys())
        target_vec = np.array([targets[k] for k in sorted_keys])
        constraints = [cp.sum(w) == n, X.T @ w == target_vec]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, max_iters=max_iter)
        if w.value is None:
            raise RuntimeError("Entropy balancing failed to converge.")
        weights = pd.Series(w.value, index=self.df.index)
        return weights.clip(self.min_weight, self.max_weight)

    def run(self, method: str, **kwargs) -> pd.Series:
        """Execute selected weighting method."""
        method_map = {
            'post_stratification': self.post_stratification,
            'inverse_probability_weighting': self.inverse_probability_weighting,
            'propensity_score_weighting': self.propensity_score_weighting,
            'calibration_weighting': self.calibration_weighting,
            'rake_weighting': self.rake_weighting
        }
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}")
        return method_map[method](**kwargs)

# ---------------------- Streamlit Interface ----------------------
def configure_targets_for_raking(df: pd.DataFrame) -> Dict[str, Dict[Any, float]]:
    """Interactive target configuration for raking."""
    targets = {}
    categorical_cols = list(df.select_dtypes('category').columns)
    for col in categorical_cols:
        st.subheader(f"Targets for {col}")
        unique_vals = df[col].unique()
        targets[col] = {
            val: st.number_input(f"{col} - {val} target", min_value=0, value=100, key=f"{col}_{val}_rake")
            for val in unique_vals
        }
    return targets

def handle_data_upload() -> pd.DataFrame:
    """Handle data upload and preprocessing."""
    uploaded_file = st.sidebar.file_uploader("Upload survey data", type=["csv", "xlsx", "sav"])
    if not uploaded_file:
        st.info("Please upload a file (CSV, XLSX, or SAV) to begin")
        return None
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    elif filename.endswith(".sav"):
        try:
            import pyreadstat
            df, meta = pyreadstat.read_sav(uploaded_file)
        except ImportError:
            st.error("pyreadstat is required to read SAV files. Please install it via pip.")
            return None
    else:
        st.error("Unsupported file type.")
        return None
    cat_cols = st.sidebar.multiselect("Select categorical columns", df.columns)
    if cat_cols:
        df = handle_missing_values(df, cat_cols)
        df[cat_cols] = df[cat_cols].astype('category')
    return df

def parse_target_csv(uploaded_file) -> Dict:
    """Parse CSV target configuration."""
    try:
        target_df = pd.read_csv(uploaded_file)
        return {
            tuple(row['grouping'].split(',')): {row['cell']: row['value']}
            for _, row in target_df.iterrows()
        }
    except Exception as e:
        st.error(f"Error parsing targets CSV: {str(e)}")
        return {}

def configure_interactive_targets(df: pd.DataFrame) -> Dict:
    """Interactive target configuration UI."""
    targets = {}
    columns = list(df.select_dtypes('category').columns)
    for col in columns:
        st.subheader(f"Targets for {col}")
        unique_vals = df[col].unique()
        targets[(col,)] = {
            val: st.number_input(f"{val} target", min_value=0, value=100, key=f"{col}_{val}")
            for val in unique_vals
        }
    return targets

def configure_targets(df: pd.DataFrame) -> Dict:
    """Configure weighting targets through UI."""
    with st.sidebar.expander("Target Configuration"):
        if st.checkbox("Upload targets CSV"):
            target_file = st.file_uploader("Targets file", type=["csv"])
            if target_file:
                return parse_target_csv(target_file)
        return configure_interactive_targets(df)

def get_weighting_params(method: str, df: pd.DataFrame) -> Dict:
    """Collect parameters based on selected method."""
    params = {'method': method}
    st.sidebar.header("Algorithm Parameters")
    # Common parameters
    params.update({
        'min_weight': st.sidebar.number_input("Minimum weight", 0.01, 10.0, 0.1),
        'max_weight': st.sidebar.number_input("Maximum weight", 0.1, 20.0, 5.0),
        'max_adj_factor': st.sidebar.slider("Max adjustment factor", 1.0, 5.0, 2.0),
        'smoothing_factor': st.sidebar.slider("Smoothing factor", 0.0, 1.0, 0.8),
        'zero_cell_strategy': st.sidebar.selectbox("Zero-cell strategy", ["default", "smooth", "ignore"])
    })
    # Method-specific parameters
    if method == 'post_stratification':
        params['targets'] = configure_targets(df)
        params['max_iter'] = st.sidebar.number_input("Max iterations", 1, 200, 50)
        params['threshold'] = st.sidebar.number_input("Convergence threshold", 0.0001, 0.1, 0.001, format="%.4f")
    elif method == 'inverse_probability_weighting':
        prob_file = st.sidebar.file_uploader("Upload probabilities CSV", type=["csv"])
        if prob_file:
            prob_df = pd.read_csv(prob_file)
            params['probabilities'] = prob_df['probability']
        else:
            st.error("Please upload probability estimates")
    elif method == 'propensity_score_weighting':
        params['outcome'] = st.sidebar.selectbox("Outcome variable", df.columns)
        params['predictors'] = st.sidebar.multiselect("Predictor variables", df.columns)
    elif method == 'calibration_weighting':
        params['targets'] = configure_targets(df)
        params['predictors'] = st.sidebar.multiselect("Predictor variables", df.columns)
        params['max_iter'] = st.sidebar.number_input("Optimization iterations", 10, 1000, 100)
    elif method == 'rake_weighting':
        params['margin_targets'] = configure_targets_for_raking(df)
        params['max_iter'] = st.sidebar.number_input("Max iterations", 1, 200, 50)
        params['threshold'] = st.sidebar.number_input("Convergence threshold", 0.0001, 0.1, 0.001, format="%.4f")
    # Advanced options
    if st.sidebar.checkbox("Perform Assumption Checks"):
        params['assumption_checks'] = True
    if st.sidebar.checkbox("Estimate Variance (Bootstrap)"):
        params['bootstrap_variance'] = True
    if st.sidebar.checkbox("Estimate Variance (Taylor Series)"):
        params['taylor_variance'] = True
    if st.sidebar.checkbox("Apply Trimmed Weights Analysis"):
        params['trim_weights'] = True
    advanced_method = st.sidebar.selectbox("Advanced Weighting Method", ["None", "MRP", "Entropy Balancing"])
    params['advanced_method'] = advanced_method
    return params

def run_weighting_engine(df: pd.DataFrame, method: str, params: Dict) -> Tuple[pd.Series, list]:
    """Execute weighting process."""
    engine = WeightingEngine(
        df,
        min_weight=params['min_weight'],
        max_weight=params['max_weight'],
        max_adj_factor=params['max_adj_factor'],
        smoothing_factor=params['smoothing_factor'],
        zero_cell_strategy=params['zero_cell_strategy']
    )
    method_args = {
        'post_stratification': lambda: engine.run(
            method,
            targets=params['targets'],
            max_iter=params['max_iter'],
            threshold=params['threshold']
        ),
        'inverse_probability_weighting': lambda: engine.run(
            method,
            probabilities=params['probabilities']
        ),
        'propensity_score_weighting': lambda: engine.run(
            method,
            outcome=params['outcome'],
            predictors=params['predictors']
        ),
        'calibration_weighting': lambda: engine.run(
            method,
            targets=params['targets'],
            predictors=params['predictors'],
            max_iter=params['max_iter']
        ),
        'rake_weighting': lambda: engine.run(
            method,
            margin_targets=params['margin_targets'],
            max_iter=params['max_iter'],
            threshold=params['threshold']
        )
    }
    weights = method_args[method]()
    # Advanced method override, if selected
    if params.get('advanced_method', "None") != "None":
        if params['advanced_method'] == "Entropy Balancing":
            weights = engine.entropy_balancing(
                targets=params['targets'],
                predictors=params.get('predictors', []),
                max_iter=params.get('max_iter', 100)
            )
        elif params['advanced_method'] == "MRP":
            weights = engine.mrp_weighting(
                outcome=params.get('outcome'),
                group_vars=params.get('group_vars', []),
                predictors=params.get('predictors', [])
            )
    # Apply trimmed weights if selected
    if params.get('trim_weights', False):
        weights = trim_weights(weights)
    return weights, engine.convergence_data

def plot_weight_distribution(weights: pd.Series):
    """Visualize weight distribution."""
    fig = px.histogram(weights, nbins=50, 
                       title="Final Weight Distribution",
                       labels={'value': 'Weight'})
    st.plotly_chart(fig, use_container_width=True)

def plot_convergence(convergence: list, threshold: float):
    """Plot convergence metrics over iterations."""
    iterations, values = zip(*convergence)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=values, name='Max Relative Difference'))
    fig.add_hline(y=threshold, line_dash='dash', line_color='red',
                  annotation_text=f'Threshold: {threshold:.4f}')
    fig.update_layout(title='Convergence Progress',
                      xaxis_title='Iteration',
                      yaxis_title='Max Relative Difference')
    st.plotly_chart(fig, use_container_width=True)

def effective_sample_size(weights: pd.Series) -> float:
    """Calculate effective sample size with validation."""
    if (weights <= 0).any():
        raise ValueError("Weights must be positive")
    return (weights.sum() ** 2) / (weights ** 2).sum()

def show_results(df: pd.DataFrame, weights: pd.Series, convergence: list, params: Dict):
    """Display weighting results and diagnostics."""
    st.header("Weighting Results")
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Effective Sample Size", f"{effective_sample_size(weights):.1f}")
    with col2:
        st.metric("Weight Range", f"{weights.min():.2f} - {weights.max():.2f}")
    with col3:
        st.metric("Convergence Iterations", len(convergence) if convergence else "N/A")
    # Plots
    col1, col2 = st.columns(2)
    with col1:
        plot_weight_distribution(weights)
    with col2:
        if convergence:
            plot_convergence(convergence, params.get('threshold', 0.001))
        else:
            st.info("No convergence data available for this method")
    # Distribution Comparisons for categorical variables
    st.subheader("Distribution Comparisons")
    for col in df.select_dtypes('category').columns:
        plot_distribution_comparison(df, col, np.ones(len(df)), weights, 50)
    # Diagnostics: Assumption Checks
    if params.get('assumption_checks', False):
        st.subheader("Assumption Diagnostics")
        numeric_cols = df.select_dtypes(include=['float', 'int']).columns
        for col in numeric_cols:
            asmd = calculate_asmd(df, weights, col)
            ks = calculate_ks_statistic(df, weights, col)
            st.write(f"{col}: ASMD = {asmd:.3f}, KS Statistic = {ks['statistic']:.3f} (p = {ks['p_value']:.3f})")
        if params['method'] == 'propensity_score_weighting':
            X = df[params['predictors']]
            y = df[params['outcome']]
            cat_cols = X.select_dtypes(include=['category', 'object']).columns
            if not cat_cols.empty:
                X = pd.get_dummies(X, drop_first=True)
            base_model = LogisticRegression(max_iter=1000)
            model = CalibratedClassifierCV(base_model, cv=5)
            model.fit(X, y)
            propensity_scores = model.predict_proba(X)[:, 1]
            overlap = check_propensity_overlap(propensity_scores)
            st.write(f"Propensity score overlap metric: {overlap:.3f}")
        if params['method'] == 'calibration_weighting':
            X = pd.get_dummies(df[params['predictors']], drop_first=True).values
            sorted_keys = sorted(params['targets'].keys())
            target_vec = np.array([params['targets'][k] for k in sorted_keys])
            residuals = calibration_residuals(X, weights.values, target_vec)
            st.write("Calibration residuals (first 10):", residuals[:10])
    # Variance Estimation
    if params.get('bootstrap_variance', False):
        var_bootstrap = bootstrap_variance_estimation(df, lambda d: run_weighting_engine(d, params['method'], params)[0])
        st.write(f"Bootstrap Variance Estimate: {var_bootstrap:.3f}")
    if params.get('taylor_variance', False):
        var_taylor = taylor_series_variance(weights)
        st.write(f"Taylor Series Variance Estimate: {var_taylor:.3f}")
    # Data Export
    st.download_button("Download Weighted Data",
                       df.assign(weight=weights).to_csv(index=False).encode(),
                       "weighted_data.csv")

def main():
    st.title("📊 Advanced Survey Weighting Suite")
    # Data Upload and Processing
    df = handle_data_upload()
    if df is None:
        return
    # Weighting Method Selection
    method = st.sidebar.selectbox("Weighting Method", [
        'post_stratification',
        'inverse_probability_weighting',
        'propensity_score_weighting',
        'calibration_weighting',
        'rake_weighting'
    ])
    # Parameter Collection
    params = get_weighting_params(method, df)
    # Run Analysis
    if st.sidebar.button("Run Weighting Analysis"):
        with st.spinner("Optimizing weights..."):
            try:
                weights, convergence = run_weighting_engine(df, method, params)
                show_results(df, weights, convergence, params)
            except Exception as e:
                st.error(f"Error in weighting process: {str(e)}")

if __name__ == "__main__":
    main()
