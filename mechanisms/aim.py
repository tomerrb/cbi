"""Implementation of CLAIM: Causally-Learned Adaptive and Iterative Mechanism for DP Synthetic Data.

This implementation supports two selection modes:
- "marginal" (default): Original L1-based selection using the exponential mechanism.
  Selects the worst-approximated marginal based on L1 distance.
- "ate": ATE-based selection for causal inference applications.
  Selects the marginal that most improves Average Treatment Effect estimation.
  Requires DoWhy library and causal structure specification.

Note that with the default settings, CLAIM can take many hours to run. You can configure
the runtime/utility tradeoff via the max_model_size flag. We recommend setting it to 1.0
for debugging, but keeping the default value of 80 for any official comparisons.

Note that we assume the data has been appropriately preprocessed so that there are no
large-cardinality categorical attributes. If there are, we recommend using something like
"compress_domain" from mst.py.
"""

import gc
import jax
import numpy as np
import itertools
from mbi import (
    Dataset,
    Domain,
    estimation,
    junction_tree,
    LinearMeasurement,
)
from mechanism import Mechanism
from collections import defaultdict
from scipy.optimize import bisect
import pandas as pd
from mbi import Factor
import argparse


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def compile_workload(workload):
    weights = {cl: wt for (cl, wt) in workload}
    workload_cliques = weights.keys()

    def score(cl):
        return sum(
            weights[workload_cl] * len(set(cl) & set(workload_cl))
            for workload_cl in workload_cliques
        )

    return {cl: score(cl) for cl in downward_closure(workload_cliques)}


def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = (
            junction_tree.hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        )
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


class CLAIM(Mechanism):
    """Causally-Learned Adaptive and Iterative Mechanism for DP Synthetic Data.
    
    Args:
        epsilon: Privacy parameter (epsilon for zCDP conversion).
        delta: Privacy parameter.
        prng: Optional random number generator.
        rounds: Number of selection rounds (default: 16 * domain size).
        max_model_size: Maximum model size in MB (default: 80).
        max_iters: Maximum iterations for mirror descent (default: 1000).
        structural_zeros: Dict of structural zeros constraints.
        selection_mode: "marginal" (L1-based) or "ate" (ATE-based) selection.
        ate_configs: List of ATE configurations (required for ATE mode). Each config is a dict:
            - "name": str, identifier for this ATE
            - "treatment": str, treatment variable name
            - "outcome": str, outcome variable name  
            - "confounders": list[str], confounder variable names
            - "alpha": float, weight in [0, 1] for this ATE in utility function
        causal_graph_path: Path to GML file with causal graph (required for ATE mode).
        ate_sample_size: Samples for final ATE computation (default: 10000).
        sim_sample_size: Samples for simulation during selection (default: 5000).
        marginal_weight: Weight for L1 marginal error in hybrid selection (default: 0.3).
            0.0 = pure ATE selection, 1.0 = pure marginal selection.
            Only used when selection_mode="ate".
    """
    
    def __init__(
        self,
        epsilon,
        delta,
        prng=None,
        rounds=None,
        max_model_size=80,
        max_iters=1000,
        structural_zeros={},
        selection_mode="marginal",
        ate_configs=None,
        causal_graph_path=None,
        ate_sample_size=10000,
        sim_sample_size=5000,
        marginal_weight=0.3,
    ):
        super(CLAIM, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros
        
        # Selection mode configuration
        self.selection_mode = selection_mode
        if selection_mode not in ("marginal", "ate"):
            raise ValueError(f"selection_mode must be 'marginal' or 'ate', got '{selection_mode}'")
        
        # ATE mode configuration
        self.ate_configs = ate_configs or []
        self.ate_sample_size = ate_sample_size
        self.sim_sample_size = sim_sample_size
        self.causal_graph = None
        self._true_ates = {}  # Dict of {name: true_ate_value}
        
        # Hybrid selection configuration
        self.marginal_weight = marginal_weight
        if not (0.0 <= marginal_weight <= 1.0):
            raise ValueError(f"marginal_weight must be in [0, 1], got {marginal_weight}")
        
        if selection_mode == "ate":
            # Validate required causal parameters
            if not ate_configs or not causal_graph_path:
                raise ValueError(
                    "ATE mode requires ate_configs (list of ATE configurations) and causal_graph_path"
                )
            
            # Validate each ATE config
            for i, config in enumerate(ate_configs):
                required_keys = ["name", "treatment", "outcome", "confounders", "alpha"]
                for key in required_keys:
                    if key not in config:
                        raise ValueError(f"ATE config {i} missing required key: '{key}'")
                if not (0 <= config["alpha"] <= 1):
                    raise ValueError(f"ATE config '{config['name']}' alpha must be in [0, 1], got {config['alpha']}")
            
            # Validate alpha weights sum to 1
            alpha_sum = sum(config["alpha"] for config in ate_configs)
            if abs(alpha_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"ATE alpha weights must sum to 1.0, got {alpha_sum:.6f}. "
                    f"Weights: {[config['alpha'] for config in ate_configs]}"
                )
            
            # Load GML causal graph (shared across all ATEs)
            with open(causal_graph_path, 'r') as f:
                self.causal_graph = f.read()

    def _binarize_for_ate(self, df, config):
        """Apply binarization to treatment and outcome columns for ATE calculation.
        
        This converts encoded (integer) data to binary (0/1) based on the config.
        Called just before ATE calculation to prepare the data for DoWhy.
        
        Args:
            df: DataFrame with encoded data.
            config: ATE config dict with optional 'binarization' field:
                {
                    "treatment": "educational-num",
                    "outcome": "income",
                    "binarization": {
                        "treatment": {"type": "threshold", "threshold": 9, "comparison": ">"},
                        "outcome": {"type": "threshold", "threshold": 0, "comparison": ">"}
                    }
                }
                If binarization not specified, assumes data is already binary.
        
        Returns:
            DataFrame: Copy with binarized treatment and outcome.
        """
        df = df.copy()
        treatment = config["treatment"]
        outcome = config["outcome"]
        binarization = config.get("binarization", {})
        
        # Binarize treatment if config provided
        if "treatment" in binarization:
            bin_config = binarization["treatment"]
            if bin_config.get("type") == "threshold":
                threshold = bin_config.get("threshold", 0)
                comparison = bin_config.get("comparison", ">")
                if comparison == ">":
                    df[treatment] = (df[treatment] > threshold).astype(int)
                elif comparison == ">=":
                    df[treatment] = (df[treatment] >= threshold).astype(int)
                elif comparison == "<":
                    df[treatment] = (df[treatment] < threshold).astype(int)
                elif comparison == "<=":
                    df[treatment] = (df[treatment] <= threshold).astype(int)
            elif bin_config.get("type") == "value":
                # For categorical: specific value = 1, others = 0
                # In encoded data, this is a specific integer
                positive_value = bin_config.get("positive_value", 1)
                df[treatment] = (df[treatment] == positive_value).astype(int)
            elif bin_config.get("type") == "categorical":
                # For categorical type: use encoded_positive_value for encoded data
                encoded_positive_value = bin_config.get("encoded_positive_value")
                if encoded_positive_value is not None:
                    df[treatment] = (df[treatment] == encoded_positive_value).astype(int)
                else:
                    # Fallback to positive_value if encoded_positive_value not specified
                    positive_value = bin_config.get("positive_value", 1)
                    df[treatment] = (df[treatment] == positive_value).astype(int)
        
        # Binarize outcome if config provided
        if "outcome" in binarization:
            bin_config = binarization["outcome"]
            if bin_config.get("type") == "threshold":
                threshold = bin_config.get("threshold", 0)
                comparison = bin_config.get("comparison", ">")
                if comparison == ">":
                    df[outcome] = (df[outcome] > threshold).astype(int)
                elif comparison == ">=":
                    df[outcome] = (df[outcome] >= threshold).astype(int)
            elif bin_config.get("type") == "value":
                positive_value = bin_config.get("positive_value", 1)
                df[outcome] = (df[outcome] == positive_value).astype(int)
            elif bin_config.get("type") == "categorical":
                # For categorical type: use encoded_positive_value for encoded data
                encoded_positive_value = bin_config.get("encoded_positive_value")
                if encoded_positive_value is not None:
                    df[outcome] = (df[outcome] == encoded_positive_value).astype(int)
                else:
                    # Fallback to positive_value if encoded_positive_value not specified
                    positive_value = bin_config.get("positive_value", 1)
                    df[outcome] = (df[outcome] == positive_value).astype(int)
        
        return df

    def compute_ate_from_data(self, df, config):
        """Compute ATE for a single treatment-outcome pair using DoWhy's backdoor adjustment.
        
        Args:
            df: DataFrame containing treatment, outcome, and confounder columns (encoded).
            config: ATE config dict with treatment, outcome, and optional binarization.
            
        Returns:
            float: Estimated ATE value.
        """
        # Late import to avoid dependency when not using ATE mode
        from dowhy import CausalModel
        
        treatment = config["treatment"]
        outcome = config["outcome"]
        
        # Apply binarization for ATE calculation
        df_binary = self._binarize_for_ate(df, config)
        
        model = CausalModel(
            data=df_binary,
            treatment=treatment,
            outcome=outcome,
            graph=self.causal_graph
        )
        
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )
        
        return estimate.value

    def compute_all_ates(self, df):
        """Compute all configured ATEs from a DataFrame.
        
        Args:
            df: DataFrame containing all required columns (encoded).
            
        Returns:
            dict: {ate_name: ate_value} for each configured ATE.
        """
        ates = {}
        for config in self.ate_configs:
            ate_value = self.compute_ate_from_data(df, config)
            ates[config["name"]] = ate_value
        return ates

    def compute_weighted_ate_error(self, true_ates, model_ates):
        """Compute weighted sum of ATE errors: Σ α_i * |true_i - model_i|.
        
        Args:
            true_ates: Dict of {name: true_ate_value}.
            model_ates: Dict of {name: model_ate_value}.
            
        Returns:
            float: Weighted ATE error.
        """
        weighted_error = 0.0
        for config in self.ate_configs:
            name = config["name"]
            alpha = config["alpha"]
            error = abs(true_ates[name] - model_ates[name])
            weighted_error += alpha * error
        return weighted_error

    def _model_to_dataframe(self, model, num_samples, seed=None):
        """Generate synthetic DataFrame from PGM model with optional seed for reproducibility.
        
        Args:
            model: Fitted PGM model.
            num_samples: Number of samples to generate.
            seed: Optional random seed for reproducibility.
            
        Returns:
            DataFrame: Synthetic data.
        """
        if seed is not None:
            np.random.seed(seed)
        synth = model.synthetic_data(rows=num_samples)
        return synth.df

    def _simulate_measurement(self, model, data, clique, measurements):
        """Simulate measuring a clique without actually adding noise.
        
        Returns a new model fitted with the additional measurement.
        Used to predict how measuring a particular clique would affect ATE.
        
        Args:
            model: Current fitted PGM model.
            data: Original Dataset.
            clique: Clique to simulate measuring.
            measurements: Current list of measurements.
            
        Returns:
            Model: New model fitted with the simulated measurement.
        """
        # Get true marginal (no noise since we're simulating)
        x = data.project(clique).datavector()
        
        # Small stddev = high confidence measurement
        temp_measurement = LinearMeasurement(x, clique, stddev=1e-10)
        
        # Copy and extend measurements
        sim_measurements = measurements.copy()
        sim_measurements.append(temp_measurement)
        
        # Warm start from current model
        pcliques = list(set(M.clique for M in sim_measurements))
        potentials = model.potentials.expand(pcliques)
        
        # Fit with fewer iterations for speed during simulation
        sim_model = estimation.mirror_descent(
            data.domain, 
            sim_measurements, 
            iters=min(self.max_iters, 500),
            potentials=potentials,
            callback_fn=lambda *_: None
        )
        
        return sim_model

    def _fallback_worst_approximated(self, candidates, answers, model, sigma):
        """Original L1-based selection as fallback (deterministic, no exponential mechanism).
        
        Args:
            candidates: Dict of candidate cliques with weights.
            answers: Dict of true marginals.
            model: Current fitted model.
            sigma: Noise standard deviation.
            
        Returns:
            tuple: Selected clique.
        """
        errors = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            xest = model.project(cl).datavector()
            errors[cl] = wgt * np.linalg.norm(x - xest, 1)
        
        # Deterministic: pick max error
        return max(errors, key=errors.get)

    def _compute_l1_scores(self, candidates, answers, model):
        """Compute L1 error scores for all candidates.
        
        Args:
            candidates: Dict of candidate cliques with weights.
            answers: Dict of true marginals.
            model: Current fitted model.
            
        Returns:
            dict: {clique: L1_error_score}
        """
        scores = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            xest = model.project(cl).datavector()
            scores[cl] = wgt * np.linalg.norm(x - xest, 1)
        return scores

    def _normalize_scores(self, scores):
        """Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: Dict of {clique: score}.
            
        Returns:
            dict: {clique: normalized_score} in [0, 1] range.
        """
        if not scores:
            return {}
        
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        
        # Avoid division by zero if all scores are equal
        if max_val - min_val < 1e-10:
            return {cl: 0.5 for cl in scores}
        
        return {cl: (s - min_val) / (max_val - min_val) for cl, s in scores.items()}

    def worst_ate_approximated(self, candidates, answers, data, model, measurements, sigma):
        """Select marginal using hybrid strategy combining ATE and L1 scores.
        
        Uses marginal_weight to balance between:
        - L1 error scores (how poorly each marginal is approximated)
        - ATE improvement scores (how much each marginal improves ATE estimation)
        
        Args:
            candidates: Dict of candidate cliques with weights.
            answers: Dict of true marginals.
            data: Original Dataset.
            model: Current fitted model.
            measurements: Current list of measurements.
            sigma: Noise standard deviation.
            
        Returns:
            tuple: Selected clique.
        """
        # Use cached true ATEs
        true_ates = self._true_ates
        
        # Step 1: Compute L1 scores (fast, no simulation)
        l1_scores = self._compute_l1_scores(candidates, answers, model)
        
        # Step 2: Compute ATE improvement scores (requires simulation)
        ate_scores = {}
        
        # Current model's ATEs (use fixed seed for consistency)
        current_model_df = self._model_to_dataframe(model, self.sim_sample_size, seed=42)
        current_ates = self.compute_all_ates(current_model_df)
        current_error = self.compute_weighted_ate_error(true_ates, current_ates)
        
        # Print current status for each ATE
        print(f"Current weighted ATE error: {current_error:.6f}")
        print(f"Hybrid selection: marginal_weight={self.marginal_weight:.2f}")
        for config in self.ate_configs:
            name = config["name"]
            alpha = config["alpha"]
            true_val = true_ates[name]
            model_val = current_ates[name]
            print(f"  {name} (α={alpha}): true={true_val:.4f}, model={model_val:.4f}, error={abs(true_val - model_val):.4f}")
        
        for cl in candidates:
            simulated_model = None
            simulated_df = None
            try:
                # Simulate measuring this clique
                simulated_model = self._simulate_measurement(model, data, cl, measurements)
                
                # Compute all ATEs with fixed seed
                simulated_df = self._model_to_dataframe(simulated_model, self.sim_sample_size, seed=42)
                simulated_ates = self.compute_all_ates(simulated_df)
                simulated_error = self.compute_weighted_ate_error(true_ates, simulated_ates)
                
                # ATE improvement score (higher = better improvement)
                # We use current_error - simulated_error so positive means improvement
                ate_scores[cl] = current_error - simulated_error
                
            except Exception as e:
                print(f"  Candidate {cl}: failed ({e})")
                # Assign zero improvement score if simulation fails
                ate_scores[cl] = 0.0
            finally:
                # Force cleanup after each candidate to prevent memory accumulation
                if simulated_model is not None:
                    del simulated_model
                if simulated_df is not None:
                    del simulated_df
                gc.collect()
                # Clear JAX JIT compilation caches to prevent LLVM memory buildup
                jax.clear_caches()
        
        # Step 3: Normalize both score sets to [0, 1]
        l1_normalized = self._normalize_scores(l1_scores)
        ate_normalized = self._normalize_scores(ate_scores)
        
        # Step 4: Combine with marginal_weight
        # combined = λ * L1 + (1-λ) * ATE
        combined_scores = {}
        for cl in candidates:
            l1_score = l1_normalized.get(cl, 0.0)
            ate_score = ate_normalized.get(cl, 0.0)
            combined_scores[cl] = (
                self.marginal_weight * l1_score + 
                (1 - self.marginal_weight) * ate_score
            )
        
        # Step 5: Select best candidate (highest combined score)
        best_candidate = max(combined_scores, key=combined_scores.get)
        best_combined = combined_scores[best_candidate]
        best_l1 = l1_normalized.get(best_candidate, 0.0)
        best_ate = ate_normalized.get(best_candidate, 0.0)
        
        print(f"Selected {best_candidate}: L1={best_l1:.3f}, ATE={best_ate:.3f}, combined={best_combined:.3f}")
        
        return best_candidate


    def worst_approximated(self, candidates, answers, model, eps, sigma):
        """Original L1-based selection using the exponential mechanism.
        
        Args:
            candidates: Dict of candidate cliques with weights.
            answers: Dict of true marginals.
            model: Current fitted model.
            eps: Epsilon for exponential mechanism.
            sigma: Noise standard deviation.
            
        Returns:
            tuple: Selected clique.
        """
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(
            sensitivity.values()
        )  # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def run(self, data, workload, num_synth_rows=None, initial_cliques=None):
        # Cache true ATEs at the start if using ATE mode
        if self.selection_mode == "ate":
            self._true_ates = self.compute_all_ates(data.df)
            print("True ATEs from data:")
            for config in self.ate_configs:
                name = config["name"]
                alpha = config["alpha"]
                print(f"  {name} (α={alpha}): {self._true_ates[name]:.6f}")
        
        rounds = self.rounds or 16 * len(data.domain)
        candidates = compile_workload(workload)
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        if not initial_cliques:
            initial_cliques = [
                cl for cl in candidates if len(cl) == 1
            ]  # use one-way marginals

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        measurements = []
        print("Initial Sigma", sigma)
        rho_used = len(oneway) * 0.5 / sigma**2
        for cl in initial_cliques:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, x.size)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))

        zeros = self.structural_zeros
        # NOTE: Haven't incorproated structural zeros back yet after refactoring
        model = estimation.mirror_descent(
                data.domain, measurements, iters=self.max_iters, callback_fn=lambda *_: None
        )

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2 * (0.5 / sigma**2 + 1.0 / 8 * epsilon**2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
            print('Budget Used', rho_used, '/', self.rho)
            size_limit = self.max_model_size * rho_used / self.rho

            small_candidates = filter_candidates(candidates, model, size_limit)
            
            # Branch on selection mode
            if self.selection_mode == "ate":
                cl = self.worst_ate_approximated(
                    small_candidates, answers, data, model, measurements, sigma
                )
            else:
                cl = self.worst_approximated(
                    small_candidates, answers, model, epsilon, sigma
                )
            print('Measuring Clique', cl)
            n = data.domain.size(cl)
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))
            z = model.project(cl).datavector()

            # Warm start potentials from prior round
            # TODO: check if it helps to call maximal_subsets here
            pcliques = list(set(M.clique for M in measurements))
            potentials = model.potentials.expand(pcliques)
            model = estimation.mirror_descent(
                    data.domain, measurements, iters=self.max_iters, potentials=potentials, callback_fn=lambda *_: None
            )
            w = model.project(cl).datavector()
            # print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                print("(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma", sigma / 2)
                sigma /= 2
                epsilon *= 2

        print("Generating Data...")
        model = estimation.mirror_descent(
            data.domain, measurements, iters=self.max_iters, potentials=potentials
        )
        synth = model.synthetic_data(rows=num_synth_rows)

        return model, synth


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params["dataset"] = "../data/adult.csv"
    params["domain"] = "../data/adult-domain.json"
    params["epsilon"] = 1.0
    params["delta"] = 1e-9
    params["noise"] = "laplace"
    params["max_model_size"] = 80
    params["max_iters"] = 1000
    params["degree"] = 2
    params["num_marginals"] = None
    params["max_cells"] = 10000
    # Selection mode: "marginal" (L1-based) or "ate" (ATE-based)
    params["selection_mode"] = "marginal"
    # Causal parameters (required when selection_mode="ate")
    params["ate_configs"] = None  # Path to JSON file with ATE configs
    params["causal_graph"] = None
    # Hybrid selection weight: 0.0 = pure ATE, 1.0 = pure marginal
    params["marginal_weight"] = 0.3

    return params


if __name__ == "__main__":
    import json

    description = "CLAIM: Causally-Learned Adaptive and Iterative Mechanism for DP Synthetic Data"
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument("--dataset", help="dataset to use")
    parser.add_argument("--domain", help="domain to use")
    parser.add_argument("--epsilon", type=float, help="privacy parameter")
    parser.add_argument("--delta", type=float, help="privacy parameter")
    parser.add_argument(
        "--max_model_size", type=float, help="maximum size (in megabytes) of model"
    )
    parser.add_argument("--max_iters", type=int, help="maximum number of iterations")
    parser.add_argument("--degree", type=int, help="degree of marginals in workload")
    parser.add_argument(
        "--num_marginals", type=int, help="number of marginals in workload"
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        help="maximum number of cells for marginals in workload",
    )
    parser.add_argument("--save", type=str, help="path to save synthetic data")
    
    # Selection mode arguments
    parser.add_argument(
        "--selection_mode",
        type=str,
        choices=["marginal", "ate"],
        help="Selection mode: 'marginal' (L1-based) or 'ate' (ATE-based)"
    )
    
    # Multi-ATE configuration (required when selection_mode="ate")
    parser.add_argument(
        "--ate_configs",
        type=str,
        help="Path to JSON file with ATE configurations. Each config should have: "
             "name, treatment, outcome, confounders (list), alpha (weight in [0,1])"
    )
    parser.add_argument(
        "--causal_graph",
        type=str,
        help="Path to GML file with causal graph (required for ATE mode)"
    )
    parser.add_argument(
        "--marginal_weight",
        type=float,
        help="Weight for L1 marginal error in hybrid selection (0.0 = pure ATE, 1.0 = pure marginal)."
             " Only used when selection_mode='ate'. Default: 0.3"
    )

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    
    # Load ATE configs from JSON file if provided
    ate_configs_list = None
    if args.ate_configs is not None:
        with open(args.ate_configs, 'r') as f:
            ate_configs_list = json.load(f)

    data = Dataset.load(args.dataset, args.domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        prng = np.random
        workload = [
            workload[i]
            for i in prng.choice(len(workload), args.num_marginals, replace=False)
        ]

    workload = [(cl, 1.0) for cl in workload]
    mech = CLAIM(
        args.epsilon,
        args.delta,
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
        selection_mode=args.selection_mode,
        ate_configs=ate_configs_list,
        causal_graph_path=args.causal_graph,
        marginal_weight=args.marginal_weight,
    )
    model, synth = mech.run(data, workload)

    if args.save is not None:
        synth.df.to_csv(args.save, index=False)

    # Print ATE comparison if in ATE mode
    if args.selection_mode == "ate":
        true_ates = mech._true_ates
        synth_ates = mech.compute_all_ates(synth.df)
        print("\nATE Comparison:")
        total_weighted_error = 0.0
        for config in mech.ate_configs:
            name = config["name"]
            alpha = config["alpha"]
            true_val = true_ates[name]
            synth_val = synth_ates[name]
            error = abs(true_val - synth_val)
            weighted_error = alpha * error
            total_weighted_error += weighted_error
            print(f"  {name} (α={alpha}): true={true_val:.6f}, synth={synth_val:.6f}, error={error:.6f}")
        print(f"  Weighted Total Error: {total_weighted_error:.6f}")

    # Print marginal errors
    synth_errors = []
    model_errors = []
    for proj, wgt in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        Z = model.project(proj).datavector()
        e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        synth_errors.append(e)
        e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Z / Z.sum(), 1)
        model_errors.append(e)
    print("Average Marginal Error: ", np.mean(model_errors), np.mean(synth_errors))


