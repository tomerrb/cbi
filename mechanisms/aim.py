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
        treatment: Treatment variable name (required for ATE mode).
        outcome: Outcome variable name (required for ATE mode).
        confounders: List of confounder variable names (required for ATE mode).
        causal_graph_path: Path to GML file with causal graph (required for ATE mode).
        ate_sample_size: Samples for final ATE computation (default: 10000).
        sim_sample_size: Samples for simulation during selection (default: 5000).
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
        treatment=None,
        outcome=None,
        confounders=None,
        causal_graph_path=None,
        ate_sample_size=10000,
        sim_sample_size=5000,
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
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.ate_sample_size = ate_sample_size
        self.sim_sample_size = sim_sample_size
        self.causal_graph = None
        self._true_ate = None
        
        if selection_mode == "ate":
            # Validate required causal parameters
            if not all([treatment, outcome, confounders, causal_graph_path]):
                raise ValueError(
                    "ATE mode requires treatment, outcome, confounders, and causal_graph_path"
                )
            # Load GML causal graph
            with open(causal_graph_path, 'r') as f:
                self.causal_graph = f.read()

    def compute_ate_from_data(self, df):
        """Compute ATE using DoWhy's backdoor adjustment.
        
        Args:
            df: DataFrame containing treatment, outcome, and confounder columns.
            
        Returns:
            float: Estimated ATE value.
        """
        # Late import to avoid dependency when not using ATE mode
        from dowhy import CausalModel
        
        model = CausalModel(
            data=df,
            treatment=self.treatment,
            outcome=self.outcome,
            graph=self.causal_graph
        )
        
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )
        
        return estimate.value

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

    def worst_ate_approximated(self, candidates, answers, data, model, measurements, sigma):
        """Select the marginal that most improves ATE estimation.
        
        Iterates over candidates, simulates measuring each one, and selects
        the one that minimizes ATE error. Falls back to L1 selection if no
        candidate improves ATE.
        
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
        # Use cached true ATE
        true_ate = self._true_ate
        
        # Current model's ATE (use fixed seed for consistency)
        current_model_df = self._model_to_dataframe(model, self.sim_sample_size, seed=42)
        current_ate = self.compute_ate_from_data(current_model_df)
        current_error = abs(true_ate - current_ate)
        
        print(f"Current ATE error: {current_error:.6f} (true={true_ate:.4f}, model={current_ate:.4f})")
        
        best_candidate = None
        best_error = current_error
        
        for cl in candidates:
            try:
                # Simulate measuring this clique
                simulated_model = self._simulate_measurement(model, data, cl, measurements)
                
                # Compute ATE with fixed seed
                simulated_df = self._model_to_dataframe(simulated_model, self.sim_sample_size, seed=42)
                simulated_ate = self.compute_ate_from_data(simulated_df)
                simulated_error = abs(true_ate - simulated_ate)
                
                if simulated_error < best_error:
                    best_error = simulated_error
                    best_candidate = cl
                    print(f"  Candidate {cl}: error={simulated_error:.6f} (BETTER)")
            except Exception as e:
                print(f"  Candidate {cl}: failed ({e})")
                continue
        
        # Fallback: use original L1-based selection if no ATE improvement
        if best_candidate is None:
            print("No ATE improvement found, falling back to L1 selection")
            best_candidate = self._fallback_worst_approximated(candidates, answers, model, sigma)
        
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
        # Cache true ATE at the start if using ATE mode
        if self.selection_mode == "ate":
            self._true_ate = self.compute_ate_from_data(data.df)
            print(f"True ATE from data: {self._true_ate:.6f}")
        
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
    params["treatment"] = None
    params["outcome"] = None
    params["confounders"] = None
    params["causal_graph"] = None

    return params


if __name__ == "__main__":

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
    
    # Causal structure arguments (required when selection_mode="ate")
    parser.add_argument(
        "--treatment",
        type=str,
        help="Treatment variable name (required for ATE mode)"
    )
    parser.add_argument(
        "--outcome",
        type=str,
        help="Outcome variable name (required for ATE mode)"
    )
    parser.add_argument(
        "--confounders",
        type=str,
        help="Comma-separated list of confounder variable names (required for ATE mode)"
    )
    parser.add_argument(
        "--causal_graph",
        type=str,
        help="Path to GML file with causal graph (required for ATE mode)"
    )

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    
    # Parse confounders from comma-separated string if provided
    confounders_list = None
    if args.confounders is not None:
        confounders_list = [c.strip() for c in args.confounders.split(',')]

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
        treatment=args.treatment,
        outcome=args.outcome,
        confounders=confounders_list,
        causal_graph_path=args.causal_graph,
    )
    model, synth = mech.run(data, workload)

    if args.save is not None:
        synth.df.to_csv(args.save, index=False)

    # Print ATE comparison if in ATE mode
    if args.selection_mode == "ate":
        true_ate = mech._true_ate
        synth_ate = mech.compute_ate_from_data(synth.df)
        print(f"True ATE: {true_ate:.6f}")
        print(f"Synthetic ATE: {synth_ate:.6f}")
        print(f"ATE Error: {abs(true_ate - synth_ate):.6f}")

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

