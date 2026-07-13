"""Implementation of CLAIM: Causally-Learned Adaptive and Iterative Mechanism for DP Synthetic Data.

This implementation supports three selection modes:
- "marginal" (default): Original L1-based selection using the exponential mechanism.
  Selects the worst-approximated marginal based on L1 distance.
- "ate": ATE-based selection for causal inference applications.
  Selects the marginal that most improves Average Treatment Effect estimation.
  Requires DoWhy library and causal structure specification.
- "claim": Blended CLAIM selection.  Selects marginals with an exponential
  mechanism using a convex tradeoff between marginal error and predicted ATE
  improvement.  The tradeoff is controlled by lambda_weight.

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
from cdp2adp import cdp_rho, cdp_eps
import warnings


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
        selection_mode: "marginal" (L1-based), "ate" (ATE-based), or
            "claim" (blended marginal/ATE) selection.
        lambda_weight: Weight on the marginal/statistical term. Smaller values
            give more weight to the causal/ATE term.
        mu_eta: Small positive constant preventing division by zero in the
            theory-derived scale parameter mu. mu is no longer a free
            hyperparameter: it is recomputed each selection round as the ratio
            between the typical (median) magnitudes of the two score terms
            over the candidate set,
                mu_t = median_r |L_hat_r| / (kappa * median_r |A_hat_r| + mu_eta),
            where L_hat_r and A_hat_r are *public proxies* of the statistical
            and causal terms obtained by evaluating them on the previous
            models instead of the private data: L_hat_r compares the current
            model's marginal on r against the previous round's model (minus
            the expected noise bias), and A_hat_r is the causal term with the
            data replaced by the current model (using an FWL linear ATE
            estimator computed from the model itself). Both proxies read only
            DP-released quantities, so mu_t is public. If no candidate clique
            covers {treatment, outcome} + confounders, all A_hat_r are zero
            and mu_t falls back to median_r |L_hat_r| / mu_eta.
        kappa_eta: Minimum-variance floor for the theory-derived normalizer
            kappa. kappa is no longer a free hyperparameter: it is recomputed
            each selection round as kappa = (sum_i beta_i / v_i)^{-1}, which
            for the single causal query used here (beta = 1) reduces to
            kappa = v, the residual treatment variance
            v = E_{p_hat}[(T - P_hat(T=1|Z))^2] under the current model. This
            is the value required by the sensitivity analysis so that the
            causal term's contribution to the quality-score sensitivity
            matches the statistical term's (2/N). To guard against the
            degenerate case v = 0, kappa is clipped below at kappa_eta.
        adaptive_lambda: If True, lambda_weight is adjusted at the end of each round
            using both the current model's ATE error and its TVD (marginal) error,
            so that chasing ATE improvements cannot come at unbounded cost to TVD
            preservation, and vice versa.
        treatment: Treatment variable name (required for ATE/CLAIM modes).
        outcome: Outcome variable name (required for ATE mode).
        confounders: List of confounder variable names (required for ATE mode).
        causal_graph_path: Path to GML file with causal graph (required for ATE mode).
        ate_sample_size: Samples for final ATE computation (default: 10000).
        sim_sample_size: Samples for simulation during selection (default: 5000).
        reference_ate: Precomputed reference ATE. If provided, treated as
            already private (e.g. computed outside this class with its own DP
            guarantee) and used as-is.
        ate_sensitivity: Assumed L2 sensitivity of the raw-data ATE estimator
            under one record changing. If `reference_ate` is omitted, a
            reference ATE is auto-released privately instead: computed on raw
            data, then perturbed with Gaussian noise calibrated for zCDP using
            `reference_ate_rho_fraction * rho`, which is deducted from the
            remaining budget before selection rounds start. Defaults to
            `(ate_outcome_range[1] - ate_outcome_range[0]) / data.records`
            (the sensitivity of a difference-in-means estimator over an
            outcome bounded to `ate_outcome_range`) if not set explicitly.
        reference_ate_rho_fraction: Fraction of the total zCDP budget spent on
            auto-releasing a private reference ATE when `reference_ate` is
            not supplied (default: 0.05).
        ate_outcome_range: (min, max) the outcome column is assumed (and
            enforced) to lie in for ATE computation. Every outcome value is
            clipped into this range before `compute_ate_from_data` runs, so
            the sensitivity default above holds by construction rather than
            by caller discipline. Defaults to (-1.0, 1.0). If the outcome's
            natural scale doesn't match this, rescale it yourself before
            passing data in, or override this range accordingly.
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
        lambda_weight=0.5,
        mu_eta=1e-6,
        kappa_eta=1e-6,
        adaptive_lambda=False,
        lambda_min=0.1,
        lambda_max=0.9,
        lambda_update_factor=1.25,
        ate_tolerance=0.01,
        tvd_tolerance=0.05,
        reference_ate=None,
        ate_sensitivity=None,
        reference_ate_rho_fraction=0.05,
        ate_outcome_range=(-1.0, 1.0),
        treatment=None,
        outcome=None,
        confounders=None,
        causal_graph_path=None,
        ate_sample_size=10000,
        sim_sample_size=5000,
    ):
        super(CLAIM, self).__init__(epsilon, delta, bounded=False, prng=prng or np.random)
        self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros

        # Selection mode configuration
        self.selection_mode = selection_mode
        if selection_mode not in ("marginal", "ate", "claim"):
            raise ValueError(
                "selection_mode must be 'marginal', 'ate', or 'claim', "
                f"got '{selection_mode}'"
            )

        # CLAIM lambda configuration
        self.lambda_weight = float(lambda_weight)
        # mu and kappa are derived from the theory each selection round
        # (see _compute_mu and _model_treatment_variance); mu_eta and
        # kappa_eta are only small-value safeguards.
        self.mu_eta = float(mu_eta)
        if self.mu_eta <= 0:
            raise ValueError("mu_eta must be positive")
        self.mu = None
        self.kappa_eta = float(kappa_eta)
        if self.kappa_eta <= 0:
            raise ValueError("kappa_eta must be positive")
        self.kappa = None
        self._prev_model = None  # p_hat_{t-2}, for the public mu proxies
        self.adaptive_lambda = bool(adaptive_lambda)
        self.lambda_schedule = "adaptive" if self.adaptive_lambda else "fixed"
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.lambda_update_factor = float(lambda_update_factor)
        self.ate_tolerance = ate_tolerance
        self.tvd_tolerance = tvd_tolerance
        if not (0.0 <= self.lambda_weight <= 1.0):
            raise ValueError("lambda_weight must be in [0, 1]")
        if not (0.0 <= self.lambda_min <= self.lambda_max <= 1.0):
            raise ValueError("lambda_min/lambda_max must satisfy 0 <= min <= max <= 1")
        if self.adaptive_lambda and self.lambda_min <= 0:
            raise ValueError("adaptive lambda requires lambda_min > 0")
        if self.lambda_update_factor <= 1.0:
            raise ValueError("lambda_update_factor must be > 1")

        # ATE mode configuration
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.ate_sample_size = ate_sample_size
        self.sim_sample_size = sim_sample_size
        self.causal_graph = None
        self.reference_ate = reference_ate
        self.ate_sensitivity = ate_sensitivity
        self.reference_ate_rho_fraction = float(reference_ate_rho_fraction)
        self.ate_outcome_range = (float(ate_outcome_range[0]), float(ate_outcome_range[1]))
        if not (0.0 < self.reference_ate_rho_fraction < 1.0):
            raise ValueError("reference_ate_rho_fraction must be in (0, 1)")
        if self.ate_sensitivity is not None and self.ate_sensitivity <= 0:
            raise ValueError("ate_sensitivity must be positive")
        if self.ate_outcome_range[0] >= self.ate_outcome_range[1]:
            raise ValueError("ate_outcome_range must satisfy min < max")

        if selection_mode in ("ate", "claim"):
            # Validate required causal parameters
            if not all([treatment, outcome, confounders, causal_graph_path]):
                raise ValueError(
                    "ATE/CLAIM mode requires treatment, outcome, confounders, and causal_graph_path"
                )
            # Load GML causal graph
            with open(causal_graph_path, 'r') as f:
                self.causal_graph = f.read()

    def compute_ate_from_data(self, df):
        """Compute ATE using DoWhy's backdoor adjustment.

        The outcome column is clipped into `self.ate_outcome_range` first, so
        the `ate_outcome_range`-derived sensitivity used by
        `_release_private_reference_ate` holds by construction rather than by
        caller discipline. If clipping actually changes any values, a warning
        is raised, since it means the outcome's natural scale doesn't match
        `ate_outcome_range` and the resulting ATE is computed on distorted
        data (rescale the outcome yourself, or override `ate_outcome_range`,
        to avoid this).

        Args:
            df: DataFrame containing treatment, outcome, and confounder columns.

        Returns:
            float: Estimated ATE value.
        """
        # Late import to avoid dependency when not using ATE mode
        from dowhy import CausalModel

        lo, hi = self.ate_outcome_range
        outcome_col = df[self.outcome]
        clipped = outcome_col.clip(lower=lo, upper=hi)
        if not clipped.equals(outcome_col):
            warnings.warn(
                f"Outcome '{self.outcome}' has values outside ate_outcome_range="
                f"{self.ate_outcome_range}; clipping into range. This distorts "
                "the ATE estimate and the sensitivity assumed for private "
                "reference-ATE release -- rescale the outcome to match "
                "ate_outcome_range, or override ate_outcome_range to match "
                "the outcome's true scale.",
                RuntimeWarning,
            )
            df = df.copy()
            df[self.outcome] = clipped

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
        true_ate = self.reference_ate

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


    def _ate_error_for_model(self, model, seed=42):
        """Compute absolute ATE error for the current PGM model."""
        if self.reference_ate is None:
            raise ValueError("A reference/true ATE is required for ATE error computation")
        df = self._model_to_dataframe(model, self.sim_sample_size, seed=seed)
        ate = self.compute_ate_from_data(df)
        return abs(self.reference_ate - ate), ate

    def _tvd_error_for_model(self, model, measurements, cliques):
        """Average total variation distance between the model and the DP-released marginals.

        This compares against the noisy `noisy_measurement` already released for
        each clique in `measurements`, not the true raw marginal. That noisy
        value's privacy cost was already paid for and accounted in `rho_used`
        when it was measured, so reading it back here is pure post-processing
        and does not touch sensitive data directly.

        Args:
            model: Current fitted PGM model.
            measurements: List of LinearMeasurement objects released so far.
            cliques: Cliques to average the TVD over (e.g. the one-way marginals).
                Only cliques with a matching entry in `measurements` are used.

        Returns:
            float: Mean (approximate) TVD across the matched cliques.
        """
        noisy_by_clique = {m.clique: m.noisy_measurement for m in measurements if m.clique in cliques}
        used_cliques = [cl for cl in cliques if cl in noisy_by_clique]
        if not used_cliques:
            raise ValueError("no noisy measurements available for the requested TVD cliques")

        total = 0.0
        for cl in used_cliques:
            y = noisy_by_clique[cl]
            xest = model.project(cl).datavector()
            total += 0.5 * np.linalg.norm(y - xest, 1) / y.sum()
        return total / len(used_cliques)

    def _adjust_lambda(self, current_lambda, tvd_error, ate_error):
        """Combine last iteration's TVD and ATE errors into a new lambda_weight.

        Both errors are judged against their own tolerance. Lambda only moves
        toward the causal term (decreases) when TVD is already within tolerance,
        i.e. there is fidelity "slack" to trade for a causal improvement. If TVD
        is not preserved, lambda moves toward (or stays at) the statistical term
        regardless of how large the ATE error is, so TVD preservation can no
        longer be sacrificed without bound to chase ATE. When neither error is
        within tolerance, lambda moves toward whichever objective is relatively
        further from its own tolerance.
        """
        ate_ok = ate_error <= self.ate_tolerance
        tvd_ok = tvd_error <= self.tvd_tolerance

        if ate_ok and tvd_ok:
            return current_lambda
        if tvd_ok and not ate_ok:
            return max(self.lambda_min, current_lambda / self.lambda_update_factor)
        if ate_ok and not tvd_ok:
            return min(self.lambda_max, current_lambda * self.lambda_update_factor)

        # Neither preserved: favor whichever is relatively worse against its tolerance.
        ate_ratio = ate_error / self.ate_tolerance
        tvd_ratio = tvd_error / self.tvd_tolerance
        if tvd_ratio >= ate_ratio:
            return min(self.lambda_max, current_lambda * self.lambda_update_factor)
        return max(self.lambda_min, current_lambda / self.lambda_update_factor)

    def _maybe_update_lambda(self, model, measurements, cliques):
        """Update lambda_weight using both ATE error and TVD error from the last iteration.

        Smaller lambda means more causal weight. See `_adjust_lambda` for the
        precise rule that keeps TVD preservation from being sacrificed without
        bound while still letting lambda decrease when there is fidelity slack.

        Note: the ATE side of this still relies on `self.reference_ate`, which
        defaults to a non-private (raw-data) computation unless a DP-released
        reference_ate is passed in explicitly (see `run()`). The TVD side is
        private post-processing, since it only reads already-released noisy
        measurements.
        """
        if not self.adaptive_lambda or self.selection_mode != "claim":
            return
        if self.ate_tolerance is None:
            raise ValueError("adaptive lambda requires ate_tolerance")
        if self.tvd_tolerance is None:
            raise ValueError("adaptive lambda requires tvd_tolerance")

        current_ate_error, current_ate = self._ate_error_for_model(model, seed=42)
        current_tvd_error = self._tvd_error_for_model(model, measurements, cliques)
        old_lambda = self.lambda_weight
        self.lambda_weight = self._adjust_lambda(
            old_lambda, current_tvd_error, current_ate_error
        )
        print(
            "Adaptive lambda: "
            f"ATE error={current_ate_error:.6f} (tol={self.ate_tolerance:.4f}), "
            f"model ATE={current_ate:.6f}, "
            f"TVD error={current_tvd_error:.6f} (tol={self.tvd_tolerance:.4f}), "
            f"lambda {old_lambda:.4f} -> {self.lambda_weight:.4f}"
        )

    def _model_treatment_variance(self, model):
        """Residual treatment variance v = E_{p_hat}[(T - E_hat[T|Z])^2] under the model.

        This is the nuisance parameter v_i from the FWL sensitivity analysis,
        computed from the current PGM model's marginal over (treatment,
        confounders). It is a function of DP-released quantities only, so it
        is public. Treatment values are taken to be their (integer) domain
        indices, which for a binary 0/1 treatment gives
        v = E_z[ P(T=1|z) * (1 - P(T=1|z)) ].

        Returns:
            float: The residual treatment variance under the model.
        """
        cl = tuple([self.treatment] + list(self.confounders))
        counts = np.asarray(model.project(cl).datavector(), dtype=float)
        shape = [model.domain.size((a,)) for a in cl]
        joint = counts.reshape(shape)
        joint = np.clip(joint, 0.0, None)
        total = joint.sum()
        if total <= 0:
            return 0.0
        p = joint / total  # p(t, z), treatment on axis 0
        t_vals = np.arange(shape[0], dtype=float).reshape([-1] + [1] * (len(shape) - 1))
        pz = p.sum(axis=0)  # p(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            tbar = np.where(pz > 0, (t_vals * p).sum(axis=0) / pz, 0.0)  # E[T|z]
        v = float(np.sum(p * (t_vals - tbar) ** 2))
        return v

    def _model_fwl_ate(self, model):
        """FWL linear ATE estimator evaluated on the model itself (public).

        Computes tau_fwl = E_p[Y * (T - E_p[T|Z])] / v over the model's
        marginal on (treatment, outcome, confounders), with the nuisances
        (conditional treatment mean and residual treatment variance v) also
        taken from the model. Treatment and outcome values are taken to be
        their integer domain indices, consistent with
        _model_treatment_variance. This is the public proxy tau_hat^r(p_hat)
        used by the mu_t computation; it reads only the DP-released model.

        Returns:
            float: The FWL linear ATE estimate under the model.
        """
        cl = tuple([self.treatment, self.outcome] + list(self.confounders))
        counts = np.asarray(model.project(cl).datavector(), dtype=float)
        shape = [model.domain.size((a,)) for a in cl]
        joint = np.clip(counts.reshape(shape), 0.0, None)
        total = joint.sum()
        if total <= 0:
            return 0.0
        p = joint / total  # p(t, y, z...), treatment axis 0, outcome axis 1
        t_vals = np.arange(shape[0], dtype=float).reshape([-1, 1] + [1] * (len(shape) - 2))
        y_vals = np.arange(shape[1], dtype=float).reshape([1, -1] + [1] * (len(shape) - 2))
        pz = p.sum(axis=(0, 1))  # p(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            tbar = np.where(pz > 0, (t_vals * p).sum(axis=(0, 1)) / pz, 0.0)  # E[T|z]
        v = max(self._model_treatment_variance(model), self.kappa_eta)
        return float(np.sum(p * y_vals * (t_vals - tbar)) / v)

    def _compute_mu(self, candidates, model, sigma, current_ate):
        """Theory-derived public scale parameter mu_t.

        Implements the paper's definition
            mu_t = median_r |L_hat_r| / (kappa * median_r |A_hat_r| + mu_eta),
        where the proxies substitute released models for the private data:
        - L_hat_r = ||M_r(p_hat_{t-1}) - M_r(p_hat_{t-2})||_1 - bias, the
          statistical term with the current and previous round's models in
          place of (data, model). On the first round p_hat_{t-2} is taken to
          be p_hat_{t-1}, so |L_hat_r| reduces to the noise bias.
        - A_hat_r = |ref - tau(p_hat_{t-1})| - |ref - tau_fwl(p_hat_{t-1})|
          for candidates covering {treatment, outcome} + confounders, and 0
          otherwise (mirroring the FWL estimator's covering condition).
        Everything here is post-processing of DP-released quantities, so mu_t
        is public and the score's sensitivity analysis is unaffected.
        """
        prev_model = self._prev_model if self._prev_model is not None else model
        stat_proxies = []
        for cl in candidates:
            x_new = model.project(cl).datavector()
            x_old = prev_model.project(cl).datavector()
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            stat_proxies.append(abs(np.linalg.norm(x_new - x_old, 1) - bias))

        causal_clique = set([self.treatment, self.outcome] + list(self.confounders))
        tau_fwl = self._model_fwl_ate(model)
        a_cover = abs(
            abs(self.reference_ate - current_ate) - abs(self.reference_ate - tau_fwl)
        )
        causal_proxies = [
            a_cover if causal_clique <= set(cl) else 0.0 for cl in candidates
        ]
        mu = float(
            np.median(stat_proxies)
            / (self.kappa * np.median(causal_proxies) + self.mu_eta)
        )
        return mu

    def _claim_candidate_scores(self, candidates, answers, data, model, measurements, sigma):
        """Compute blended CLAIM scores for candidate cliques.

        The statistical term is AIM's L1 marginal error minus expected Gaussian
        noise bias.  The causal term is the predicted ATE-error improvement from
        simulating that candidate measurement.  Higher is better.
        """
        if self.reference_ate is None:
            raise ValueError("CLAIM mode requires a reference ATE")

        current_error, current_ate = self._ate_error_for_model(model, seed=42)
        print(
            f"Current ATE error: {current_error:.6f} "
            f"(reference={self.reference_ate:.4f}, model={current_ate:.4f})"
        )

        # Theory-derived kappa = (sum_i beta_i / v_i)^{-1}; with the single
        # causal query here (beta = 1) this is the residual treatment variance
        # under the current model, floored at kappa_eta (degenerate case v=0).
        self.kappa = max(self._model_treatment_variance(model), self.kappa_eta)
        print(f"Theory-derived kappa (residual treatment variance): {self.kappa:.6g}")

        # Theory-derived mu_t: median statistical proxy over median causal
        # proxy (see _compute_mu). Public post-processing of released models.
        self.mu = self._compute_mu(candidates, model, sigma, current_ate)
        print(f"Theory-derived mu_t (median stat/causal proxy ratio): {self.mu:.6g}")

        scores = {}
        stat_scores = {}
        causal_scores = {}
        lam = self.lambda_weight
        denom = lam + (1.0 - lam) * self.mu

        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            xest = model.project(cl).datavector()
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            stat = wgt * (np.linalg.norm(x - xest, 1) - bias)
            stat_scores[cl] = stat

            causal = 0.0
            try:
                simulated_model = self._simulate_measurement(model, data, cl, measurements)
                simulated_error, simulated_ate = self._ate_error_for_model(simulated_model, seed=42)
                causal = current_error - simulated_error
                if causal > 0:
                    print(
                        f"  Candidate {cl}: ATE error={simulated_error:.6f}, "
                        f"improvement={causal:.6f}"
                    )
            except Exception as exc:
                print(f"  Candidate {cl}: ATE simulation failed ({exc})")
            causal_scores[cl] = causal
            scores[cl] = (lam * stat + (1.0 - lam) * self.mu * self.kappa * causal) / denom

        return scores, stat_scores, causal_scores

    def claim_approximated(self, candidates, answers, data, model, measurements, eps, sigma):
        """Blended CLAIM selection with the exponential mechanism."""
        scores, _, _ = self._claim_candidate_scores(
            candidates, answers, data, model, measurements, sigma
        )
        # This follows the same practical sensitivity scale as AIM's weighted
        # marginal score.  If the causal/reference ATE is computed directly from
        # private data, a formal proof must also bound that contribution.
        sensitivity = max(abs(wgt) for wgt in candidates.values())
        sensitivity = max(sensitivity, 1e-12)
        return self.exponential_mechanism(scores, eps, sensitivity)

    def _release_private_reference_ate(self, data):
        """Auto-release a one-shot private reference ATE from a slice of rho.

        Spends `reference_ate_rho_fraction * self.rho` of the total zCDP
        budget on a single Gaussian-mechanism release of the raw-data ATE
        estimate. Calibrated using `self.ate_sensitivity` as the assumed L2
        sensitivity of the estimator under one record changing, defaulting to
        `(ate_outcome_range[1] - ate_outcome_range[0]) / data.records` when
        not set explicitly (the sensitivity of a difference-in-means
        estimator over an outcome bounded to `ate_outcome_range`, which
        `compute_ate_from_data` enforces by clipping). The spent amount is
        deducted from `self.rho` so the selection rounds that follow see the
        reduced remaining budget; this is a genuine privacy expenditure, not
        post-processing.
        """
        sensitivity = self.ate_sensitivity
        if sensitivity is None:
            lo, hi = self.ate_outcome_range
            sensitivity = (hi - lo) / data.records
        ate_rho = self.reference_ate_rho_fraction * self.rho
        raw_ate = self.compute_ate_from_data(data.df)
        sigma_ate = sensitivity / np.sqrt(2 * ate_rho)
        private_ate = float(raw_ate + self.gaussian_noise(sigma_ate, 1)[0])
        self.rho -= ate_rho
        print(
            "Auto-released private reference ATE: "
            f"{private_ate:.6f} (raw={raw_ate:.6f}, sensitivity={sensitivity:.6g}, "
            f"rho spent={ate_rho:.6g}, sigma={sigma_ate:.6g}, "
            f"remaining rho={self.rho:.6g})"
        )
        return private_ate

    def run(self, data, workload, num_synth_rows=None, initial_cliques=None):
        # Cache reference ATE at the start if using ATE/CLAIM mode. If
        # reference_ate is omitted, one is auto-released privately from a
        # fraction of rho instead of being computed non-privately.
        if self.selection_mode in ("ate", "claim"):
            if self.reference_ate is None:
                self.reference_ate = self._release_private_reference_ate(data)
            print(f"Reference ATE: {self.reference_ate:.6f}")

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
            elif self.selection_mode == "claim":
                cl = self.claim_approximated(
                    small_candidates, answers, data, model, measurements, epsilon, sigma
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
            # Keep the pre-update model (p_hat_{t-2} for the next round's
            # selection): it is used by the public mu_t proxies.
            self._prev_model = model
            model = estimation.mirror_descent(
                    data.domain, measurements, iters=self.max_iters, potentials=potentials, callback_fn=lambda *_: None
            )

            # Optional one-run adaptive lambda update. This happens after the
            # model update, so lambda_{t+1} reacts to the newest DP transcript.
            # Decreasing lambda gives more weight to the causal term; increasing
            # lambda gives more weight to the statistical/marginal term.
            self._maybe_update_lambda(model, measurements, oneway)

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



def _epsilon_for_rho(rho, delta):
    """Return an epsilon whose cdp_rho(epsilon, delta) is approximately rho."""
    if rho <= 0:
        return 0.0
    return cdp_eps(rho, delta)


def pilot_select_lambda(
    data,
    workload,
    epsilon,
    delta,
    lambda_values=(0.1, 0.5, 0.9),
    pilot_fraction=0.15,
    pilot_samples=5,
    num_synth_rows=None,
    **claim_kwargs,
):
    """Select lambda with cheap pilot CLAIM runs, then run the final CLAIM.

    The procedure spends a small zCDP budget on one pilot run per lambda, scores
    each pilot model by average synthetic ATE error, and then spends the remaining
    budget on a final run with the selected lambda.

    If `reference_ate` is supplied in `claim_kwargs`, scoring is post-processing
    of DP outputs.  If it is omitted, the reference ATE is computed from the raw
    data for experimental use only and should not be claimed as private without
    additional accounting.
    """
    if not lambda_values:
        raise ValueError("lambda_values must contain at least one value")
    if not (0.0 < pilot_fraction < 1.0):
        raise ValueError("pilot_fraction must be in (0, 1)")

    total_rho = cdp_rho(epsilon, delta)
    pilot_rho_each = total_rho * pilot_fraction / len(lambda_values)
    final_rho = total_rho * (1.0 - pilot_fraction)
    pilot_epsilon = _epsilon_for_rho(pilot_rho_each, delta)
    final_epsilon = _epsilon_for_rho(final_rho, delta)

    reference_ate = claim_kwargs.get("reference_ate")
    if reference_ate is None:
        tmp = CLAIM(epsilon, delta, **claim_kwargs)
        warnings.warn(
            "Computing pilot reference ATE directly from private data. "
            "For a formal privacy guarantee, pass a DP reference_ate.",
            RuntimeWarning,
        )
        reference_ate = tmp.compute_ate_from_data(data.df)
        claim_kwargs["reference_ate"] = reference_ate

    base_claim_kwargs = dict(claim_kwargs)
    base_claim_kwargs.pop("lambda_weight", None)
    base_claim_kwargs.pop("adaptive_lambda", None)
    base_claim_kwargs.pop("lambda_schedule", None)

    scores = {}
    pilot_outputs = {}
    for lam in lambda_values:
        print(f"\n[Pilot lambda={lam}] budget rho={pilot_rho_each:.6g}")
        mech = CLAIM(
            pilot_epsilon,
            delta,
            lambda_weight=lam,
            adaptive_lambda=False,
            **base_claim_kwargs,
        )
        model, synth = mech.run(data, workload, num_synth_rows=num_synth_rows)
        errors = []
        for b in range(pilot_samples):
            rows = num_synth_rows or data.records
            pilot_df = model.synthetic_data(rows=rows).df
            pilot_ate = mech.compute_ate_from_data(pilot_df)
            errors.append(abs(reference_ate - pilot_ate))
        avg_error = float(np.mean(errors))
        scores[lam] = avg_error
        pilot_outputs[lam] = (model, synth)
        print(f"[Pilot lambda={lam}] average ATE error={avg_error:.6f}")

    selected_lambda = min(scores, key=scores.get)
    print(f"\nSelected lambda={selected_lambda} with pilot ATE error={scores[selected_lambda]:.6f}")
    print(f"[Final lambda={selected_lambda}] budget rho={final_rho:.6g}")

    final_mech = CLAIM(
        final_epsilon,
        delta,
        lambda_weight=selected_lambda,
        adaptive_lambda=False,
        **base_claim_kwargs,
    )
    final_model, final_synth = final_mech.run(
        data, workload, num_synth_rows=num_synth_rows
    )
    return selected_lambda, scores, final_model, final_synth

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
    # Selection mode: "marginal", "ate", or "claim"
    params["selection_mode"] = "marginal"
    params["lambda_weight"] = 0.5
    params["mu_eta"] = 1e-6
    params["kappa_eta"] = 1e-6
    params["adaptive_lambda"] = False
    params["lambda_min"] = 0.1
    params["lambda_max"] = 0.9
    params["lambda_update_factor"] = 1.25
    params["ate_tolerance"] = 0.01
    params["tvd_tolerance"] = 0.05
    params["reference_ate"] = None
    params["ate_sensitivity"] = None
    params["reference_ate_rho_fraction"] = 0.05
    params["ate_outcome_min"] = -1.0
    params["ate_outcome_max"] = 1.0
    params["fixed_lambda_from_list"] = False
    params["lambda_values"] = "0.1,0.5,0.9"
    params["pilot_fraction"] = 0.15
    params["pilot_samples"] = 5
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
        choices=["marginal", "ate", "claim"],
        help="Selection mode: 'marginal' (L1-based), 'ate' (ATE-based), or 'claim' (blended)"
    )

    parser.add_argument(
        "--lambda_weight",
        type=float,
        help="Weight on marginal/statistical term; smaller gives more causal weight"
    )
    parser.add_argument(
        "--mu_eta",
        type=float,
        help="Small positive constant preventing division by zero in the "
        "theory-derived scale parameter mu (mu itself is computed each round "
        "as the ratio of median statistical to median causal proxy magnitudes "
        "and is no longer a free parameter)"
    )
    parser.add_argument(
        "--kappa_eta",
        type=float,
        help="Variance floor for the theory-derived normalizer kappa "
        "(kappa itself is computed each round as the model's residual "
        "treatment variance and is no longer a free parameter)"
    )
    parser.add_argument(
        "--adaptive_lambda",
        action="store_true",
        help="Adapt lambda during one CLAIM run using the current model ATE error"
    )
    parser.add_argument("--lambda_min", type=float, help="Minimum adaptive lambda")
    parser.add_argument("--lambda_max", type=float, help="Maximum adaptive lambda")
    parser.add_argument(
        "--lambda_update_factor",
        type=float,
        help="Multiplicative adaptive lambda update factor"
    )
    parser.add_argument(
        "--ate-tolerance",
        "--ate_tolerance",
        dest="ate_tolerance",
        type=float,
        help="ATE-error tolerance for adaptive lambda"
    )
    parser.add_argument(
        "--tvd-tolerance",
        "--tvd_tolerance",
        dest="tvd_tolerance",
        type=float,
        help="TVD-error tolerance for adaptive lambda"
    )
    parser.add_argument(
        "--reference_ate",
        type=float,
        help="Precomputed/privately-released reference ATE. If omitted in ATE/CLAIM mode, one is auto-released privately from a fraction of rho instead."
    )
    parser.add_argument(
        "--ate_sensitivity",
        type=float,
        help="Assumed L2 sensitivity of the ATE estimator, used when auto-releasing a private reference ATE. Defaults to (ate_outcome_max - ate_outcome_min)/records if omitted."
    )
    parser.add_argument(
        "--reference_ate_rho_fraction",
        type=float,
        help="Fraction of total zCDP budget spent auto-releasing a private reference ATE (used only when --reference_ate is omitted)"
    )
    parser.add_argument(
        "--ate_outcome_min",
        type=float,
        help="Minimum value the outcome column is clipped to before ATE computation (enforces the sensitivity assumption)"
    )
    parser.add_argument(
        "--ate_outcome_max",
        type=float,
        help="Maximum value the outcome column is clipped to before ATE computation (enforces the sensitivity assumption)"
    )
    parser.add_argument(
        "--fixed_lambda_from_list",
        action="store_true",
        help="Select lambda from --lambda_values using small pilot runs before the final run"
    )
    parser.add_argument(
        "--lambda_values",
        type=str,
        help="Comma-separated lambda grid for pilot selection, e.g. 0.1,0.5,0.9"
    )
    parser.add_argument(
        "--pilot_fraction",
        type=float,
        help="Fraction of zCDP budget spent on all pilot lambda runs"
    )
    parser.add_argument(
        "--pilot_samples",
        type=int,
        help="Number of synthetic samples used to average pilot ATE error"
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

    if args.fixed_lambda_from_list and args.adaptive_lambda:
        raise ValueError("Use either --fixed_lambda_from_list or --adaptive_lambda, not both")

    claim_kwargs = dict(
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
        selection_mode=args.selection_mode,
        treatment=args.treatment,
        outcome=args.outcome,
        confounders=confounders_list,
        causal_graph_path=args.causal_graph,
        lambda_weight=args.lambda_weight,
        mu_eta=args.mu_eta,
        kappa_eta=args.kappa_eta,
        adaptive_lambda=args.adaptive_lambda,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        lambda_update_factor=args.lambda_update_factor,
        ate_tolerance=args.ate_tolerance,
        tvd_tolerance=args.tvd_tolerance,
        reference_ate=args.reference_ate,
        ate_sensitivity=args.ate_sensitivity,
        reference_ate_rho_fraction=args.reference_ate_rho_fraction,
        ate_outcome_range=(args.ate_outcome_min, args.ate_outcome_max),
    )

    selected_lambda = None
    lambda_scores = None
    if args.fixed_lambda_from_list:
        lambda_values = [float(x.strip()) for x in args.lambda_values.split(',') if x.strip()]
        selected_lambda, lambda_scores, model, synth = pilot_select_lambda(
            data,
            workload,
            args.epsilon,
            args.delta,
            lambda_values=lambda_values,
            pilot_fraction=args.pilot_fraction,
            pilot_samples=args.pilot_samples,
            **claim_kwargs,
        )
        mech = CLAIM(
            args.epsilon,
            args.delta,
            **{**claim_kwargs, "lambda_weight": selected_lambda, "adaptive_lambda": False},
        )
    else:
        mech = CLAIM(args.epsilon, args.delta, **claim_kwargs)
        model, synth = mech.run(data, workload)

    if args.save is not None:
        synth.df.to_csv(args.save, index=False)

    if selected_lambda is not None:
        print(f"Selected Lambda: {selected_lambda}")
        print(f"Pilot Lambda Scores: {lambda_scores}")

    # Print ATE comparison if in ATE/CLAIM mode
    if args.selection_mode in ("ate", "claim"):
        true_ate = mech.reference_ate
        if true_ate is None:
            true_ate = mech.compute_ate_from_data(data.df)
        synth_ate = mech.compute_ate_from_data(synth.df)
        print(f"Reference ATE: {true_ate:.6f}")
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

