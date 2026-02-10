#!/usr/bin/env python
"""Sample script for generating synthetic datasets using CLAIM from CBI library.

This script uses the mechanisms/aim.py module (CLAIM class) from the CBI library.
It handles preprocessing of adult.csv and generation of multiple synthetic datasets.

CLAIM = Causally-Learned Adaptive and Iterative Mechanism

Requirements:
    - Virtual environment with JAX, MBI (CBI), DoWhy installed
    - adult.csv and causal_graph.dot in the same directory

Usage:
    source /path/to/your/venv/bin/activate
    python sampleAdult_CLAIM.py --num_synth_datasets 100
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd

# Add mechanisms directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'mechanisms'))

from mbi import Dataset, Domain
from aim import CLAIM


# ============================================================================
# Preprocessing
# ============================================================================

class AdultPreprocessor:
    """Preprocess adult.csv for use with CLAIM."""
    
    CATEGORICAL_COLS = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'gender', 'native-country', 'income'
    ]
    
    BINNING_CONFIG = {
        'age': 85,
        'fnlwgt': 100,
        'educational-num': 16,
        'capital-gain': 100,
        'capital-loss': 100,
        'hours-per-week': 99,
    }
    
    def __init__(self):
        self.encoders = {}
        self.bin_edges = {}
        self.domain_config = {}
        self.age_min = None
        
    def fit_transform(self, df):
        """Fit encoders and transform data to integer-encoded format."""
        result = pd.DataFrame()
        
        for col in df.columns:
            if col in self.CATEGORICAL_COLS:
                df[col] = df[col].astype(str)
                unique_vals = sorted(df[col].unique())
                self.encoders[col] = {val: i for i, val in enumerate(unique_vals)}
                result[col] = df[col].map(self.encoders[col]).astype(int)
                self.domain_config[col] = len(unique_vals)
                
            elif col in self.BINNING_CONFIG:
                n_bins = self.BINNING_CONFIG[col]
                if col == 'age':
                    self.age_min = int(df[col].min())
                    result[col] = (df[col] - self.age_min).clip(0, n_bins - 1).astype(int)
                    self.domain_config[col] = n_bins
                elif col == 'educational-num':
                    result[col] = (df[col] - 1).clip(0, n_bins - 1).astype(int)
                    self.domain_config[col] = n_bins
                elif col == 'hours-per-week':
                    result[col] = (df[col] - 1).clip(0, n_bins - 1).astype(int)
                    self.domain_config[col] = n_bins
                else:
                    if df[col].nunique() <= n_bins:
                        result[col] = pd.qcut(df[col], q=min(n_bins, df[col].nunique()),
                                              labels=False, duplicates='drop')
                    else:
                        result[col] = pd.qcut(df[col].rank(method='first'), q=n_bins,
                                              labels=False)
                    result[col] = result[col].fillna(0).astype(int)
                    self.domain_config[col] = n_bins
            else:
                print(f"Warning: Unknown column {col}")
                
        return result
    
    def inverse_transform(self, df, original_df=None):
        """Convert integer-encoded data back to original format.
        
        This is the inverse of fit_transform.
        """
        result = pd.DataFrame()
        
        for col in df.columns:
            # Check categorical columns first (same order as fit_transform)
            if col in self.CATEGORICAL_COLS:
                if col in self.encoders:
                    inverse_encoder = {v: k for k, v in self.encoders[col].items()}
                    result[col] = df[col].apply(
                        lambda x: inverse_encoder.get(int(x) % len(inverse_encoder), 'Unknown')
                    )
                else:
                    # Encoder not found - keep as-is
                    result[col] = df[col]
                    
            elif col in self.BINNING_CONFIG:
                if col == 'age':
                    # Inverse of: (value - age_min).clip(0, n_bins-1)
                    result[col] = df[col].astype(int) + (self.age_min if self.age_min else 17)
                elif col == 'educational-num':
                    # Inverse of: (value - 1).clip(0, n_bins-1)
                    result[col] = df[col].astype(int) + 1
                elif col == 'hours-per-week':
                    # Inverse of: (value - 1).clip(0, n_bins-1)
                    result[col] = df[col].astype(int) + 1
                else:
                    # For other binned columns, keep encoded value
                    result[col] = df[col].astype(int)
            else:
                # Keep as-is
                result[col] = df[col]
                
        return result
    
    def save_domain(self, path):
        """Save domain configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.domain_config, f, indent=2)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic datasets using CLAIM from CBI library',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', type=str, default='adult.csv',
                        help='Path to adult.csv')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Privacy budget epsilon')
    parser.add_argument('--delta', type=float, default=1e-9,
                        help='Privacy budget delta')
    parser.add_argument('--degree', type=int, default=2,
                        help='Degree of marginals')
    parser.add_argument('--max_cells', type=int, default=10000,
                        help='Maximum cells per marginal')
    parser.add_argument('--max_iters', type=int, default=100,
                        help='Maximum iterations for mirror descent (lower = faster/less memory)')
    parser.add_argument('--max_model_size', type=float, default=20,
                        help='Maximum model size in MB (lower = faster/less memory)')
    parser.add_argument('--num_synth_datasets', type=int, default=100,
                        help='Number of synthetic datasets to generate')
    parser.add_argument('--output_dir', type=str, default='CLAIM_syntheticData',
                        help='Output directory for synthetic data')
    parser.add_argument('--marginals', type=str, nargs='+', default=None,
                        help='Custom marginals (e.g., "age" "age,gender" "age,gender,income")')
    
    # Selection mode
    parser.add_argument('--selection_mode', type=str, default='ate',
                        choices=['marginal', 'ate'],
                        help='Selection mode: "marginal" (L1-based) or "ate" (ATE-based)')
    
    # Multi-ATE configuration
    parser.add_argument('--ate_configs_file', type=str, default=None,
                        help='Path to JSON file with ATE configurations (overrides defaults)')
    parser.add_argument('--causal_graph', type=str, default='causal_graph.dot',
                        help='Path to causal graph DOT file')
    
    # Hybrid selection (ATE mode only)
    parser.add_argument('--marginal_weight', type=float, default=0.3,
                        help='Weight for L1 marginal error in hybrid selection '
                             '(0.0 = pure ATE, 1.0 = pure marginal). Only used in ATE mode.')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CLAIM Synthetic Data Generation (CBI Library)")
    print("Causally-Learned Adaptive and Iterative Mechanism")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Preprocess data
    print("\n[1/4] Preprocessing data...")
    df = pd.read_csv(args.data)
    print(f"  Original records: {len(df)}")
    
    # Define default ATE configs for Adult dataset
    # ATE 1: marital-status (Married-civ-spouse) -> income
    # ATE 2: educational-num (>9) -> income
    if args.ate_configs_file:
        import json
        with open(args.ate_configs_file, 'r') as f:
            ate_configs = json.load(f)
        print(f"  Loaded ATE configs from {args.ate_configs_file}")
    else:
        # Default: two ATEs with equal weight, including binarization config
        ate_configs = [
            {
                "name": "marital_status_ate",
                "treatment": "marital-status",
                "outcome": "income",
                "confounders": ["age", "gender", "native-country"],
                "alpha": 0.5,
                "binarization": {
                    "treatment": {"type": "value", "positive_value": 1},  # Encoded: Married-civ-spouse = 1
                    "outcome": {"type": "value", "positive_value": 1}      # Encoded: >50K = 1
                }
            },
            {
                "name": "educational_num_ate",
                "treatment": "educational-num",
                "outcome": "income",
                "confounders": ["age", "gender", "native-country"],
                "alpha": 0.5,
                "binarization": {
                    "treatment": {"type": "threshold", "threshold": 8, "comparison": ">"},  # Encoded: >8 means original >9
                    "outcome": {"type": "value", "positive_value": 1}
                }
            }
        ]
        print("  Using default ATE configs with binarization: marital-status (α=0.5) + educational-num (α=0.5)")
    
    print(f"\nConfiguration:")
    print(f"  Selection Mode: {args.selection_mode}")
    print(f"  Epsilon: {args.epsilon}")
    if args.selection_mode == 'ate':
        print(f"  ATE Configurations:")
        for config in ate_configs:
            print(f"    {config['name']}: T={config['treatment']}, Y={config['outcome']}, α={config['alpha']}")
        print(f"  Marginal Weight (λ): {args.marginal_weight}")
    
    preprocessor = AdultPreprocessor()
    df_encoded = preprocessor.fit_transform(df)
    
    # Save domain
    domain_path = os.path.join(args.output_dir, 'domain.json')
    preprocessor.save_domain(domain_path)
    
    # Save encoded data
    encoded_path = os.path.join(args.output_dir, 'adult_encoded.csv')
    df_encoded.to_csv(encoded_path, index=False)
    print(f"  Saved encoded data to {encoded_path}")
    
    # Step 2: Create Dataset
    print("\n[2/4] Creating MBI Dataset...")
    domain = Domain.fromdict(preprocessor.domain_config)
    print(f"  Domain: {domain}")
    
    data = Dataset.load(encoded_path, domain)
    print(f"  Records: {data.records}")
    
    # Step 3: Create workload and run CLAIM
    print(f"\n[3/4] Running CLAIM (selection_mode={args.selection_mode})...")
    import itertools
    
    if args.marginals:
        # Custom marginals specified
        workload = []
        for m in args.marginals:
            cl = tuple(c.strip() for c in m.split(','))
            workload.append((cl, 1.0))
        print(f"  Custom workload: {len(workload)} marginals")
        for cl, _ in workload:
            print(f"    {cl}")
    else:
        # Default: use all degree-way marginals
        workload = list(itertools.combinations(domain.attrs, args.degree))
        workload = [cl for cl in workload if domain.size(cl) <= args.max_cells]
        workload = [(cl, 1.0) for cl in workload]
        print(f"  Workload: {len(workload)} {args.degree}-way marginals")
    
    # Create CLAIM mechanism with appropriate mode
    if args.selection_mode == 'ate':
        mech = CLAIM(
            epsilon=args.epsilon,
            delta=args.delta,
            selection_mode='ate',
            ate_configs=ate_configs,
            causal_graph_path=args.causal_graph,
            max_model_size=args.max_model_size,
            max_iters=args.max_iters,
            marginal_weight=args.marginal_weight,
        )
    else:
        mech = CLAIM(
            epsilon=args.epsilon,
            delta=args.delta,
            selection_mode='marginal',
            max_model_size=args.max_model_size,
            max_iters=args.max_iters,
        )
    
    # Run to get model and one synthetic dataset
    model, synth = mech.run(data, workload)
    
    # Step 4: Generate multiple synthetic datasets
    print(f"\n[4/4] Generating {args.num_synth_datasets} synthetic datasets...")
    
    for i in range(1, args.num_synth_datasets + 1):
        np.random.seed(i)
        synth_data = model.synthetic_data(rows=data.records)
        synth_original = preprocessor.inverse_transform(synth_data.df, original_df=df)
        
        output_path = os.path.join(args.output_dir, f'synthetic_adult_{i}.csv')
        synth_original.to_csv(output_path, index=False)
        
        if i % 10 == 0 or i == 1:
            print(f"  Generated {i}/{args.num_synth_datasets}")
    
    # Final comparison
    print("\n" + "="*60)
    if args.selection_mode == 'ate':
        print("MULTI-ATE COMPARISON")
        print("="*60)
        true_ates = mech._true_ates
        synth_ates = mech.compute_all_ates(synth.df)
        total_weighted_error = 0.0
        for config in ate_configs:
            name = config["name"]
            alpha = config["alpha"]
            true_val = true_ates[name]
            synth_val = synth_ates[name]
            error = abs(true_val - synth_val)
            weighted_error = alpha * error
            total_weighted_error += weighted_error
            print(f"  {name} (α={alpha}):")
            print(f"    True ATE:      {true_val:.6f}")
            print(f"    Synthetic ATE: {synth_val:.6f}")
            print(f"    Error:         {error:.6f}")
        print(f"\n  Total Weighted Error: {total_weighted_error:.6f}")
    else:
        print("MARGINAL MODE - No ATE comparison")
        print("="*60)
    
    print("\n" + "="*60)
    print(f"COMPLETE!")
    print(f"Generated {args.num_synth_datasets} datasets in {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

