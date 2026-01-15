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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mechanisms'))

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
        """Convert integer-encoded data back to original format."""
        result = pd.DataFrame()
        
        for col in df.columns:
            if col in self.encoders:
                inverse_encoder = {v: k for k, v in self.encoders[col].items()}
                result[col] = df[col].map(lambda x: inverse_encoder.get(int(x) % len(inverse_encoder), 'Unknown'))
                
            elif col in self.BINNING_CONFIG:
                if col == 'age':
                    result[col] = df[col].astype(int) + self.age_min
                elif col == 'educational-num':
                    result[col] = df[col].astype(int) + 1
                elif col == 'hours-per-week':
                    result[col] = df[col].astype(int) + 1
                else:
                    # For other binned columns, use random value within bin
                    result[col] = df[col].astype(int)
            else:
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
    
    # Causal structure arguments (required for ate mode)
    parser.add_argument('--treatment', type=str, default='marital-status',
                        help='Treatment variable name')
    parser.add_argument('--treatment_value', type=str, default='Married-civ-spouse',
                        help='Value of treatment that maps to 1 (others become 0)')
    parser.add_argument('--outcome', type=str, default='income',
                        help='Outcome variable name')
    parser.add_argument('--confounders', type=str, default='age,gender,native-country',
                        help='Comma-separated list of confounders')
    parser.add_argument('--causal_graph', type=str, default='causal_graph.dot',
                        help='Path to causal graph DOT file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CLAIM Synthetic Data Generation (CBI Library)")
    print("Causally-Learned Adaptive and Iterative Mechanism")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse confounders
    confounders_list = [c.strip() for c in args.confounders.split(',')]
    
    print(f"\nConfiguration:")
    print(f"  Selection Mode: {args.selection_mode}")
    print(f"  Epsilon: {args.epsilon}")
    if args.selection_mode == 'ate':
        print(f"  Treatment: {args.treatment}")
        print(f"  Outcome: {args.outcome}")
        print(f"  Confounders: {confounders_list}")
    
    # Step 1: Preprocess data
    print("\n[1/4] Preprocessing data...")
    df = pd.read_csv(args.data)
    print(f"  Original records: {len(df)}")
    
    # Binarize treatment column if in ATE mode
    if args.selection_mode == 'ate' and args.treatment_value:
        original_values = df[args.treatment].nunique()
        treated_count = (df[args.treatment].str.strip() == args.treatment_value).sum()
        df[args.treatment] = (df[args.treatment].str.strip() == args.treatment_value).astype(int)
        print(f"  Binarized '{args.treatment}': {treated_count} treated ('{args.treatment_value}'), {len(df) - treated_count} control")
    
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
            treatment=args.treatment,
            outcome=args.outcome,
            confounders=confounders_list,
            causal_graph_path=args.causal_graph,
            max_model_size=args.max_model_size,
            max_iters=args.max_iters,
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
        print("ATE COMPARISON")
        print("="*60)
        true_ate = mech._true_ate
        synth_ate = mech.compute_ate_from_data(synth.df)
        print(f"True ATE: {true_ate:.6f}")
        print(f"Synthetic ATE: {synth_ate:.6f}")
        print(f"ATE Error: {abs(true_ate - synth_ate):.6f}")
    else:
        print("MARGINAL MODE - No ATE comparison")
        print("="*60)
    
    print("\n" + "="*60)
    print(f"COMPLETE!")
    print(f"Generated {args.num_synth_datasets} datasets in {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
