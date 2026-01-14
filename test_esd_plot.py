#!/usr/bin/env python3
"""Quick diagnostic to test ESD plotting and data inspection."""

import sys
import os
sys.path.insert(0, 'algorithmic/pyhessian_patched')
sys.path.insert(0, 'algorithmic')

import torch
import matplotlib.pyplot as plt
from compute_hessian_analysis import load_model_and_config, create_hessian_dataloaders, compute_hessian_metrics

# Load model and compute Hessian
device = torch.device('cpu')
model_path = './lm-out-new-unique_copy/2l1h64d_n10000_run0'

print("Loading model...")
model, config, model_type = load_model_and_config(model_path, device)

print("\nCreating small dataset...")
dataloaders, tokenizer = create_hessian_dataloaders(
    'unique_copy',
    (1, 10),
    [(1, 10), (11, 20)],
    hessian_batch_size=50,
    max_samples=50,  # Very small for quick test
    train_seed=42,
    test_seed=1337
)

print("\nComputing Hessian metrics...")
dataloader = dataloaders['len1-10_train']
results = compute_hessian_metrics(model, dataloader, device, verbose=True)

print("\n" + "="*60)
print("DIAGNOSTIC RESULTS")
print("="*60)

print(f"\nTop eigenvalue: {results['top_eigenvalue']}")
print(f"Trace: {results['trace']}")

density_eigen = results['density_eigen']
density_weight = results['density_weight']

print(f"\nDensity eigenvalues type: {type(density_eigen)}")
print(f"Density weights type: {type(density_weight)}")

if isinstance(density_eigen, list):
    print(f"Number of runs (n_v): {len(density_eigen)}")
    if len(density_eigen) > 0:
        print(f"First run has {len(density_eigen[0])} eigenvalues")
        print(f"First 10 eigenvalues: {density_eigen[0][:10]}")
        print(f"First 10 weights: {density_weight[0][:10]}")

        # Check for issues
        if all(w == 0 for w in density_weight[0]):
            print("⚠ WARNING: All weights are zero!")
        if len(density_eigen[0]) == 0:
            print("⚠ WARNING: No eigenvalues computed!")

print("\nCreating test plot...")
plt.figure(figsize=(10, 6))

if isinstance(density_eigen, list) and len(density_eigen) > 0 and isinstance(density_eigen[0], list):
    for i, (eigen_vals, weights) in enumerate(zip(density_eigen, density_weight)):
        print(f"  Plotting run {i+1}: {len(eigen_vals)} points")
        print(f"    Eigenvalue range: [{min(eigen_vals):.2f}, {max(eigen_vals):.2f}]")
        print(f"    Weight range: [{min(weights):.6f}, {max(weights):.6f}]")
        plt.semilogy(eigen_vals, weights, linewidth=2, marker='o', markersize=4, alpha=0.7)

plt.xlabel('Eigenvalue', fontsize=14)
plt.ylabel('Density (log scale)', fontsize=14)
plt.title('Diagnostic ESD Plot', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = 'diagnostic_esd.pdf'
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"\nDiagnostic plot saved to: {output_path}")
print(f"File size: {os.path.getsize(output_path)} bytes")

# Also save as PNG for easier viewing
plt.savefig('diagnostic_esd.png', format='png', dpi=150, bbox_inches='tight')
print(f"PNG version: diagnostic_esd.png")

plt.close()
print("\n✓ Diagnostic complete!")
