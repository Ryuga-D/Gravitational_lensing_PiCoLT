import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.gridspec as gridspec

# Paths
DATASET_PATH = "picolt_dataset_enhanced/dataset.h5"
PARAMS_PATH = "picolt_dataset_enhanced/parameters.csv"

# Load dataset
with h5py.File(DATASET_PATH, "r") as f:
    lensed_images = f["lensed_images"][:]
    source_images = f["source_images"][:]

# Load parameters
params_df = pd.read_csv(PARAMS_PATH)

def visualize_strong_lenses(n_samples=5, min_theta_E=1.2, save_path=None):
    """
    Enhanced visualization of strong lensing samples with proper formatting
    """
    # Filter for strong lenses
    strong_lens_mask = params_df["lens_theta_E"] > min_theta_E
    strong_lens_indices = params_df[strong_lens_mask].index.tolist()
    
    if not strong_lens_indices:
        print(f"No lenses found with θ_E > {min_theta_E}")
        return
    
    print(f"Found {len(strong_lens_indices)} strong lens samples (θ_E > {min_theta_E})")
    
    # Select samples to visualize
    if len(strong_lens_indices) < n_samples:
        indices = strong_lens_indices
    else:
        indices = np.random.choice(strong_lens_indices, n_samples, replace=False)
    
    # Create figure with proper spacing
    fig = plt.figure(figsize=(15, 5 * n_samples))
    
    # Use GridSpec for better layout control
    gs = gridspec.GridSpec(n_samples, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    for i, idx in enumerate(indices):
        lensed_img = lensed_images[idx, :, :, 0]
        source_img = source_images[idx, :, :, 0]
        
        # Get parameters for this sample
        sample_params = params_df.iloc[idx]
        theta_E = sample_params['lens_theta_E']
        magnification = sample_params['magnification']
        source_type = sample_params['source_type']
        
        # Lensed image
        ax1 = fig.add_subplot(gs[i, 0])
        im1 = ax1.imshow(lensed_img, cmap='viridis', origin='lower')
        ax1.set_title(f"Lensed Image\nθ_E = {theta_E:.2f}\"", fontsize=12, fontweight='bold')
        ax1.set_xlabel('Pixels')
        ax1.set_ylabel('Pixels')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Source image
        ax2 = fig.add_subplot(gs[i, 1])
        im2 = ax2.imshow(source_img, cmap='viridis', origin='lower')
        ax2.set_title(f"Source Galaxy\nType: {source_type}", fontsize=12, fontweight='bold')
        ax2.set_xlabel('Pixels')
        ax2.set_ylabel('Pixels')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Difference image with enhanced contrast
        ax3 = fig.add_subplot(gs[i, 2])
        diff = lensed_img - source_img
        
        # Use symmetric logarithmic scale for better visualization of small differences
        vmax = np.max(np.abs(diff))
        im3 = ax3.imshow(diff, cmap='RdBu_r', origin='lower', 
                        norm=SymLogNorm(linthresh=0.01 * vmax, vmin=-vmax, vmax=vmax))
        ax3.set_title(f"Difference\nMagnification: {magnification:.1f}×", fontsize=12, fontweight='bold')
        ax3.set_xlabel('Pixels')
        ax3.set_ylabel('Pixels')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Strong Gravitational Lensing Samples (θ_E > {min_theta_E}\")", 
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.savefig("strong_lensing_visualization.pdf", dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Print detailed information about the shown samples
    print("\nDetailed sample information:")
    for i, idx in enumerate(indices):
        sample_params = params_df.iloc[idx]
        print(f"\nSample {i} (Index {idx}):")
        print(f"  θ_E: {sample_params['lens_theta_E']:.2f}\"")
        print(f"  e1: {sample_params['lens_e1']:.3f}, e2: {sample_params['lens_e2']:.3f}")
        print(f"  Source type: {sample_params['source_type']}")
        print(f"  Magnification: {sample_params['magnification']:.1f}×")
        print(f"  Complexity: {sample_params['complexity']}")

def visualize_parameter_correlations():
    """
    Create correlation plots between key parameters
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # θ_E vs Magnification
    sc1 = axes[0, 0].scatter(params_df['lens_theta_E'], params_df['magnification'], 
                           c=params_df['lens_theta_E'], cmap='viridis', alpha=0.6)
    axes[0, 0].set_xlabel('Einstein Radius (arcsec)')
    axes[0, 0].set_ylabel('Magnification')
    axes[0, 0].set_title('θ_E vs Magnification')
    plt.colorbar(sc1, ax=axes[0, 0], label='θ_E (arcsec)')
    
    # Ellipticity vs Magnification
    total_e = np.sqrt(params_df['lens_e1']**2 + params_df['lens_e2']**2)
    sc2 = axes[0, 1].scatter(total_e, params_df['magnification'], 
                           c=total_e, cmap='plasma', alpha=0.6)
    axes[0, 1].set_xlabel('Total Ellipticity')
    axes[0, 1].set_ylabel('Magnification')
    axes[0, 1].set_title('Ellipticity vs Magnification')
    plt.colorbar(sc2, ax=axes[0, 1], label='Ellipticity')
    
    # Source size vs Magnification
    sc3 = axes[1, 0].scatter(params_df['source_R_sersic'], params_df['magnification'], 
                           c=params_df['source_R_sersic'], cmap='inferno', alpha=0.6)
    axes[1, 0].set_xlabel('Source Size (arcsec)')
    axes[1, 0].set_ylabel('Magnification')
    axes[1, 0].set_title('Source Size vs Magnification')
    plt.colorbar(sc3, ax=axes[1, 0], label='Source Size (arcsec)')
    
    # Source position vs Magnification
    source_r = np.sqrt(params_df['source_center_x']**2 + params_df['source_center_y']**2)
    sc4 = axes[1, 1].scatter(source_r, params_df['magnification'], 
                           c=source_r, cmap='cool', alpha=0.6)
    axes[1, 1].set_xlabel('Source Distance from Center (arcsec)')
    axes[1, 1].set_ylabel('Magnification')
    axes[1, 1].set_title('Source Position vs Magnification')
    plt.colorbar(sc4, ax=axes[1, 1], label='Distance from Center (arcsec)')
    
    plt.tight_layout()
    plt.savefig("parameter_correlations.pdf", dpi=150, bbox_inches='tight')
    plt.show()

# Run the visualizations
visualize_strong_lenses(n_samples=5, min_theta_E=1.2, save_path="strong_lenses_visualization.pdf")
visualize_parameter_correlations()


# import h5py

# DATASET_PATH = "picolt_dataset_enhanced/dataset.h5"

# with h5py.File(DATASET_PATH, "r") as f:
#     def printname(name):
#         print(name)
#     f.visit(printname)


# import pandas as pd

# PARAMS_PATH = "picolt_dataset_enhanced/parameters.csv"
# params = pd.read_csv(PARAMS_PATH)

# print("Available columns:\n", params.columns.tolist())
# print("\nFirst few rows:\n", params.head())
