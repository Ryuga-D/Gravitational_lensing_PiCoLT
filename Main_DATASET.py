import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import h5py
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import kernel_util
import lenstronomy.Util.util as util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

from astropy.convolution import Gaussian2DKernel
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')


class EnhancedPICoLTDataGenerator:
    """
    Enhanced dataset generator for Physics-Informed Continuous Lens Transformer (PICoLT)
    Optimized for Vision Transformer training with physics-informed constraints
    """
    
    def __init__(self, 
                 output_dir,
                 num_samples=50000,
                 image_size=128,  # Larger for ViT
                 pixel_scale=0.2,  # HSC-like pixel scale
                 survey='HSC',  # Can be 'HSC', 'LSST', 'Euclid'
                 include_lens_light=True,
                 include_noise_variations=True,
                 augmentation=True,
                 num_workers=None):
        
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.image_size = image_size
        self.pixel_scale = pixel_scale #used to convert FWHM in arcsec to sigma in pixels.
        self.survey = survey
        self.include_lens_light = include_lens_light
        self.include_noise_variations = include_noise_variations
        self.augmentation = augmentation
        self.num_workers = num_workers or max(1, cpu_count() - 2)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'parameters'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        
        # Initialize grid which is used to evaluate surface brightness and ray-shooting.
        self.num_pix = self.image_size
        self.x_grid, self.y_grid = util.make_grid(
            numPix=self.num_pix, 
            deltapix=self.pixel_scale
        )
        
        # Survey-specific configurations
        self.survey_config = self._get_survey_config()
        
        # Initialize PSF
        self.psf_kernel = self._create_psf()
        
        # Storage for parameters
        self.parameters = []
        self.metadata = {
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_samples': num_samples,
            'image_size': image_size,
            'pixel_scale': pixel_scale,
            'survey': survey,
            'include_lens_light': include_lens_light
        }
    
    def _get_survey_config(self):
        """Get survey-specific configuration parameters"""

        """
        observational realism parameters, all connected to noise & resolution:
        psf_fwhm: how much the atmosphere + optics blur the image.
        sky_brightness: how bright the sky background is (main noise source).
        exposure_time: how long telescope integrates → higher = more photons, less noisy.
        gain, read_noise: electronic noise of the CCD camera.
        zeropoint: converts flux ↔ magnitude.
        limiting_mag: faintest detectable object (sets scaling)
        """
        configs = {
            'HSC': {
                'psf_fwhm': 0.7,  # arcsec
                'sky_brightness': 22.0,  # mag/arcsec^2
                'exposure_time': 1200,  # seconds
                'gain': 3.0,
                'read_noise': 4.5,
                'zeropoint': 27.0,
                'limiting_mag': 26.5
            },
            'LSST': {
                'psf_fwhm': 0.7,
                'sky_brightness': 21.8,
                'exposure_time': 3600,
                'gain': 2.3,
                'read_noise': 2.0,
                'zeropoint': 28.0,
                'limiting_mag': 27.5
            },
            'Euclid': {
                'psf_fwhm': 0.18,
                'sky_brightness': 22.5,
                'exposure_time': 565,
                'gain': 3.1,
                'read_noise': 4.2,
                'zeropoint': 25.5,
                'limiting_mag': 24.5
            }
        }
        return configs.get(self.survey, configs['HSC'])
    
    def _create_psf(self):
        """Create PSF kernel with potential variations"""
        fwhm = self.survey_config['psf_fwhm']
        
        if self.include_noise_variations:
            # Add slight PSF variations
            fwhm *= np.random.uniform(0.9, 1.1)
        
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2))) / self.pixel_scale
        
        # Create more realistic PSF with potential asymmetry
        psf_size = 25
        psf = Gaussian2DKernel(x_stddev=sigma, y_stddev=sigma, 
                              x_size=psf_size, y_size=psf_size)
        
        psf_arr = np.array(psf)  # extract kernel as writable array
        if self.include_noise_variations and np.random.random() > 0.7:
            psf_arr += np.random.normal(0, 0.001, psf_arr.shape)
            psf_arr = np.maximum(psf_arr, 0)

        psf_arr /= np.sum(psf_arr)  # normalize
        return psf_arr

    
    def generate_lens_parameters(self, complexity='moderate'):
        """
        Generate lens mass parameters with varying complexity levels
        """
        if complexity == 'simple':
            # Simple SIE lens
            params = {
                'theta_E': np.random.uniform(0.8, 1.5),
                'e1': np.random.uniform(-0.2, 0.2),
                'e2': np.random.uniform(-0.2, 0.2),
                'center_x': np.random.uniform(-0.1, 0.1),
                'center_y': np.random.uniform(-0.1, 0.1),
                'gamma1': 0.0,
                'gamma2': 0.0
            }
        elif complexity == 'moderate':
            # SIE + external shear
            params = {
                'theta_E': np.random.uniform(0.8, 2.0),
                'e1': np.random.uniform(-0.3, 0.3),
                'e2': np.random.uniform(-0.3, 0.3),
                'center_x': np.random.uniform(-0.3, 0.3),
                'center_y': np.random.uniform(-0.3, 0.3),
                'gamma1': np.random.uniform(-0.05, 0.05),
                'gamma2': np.random.uniform(-0.05, 0.05)
            }
        else:  # complex
            # More extreme parameters for challenging cases
            params = {
                'theta_E': np.random.uniform(0.5, 2.5),
                'e1': np.random.uniform(-0.4, 0.4),
                'e2': np.random.uniform(-0.4, 0.4),
                'center_x': np.random.uniform(-0.5, 0.5),
                'center_y': np.random.uniform(-0.5, 0.5),
                'gamma1': np.random.uniform(-0.1, 0.1),
                'gamma2': np.random.uniform(-0.1, 0.1)
            }
        
        # Add lens light parameters if requested
        if self.include_lens_light:
            params.update({
                'lens_amp': np.random.uniform(500, 2000),
                'lens_R_sersic': np.random.uniform(2.0, 5.0),
                'lens_n_sersic': np.random.uniform(2.0, 5.0),  # Elliptical-like
                'lens_e1': params['e1'] * 0.5,  # Aligned with mass
                'lens_e2': params['e2'] * 0.5
            })
        
        return params
    
    def generate_source_parameters(self, source_type='single'):
        """
        Generate source parameters with different morphologies
        """
        if source_type == 'single':
            # Single Sersic source
            params = {
                'amp': np.random.uniform(100, 500),
                'R_sersic': np.random.uniform(0.2, 0.8),
                'n_sersic': np.random.uniform(0.5, 4.0),
                'e1': np.random.uniform(-0.4, 0.4),
                'e2': np.random.uniform(-0.4, 0.4),
                'center_x': np.random.uniform(-0.5, 0.5),
                'center_y': np.random.uniform(-0.5, 0.5)
            }
        elif source_type == 'double':
            # Double source (merging galaxies)
            params = {
                'amp1': np.random.uniform(50, 250),
                'R_sersic1': np.random.uniform(0.1, 0.5),
                'n_sersic1': np.random.uniform(0.5, 2.0),
                'e1_1': np.random.uniform(-0.3, 0.3),
                'e2_1': np.random.uniform(-0.3, 0.3),
                'center_x1': np.random.uniform(-0.3, 0.3),
                'center_y1': np.random.uniform(-0.3, 0.3),
                'amp2': np.random.uniform(50, 250),
                'R_sersic2': np.random.uniform(0.1, 0.5),
                'n_sersic2': np.random.uniform(0.5, 2.0),
                'e1_2': np.random.uniform(-0.3, 0.3),
                'e2_2': np.random.uniform(-0.3, 0.3),
                'center_x2': np.random.uniform(-0.3, 0.3),
                'center_y2': np.random.uniform(-0.3, 0.3)
            }
        else:  # irregular
            # Irregular source (disk + bulge)
            params = {
                'amp_bulge': np.random.uniform(50, 200),
                'R_bulge': np.random.uniform(0.05, 0.2),
                'n_bulge': np.random.uniform(2.0, 4.0),
                'amp_disk': np.random.uniform(100, 300),
                'R_disk': np.random.uniform(0.3, 0.8),
                'n_disk': np.random.uniform(0.5, 1.5),
                'e1': np.random.uniform(-0.5, 0.5),
                'e2': np.random.uniform(-0.5, 0.5),
                'center_x': np.random.uniform(-0.4, 0.4),
                'center_y': np.random.uniform(-0.4, 0.4)
            }
        
        return params, source_type
    
    def create_lensed_image(self, lens_params, source_params, source_type='single'):
        """
        Create lensed image with proper physics simulation
        """
        # Setup lens mass model
        lens_model_list = ['SIE', 'SHEAR']
        lens_model = LensModel(lens_model_list=lens_model_list)
        
        # Setup light models
        if source_type == 'single':
            source_model_list = ['SERSIC_ELLIPSE']
        elif source_type == 'double':
            source_model_list = ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']
        else:  # irregular
            source_model_list = ['SERSIC_ELLIPSE', 'SERSIC']
        
        source_model = LightModel(light_model_list=source_model_list)
        
        # Add lens light if requested
        if self.include_lens_light:
            lens_light_model_list = ['SERSIC_ELLIPSE']
            lens_light_model = LightModel(light_model_list=lens_light_model_list)
        else:
            lens_light_model = None
            lens_light_model_list = []
        
        # Prepare kwargs
        lens_kwargs = [
            {
                'theta_E': lens_params['theta_E'],
                'e1': lens_params['e1'],
                'e2': lens_params['e2'],
                'center_x': lens_params['center_x'],
                'center_y': lens_params['center_y']
            },
            {
                'gamma1': lens_params['gamma1'],
                'gamma2': lens_params['gamma2']
            }
        ]
        
        # Prepare source kwargs based on type
        if source_type == 'single':
            source_kwargs = [{
                'amp': source_params['amp'] * 10.0,
                'R_sersic': source_params['R_sersic'],
                'n_sersic': source_params['n_sersic'],
                'e1': source_params['e1'],
                'e2': source_params['e2'],
                'center_x': source_params['center_x'],
                'center_y': source_params['center_y']
            }]
        elif source_type == 'double':
            source_kwargs = [
                {
                    'amp': source_params['amp1'] * 10.0,
                    'R_sersic': source_params['R_sersic1'],
                    'n_sersic': source_params['n_sersic1'],
                    'e1': source_params['e1_1'],
                    'e2': source_params['e2_1'],
                    'center_x': source_params['center_x1'],
                    'center_y': source_params['center_y1']
                },
                {
                    'amp': source_params['amp2'] * 10.0,
                    'R_sersic': source_params['R_sersic2'],
                    'n_sersic': source_params['n_sersic2'],
                    'e1': source_params['e1_2'],
                    'e2': source_params['e2_2'],
                    'center_x': source_params['center_x2'],
                    'center_y': source_params['center_y2']
                }
            ]
        else:  # irregular
            source_kwargs = [
                {
                    'amp': source_params['amp_bulge'] * 10.0,
                    'R_sersic': source_params['R_bulge'],
                    'n_sersic': source_params['n_bulge'],
                    'e1': source_params['e1'],
                    'e2': source_params['e2'],
                    'center_x': source_params['center_x'],
                    'center_y': source_params['center_y']
                },
                {
                    'amp': source_params['amp_disk'] * 10.0,
                    'R_sersic': source_params['R_disk'],
                    'n_sersic': source_params['n_disk'],
                    'center_x': source_params['center_x'],
                    'center_y': source_params['center_y']
                }
            ]
        
        # Lens light kwargs
        if self.include_lens_light:
            lens_light_kwargs = [{
                'amp': lens_params['lens_amp'] * 20.0,
                'R_sersic': lens_params['lens_R_sersic'],
                'n_sersic': lens_params['lens_n_sersic'],
                'e1': lens_params['lens_e1'],
                'e2': lens_params['lens_e2'],
                'center_x': lens_params['center_x'],
                'center_y': lens_params['center_y']
            }]
        else:
            lens_light_kwargs = []
        
        # Ray-tracing for lensed source
        beta_x, beta_y = lens_model.ray_shooting(self.x_grid, self.y_grid, lens_kwargs)
        #computes mapping from image-plane coordinates to source-plane coordinates (i.e., where a pixel in the observed image maps to in the unlensed source).
        
        # Get surface brightness
        source_lensed = source_model.surface_brightness(beta_x, beta_y, source_kwargs)
        source_only = source_model.surface_brightness(self.x_grid, self.y_grid, source_kwargs)
        
        # Add lens light
        if self.include_lens_light:
            lens_light = lens_light_model.surface_brightness(self.x_grid, self.y_grid, lens_light_kwargs)
            lensed_total = source_lensed + lens_light
        else:
            lensed_total = source_lensed
        
        # Convert to images
        lensed_image = util.array2image(lensed_total)
        source_image = util.array2image(source_only)
        
        # Apply PSF convolution
        lensed_image = convolve2d(lensed_image, self.psf_kernel, mode='same', boundary='fill')
        source_image = convolve2d(source_image, self.psf_kernel, mode='same', boundary='fill')
        
        # Add realistic noise
        lensed_image = self._add_noise(lensed_image)
        source_image = self._add_noise(source_image, is_source=True)
        
        # Data augmentation if requested
        if self.augmentation:
            lensed_image, source_image = self._augment_pair(lensed_image, source_image)
        
        # Normalize images for neural network training
        lensed_image = self._normalize_image(lensed_image)
        source_image = self._normalize_image(source_image)
        
        return lensed_image, source_image
    
    def _add_noise(self, image, is_source=False):
        """Add realistic noise based on survey parameters"""
        config = self.survey_config
        
        # Convert to electron counts
        exposure_time = config['exposure_time']
        gain = config['gain']
        read_noise = config['read_noise']
        
        # Scale for source (dimmer without lensing)
        if is_source:
            exposure_time *= 0.3  # Reduced effective exposure for unlensed source
        
        # Add sky background
        sky_counts = 10 ** ((config['zeropoint'] - config['sky_brightness']) / 2.5)
        sky_counts *= exposure_time * self.pixel_scale ** 2
        
        # Total counts
        image_electrons = image * exposure_time + sky_counts
        
        # Poisson noise (photon noise)
        # Poisson noise (photon noise) with Gaussian fallback
        lam = np.maximum(image_electrons, 0).astype(float)

        large_mask = lam > 1e7
        image_electrons = np.empty_like(lam)

        # Poisson for safe values
        image_electrons[~large_mask] = np.random.poisson(lam[~large_mask])

        # Normal approximation for large λ
        image_electrons[large_mask] = lam[large_mask] + np.random.normal(
            0, np.sqrt(lam[large_mask])
        )
        
        # Read noise
        read_noise_electrons = np.random.normal(0, read_noise, image.shape)
        
        # Total noisy image
        noisy_image = (image_electrons + read_noise_electrons) / gain
        
        # Subtract sky background estimate
        noisy_image -= sky_counts / gain
        
        return noisy_image
    
    def _normalize_image(self, image):
        """Normalize image for neural network training"""
        # Asinh stretch for better dynamic range
        stretch_factor = 10.0
        image_norm = np.arcsinh(image * stretch_factor) / np.arcsinh(stretch_factor)
        
        # Clip extreme values
        image_norm = np.clip(image_norm, -1, 1)
        
        # Scale to [0, 1]
        image_norm = (image_norm + 1) / 2
        
        return image_norm
    
    def _augment_pair(self, lensed, source):
        """Apply consistent augmentation to lensed-source pair"""
        if np.random.random() > 0.5:
            # Random rotation (90 degree increments)
            k = np.random.randint(0, 4)
            lensed = np.rot90(lensed, k)
            source = np.rot90(source, k)
        
        if np.random.random() > 0.5:
            # Horizontal flip
            lensed = np.fliplr(lensed)
            source = np.fliplr(source)
        
        if np.random.random() > 0.5:
            # Vertical flip
            lensed = np.flipud(lensed)
            source = np.flipud(source)
        
        return lensed, source
    
    def generate_sample(self, index):
        """Generate a single sample with error handling"""
        try:
            # Vary complexity
            complexity = np.random.choice(['simple', 'moderate', 'complex'], 
                                        p=[0.2, 0.6, 0.2])
            
            # Vary source type
            source_type = np.random.choice(['single', 'double', 'irregular'],
                                          p=[0.6, 0.2, 0.2])
            
            lens_params = self.generate_lens_parameters(complexity)
            source_params, s_type = self.generate_source_parameters(source_type)
            
            lensed_image, source_image = self.create_lensed_image(
                lens_params, source_params, s_type
            )
            
            # Compile all parameters
            sample_params = {
                'sample_id': index,
                'complexity': complexity,
                'source_type': s_type,
                **{f'lens_{k}': v for k, v in lens_params.items()},
                **{f'source_{k}': v for k, v in source_params.items()}
            }
            
            # Calculate additional physics quantities
            sample_params['magnification'] = self._estimate_magnification(lens_params, source_params)
            sample_params['num_images'] = self._count_images(lensed_image)
            
            return lensed_image, source_image, sample_params
            
        except Exception as e:
            print(f"Error generating sample {index}: {str(e)}")
            return None, None, None
    
    def _estimate_magnification(self, lens_params, source_params):
        """Estimate total magnification of the system"""
        # Simple estimation based on source position and Einstein radius
        source_r = np.sqrt(source_params.get('center_x', 0)**2 + 
                          source_params.get('center_y', 0)**2)
        theta_E = lens_params['theta_E']
        
        if source_r < 0.1 * theta_E:
            return np.random.uniform(10, 30)  # High magnification
        elif source_r < 0.5 * theta_E:
            return np.random.uniform(3, 10)   # Moderate magnification
        else:
            return np.random.uniform(1.5, 3)  # Low magnification
    
    def _count_images(self, lensed_image):
        """Count approximate number of lensed images"""
        # Simple peak counting after smoothing
        from scipy.ndimage import label, gaussian_filter
        
        smoothed = gaussian_filter(lensed_image, sigma=2)
        threshold = np.percentile(smoothed, 90)
        binary = smoothed > threshold
        labeled, num_features = label(binary)
        
        return min(num_features, 4)  # Max 4 images for quad systems
    
    def generate_dataset(self, batch_size=100):
        """Generate full dataset with progress tracking and error recovery"""
        print(f"Generating {self.num_samples} lensing examples...")
        print(f"Configuration: {self.survey} survey, {self.image_size}x{self.image_size} pixels")
        print(f"Include lens light: {self.include_lens_light}")
        print(f"Augmentation: {self.augmentation}")
        
        start_time = time.time()
        
        # Initialize HDF5 file for efficient storage
        h5_path = os.path.join(self.output_dir, 'dataset.h5')
        
        with h5py.File(h5_path, 'w') as h5f:
            # Create datasets
            lensed_dset = h5f.create_dataset(
                'lensed_images',
                shape=(self.num_samples, self.image_size, self.image_size, 1),
                dtype='float32',
                chunks=(batch_size, self.image_size, self.image_size, 1),
                compression='gzip'
            )
            
            source_dset = h5f.create_dataset(
                'source_images',
                shape=(self.num_samples, self.image_size, self.image_size, 1),
                dtype='float32',
                chunks=(batch_size, self.image_size, self.image_size, 1),
                compression='gzip'
            )
            
            # Generate samples in batches
            successful_samples = 0
            all_params = []
            
            pbar = tqdm(total=self.num_samples)
            
            while successful_samples < self.num_samples:
                batch_lensed = []
                batch_source = []
                batch_params = []
                
                # Generate batch
                for _ in range(min(batch_size, self.num_samples - successful_samples)):
                    lensed, source, params = self.generate_sample(successful_samples)
                    
                    if lensed is not None:
                        batch_lensed.append(lensed)
                        batch_source.append(source)
                        batch_params.append(params)
                        successful_samples += 1
                        pbar.update(1)
                
                # Save batch to HDF5
                if batch_lensed:
                    start_idx = successful_samples - len(batch_lensed)
                    end_idx = successful_samples
                    
                    lensed_dset[start_idx:end_idx] = np.expand_dims(
                        np.array(batch_lensed), axis=-1
                    )
                    source_dset[start_idx:end_idx] = np.expand_dims(
                        np.array(batch_source), axis=-1
                    )
                    all_params.extend(batch_params)
                
                # Save checkpoint
                if successful_samples % 1000 == 0 and successful_samples > 0:
                    self._save_checkpoint(all_params, successful_samples)
            
            pbar.close()
            
            # Save metadata
            h5f.attrs['num_samples'] = successful_samples
            h5f.attrs['image_size'] = self.image_size
            h5f.attrs['pixel_scale'] = self.pixel_scale
            h5f.attrs['survey'] = self.survey
            
        # Save parameters
        df_params = pd.DataFrame(all_params)
        df_params.to_csv(
            os.path.join(self.output_dir, 'parameters.csv'),
            index=False
        )
        
        # Save metadata
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Generate statistics
        self._generate_statistics(df_params)
        
        end_time = time.time()
        print(f"\nDataset generation complete!")
        print(f"Generated {successful_samples} samples in {end_time - start_time:.2f} seconds")
        print(f"Dataset saved to {self.output_dir}")
        
        return h5_path, df_params
    
    def _save_checkpoint(self, params, num_samples):
        """Save checkpoint for recovery"""
        checkpoint_path = os.path.join(
            self.output_dir, 
            'checkpoints',
            f'checkpoint_{num_samples}.csv'
        )
        pd.DataFrame(params).to_csv(checkpoint_path, index=False)
    
    def _generate_statistics(self, df_params):
        """Generate and save dataset statistics"""
        stats = {
            'total_samples': len(df_params),
            'complexity_distribution': df_params['complexity'].value_counts().to_dict(),
            'source_type_distribution': df_params['source_type'].value_counts().to_dict(),
            'theta_E_stats': {
                'mean': df_params['lens_theta_E'].mean(),
                'std': df_params['lens_theta_E'].std(),
                'min': df_params['lens_theta_E'].min(),
                'max': df_params['lens_theta_E'].max()
            },
            'magnification_stats': {
                'mean': df_params['magnification'].mean(),
                'std': df_params['magnification'].std(),
                'min': df_params['magnification'].min(),
                'max': df_params['magnification'].max()
            }
        }
        
        with open(os.path.join(self.output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate parameter distribution plots
        self._plot_parameter_distributions(df_params)
    
    def _plot_parameter_distributions(self, df_params):
        """Plot parameter distributions for quality control"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Einstein radius distribution
        axes[0, 0].hist(df_params['lens_theta_E'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Einstein Radius (arcsec)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Einstein Radius Distribution')
        
        # Ellipticity distribution
        e_total = np.sqrt(df_params['lens_e1']**2 + df_params['lens_e2']**2)
        axes[0, 1].hist(e_total, bins=30, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Total Ellipticity')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Lens Ellipticity Distribution')
        
        # Shear distribution
        gamma_total = np.sqrt(df_params['lens_gamma1']**2 + df_params['lens_gamma2']**2)
        axes[0, 2].hist(gamma_total, bins=30, alpha=0.7, color='red')
        axes[0, 2].set_xlabel('Total Shear')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('External Shear Distribution')
        
        # Source size distribution
        axes[1, 0].hist(df_params['source_R_sersic'], bins=30, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Source Size (arcsec)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Source Size Distribution')
        
        # Magnification distribution
        axes[1, 1].hist(df_params['magnification'], bins=30, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Magnification')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Magnification Distribution')
        axes[1, 1].set_yscale('log')
        
        # Number of images distribution
        axes[1, 2].hist(df_params['num_images'], bins=np.arange(0.5, 5.5, 1), 
                       alpha=0.7, color='brown')
        axes[1, 2].set_xlabel('Number of Images')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Number of Lensed Images')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_distributions.pdf'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_samples(self, h5_path=None, num_samples=5, save_path=None):
        """Visualize sample images from the dataset"""
        if h5_path is None:
            h5_path = os.path.join(self.output_dir, 'dataset.h5')
        
        with h5py.File(h5_path, 'r') as h5f:
            lensed = h5f['lensed_images'][:num_samples]
            source = h5f['source_images'][:num_samples]
        
        params_df = pd.read_csv(os.path.join(self.output_dir, 'parameters.csv'))
        params = params_df.iloc[:num_samples].to_dict('records')
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples*4))
        
        for i in range(num_samples):
            # Lensed image
            im0 = axes[i, 0].imshow(lensed[i, :, :, 0], cmap='viridis',
                                    origin='lower')
            axes[i, 0].set_title(f"Lensed Image {i}\nθ_E={params[i]['lens_theta_E']:.2f}\"")
            axes[i, 0].axis('off')
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
            
            # Source image
            im1 = axes[i, 1].imshow(source[i, :, :, 0], cmap='viridis',
                                    origin='lower')
            axes[i, 1].set_title(f"Source Galaxy {i}\nType: {params[i]['source_type']}")
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
            
            # Residual (for quality check)
            # Simulate re-lensing for physics check
            residual = lensed[i, :, :, 0] - source[i, :, :, 0]
            im2 = axes[i, 2].imshow(residual, cmap='RdBu_r', origin='lower')
            axes[i, 2].set_title(f"Difference\nMag={params[i]['magnification']:.1f}")
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'sample_preview.pdf'), 
                       dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_training_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Create train/validation/test splits for the dataset"""
        params_df = pd.read_csv(os.path.join(self.output_dir, 'parameters.csv'))
        n_samples = len(params_df)
        
        # Calculate split indices
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val
        
        # Create random indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Save split indices
        splits = {
            'train': train_idx.tolist(),
            'validation': val_idx.tolist(),
            'test': test_idx.tolist(),
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test
        }
        
        with open(os.path.join(self.output_dir, 'splits.json'), 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"Created splits: Train={n_train}, Val={n_val}, Test={n_test}")
        
        return splits
    
    def create_physics_validation_set(self, n_samples=100):
        """
        Create a special validation set for physics-informed loss validation
        with known analytical solutions
        """
        print(f"Creating physics validation set with {n_samples} samples...")
        
        physics_samples = []
        
        with h5py.File(os.path.join(self.output_dir, 'physics_validation.h5'), 'w') as h5f:
            lensed_dset = h5f.create_dataset(
                'lensed_images',
                shape=(n_samples, self.image_size, self.image_size, 1),
                dtype='float32'
            )
            source_dset = h5f.create_dataset(
                'source_images',
                shape=(n_samples, self.image_size, self.image_size, 1),
                dtype='float32'
            )
            
            for i in tqdm(range(n_samples)):
                # Generate simple, well-controlled systems
                lens_params = {
                    'theta_E': np.random.uniform(1.0, 1.5),
                    'e1': 0.0,  # Circular lens for simplicity
                    'e2': 0.0,
                    'center_x': 0.0,
                    'center_y': 0.0,
                    'gamma1': 0.0,
                    'gamma2': 0.0
                }
                
                source_params = {
                    'amp': 200,
                    'R_sersic': 0.3,
                    'n_sersic': 1.0,
                    'e1': 0.0,
                    'e2': 0.0,
                    'center_x': np.random.uniform(-0.3, 0.3),
                    'center_y': np.random.uniform(-0.3, 0.3)
                }
                
                lensed, source = self.create_lensed_image(
                    lens_params, source_params, source_type='single'
                )
                
                lensed_dset[i] = np.expand_dims(lensed, axis=-1)
                source_dset[i] = np.expand_dims(source, axis=-1)
                
                physics_samples.append({
                    'sample_id': i,
                    **{f'lens_{k}': v for k, v in lens_params.items()},
                    **{f'source_{k}': v for k, v in source_params.items()}
                })
        
        # Save parameters
        pd.DataFrame(physics_samples).to_csv(
            os.path.join(self.output_dir, 'physics_validation_params.csv'),
            index=False
        )
        
        print(f"Physics validation set created with {n_samples} samples")


class PICoLTDataLoader:
    """
    Custom data loader for PICoLT training with TensorFlow/PyTorch compatibility
    """
    
    def __init__(self, dataset_path, batch_size=32, shuffle=True):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load dataset info
        with h5py.File(os.path.join(dataset_path, 'dataset.h5'), 'r') as h5f:
            self.n_samples = h5f.attrs['num_samples']
            self.image_size = h5f.attrs['image_size']
        
        # Load splits
        with open(os.path.join(dataset_path, 'splits.json'), 'r') as f:
            self.splits = json.load(f)
        
        # Load parameters
        self.params_df = pd.read_csv(os.path.join(dataset_path, 'parameters.csv'))
    
    def get_tensorflow_dataset(self, split='train'):
        """Create TensorFlow dataset"""
        import tensorflow as tf
        
        indices = self.splits[split]
        
        def generator():
            if self.shuffle and split == 'train':
                np.random.shuffle(indices)
            
            with h5py.File(os.path.join(self.dataset_path, 'dataset.h5'), 'r') as h5f:
                for idx in indices:
                    lensed = h5f['lensed_images'][idx]
                    source = h5f['source_images'][idx]
                    
                    # Get parameters for this sample
                    params = self.params_df.iloc[idx]
                    
                    # Prepare parameter vector for regression
                    param_vector = np.array([
                        params['lens_theta_E'],
                        params['lens_e1'],
                        params['lens_e2'],
                        params['lens_center_x'],
                        params['lens_center_y'],
                        params['lens_gamma1'],
                        params['lens_gamma2']
                    ], dtype=np.float32)
                    
                    yield lensed, (source, param_vector)
        
        # Create dataset
        output_signature = (
            tf.TensorSpec(shape=(self.image_size, self.image_size, 1), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(self.image_size, self.image_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(7,), dtype=tf.float32)
            )
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    # def get_pytorch_dataset(self, split='train'):
    #     """Create PyTorch dataset"""
    #     import torch
    #     from torch.utils.data import Dataset, DataLoader
        
    #     class PICoLTDataset(Dataset):
    #         def __init__(self, h5_path, indices, params_df):
    #             self.h5_path = h5_path
    #             self.indices = indices
    #             self.params_df = params_df
            
    #         def __len__(self):
    #             return len(self.indices)
            
    #         def __getitem__(self, idx):
    #             real_idx = self.indices[idx]
                
    #             with h5py.File(self.h5_path, 'r') as h5f:
    #                 lensed = torch.from_numpy(h5f['lensed_images'][real_idx]).float()
    #                 source = torch.from_numpy(h5f['source_images'][real_idx]).float()
                
    #             params = self.params_df.iloc[real_idx]
    #             param_vector = torch.tensor([
    #                 params['lens_theta_E'],
    #                 params['lens_e1'],
    #                 params['lens_e2'],
    #                 params['lens_center_x'],
    #                 params['lens_center_y'],
    #                 params['lens_gamma1'],
    #                 params['lens_gamma2']
    #             ], dtype=torch.float32)
                
    #             return lensed, source, param_vector
        
    #     dataset = PICoLTDataset(
    #         os.path.join(self.dataset_path, 'dataset.h5'),
    #         self.splits[split],
    #         self.params_df
    #     )
        
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=self.batch_size,
    #         shuffle=(self.shuffle and split == 'train'),
    #         num_workers=4,
    #         pin_memory=True
    #     )
        
    #     return dataloader


# Main execution
if __name__ == "__main__":
    # Configuration
    OUTPUT_DIR = "picolt_dataset_enhanced"
    NUM_SAMPLES = 60000
    IMAGE_SIZE = 128  # Larger for Vision Transformer
    SURVEY = 'HSC'  # Can be 'HSC', 'LSST', or 'Euclid'
    
    # Create generator
    generator = EnhancedPICoLTDataGenerator(
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        image_size=IMAGE_SIZE,
        pixel_scale=0.2,
        survey=SURVEY,
        include_lens_light=False,
        include_noise_variations=True,
        augmentation=True,
        num_workers=4
    )
    
    # Generate dataset
    h5_path, params_df = generator.generate_dataset(batch_size=100)
    
    # Create train/val/test splits
    splits = generator.create_training_splits(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Create physics validation set
    generator.create_physics_validation_set(n_samples=500)
    
    # Visualize samples
    print("\nGenerating sample visualizations...")
    generator.visualize_samples(num_samples=5)
    
    # Example: Load data for training
    print("\nTesting data loader...")
    loader = PICoLTDataLoader(OUTPUT_DIR, batch_size=32)
    
    # For TensorFlow
    tf_dataset = loader.get_tensorflow_dataset('train')
    
    # For PyTorch
    # pytorch_loader = loader.get_pytorch_dataset('train')
    
    print("\nDataset generation complete!")
    print(f"Dataset location: {OUTPUT_DIR}")
    print(f"Total samples: {NUM_SAMPLES}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Survey configuration: {SURVEY}")