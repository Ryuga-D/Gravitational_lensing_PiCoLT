import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import kernel_util
import lenstronomy.Util.util as util
from astropy.convolution import Gaussian2DKernel
from matplotlib.colors import LogNorm

class StrongLensDataGenerator:
    """
    A complete pipeline to generate strong gravitational lensing data
    using Lenstronomy for the PICOLT project
    """
    
    def __init__(self, output_dir, num_samples=50000):
        self.output_dir = output_dir
        self.num_samples = num_samples
        os.makedirs(output_dir, exist_ok=True)
        self.parameters = []

        # Rubin Observatory characteristics
        self.pixel_scale = 0.2  
        self.image_size = 64    
        self.num_pix = self.image_size
        self.x_grid, self.y_grid = util.make_grid(numPix=self.num_pix, deltapix=self.pixel_scale)

        # Gaussian PSF
        self.psf_fwhm = 0.7  
        sigma = self.psf_fwhm / (2 * np.sqrt(2 * np.log(2))) 
        self.psf_kernel = Gaussian2DKernel(x_stddev=sigma, x_size=21, y_size=21)
        self.psf_kernel.normalize()

    def generate_lens_parameters(self):
        return {
            'theta_E': np.random.uniform(0.8, 2.0),
            'e1': np.random.uniform(-0.3, 0.3),
            'e2': np.random.uniform(-0.3, 0.3),
            'center_x': np.random.uniform(-0.3, 0.3),
            'center_y': np.random.uniform(-0.3, 0.3),
            'gamma1': np.random.uniform(-0.05, 0.05),
            'gamma2': np.random.uniform(-0.05, 0.05)
        }
    
    def generate_source_parameters(self):
        source_size_arcsec = np.random.uniform(1.5, 3.0)  
        return {
            'amp': np.random.uniform(100, 300),
            'R_sersic': source_size_arcsec,
            'n_sersic': np.random.uniform(1.0, 4.0),
            'e1': np.random.uniform(-0.4, 0.4),
            'e2': np.random.uniform(-0.4, 0.4),
            'center_x': np.random.uniform(-1.0, 1.0),
            'center_y': np.random.uniform(-1.0, 1.0)
        }

    def safe_poisson(self, lam):
        """
        Robust Poisson sampler:
        - Use Poisson when lam < 1e7
        - Use Normal approximation when lam >= 1e7
        """
        lam = np.asarray(lam, dtype=np.float64)
        out = np.empty_like(lam, dtype=np.float64)

        mask_small = lam < 1e7
        mask_large = ~mask_small

        if np.any(mask_small):
            out[mask_small] = np.random.poisson(lam[mask_small])
        if np.any(mask_large):
            out[mask_large] = np.random.normal(lam[mask_large], np.sqrt(lam[mask_large]))

        return out

    def create_lensed_image(self, lens_params, source_params):
        lens_model = LensModel(lens_model_list=['SIE', 'SHEAR'])
        source_model = LightModel(light_model_list=['SERSIC_ELLIPSE'])

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
        source_kwargs = [{
            'amp': source_params['amp'] * 10.0,
            'R_sersic': source_params['R_sersic'],
            'n_sersic': source_params['n_sersic'],
            'e1': source_params['e1'],
            'e2': source_params['e2'],
            'center_x': source_params['center_x'],
            'center_y': source_params['center_y']
        }]

        beta_x, beta_y = lens_model.ray_shooting(self.x_grid, self.y_grid, lens_kwargs)
        source_surface_brightness = source_model.surface_brightness(beta_x, beta_y, source_kwargs)
        source_only_brightness = source_model.surface_brightness(self.x_grid, self.y_grid, source_kwargs)

        lensed_image = util.array2image(source_surface_brightness)
        source_image = util.array2image(source_only_brightness)

        from scipy.signal import convolve2d
        lensed_image = convolve2d(lensed_image, self.psf_kernel, mode='same', boundary='fill')
        source_image = convolve2d(source_image, self.psf_kernel, mode='same', boundary='fill')

        # Noise model
        exposure_time = 3600  
        gain = 2.3  
        read_noise = 2.0  

        lensed_electrons = lensed_image * exposure_time
        source_electrons = source_image * exposure_time

        # Use safe Poisson sampling
        lensed_electrons = self.safe_poisson(np.maximum(lensed_electrons, 0))
        source_electrons = self.safe_poisson(np.maximum(source_electrons, 0))

        lensed_image = lensed_electrons / gain + np.random.normal(0, read_noise/gain, lensed_image.shape)
        source_image = source_electrons / gain + np.random.normal(0, read_noise/gain, source_image.shape)

        # Normalize
        if np.max(lensed_image) > 0:
            lensed_image /= np.max(lensed_image)
        if np.max(source_image) > 0:
            source_image /= np.max(source_image)

        lensed_image = np.arcsinh(lensed_image * 10) / np.arcsinh(10)
        source_image = np.arcsinh(source_image * 10) / np.arcsinh(10)

        return lensed_image, source_image
    
    def generate_sample(self, index):
        try:
            lens_params = self.generate_lens_parameters()
            source_params = self.generate_source_parameters()
            lensed_image, source_image = self.create_lensed_image(lens_params, source_params)
            sample_params = {
                'sample_id': index,
                **{f'lens_{k}': v for k, v in lens_params.items()},
                **{f'source_{k}': v for k, v in source_params.items()}
            }
            self.parameters.append(sample_params)
            return lensed_image, source_image, sample_params
        except Exception as e:
            print(f"Error generating sample {index}: {str(e)}")
            return None, None, None
    
    def generate_dataset(self):
        print(f"Generating {self.num_samples} lensing examples...")
        start_time = time.time()
        all_lensed, all_source = [], []
        successful_samples = 0

        for i in tqdm(range(self.num_samples)):
            lensed_img, source_img, params = self.generate_sample(i)
            if lensed_img is not None:
                all_lensed.append(lensed_img)
                all_source.append(source_img)
                successful_samples += 1
                if (i + 1) % 1000 == 0:
                    self.save_progress(all_lensed, all_source, i)
        
        X_lensed = np.expand_dims(np.array(all_lensed), axis=-1)
        y_source = np.expand_dims(np.array(all_source), axis=-1)
        self.save_dataset(X_lensed, y_source)

        end_time = time.time()
        print(f"Dataset generation complete! {successful_samples} samples in {end_time - start_time:.2f}s")
        return X_lensed, y_source, self.parameters
    
    def save_progress(self, lensed, source, index):
        np.save(os.path.join(self.output_dir, f'temp_lensed_{index}.npy'), np.array(lensed))
        np.save(os.path.join(self.output_dir, f'temp_source_{index}.npy'), np.array(source))
        pd.DataFrame(self.parameters).to_csv(os.path.join(self.output_dir, 'temp_parameters.csv'), index=False)
    
    def save_dataset(self, X_lensed, y_source):
        if len(X_lensed) == 0:
            print("Error: No samples generated!")
            return
        np.save(os.path.join(self.output_dir, 'X_lensed_images.npy'), X_lensed)
        np.save(os.path.join(self.output_dir, 'y_source_images.npy'), y_source)
        pd.DataFrame(self.parameters).to_csv(os.path.join(self.output_dir, 'lens_parameters.csv'), index=False)
        with open(os.path.join(self.output_dir, 'dataset_info.txt'), 'w') as f:
            f.write(f"num_samples: {len(X_lensed)}\n")
            f.write(f"image_shape: {X_lensed[0].shape}\n")
            f.write(f"pixel_scale: {self.pixel_scale} arcsec/pixel\n")
            f.write(f"image_size: {self.image_size}x{self.image_size} pixels\n")
            f.write(f"psf_fwhm: {self.psf_fwhm} arcsec\n")
            f.write(f"generation_date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("notes: Generated for PICOLT model training\n")
        print(f"Dataset saved to {self.output_dir}")

# Run
if __name__ == "__main__":
    OUTPUT_DIR = "picolt_dataset_1"
    NUM_SAMPLES = 50000
    
    generator = StrongLensDataGenerator(OUTPUT_DIR, NUM_SAMPLES)
    X_lensed, y_source, parameters = generator.generate_dataset()
    
    # Visualization
    def visualize_samples(X_lensed, y_source, parameters, num_samples=5):
        if isinstance(parameters, pd.DataFrame):
            parameters = parameters.to_dict('records')
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples*3))
        for i in range(num_samples):
            im0 = axes[i, 0].imshow(X_lensed[i, :, :, 0], cmap='viridis',
                                    norm=LogNorm(vmin=0.01, vmax=1.0), origin='lower')
            axes[i, 0].set_title(f"Lensed {i}\nθ_E={parameters[i]['lens_theta_E']:.2f}, e1={parameters[i]['lens_e1']:.2f}")
            axes[i, 0].axis('off')
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
            
            im1 = axes[i, 1].imshow(y_source[i, :, :, 0], cmap='viridis',
                                    norm=LogNorm(vmin=0.01, vmax=1.0), origin='lower')
            axes[i, 1].set_title(f"Source {i}\nSize={parameters[i]['source_R_sersic']:.2f} arcsec")
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'sample_preview_log.pdf'), dpi=150, bbox_inches='tight')
        plt.show()
    
    print("Generating sample preview...")
    visualize_samples(X_lensed, y_source, parameters, num_samples=5)
