"""
Benchmark script for pypass PSF fitting on jbjm01kkq_flc.fits.
Run with: conda run -n bp3m-test python benchmark_psf.py
"""
import os
import sys
import time

# Set thread limits before any numpy/jax imports (matches what bp3m_run.py does)
N_PROCESSES = 10
os.environ['OMP_NUM_THREADS']      = str(N_PROCESSES)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_PROCESSES)
os.environ['MKL_NUM_THREADS']      = str(N_PROCESSES)

try:
    import threadpoolctl as _tpc
    _tpc.threadpool_limits(limits=N_PROCESSES)
except ImportError:
    pass

sys.path.insert(0, '/bootes_raid6/users/kmckinnon/claude/bp3m')

IMAGE_PATH = '/home/jupyter-kmckinnon/data_bootes/bp3m/GaiaHub_results/Leo_I/HST/mastDownload/HST/jbjm01kkq/jbjm01kkq_flc.fits'
LIB_DIR    = '/home/jupyter-kmckinnon/data_bootes/bp3m_lib'

params = dict(
    # Exactly psf_params.json values
    fmin_thresh        = 70.0,
    hmin               = 4,
    n_passes           = 2,
    n_discovery_passes = 1,
    sat_threshold      = 60000.0,
    max_iter_fit       = 100,
    half_width         = 3,
    sky_inner          = 4,
    sky_outer          = 8,
    tol                = 1e-3,
    sigma_clip         = True,
    sigma_clip_sigma   = 4.0,
    conc_limit         = 0.9,
    backend            = 'auto',
    # User-specified benchmark parameter
    n_jobs             = N_PROCESSES,
)

print("=" * 60)
print("pypass PSF fitting benchmark")
print("=" * 60)
print(f"Image      : {IMAGE_PATH}")
print(f"lib_dir    : {LIB_DIR}")
print(f"n_processes: {N_PROCESSES}")
print(f"fmin_thresh: {params['fmin_thresh']}")
print(f"n_passes   : {params['n_passes']}")
print(f"half_width : {params['half_width']}")
print(f"backend    : {params['backend']}")
print()

from pypass.io import run_photometry_fits
import jax
# Create N virtual CPU devices so pmap can shard the star batch across them.
# Must be set before any JAX computation.
jax.config.update('jax_num_cpu_devices', N_PROCESSES)
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print()

t_start = time.perf_counter()

records, residuals, var_images, psf_path, gdc_path = run_photometry_fits(
    image_path   = IMAGE_PATH,
    psf_path     = None,
    lib_dir      = LIB_DIR,
    return_residual = True,
    verbose      = True,
    **params,
)

t_end = time.perf_counter()
elapsed = t_end - t_start

print()
print("=" * 60)
print(f"Total stars fitted : {len(records)}")
print(f"Total wall time    : {elapsed:.2f} s  ({elapsed/60:.2f} min)")
print("=" * 60)
