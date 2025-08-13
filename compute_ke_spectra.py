import numpy as np
from utils import load_box_timeseries
import matplotlib.pyplot as plt

def compute_isotropic_spectrum(field, Lx, Ly=None, Lz=None, tke_normalize=False, spectral_dealias=False):
    """
    field: (T, 3, Nx, Ny, Nz) velocity field [m/s]
    Returns:
      k_plot : 1D array of kappa (rad/length)
      E_plot : normalized spectrum (unitless), integrates to 1 over kept bins
    """
    if Ly is None: 
        Ly = Lx
    if Lz is None:
        Lz = Lx

    T, C, Nx, Ny, Nz = field.shape

    field_hat = np.fft.fftn(field, axes=(2,3,4), norm='forward') # Forward divides by N
    E3   = 0.5 * np.sum(field_hat * np.conj(field_hat), axis=1).real 

    # Wavenumbers (radians/length)
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    kz = 2*np.pi*np.fft.fftfreq(Nz, d=Lz/Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()

    # Bin setup
    dk = float(min(abs(kx[1]-kx[0]), abs(ky[1]-ky[0]), abs(kz[1]-kz[0])))
    edges   = np.arange(0.0, k_mag.max() + dk, dk)
    centers = 0.5*(edges[1:] + edges[:-1])

    if spectral_dealias:
        # Mimic spectral solver: 2/3 rule per direction
        kx_cut = (2/3) * np.max(np.abs(kx))
        ky_cut = (2/3) * np.max(np.abs(ky))
        kz_cut = (2/3) * np.max(np.abs(kz))
        k_cut_mag = min(kx_cut, ky_cut, kz_cut)  # Conservative: all directions satisfied
        mask = (centers > 0) & (centers <= k_cut_mag)
    else:
        mask = centers > 0

    # Shell sum only for masked bins
    E_k_masked = []
    widths = np.diff(edges)[mask]
    E3_flat = E3.reshape(T, -1)
    for t in range(T):
        shell, _ = np.histogram(k_mag, bins=edges, weights=E3_flat[t])
        shell_masked = shell[mask] / widths
        E_k_masked.append(shell_masked)
    E_k_masked = np.stack(E_k_masked, axis=0)  # (T, K_masked)

    # Normalize by masked-bin TKE so ∫ E_norm dk = 1
    if tke_normalize:
        TKE_masked = np.sum(E_k_masked * widths, axis=1)  # (T,)
        E_k_norm = (E_k_masked.T / TKE_masked).T
        E_1D = E_k_norm.mean(axis=0)
    else:
        E_1D = E_k_masked.mean(axis=0)
    tke_physical = 0.5 * np.mean(np.sum(field**2, axis=1))

    if sanity_check:
        tke_spectral = np.sum(E_1D * widths)
        print(f'TKE physical: {tke_physical}, TKE spectral: {tke_spectral}')
        print(f'Parseval: {np.allclose(np.sum(field**2), (Nx*Ny*Nz) * np.sum(np.abs(np.fft.fftn(field, axes=(2,3,4), norm='forward'))**2))}')

    return centers[mask], E_1D

plt.figure()

data = load_box_timeseries(prefix = 'forced_boxfilter_fs8_highres', box_number=0)[0:10,:,:,:]
T, C, Nx, Ny, Nz = data.shape
Lx = Ly = Lz = 0.0256
V  = Lx*Ly*Lz
N  = Nx*Ny*Nz

print(f'Velocity field shape u(t,u,x,y,z): {data.shape}')
k_plot, E_plot = compute_isotropic_spectrum(data, Lx, Ly, Lz, spectral_dealias=True)
plt.loglog(k_plot, E_plot, marker='o', linestyle='-')


data = load_box_timeseries(prefix = 'forced_boxfilter_fs8_highres', box_number=0)[0:10,0:1,:,:]
T, C, Nx, Ny, Nz = data.shape
Lx = Ly = Lz = 0.0256
V  = Lx*Ly*Lz
N  = Nx*Ny*Nz

print(f'Velocity field shape u(t,u,x,y,z): {data.shape}')
k_plot, E_plot = compute_isotropic_spectrum(data, Lx, Ly, Lz, spectral_dealias=True)
plt.loglog(k_plot, E_plot, marker='o', linestyle='-')

data = load_box_timeseries(prefix = 'forced_boxfilter_fs8_highres', box_number=0)[0:10,1:2,:,:]
T, C, Nx, Ny, Nz = data.shape
Lx = Ly = Lz = 0.0256
V  = Lx*Ly*Lz
N  = Nx*Ny*Nz

print(f'Velocity field shape u(t,u,x,y,z): {data.shape}')
k_plot, E_plot = compute_isotropic_spectrum(data, Lx, Ly, Lz, spectral_dealias=True)
plt.loglog(k_plot, E_plot, marker='o', linestyle='-')

data = load_box_timeseries(prefix = 'forced_boxfilter_fs8_highres', box_number=0)[0:10,2:,:,:]
T, C, Nx, Ny, Nz = data.shape
Lx = Ly = Lz = 0.0256
V  = Lx*Ly*Lz
N  = Nx*Ny*Nz

print(f'Velocity field shape u(t,u,x,y,z): {data.shape}')
k_plot, E_plot = compute_isotropic_spectrum(data, Lx, Ly, Lz, spectral_dealias=True)
plt.loglog(k_plot, E_plot, marker='o', linestyle='-')


data = load_box_timeseries(prefix = 'forced_boxfilter_fs8_lowres', box_number=0)[0:10,:,:,:]
T, C, Nx, Ny, Nz = data.shape
Lx = Ly = Lz = 0.0256
V  = Lx*Ly*Lz
N  = Nx*Ny*Nz

k_plot, E_plot = compute_isotropic_spectrum(data, Lx, Ly, Lz)
plt.loglog(k_plot, E_plot, marker='o', linestyle='-')
plt.xlabel(r'$k$')
plt.ylabel(r'$E(k)$')
plt.title(f'1D Energy Spectrum (t=0)')
plt.grid(True, which='both')
# reference slope line (shifted to match a mid-spectrum point)
#ref_k = 1000   # skip small-k bins
#ref_amp = E_k_1D_timeseries[1][5] * (ref_k / centers[5])**(-5/3)
#plt.loglog(ref_k, ref_amp, 'k--', label=r'$k^{-5/3}$')

plt.savefig('spectrum_t0.png', dpi=300)
plt.close()

# --- Parseval sanity check (do this!) ---
# Physical-space TKE per snapshot:
#TKE_phys = 0.5 * np.mean(np.sum(data**2, axis=1), axis=(1,2,3))  # (t,)
# k-space TKE from 1D spectrum:
#TKE_spec = np.sum(E_k_1D_timeseries * np.diff(edges), axis=1)     # (t,)
#print('TKE_phys:', TKE_phys)
#print('TKE_spec:', TKE_spec)
print('done')






