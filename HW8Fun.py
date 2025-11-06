import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf

def produce_trun_mean_cov(input_signal, input_type, E_val):

    X = np.asarray(input_signal)
    y = np.asarray(input_type).squeeze()
    n_samples, feature_len = X.shape
    L = feature_len // E_val
    X_reshaped = X.reshape(n_samples, E_val, L)
    mask_tar = (y == 1)
    mask_ntar = (y == 0)
    X_tar = X_reshaped[mask_tar]
    X_ntar = X_reshaped[mask_ntar]

    signal_tar_mean = X_tar.mean(axis=0) if X_tar.size else np.zeros((E_val, L))
    signal_ntar_mean = X_ntar.mean(axis=0) if X_ntar.size else np.zeros((E_val, L))

    def per_electrode_cov(stack):
        covs = np.zeros((E_val, L, L))
        for e in range(E_val):
            if stack.shape[0] >= 2:
                covs[e] = np.cov(stack[:, e, :], rowvar=False)  # ddof=1 by default (sample cov)
            else:
                covs[e] = np.zeros((L, L))
        return covs

    signal_tar_cov = per_electrode_cov(X_tar)
    signal_ntar_cov = per_electrode_cov(X_ntar)
    signal_all_cov = per_electrode_cov(X_reshaped)

    return [signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov]


def plot_trunc_mean(
        eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls,
        y_limit=None, fig_size=(12, 12)
):
    desired_first_col = ["F3", "C3", "CP3", "P4"]
    grid = [[None] * 4 for _ in range(4)]
    used = []

    for r, name in enumerate(desired_first_col):
        if name in electrode_name_ls:
            grid[r][0] = electrode_name_ls.index(name)
            used.append(electrode_name_ls.index(name))

    remaining = [i for i in range(E_val) if i not in used]
    k = 0
    for r in range(4):
        for c in range(4):
            if grid[r][c] is None:
                grid[r][c] = remaining[k]
                k += 1

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for r in range(4):
        for c in range(4):
            idx = grid[r][c]
            ax = axes[r, c]
            ax.plot(time_index, eeg_tar_mean[idx], color='red', label='Target')
            ax.plot(time_index, eeg_ntar_mean[idx], color='blue', label='Non-Target')
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (μV)")
            ax.set_title(electrode_name_ls[idx])
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend()

    fig.suptitle(f"Subject: {subject_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig("K114/Mean.png")



def plot_trunc_cov(
        eeg_cov, cov_type, time_index, subject_name, E_val, electrode_name_ls, fig_size=(12,12)
):
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    X, Y = np.meshgrid(time_index, time_index)
    first_col = ["F3", "C3", "CP3", "P4"]
    grid = [[None]*4 for _ in range(4)]
    used = []
    for r, name in enumerate(first_col):
        if name in electrode_name_ls:
            grid[r][0] = electrode_name_ls.index(name)
            used.append(electrode_name_ls.index(name))
    remaining = [i for i in range(E_val) if i not in used]
    for r in range(4):
        for c in range(4):
            if grid[r][c] is None:
                grid[r][c] = remaining.pop(0)

    for r in range(4):
        for c in range(4):
            idx = grid[r][c]
            ax = axes[r][c]
            cs = ax.contourf(X, Y, eeg_cov[idx], cmap='viridis')
            ax.set_title(electrode_name_ls[idx])
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Time (ms)")
            ax.invert_yaxis()  # so time increases top→bottom
    fig.colorbar(cs, ax=axes, shrink=0.8, label="Covariance")
    fig.suptitle(f"{cov_type} — Subject: {subject_name}")
    if cov_type == "Target":
        fname = "Covariance_Target.png"
    elif cov_type == "Non-Target":
        fname = "Covariance_Non-Target.png"
    else:
        fname = "Covariance_All.png"

    plt.savefig(os.path.join("K114", fname))
