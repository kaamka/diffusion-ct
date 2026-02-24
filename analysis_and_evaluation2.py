import numpy as np
import nibabel as nib
import glob
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from monai.transforms import Compose, Resize, ScaleIntensityRange
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from monai.metrics import MultiScaleSSIMMetric
from scipy.stats import wasserstein_distance
import SimpleITK as sitk
import pymedphys
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ===================================================
# UTILS / METRYKI
# ===================================================

def histogram_wasserstein(vol1, vol2, num_bins=100, range=None):
    v1 = vol1.ravel()
    v2 = vol2.ravel()

    if range is None:
        min_val = min(v1.min(), v2.min())
        max_val = max(v1.max(), v2.max())
        range = (min_val, max_val)

    hist1, bin_edges = np.histogram(v1, bins=num_bins, range=range, density=True)
    hist2, _ = np.histogram(v2, bins=num_bins, range=range, density=True)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    wd = wasserstein_distance(bin_centers, bin_centers, u_weights=hist1, v_weights=hist2)
    return wd


def compute_ms_ssim(vol1, vol2, mode="2d", slice_axis=2, data_range=None, device="cpu"):
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.float()

    vol1 = to_tensor(vol1).to(device)
    vol2 = to_tensor(vol2).to(device)

    if vol1.ndim != 3:
        raise ValueError("Expected input shape (H, W, D)")

    vol1 = vol1.unsqueeze(0).unsqueeze(0)
    vol2 = vol2.unsqueeze(0).unsqueeze(0)

    if data_range is None:
        min_val = torch.min(vol1.min(), vol2.min())
        max_val = torch.max(vol1.max(), vol2.max())
        data_range = max_val - min_val + 1e-6

    vol1 = (vol1 - min_val) / data_range
    vol2 = (vol2 - min_val) / data_range

    if mode == "2d":
        metric = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0)
        scores = []
        num_slices = vol1.shape[2 + slice_axis]

        for i in range(num_slices):
            sl1 = torch.index_select(vol1, 2 + slice_axis, torch.tensor(i, device=device)).squeeze(2 + slice_axis)
            sl2 = torch.index_select(vol2, 2 + slice_axis, torch.tensor(i, device=device)).squeeze(2 + slice_axis)

            if torch.std(sl1) < 1e-6 or torch.std(sl2) < 1e-6:
                continue

            score = metric(sl1, sl2)
            scores.append(score.item())

        if len(scores) == 0:
            return float("nan"), float("nan")

        return float(np.mean(scores)), float(np.std(scores))

    elif mode == "3d":
        metric = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7, weights=(0.5, 0.5))
        score = metric(vol1, vol2)
        return float(score.item()), None

    else:
        raise ValueError("mode must be '2d' or '3d'")


def mattes_mutual_information(reference: np.ndarray, evaluation: np.ndarray, num_bins: int = 50) -> float:
    ref_img = sitk.GetImageFromArray(reference.astype(np.float32))
    eval_img = sitk.GetImageFromArray(evaluation.astype(np.float32))

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=num_bins)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(learningRate=0.0, numberOfIterations=1)
    registration.SetInitialTransform(sitk.Transform(), inPlace=False)
    registration.Execute(ref_img, eval_img)
    mi = -registration.GetMetricValue()
    return mi


def dice_coefficient(mask1, mask2, epsilon=1e-6):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.sum(mask1 & mask2)
    return (2. * intersection) / (np.sum(mask1) + np.sum(mask2) + epsilon)


def jaccard_index(mask1, mask2, epsilon=1e-6):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return (intersection + epsilon) / (union + epsilon)


# ===================================================
# KLASA VOLUMEINFO
# ===================================================

@dataclass
class VolumeInfo:
    path: str
    data: np.ndarray = None
    patient_id: Optional[str] = None
    is_real: bool = True
    source: str = "real"


# ===================================================
# KLASA COMPAREMEDICALIMAGES
# ===================================================

class CompareMedicalImages:
    def __init__(self, volumes: list[VolumeInfo], do_preprocessing=True, spatial_size=(400,400,32), win_lev=60, win_wid=400):
        self.do_preprocessing = do_preprocessing
        self.spatial_size = spatial_size
        self.win_lev = win_lev
        self.win_wid = win_wid
        self.volumes: list[VolumeInfo] = []

        for v in volumes:
            data = self.load_volume(v.path, do_preprocessing if v.is_real else False, spatial_size, win_lev, win_wid)
            self.volumes.append(VolumeInfo(path=v.path, data=data, patient_id=v.patient_id, is_real=v.is_real, source=v.source))

        self.images_metrics_results = []

    @staticmethod
    def load_volume(path, do_preprocessing=False, spatial_size=(400,400,32), win_lev=60, win_wid=400):
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)

        if do_preprocessing:
            data = np.expand_dims(data, axis=0)
            transforms = Compose([
                Resize(spatial_size=spatial_size),
                ScaleIntensityRange(
                    a_min=win_lev - win_wid/2,
                    a_max=win_lev + win_wid/2,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                )
            ])
            data = transforms(data)
            data = np.squeeze(data, axis=0)

        return data

    @staticmethod
    def compute_image_metrics_pair(img1, img2):
        mse = compare_mse(img1, img2)
        psnr = compare_psnr(img1, img2, data_range=img2.max() - img2.min())
        mi = mattes_mutual_information(img1, img2)
        ms_ssim = compute_ms_ssim(img1, img2)
        ms_ssim_3d = compute_ms_ssim(img1, img2, mode="3d")
        wd = histogram_wasserstein(img1, img2)

        return {
            "MSE": mse,
            "PSNR": psnr,
            "Mutual-Information": mi,
            "MS-SSIM_mean": ms_ssim[0],
            "MS-SSIM_std": ms_ssim[1],
            "MS-SSIM-3D": ms_ssim_3d[0],
            "Wasserstein-Distance": wd
        }

    def results_to_dataframe(self):
        if not self.images_metrics_results:
            raise RuntimeError("Run compute_image_similarity_metrics() first")
        df = pd.DataFrame(self.images_metrics_results)
        df["pair_id"] = df["i"].astype(str) + "-" + df["j"].astype(str)
        return df

    def compute_image_similarity_metrics(self, skip_comparison_types=[]):
        results = []
        N = len(self.volumes)

        for i in range(N):
            for j in range(i, N):
                v1 = self.volumes[i]
                v2 = self.volumes[j]

                if self._comparison_type(v1, v2) in skip_comparison_types:
                    continue


                metrics = self.compute_image_metrics_pair(v1.data, v2.data)

                record = {
                    "i": i,
                    "j": j,
                    "i_path_short": os.path.basename(v1.path),
                    "j_path_short": os.path.basename(v2.path),
                    "pair": (i,j),
                    "pair_type": "self" if i==j else "cross",
                    "same_patient": v1.patient_id==v2.patient_id,
                    "source_i": v1.source,
                    "source_j": v2.source,
                    "is_real_i": v1.is_real,
                    "is_real_j": v2.is_real,
                    "comparison_type": self._comparison_type(v1,v2),
                }
                record.update(metrics)
                results.append(record)

        self.images_metrics_results = results
        return results

    @staticmethod
    def _comparison_type(v1, v2):
        if v1.is_real and v2.is_real:
            return "real-real"
        if not v1.is_real and not v2.is_real:
            return "gen-gen"
        return "real-gen"

    def plot_metric_boxplot(self, metric_name="PSNR", groupby="comparison_type", drop_self=True, figsize=(7,5), ylim=None):
        df = self.results_to_dataframe()
        if drop_self:
            df = df[df["pair_type"] != "self"]

        plt.figure(figsize=figsize)
        sns.boxplot(data=df, x=groupby, y=metric_name)
        sns.stripplot(data=df, x=groupby, y=metric_name, color="black", size=3, alpha=0.4)

        if ylim is not None:
            plt.ylim(ylim)

        plt.title(f"{metric_name} by Comparison Type")
        plt.tight_layout()
        return plt.gcf()


# ===================================================
# MAIN
# ===================================================

def main():
    output_csv = "image_similarity_metrics_1genvs_all_real.csv"
    base_name = Path(output_csv).stem

    real_paths = sorted(glob.glob("/ravana/d3d_work/common/DATA_med_img/augm/images_all/data_v2_prostate_32slices_34_plus_63/*.nii.gz"))
    gen_paths = ["/home/kamkal/diffusion-ct/augm/DM/diff_model_gen/ct_512_data_134_4399/default/sample_11.nii.gz"]

    real_volumes = [VolumeInfo(path=p, patient_id=Path(p).stem, is_real=True) for p in real_paths]
    gen_volumes = [VolumeInfo(path=p, patient_id=Path(p).stem, is_real=False) for p in gen_paths]

    volumes = real_volumes + gen_volumes

    compare_imgs = CompareMedicalImages(volumes, do_preprocessing=True, spatial_size=(512,512,32), win_lev=60, win_wid=400)

    compare_imgs.compute_image_similarity_metrics(skip_comparison_types=["real-real","gen-gen"])

    df = compare_imgs.results_to_dataframe()
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV → {output_csv}")

    metrics_to_plot = ["PSNR", "MS-SSIM_mean", "Wasserstein-Distance"]
    for metric in metrics_to_plot:
        fig = compare_imgs.plot_metric_boxplot(metric_name=metric, groupby="comparison_type")
        png_name = f"{base_name}_{metric}.png"
        fig.savefig(png_name, dpi=300)
        plt.close(fig)
        print(f"Saved plot → {png_name}")


if __name__ == "__main__":
    main()
