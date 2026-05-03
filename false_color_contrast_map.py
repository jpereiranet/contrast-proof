import os
from dataclasses import dataclass

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FalseColorContrastMapResult:
    output_path: str
    total_pixels: int
    shadow_clipped_pixels: int
    highlight_clipped_pixels: int
    mapped_pixels: int
    mean_contrast: float
    p95_contrast: float


class FalseColorContrastMap:
    """
    Traduce el mapa de contraste por CV a una imagen en falso color.

    El script principal calcula la curva, los umbrales y |dJND/dCV|. Esta clase
    solo proyecta esos resultados sobre los tonos reales de la imagen.
    """

    SHADOW_COLOR = np.array([0, 0, 0], dtype=np.uint8)
    HIGHLIGHT_COLOR = np.array([180, 180, 180], dtype=np.uint8)

    def __init__(
        self,
        gray_image,
        contrast_by_cv,
        contrast_norm_by_cv,
        contrast_vmax,
        heatmap_label,
        shadow_clip_cv=None,
        highlight_clip_cv=None,
        cmap_name="plasma",
    ):
        self.gray_image = np.asarray(gray_image)
        self.contrast_by_cv = self._as_cv_lut(contrast_by_cv, fill_value=0.0)
        self.contrast_norm_by_cv = np.clip(
            self._as_cv_lut(contrast_norm_by_cv, fill_value=0.0),
            0.0,
            1.0,
        )
        self.contrast_vmax = float(contrast_vmax) if np.isfinite(contrast_vmax) else 1.0
        if self.contrast_vmax <= 0:
            self.contrast_vmax = 1.0
        self.heatmap_label = heatmap_label
        self.shadow_clip_cv = shadow_clip_cv
        self.highlight_clip_cv = highlight_clip_cv
        self.cmap = plt.get_cmap(cmap_name)

        if self.gray_image.ndim != 2:
            raise ValueError("gray_image debe ser una imagen monocanal.")

    @staticmethod
    def _as_cv_lut(values, fill_value=0.0):
        lut = np.asarray(values, dtype=float)
        if lut.shape != (256,):
            raise ValueError("El mapa de contraste debe tener 256 valores, uno por CV.")
        return np.nan_to_num(lut, nan=fill_value, posinf=fill_value, neginf=fill_value)

    def render_rgb(self):
        gray_u8 = np.clip(self.gray_image, 0, 255).astype(np.uint8)
        color_lut = (self.cmap(self.contrast_norm_by_cv)[:, :3] * 255).astype(np.uint8)
        false_color = color_lut[gray_u8]

        shadow_mask = self.shadow_mask(gray_u8)
        highlight_mask = self.highlight_mask(gray_u8)
        false_color[shadow_mask] = self.SHADOW_COLOR
        false_color[highlight_mask] = self.HIGHLIGHT_COLOR

        return false_color

    def shadow_mask(self, gray_u8=None):
        if gray_u8 is None:
            gray_u8 = np.clip(self.gray_image, 0, 255).astype(np.uint8)
        if self.shadow_clip_cv is None:
            return np.zeros(gray_u8.shape, dtype=bool)
        return gray_u8 <= float(self.shadow_clip_cv)

    def highlight_mask(self, gray_u8=None):
        if gray_u8 is None:
            gray_u8 = np.clip(self.gray_image, 0, 255).astype(np.uint8)
        if self.highlight_clip_cv is None:
            return np.zeros(gray_u8.shape, dtype=bool)
        return gray_u8 > float(self.highlight_clip_cv)

    def mapped_mask(self, gray_u8=None):
        if gray_u8 is None:
            gray_u8 = np.clip(self.gray_image, 0, 255).astype(np.uint8)
        return ~(self.shadow_mask(gray_u8) | self.highlight_mask(gray_u8))

    def summarize(self):
        gray_u8 = np.clip(self.gray_image, 0, 255).astype(np.uint8)
        shadow_mask = self.shadow_mask(gray_u8)
        highlight_mask = self.highlight_mask(gray_u8)
        mapped_mask = ~(shadow_mask | highlight_mask)
        mapped_contrast = self.contrast_by_cv[gray_u8[mapped_mask]]
        mapped_contrast = mapped_contrast[np.isfinite(mapped_contrast)]

        if mapped_contrast.size:
            mean_contrast = float(np.mean(mapped_contrast))
            p95_contrast = float(np.percentile(mapped_contrast, 95))
        else:
            mean_contrast = float("nan")
            p95_contrast = float("nan")

        return {
            "total_pixels": int(gray_u8.size),
            "shadow_clipped_pixels": int(np.sum(shadow_mask)),
            "highlight_clipped_pixels": int(np.sum(highlight_mask)),
            "mapped_pixels": int(np.sum(mapped_mask)),
            "mean_contrast": mean_contrast,
            "p95_contrast": p95_contrast,
        }

    def save(self, output_path, title=None, dpi=150):
        false_color = self.render_rgb()
        summary = self.summarize()

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        height, width = self.gray_image.shape
        aspect = width / max(height, 1)
        fig_width = min(14.0, max(8.0, aspect * 7.0))
        fig_height = min(10.0, max(5.0, fig_width / max(aspect, 0.1)))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")
        ax.imshow(false_color)
        ax.set_axis_off()
        if title:
            ax.set_title(title, color="white", fontsize=11, pad=10)

        sm = plt.cm.ScalarMappable(
            cmap=self.cmap,
            norm=plt.Normalize(vmin=0.0, vmax=self.contrast_vmax),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label(self.heatmap_label, color="white", fontsize=9)
        cbar.ax.tick_params(colors="white", labelsize=8)

        legend_items = [
            mpatches.Patch(color="#000000", label="shadow clipping"),
            mpatches.Patch(color="#b4b4b4", label="highlight clipping"),
        ]
        legend = ax.legend(
            handles=legend_items,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=True,
            fontsize=8,
        )
        legend.get_frame().set_facecolor("#111111")
        legend.get_frame().set_edgecolor("#555555")
        for text in legend.get_texts():
            text.set_color("white")

        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return FalseColorContrastMapResult(output_path=output_path, **summary)
