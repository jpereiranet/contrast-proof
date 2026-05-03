"""
verify_curve_vs_histo.py
========================
Curva característica del papel vs histograma de imagen.

Concepto
--------
La curva muestra la DENSIDAD MEDIDA (D_VIS) del papel en función de los
tonos de entrada de la imagen (CV 0-255).  La densidad es una magnitud
logarítmica por naturaleza, lo que permite revelar la compresión tonal
en las sombras que una escala lineal de CV oculta.

Ejes:
  X = CV de entrada (0-255), escala lineal — compartido con el histograma.
      Cada parche del step-wedge se asigna a un CV deducido de su densidad
      nominal: CV = 255 * 10^(-D_ref/gamma).
  Y = Densidad óptica medida (D_VIS) — escala que crece hacia abajo
      (más oscuro = más densidad = más abajo en el gráfico).

Comportamiento esperado de la curva:
  - Sombras (izquierda, CV bajos):  baja diferenciación local cuando la
    densidad medida apenas cambia entre tonos enviados próximos.
  - Altas luces (derecha, CV altos): curva EMPINADA → la densidad cae
    rápidamente.  Alto contraste, buena diferenciación tonal.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse
import re
import sys
import os
import matplotlib.ticker as ticker
from dataclasses import dataclass

from false_color_contrast_map import FalseColorContrastMap

# ─────────────────────── CONFIGURACIÓN ───────────────────────
CGATS_PATH = r"papeles\H_FineArte_Baryta_FB_350\Patch-Reader_chart.txt"
IMAGE_PATH = r"test-imgs\rbdey_antilope_2.tiff"
PLOT_OUTPUT_PATH = r"output\verify_curve_vs_histo.png"
FALSE_COLOR_OUTPUT_PATH = r"output\verify_curve_vs_histo_false_color.png"
GAMMA = 2.2

# Muestras de color a descartar por defecto (por SAMPLE_NAME)
DEFAULT_EXCLUDED_SAMPLES = ""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Paper characteristic curve vs image histogram."
    )
    parser.add_argument("--cgats-path", "--cgats", default=CGATS_PATH, help="Path to the CGATS file with chart measurements.")
    parser.add_argument("--image-path", "--image", default=IMAGE_PATH, help="Path to the image (tiff/jpg/png) to analyze.")
    parser.add_argument(
        "--plot-output",
        default=PLOT_OUTPUT_PATH,
        help="Output path for the curve and histogram plot."
    )
    parser.add_argument(
        "--false-color-output",
        default=FALSE_COLOR_OUTPUT_PATH,
        help=(
            "Output path for the false color image that projects the local "
            "contrast map onto the real tones of the image."
        )
    )
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Gamma value used for Density/CV relationship.")
    parser.add_argument(
        "--nominal-density-step",
        type=float,
        default=0.1,
        help="Nominal density increment between consecutive patches."
    )
    parser.add_argument(
        "--show-lstar",
        action="store_true",
        help="Also show the measured L* curve as a perceptual lightness reading."
    )
    parser.add_argument(
        "--exclude-samples",
        type=str,
        default=DEFAULT_EXCLUDED_SAMPLES,
        help="Comma-separated list of samples to exclude from CGATS (e.g., P2,Q2,R2,S2,T2,U2)"
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear scale for the Y-axis (density) instead of the default logarithmic scale."
    )
    parser.add_argument(
        "--relative-luminance-y",
        action="store_true",
        help=(
            "Draw the Y-axis as relative luminance 10^-D instead of optical density. "
            "This mode uses a linear scale."
        )
    )
    parser.add_argument(
        "--shadow-dmax-mode",
        choices=["tail-median", "percentile", "manual"],
        default="tail-median",
        help="Method to estimate the practical maximum shadow density."
    )
    parser.add_argument(
        "--shadow-dmax-manual",
        type=float,
        default=None,
        help="Manually entered practical maximum shadow density."
    )
    parser.add_argument(
        "--shadow-plateau-tolerance",
        type=float,
        default=0.08,
        help="Density tolerance to consider a sample as belonging to the shadow plateau."
    )
    parser.add_argument(
        "--shadow-gain-threshold",
        type=float,
        default=0.25,
        help="Maximum relative tonal gain to consider low differentiation in shadows."
    )
    parser.add_argument(
        "--shadow-min-patches",
        type=int,
        default=4,
        help=(
            "Minimum number of consecutive sub-threshold transitions to declare "
            "effective tonal clipping in shadows."
        )
    )
    parser.add_argument(
        "--contrast-model",
        choices=["dicom-jnd", "density-gain"],
        default="dicom-jnd",
        help=(
            "Model used to detect low tonal contrast and color the map: "
            "dicom-jnd uses perceptual JND increments; density-gain retains the previous criterion."
        )
    )
    parser.add_argument(
        "--paper-white-luminance",
        type=float,
        default=150.0,
        help=(
            "Estimated paper white luminance under viewing conditions, in cd/m². "
            "DICOM PS3.14 uses 150 cd/m² as a typical value for reflective hardcopy."
        )
    )
    parser.add_argument(
        "--jnd-threshold",
        type=float,
        default=1.5,
        help=(
            "Minimum increment between patches, in DICOM JND units. "
            "1.0 represents a barely perceptible difference; 1.5-2.0 is more conservative "
            "for visible tonal detail in reflective copy."
        )
    )
    parser.add_argument(
        "--lstar-step-threshold",
        type=float,
        default=1.5,
        help=(
            "Maximum increment |ΔL*| between consecutive samples to declare "
            "low tonal differentiation."
        )
    )
    parser.add_argument(
        "--contrast-decision",
        choices=["any", "all"],
        default="any",
        help=(
            "Rule to combine ΔJND and ΔL*. "
            "'any' marks low differentiation if either metric falls below threshold; "
            "'all' requires both metrics to be sub-threshold."
        )
    )
    parser.add_argument(
        "--no-lstar-evidence",
        action="store_true",
        help="Disable the use of |ΔL*| as auxiliary evidence of sub-threshold contrast."
    )
    parser.add_argument(
        "--shadow-max-gap-patches",
        type=int,
        default=1,
        help=(
            "Maximum number of isolated non-sub-threshold transitions allowed within a "
            "low-contrast shadow region. Corrects false clips due to measurement noise."
        )
    )
    return parser.parse_args(argv)


def resolve_path(path, base_dir):
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def column_name_to_index(name):
    index = 0
    for char in name.upper():
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def sample_order_key(record):
    sample_name = record["SAMPLE_NAME"]
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", sample_name)
    if not match:
        return (sys.maxsize, sys.maxsize, sample_name)
    col_name, row_name = match.groups()
    return (int(row_name) - 1, column_name_to_index(col_name), sample_name)


def parse_cgats(filepath):
    """Lee un archivo CGATS.17."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    fields, in_data, records = [], False, []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "BEGIN_DATA_FORMAT":
            i += 1; fields = lines[i].strip().split(); i += 1; continue
        if line == "BEGIN_DATA":
            in_data = True; i += 1; continue
        if line == "END_DATA":
            break
        if in_data and fields:
            parts = line.split()
            if len(parts) == len(fields):
                records.append({k: v for k, v in zip(fields, parts)})
        i += 1
    return fields, records


def density_to_cv(density, gamma=2.2):
    """D_VIS absoluta → CV (0-255).  SIN normalizar al blanco del papel."""
    return 255.0 * (10.0 ** (-density)) ** (1.0 / gamma)


def monotonic_shadow_envelope(d_values):
    """
    Fuerza una envolvente no decreciente desde luces hacia sombras.
    No sustituye la curva medida; solo sirve para estimar Dmax práctico
    y aplicar criterios de bajo contraste heredados.
    """
    d_values = np.asarray(d_values, dtype=float)
    return np.maximum.accumulate(d_values)


def fill_short_false_gaps(mask, max_gap=0):
    """
    Rellena huecos falsos cortos dentro de una región positiva.

    En la cola de sombras, las mediciones densitométricas y L* pueden oscilar
    por ruido instrumental. Un único parche con ΔJND ligeramente alto no debe
    fragmentar una meseta que, considerada como región, ya no reproduce detalle
    de forma fiable. Solo se rellenan huecos acotados por valores verdaderos.
    """
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or mask.size == 0:
        return mask

    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue

        start = i
        while i < n and not mask[i]:
            i += 1
        end = i
        gap_len = end - start

        has_true_left = start > 0 and mask[start - 1]
        has_true_right = end < n and mask[end]
        if has_true_left and has_true_right and gap_len <= max_gap:
            mask[start:end] = True

    return mask


def find_first_shadow_low_contrast_transition(mask, min_run=4):
    """
    Busca desde las sombras hacia las luces el comienzo efectivo de una región
    de baja diferenciación tonal. El índice devuelto corresponde a la muestra i,
    cuya transición respecto a la muestra i-1 es subumbral.
    """
    mask = np.asarray(mask, dtype=bool)

    run = 0
    start_idx = None

    for i in range(len(mask) - 1, -1, -1):
        if mask[i]:
            run += 1
            start_idx = i
        else:
            if run >= min_run:
                return start_idx
            run = 0
            start_idx = None

    if run >= min_run:
        return start_idx

    return None


def density_to_print_luminance(density, d_min, paper_white_luminance=150.0):
    """
    Densidad óptica reflectiva -> luminancia de observación estimada.

    Se toma la primera muestra neutra como blanco efectivo del papel:
        L(D) = L_white * 10^(-(D - D_min))

    Esta forma mantiene la física densitométrica y evita depender de un
    L0 ideal no medido. Si se dispone de medición fotométrica real del
    blanco del papel, debe pasarse mediante --paper-white-luminance.
    """
    density = np.asarray(density, dtype=float)
    return paper_white_luminance * np.power(10.0, -(density - d_min))


def density_to_relative_luminance(density):
    """Densidad óptica absoluta -> luminancia relativa."""
    density = np.asarray(density, dtype=float)
    return np.power(10.0, -density)


def dicom_jnd_index(luminance):
    """
    Índice JND de DICOM PS3.14 a partir de luminancia absoluta (cd/m²).
    Implementa la ecuación inversa j(L) del GSDF.
    """
    luminance = np.asarray(luminance, dtype=float)
    safe_l = np.clip(luminance, 0.05, 4000.0)
    log_l = np.log10(safe_l)

    A = 71.498068
    B = 94.593053
    C = 41.912053
    D = 9.8247004
    E = 0.28175407
    F = -1.1878455
    G = -0.18014349
    H = 0.14710899
    I = -0.017046845

    return (
        A + B * log_l + C * log_l**2 + D * log_l**3 + E * log_l**4
        + F * log_l**5 + G * log_l**6 + H * log_l**7 + I * log_l**8
    )


def compute_adjacent_perceptual_metrics(density_values, d_min, paper_white_luminance):
    """
    Calcula métricas de contraste entre cada muestra y la muestra previa
    en el eje de la cuña: luminancia, índice JND, ΔJND, Weber y Michelson.
    """
    luminance = density_to_print_luminance(
        density_values, d_min=d_min, paper_white_luminance=paper_white_luminance
    )
    jnd_index = dicom_jnd_index(luminance)

    delta_jnd = np.full_like(jnd_index, np.nan, dtype=float)
    weber = np.full_like(jnd_index, np.nan, dtype=float)
    michelson = np.full_like(jnd_index, np.nan, dtype=float)

    if len(jnd_index) > 1:
        l0 = luminance[:-1]
        l1 = luminance[1:]
        delta_l = np.abs(l1 - l0)
        delta_jnd[1:] = np.abs(np.diff(jnd_index))
        weber[1:] = delta_l / np.maximum(np.maximum(l0, l1), 1e-12)
        michelson[1:] = delta_l / np.maximum(l0 + l1, 1e-12)

    return luminance, jnd_index, delta_jnd, weber, michelson


def configure_cli_encoding():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# LIBRARY API
@dataclass
class CurveVsHistoConfig:
    cgats_path: str = CGATS_PATH
    image_path: str = IMAGE_PATH
    plot_output: str = PLOT_OUTPUT_PATH
    false_color_output: str = FALSE_COLOR_OUTPUT_PATH
    gamma: float = GAMMA
    nominal_density_step: float = 0.1
    show_lstar: bool = False
    exclude_samples: str = DEFAULT_EXCLUDED_SAMPLES
    linear_y: bool = False
    relative_luminance_y: bool = False
    shadow_dmax_mode: str = "tail-median"
    shadow_dmax_manual: object = None
    shadow_plateau_tolerance: float = 0.08
    shadow_gain_threshold: float = 0.25
    shadow_min_patches: int = 4
    contrast_model: str = "dicom-jnd"
    paper_white_luminance: float = 150.0
    jnd_threshold: float = 1.5
    lstar_step_threshold: float = 1.5
    contrast_decision: str = "any"
    no_lstar_evidence: bool = False
    shadow_max_gap_patches: int = 1


@dataclass
class CurveVsHistoResult:
    plot_output_path: object
    false_color_output_path: object
    cgats_path: str
    image_path: str
    cv_shadow_clip: object
    d_shadow_clip: object
    shadow_clip_transition: object
    cv_highlight_clip: float
    d_highlight_clip: float
    dmax_practical: float
    dmax_practical_source: str
    contrast_model_description: str
    shadow_clipped_pixels: int
    highlight_clipped_pixels: int
    main_region_pixels: int
    false_color_result: object
    curve_data: object
    image_data: object


class CurveVsHistoVerifier:
    def __init__(
        self,
        config=None,
        base_dir=None,
        show_plot=True,
        save_plot=True,
        save_false_color=True,
        print_report=True,
        **overrides,
    ):
        if config is None:
            config = CurveVsHistoConfig()
        elif isinstance(config, argparse.Namespace):
            config = CurveVsHistoConfig(**vars(config))
        elif isinstance(config, dict):
            config = CurveVsHistoConfig(**config)
        elif not isinstance(config, CurveVsHistoConfig):
            raise TypeError("config debe ser CurveVsHistoConfig, argparse.Namespace, dict o None.")

        for key, value in overrides.items():
            if not hasattr(config, key):
                raise TypeError(f"Parametro de configuracion desconocido: {key}")
            setattr(config, key, value)

        self.config = config
        self.args = config
        self.base_dir = os.path.abspath(base_dir) if base_dir else os.path.dirname(os.path.abspath(__file__))
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.save_false_color = save_false_color
        self.print_report = print_report
        self.curve_data = None
        self.image_data = None
        self.result = None

    @classmethod
    def from_args(cls, args, **runtime_options):
        return cls(config=vars(args), **runtime_options)

    def run(self):
        args = self.args
        emit = print if self.print_report else (lambda *args, **kwargs: None)
        script_dir = self.base_dir
        cgats_full = resolve_path(args.cgats_path, script_dir)
        image_full = resolve_path(args.image_path, script_dir)
        false_color_output_full = resolve_path(args.false_color_output, script_dir)
        gamma = args.gamma

        # ── Leer CGATS ──
        emit(f"Leyendo CGATS: {cgats_full}")
        _fields, records = parse_cgats(cgats_full)

        # ── Filtrar muestras neutras ──
        excluded_samples = {s.strip() for s in args.exclude_samples.split(",") if s.strip()}
        emit(f"Excluyendo muestras: {', '.join(sorted(excluded_samples))}")

        neutral = sorted(
            [r for r in records if r["SAMPLE_NAME"] not in excluded_samples],
            key=sample_order_key,
        )
        if not neutral:
            raise ValueError("No se encontraron muestras neutras.")

        n = len(neutral)
        names  = [r["SAMPLE_NAME"] for r in neutral]
        d_vis  = np.array([float(r["D_VIS"]) for r in neutral])
        lab_l  = np.array([float(r["LAB_L"]) for r in neutral])
        emit(f"  Muestras neutras: {n}  ({names[0]} → {names[-1]})")

        # ─────────────────────────────────────────────────────────────
        # CONSTRUCCIÓN CORRECTA DE LA CURVA DENSITOMÉTRICA
        # ─────────────────────────────────────────────────────────────

        # 1. Densidad nominal relativa de entrada de la carta:
        #    0, 0.1, 0.2, 0.3, ...
        d_ref = np.arange(n, dtype=float) * args.nominal_density_step

        # 2. Valores digitales realmente enviados, deducidos de la densidad nominal
        cv_in = 255.0 * 10.0 ** (-d_ref / gamma)

        # 3. Salidas medidas
        d_abs = d_vis.copy()
        d_min = d_abs[0]
        d_max = np.nanmax(d_abs)
        lstar = lab_l.copy()

        # ───────────────── UMBRAL DE CONTRASTE SUBUMBRAL EN SOMBRAS ─────────────────
        d_analysis = monotonic_shadow_envelope(d_abs)

        if args.shadow_dmax_mode == "manual":
            if args.shadow_dmax_manual is None:
                raise ValueError("--shadow-dmax-mode manual requiere --shadow-dmax-manual.")
            dmax_practical = args.shadow_dmax_manual
            dmax_practical_source = "manual"
        elif args.shadow_dmax_mode == "percentile":
            dmax_practical = np.nanpercentile(d_analysis, 95)
            dmax_practical_source = "percentil 95"
        else:
            tail_count = max(5, int(round(0.20 * len(d_analysis))))
            dmax_practical = np.nanmedian(d_analysis[-tail_count:])
            dmax_practical_source = f"mediana cola {tail_count}"

        if dmax_practical < d_min:
            raise ValueError(
                f"la densidad máxima práctica ({dmax_practical:.3f}) "
                f"no puede ser menor que la densidad del papel ({d_min:.3f})."
            )

        near_dmax = d_analysis >= (dmax_practical - args.shadow_plateau_tolerance)
        delta_d = np.diff(d_analysis, prepend=np.nan)
        gain = np.clip(delta_d / args.nominal_density_step, 0, None)
        low_gain = gain <= args.shadow_gain_threshold
        legacy_density_gain_low_contrast_mask = near_dmax & low_gain

        (
            patch_luminance,
            patch_jnd_index,
            patch_delta_jnd,
            patch_weber,
            patch_michelson,
        ) = compute_adjacent_perceptual_metrics(
            d_analysis,
            d_min=d_min,
            paper_white_luminance=args.paper_white_luminance,
        )

        patch_delta_lstar = np.full_like(lstar, np.nan, dtype=float)
        if len(lstar) > 1:
            patch_delta_lstar[1:] = np.abs(np.diff(lstar))

        low_jnd = np.isfinite(patch_delta_jnd) & (patch_delta_jnd <= args.jnd_threshold)
        low_lstar = (
            np.isfinite(patch_delta_lstar)
            & (patch_delta_lstar <= args.lstar_step_threshold)
        )

        if args.contrast_model == "density-gain":
            shadow_low_contrast_mask_raw = legacy_density_gain_low_contrast_mask
            contrast_step = gain
            contrast_step_label = "Gain"
            contrast_model_description = (
                f"ganancia densitométrica: Dmax±{args.shadow_plateau_tolerance:.3f}, "
                f"gain≤{args.shadow_gain_threshold:.2f}"
            )
        else:
            if args.no_lstar_evidence:
                shadow_low_contrast_mask_raw = low_jnd
            elif args.contrast_decision == "all":
                shadow_low_contrast_mask_raw = low_jnd & low_lstar
            else:
                shadow_low_contrast_mask_raw = low_jnd | low_lstar

            contrast_step = patch_delta_jnd
            contrast_step_label = "ΔJND"
            lstar_part = "" if args.no_lstar_evidence else (
                f", ΔL*≤{args.lstar_step_threshold:.2f}, decisión={args.contrast_decision}"
            )
            contrast_model_description = (
                f"DICOM GSDF/JND: L_blanco={args.paper_white_luminance:.1f} cd/m², "
                f"ΔJND≤{args.jnd_threshold:.2f}{lstar_part}"
            )

        shadow_low_contrast_mask = fill_short_false_gaps(
            shadow_low_contrast_mask_raw,
            max_gap=args.shadow_max_gap_patches,
        )

        shadow_low_contrast_idx = find_first_shadow_low_contrast_transition(
            shadow_low_contrast_mask,
            min_run=args.shadow_min_patches
        )
        if shadow_low_contrast_idx is not None:
            if shadow_low_contrast_idx > 0:
                cv_shadow_clip = float(
                    (cv_in[shadow_low_contrast_idx - 1] + cv_in[shadow_low_contrast_idx]) / 2.0
                )
                d_shadow_clip = float(
                    (d_abs[shadow_low_contrast_idx - 1] + d_abs[shadow_low_contrast_idx]) / 2.0
                )
                shadow_clip_transition = (
                    names[shadow_low_contrast_idx - 1],
                    names[shadow_low_contrast_idx],
                )
            else:
                cv_shadow_clip = float(cv_in[shadow_low_contrast_idx])
                d_shadow_clip = float(d_abs[shadow_low_contrast_idx])
                shadow_clip_transition = (None, names[shadow_low_contrast_idx])
        else:
            cv_shadow_clip = None
            d_shadow_clip = None
            shadow_clip_transition = None

        # 4. Límites absolutos equivalentes del medio
        #    Estos NO sirven para colocar los parches.
        #    Solo sirven para marcar límites absolutos de blanco y negro.
        cv_equiv_white_abs = 255.0 * 10.0 ** (-d_min / gamma)
        cv_equiv_black_abs = 255.0 * 10.0 ** (-d_max / gamma)
        cv_highlight_clip = float(cv_equiv_white_abs)
        d_highlight_clip = float(d_min)

        # 5. Curva objetivo nominal esperada, expresada en densidad absoluta
        d_target_abs = d_min + d_ref

        # 6. Ordenar por CV ascendente para interpolar la curva medida.
        sort_idx = np.argsort(cv_in)

        x_sorted = cv_in[sort_idx]
        d_abs_sorted = d_abs[sort_idx]
        lstar_sorted = lstar[sort_idx]

        full_x = np.arange(256, dtype=float)

        # 7. Curva característica medida. No se impone meseta artificial.
        full_d_abs_raw = np.interp(
            full_x,
            x_sorted,
            d_abs_sorted,
            left=np.nan,
            right=np.nan
        )

        # 8. Curva de visualización fuera del rango caracterizado.
        #    No representa nueva medición; solo prolonga el último valor oscuro.
        full_d_abs_display = full_d_abs_raw.copy()
        cv_min_sent = np.nanmin(cv_in)
        cv_max_sent = np.nanmax(cv_in)
        darkest_density_measured = d_abs_sorted[0]
        full_d_abs_display[full_x < cv_min_sent] = darkest_density_measured

        # 9. Curva L* auxiliar
        full_lstar = np.interp(full_x, x_sorted, lstar_sorted, left=np.nan, right=np.nan)

        # 10. Recorte de curvas según el rango de valores enviados
        invalid_curve_mask = (full_x < cv_min_sent) | (full_x > cv_max_sent)

        # 11. Pendientes sobre la curva medida, sin imponer meseta.
        slope_d = np.abs(np.gradient(full_d_abs_display))
        full_luminance = density_to_print_luminance(
            full_d_abs_display,
            d_min=d_min,
            paper_white_luminance=args.paper_white_luminance,
        )
        full_jnd_index = dicom_jnd_index(full_luminance)
        slope_jnd = np.abs(np.gradient(full_jnd_index))
        full_relative_luminance = density_to_relative_luminance(full_d_abs_display)
        slope_relative_luminance = np.abs(np.gradient(full_relative_luminance))

        # Normalización del mapa de calor. Si el gráfico se representa en
        # luminancia relativa, el color sigue la pendiente visible de esa misma
        # curva. En la vista densitométrica se conserva el modelo de contraste
        # elegido. El detector de subumbral sigue usando el criterio perceptual.
        if args.relative_luminance_y:
            heatmap_slope = slope_relative_luminance.copy()
            heatmap_label = "Local relative-luminance contrast: |d(10^-D)/dCV|"
        elif args.contrast_model == "dicom-jnd":
            heatmap_slope = slope_jnd.copy()
            heatmap_label = "Local perceptual contrast: |dJND/dCV|"
        else:
            heatmap_slope = slope_d.copy()
            heatmap_label = "Local densitometric contrast: |dD_VIS/dCV|"
        heatmap_valid = (
            np.isfinite(heatmap_slope) &
            (full_x >= cv_min_sent) &
            (full_x <= cv_max_sent) &
            (full_x <= cv_highlight_clip)
        )
        if cv_shadow_clip is not None:
            heatmap_valid &= full_x > cv_shadow_clip

        heatmap_reference = heatmap_slope[heatmap_valid]
        heatmap_reference = heatmap_reference[np.isfinite(heatmap_reference)]
        v_max = np.percentile(heatmap_reference, 99) if len(heatmap_reference) > 0 else 1
        if v_max <= 0:
            v_max = np.nanmax(heatmap_reference) if len(heatmap_reference) > 0 else 1
        if not np.isfinite(v_max) or v_max <= 0:
            v_max = 1

        slope_norm = np.zeros_like(heatmap_slope)
        slope_norm[heatmap_valid] = np.clip(heatmap_slope[heatmap_valid] / v_max, 0, 1)

        # 12. Referencia nominal gamma en términos absolutos
        cv_safe = np.clip(full_x, 1, 255)
        d_ref_from_cv = -gamma * np.log10(cv_safe / 255.0)
        d_target_abs_interp = d_min + d_ref_from_cv
        d_target_abs_interp[invalid_curve_mask] = np.nan

        # ── Tabla ──
        emit("\n  Límites efectivos del papel:")
        emit(
            f"  Blanco papel: Dmin={d_min:.3f} -> "
            f"umbral altas luces CV={cv_highlight_clip:.1f} "
            f"(valores >CV {cv_highlight_clip:.1f} recortan al blanco del papel)"
        )
        emit(
            f"  Negro práctico ({dmax_practical_source}): "
            f"Dmax práctico={dmax_practical:.3f}"
        )
        emit(
            f"  Negro medido absoluto: Dmax medido={d_max:.3f}, "
            f"CV equivalente={cv_equiv_black_abs:.1f}"
        )
        emit(f"  Modelo contraste: {contrast_model_description}")
        if args.shadow_max_gap_patches > 0:
            emit(f"  Detector robusto: se toleran huecos aislados ≤{args.shadow_max_gap_patches} transición(es)")
        if cv_shadow_clip is not None:
            emit(
                f"  Umbral de contraste subumbral en sombras: CV={cv_shadow_clip:.1f}, "
                f"D≈{d_shadow_clip:.3f}, transición={shadow_clip_transition}, "
                f"índice={shadow_low_contrast_idx}"
            )
        else:
            emit("  No se detectó una región continua de contraste subumbral en sombras.")

        emit(f"\n  {'Muestra':<8} {'D_ref':>7} {'CV_in':>8} {'D_target':>9} {'D_abs':>8} {contrast_step_label:>7} {'ΔL*':>7} {'Cw':>7} {'BajoC':>6} {'Error_D':>9} {'L*':>8}")
        emit(f"  {'─'*8} {'─'*7} {'─'*8} {'─'*9} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*9} {'─'*8}")
        for idx, (nm, dr, ci, dt, da, lv) in enumerate(zip(names, d_ref, cv_in, d_target_abs, d_abs, lstar)):
            err = da - dt
            low_contrast = "sí" if shadow_low_contrast_mask[idx] else "no"
            metric_text = "nan" if not np.isfinite(contrast_step[idx]) else f"{contrast_step[idx]:.2f}"
            weber_text = "nan" if not np.isfinite(patch_weber[idx]) else f"{patch_weber[idx]:.3f}"
            dlstar_text = "nan" if not np.isfinite(patch_delta_lstar[idx]) else f"{patch_delta_lstar[idx]:.2f}"
            emit(f"  {nm:<8} {dr:>7.3f} {ci:>8.2f} {dt:>9.3f} {da:>8.3f} {metric_text:>7} {dlstar_text:>7} {weber_text:>7} {low_contrast:>6} {err:>9.3f} {lv:>8.2f}")

        # ── Cargar imagen ──
        emit(f"\nCargando imagen: {image_full}")
        img = cv.imread(image_full)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar: {image_full}")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hist = cv.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        false_color_result = None
        if self.save_false_color:
            false_color_result = FalseColorContrastMap(
                gray_image=gray,
                contrast_by_cv=heatmap_slope,
                contrast_norm_by_cv=slope_norm,
                contrast_vmax=v_max,
                heatmap_label=heatmap_label,
                shadow_clip_cv=cv_shadow_clip,
                highlight_clip_cv=cv_highlight_clip,
            ).save(
                false_color_output_full,
                title=(
                    "False-color tonal contrast map | "
                    f"{os.path.basename(image_full)}"
                ),
            )
            emit(f"\nImagen falso color guardada en: {false_color_result.output_path}")
            emit(
                f"  Mapa falso color: "
                f"{false_color_result.mapped_pixels:>10,} px con contraste mapeado, "
                f"{false_color_result.shadow_clipped_pixels:>10,} px sombras recortadas, "
                f"{false_color_result.highlight_clipped_pixels:>10,} px altas luces recortadas"
            )

        # ═══════════════════════ GRÁFICO ═══════════════════════
        # Expandimos el ancho de la ventana para dar espacio a la barra
        fig, ax_hist = plt.subplots(figsize=(16, 8))
        fig.patch.set_facecolor('#1a1a2e')
        ax_hist.set_facecolor('#16213e')

        # Dejamos un margen más amplio a la derecha para la etiqueta del eje Y y la colorbar
        fig.subplots_adjust(right=0.82)

        int_shadow_clip = None
        if cv_shadow_clip is not None:
            int_shadow_clip = int(np.clip(round(cv_shadow_clip), 0, 255))
        int_white = int(np.clip(round(cv_equiv_white_abs), 0, 255))

        # ── Fondo coloreado por contraste ──
        plasma = plt.get_cmap('plasma')
        for i in range(256):
            if int_shadow_clip is not None and i <= int_shadow_clip:
                color = (0.08, 0.08, 0.12)
            elif i > int_white:
                color = (0.75, 0.75, 0.80)
            else:
                color = plasma(slope_norm[i])[:3]
            ax_hist.axvspan(i, i + 1, color=color, alpha=0.55)

        # ── Histograma (eje izquierdo) ──
        hist_max = np.max(hist) if np.max(hist) > 0 else 1
        hist_norm = hist / hist_max
        ax_hist.fill_between(np.arange(256), 0, hist_norm,
                             color='white', alpha=0.35)
        ax_hist.plot(np.arange(256), hist_norm, color='white',
                     alpha=0.5, linewidth=0.5)
        ax_hist.set_ylabel('Histogram (relative frequency)',
                           fontsize=11, color='white')
        ax_hist.set_ylim(0, 1.05)
        ax_hist.tick_params(axis='y', colors='white', labelsize=9)

        # ── Curva característica (eje derecho — DENSIDAD) ──
        ax_curve = ax_hist.twinx()
        relative_luminance_y = args.relative_luminance_y

        d_min_plot = max(d_vis[0] * 0.5, 0.01)   # límite superior (claro)
        d_max_plot = max(np.nanmax(d_vis), dmax_practical) * 1.3  # límite inferior (oscuro)

        if relative_luminance_y:
            y_ref_plot = density_to_relative_luminance(d_target_abs_interp)
            y_curve_plot = density_to_relative_luminance(full_d_abs_display)
            y_measured = density_to_relative_luminance(d_abs)
            y_dmax_practical = float(density_to_relative_luminance(dmax_practical))
            y_dmin = float(density_to_relative_luminance(d_min))
            y_shadow_clip = (
                float(density_to_relative_luminance(d_shadow_clip))
                if d_shadow_clip is not None else None
            )
            y_axis_floor = max(0.0, float(density_to_relative_luminance(d_max_plot)) * 0.85)
            y_axis_top = min(1.05, float(density_to_relative_luminance(d_min_plot)) * 1.03)
            ax_curve.set_ylim(y_axis_floor, y_axis_top)
            measured_curve_label = 'Measured curve (10^-D_VIS)'
            nominal_reference_label = 'Nominal reference (relative luminance)'
            y_axis_label = 'Relative luminance (10^-D)'
            title_quantity = 'Relative Luminance'
        else:
            if not args.linear_y:
                ax_curve.set_yscale('log')
            y_ref_plot = d_target_abs_interp.copy()
            y_curve_plot = full_d_abs_display
            y_measured = d_abs
            y_dmax_practical = dmax_practical
            y_dmin = d_min
            y_shadow_clip = d_shadow_clip
            y_axis_floor = d_max_plot
            ax_curve.set_ylim(d_max_plot, d_min_plot)  # invertido
            measured_curve_label = 'Measured curve (D_VIS)'
            nominal_reference_label = 'Nominal reference: D_min - γ log10(CV/255)'
            scale_label = 'linear axis' if args.linear_y else 'log axis'
            y_axis_label = f'Absolute density D_VIS ({scale_label})'
            title_quantity = 'Density'

        # Referencia nominal gamma
        # Ocultamos la referencia en la zona que no fue enviada a imprimir
        ax_curve.plot(full_x[1:], y_ref_plot[1:], color='gray', ls=':', lw=1.2,
                      alpha=0.5, zorder=4, label=nominal_reference_label)

        # Curva densitométrica medida del papel
        ax_curve.plot(full_x, y_curve_plot, color='#00ffff', linewidth=2.5, zorder=5,
                      label=measured_curve_label)

        if cv_min_sent > 0:
            mask_unmeasured_dark = full_x < cv_min_sent
            ax_curve.plot(
                full_x[mask_unmeasured_dark],
                y_curve_plot[mask_unmeasured_dark],
                color='#00ffff',
                linewidth=1.2,
                linestyle=':',
                alpha=0.7,
                zorder=4,
                label='Uncharacterized extension'
            )

        # Curva L* CIELAB (opcional)
        if args.show_lstar:
            ax_lstar = ax_hist.twinx()
            ax_lstar.spines['right'].set_position(('outward', 60))
            ax_lstar.plot(full_x, full_lstar, color='#ff00ff', ls='--', lw=1.5,
                          alpha=0.7, label='Lightness (L*)')
            ax_lstar.set_ylabel('CIELAB lightness (L*)', color='#ff00ff', fontsize=11)
            ax_lstar.tick_params(axis='y', colors='#ff00ff', labelsize=9)
            ax_lstar.set_ylim(0, 105)

        # Puntos medidos (D_abs)
        ax_curve.scatter(cv_in, y_measured, color='#ff6b6b', s=45, zorder=6,
                         edgecolors='white', linewidth=0.5, label='Measurements')

        # ── Líneas de contraste subumbral y blanco equivalente ──
        if cv_shadow_clip is not None:
            ax_hist.axvline(cv_shadow_clip, color='#ff4444', ls='--', lw=1.5, alpha=0.9)
        ax_hist.axvline(cv_equiv_white_abs, color='#44ff44', ls='--', lw=1.5, alpha=0.8)
        ax_curve.axhline(y_dmax_practical, color='#ff4444', ls=':', lw=1.4, alpha=0.85)
        ax_curve.axhline(y_dmin, color='#44ff44', ls=':', lw=1.2, alpha=0.75)
        ax_curve.annotate(
            f'Practical Dmax {dmax_practical:.2f}',
            xy=(248, y_dmax_practical),
            xytext=(0, 4),
            textcoords='offset points',
            ha='right',
            va='bottom',
            fontsize=8,
            color='white',
            fontweight='bold',
            alpha=0.95,
            zorder=8
        )
        if d_shadow_clip is not None:
            ax_curve.axhline(
                y_shadow_clip,
                color='#ff9999',
                ls='--',
                lw=1.0,
                alpha=0.65
            )
            ax_curve.annotate(
                f'contrast threshold D≈{d_shadow_clip:.2f}',
                xy=(248, y_shadow_clip),
                xytext=(0, 4),
                textcoords='offset points',
                ha='right',
                va='bottom',
                fontsize=8,
                color='white',
                alpha=0.95,
                zorder=8
            )
        ax_curve.text(
            255,
            y_dmin,
            f'  Paper Dmin {d_min:.2f}',
            ha='left',
            va='center',
            fontsize=8,
            color='#88ff88',
            fontweight='bold',
            alpha=0.9,
            clip_on=False
        )

        # ── Líneas de rango efectivamente enviado ──
        ax_hist.axvline(cv_min_sent, color='white', ls=':', lw=1.0, alpha=0.6)
        ax_hist.axvline(cv_max_sent, color='white', ls=':', lw=1.0, alpha=0.6)

        # Etiquetas de regiones tonales
        if cv_shadow_clip is not None and cv_shadow_clip > 10:
            ax_hist.text(cv_shadow_clip / 2, 0.95, f'SUB-THRESHOLD\nSHADOW CONTRAST\n≤ CV {cv_shadow_clip:.1f}',
                         ha='center', va='top', fontsize=8, color='#ff6666',
                         fontweight='bold', alpha=0.8)
        if int_white < 250:
            highlight_label_x = max(15, cv_highlight_clip - 20)
            ax_hist.text(highlight_label_x, 0.95, f'EFFECTIVE\nHIGHLIGHT CLIP\n> CV {cv_highlight_clip:.1f}',
                         ha='center', va='top', fontsize=8, color='#88cccc',
                         fontweight='bold', alpha=0.8)

        # ── Referencia Gris Medio (D = 0.73) ──
        target_d = 0.73
        target_y = float(density_to_relative_luminance(target_d)) if relative_luminance_y else target_d
        valid_mask = np.isfinite(full_d_abs_raw)
        if np.any(valid_mask):
            valid_x = full_x[valid_mask]
            valid_d = full_d_abs_raw[valid_mask]
            
            # valid_d decrece conforme valid_x crece. Invertimos para np.interp
            target_cv = np.interp(target_d, valid_d[::-1], valid_x[::-1])
            
            if 0 <= target_cv <= 255:
                # Trazar línea horizontal hacia la derecha (eje de densidad)
                ax_curve.plot([target_cv, 255], [target_y, target_y],
                              color='#44ff44', ls=':', lw=1.2, alpha=0.8, zorder=4)
                # Trazar línea vertical hacia abajo (eje CV)
                ax_curve.plot([target_cv, target_cv], [y_axis_floor, target_y],
                              color='#44ff44', ls=':', lw=1.2, alpha=0.8, zorder=4)

                mid_x = (target_cv + 255) / 2
                ax_curve.annotate(
                    f'count value ≈ {target_cv:.0f}',
                    xy=(mid_x, target_y),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=10, color='#44ff44', fontweight='bold',
                    zorder=7)

        # ── Formato ──
        ax_hist.set_xlim(0, 255)
        ax_hist.set_xlabel(
            'count values 8bits',
            fontsize=11, color='white')
        ax_curve.set_ylabel(y_axis_label, fontsize=11, color='white')

        if not relative_luminance_y and not args.linear_y:
            # Formatear la escala logarítmica para mostrar valores decimales claros (0.1, 0.5, 1.0...)
            ax_curve.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax_curve.yaxis.get_major_formatter().set_scientific(False)
            ax_curve.yaxis.get_major_formatter().set_useOffset(False)
            
            # Definir tics lógicos para densidad en escala logarítmica
            d_ticks = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0]
            # Filtrar solo los que estén en el rango visible
            d_ticks = [t for t in d_ticks if d_min_plot <= t <= d_max_plot]
            ax_curve.set_yticks(d_ticks)

        ax_curve.tick_params(axis='y', colors='#00ffff', labelsize=9)

        ax_hist.set_title(
            f'Paper Characteristic Curve ({title_quantity}) vs Image Histogram\n'
            f'Paper: {os.path.basename(os.path.dirname(cgats_full))}  |  '
            f'Image: {os.path.basename(image_full)}  |  γ={gamma}  |  model={args.contrast_model}',
            fontsize=13, color='white', fontweight='bold')

        ax_hist.set_xticks(np.arange(0, 256, 25))
        ax_hist.tick_params(axis='x', colors='white', labelsize=9)
        ax_hist.grid(True, ls='--', alpha=0.15, color='white')

        # Colorbar de contraste (colocada en un eje dedicado fuera del gráfico)
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
        sm.set_array([])

        # Creamos un eje específico para la barra de color aún más a la derecha
        # [left, bottom, width, height] en coordenadas de la figura
        cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(heatmap_label, fontsize=10, color='white')
        cbar.ax.tick_params(colors='white')
        output_path = resolve_path(args.plot_output, script_dir)
        if self.save_plot:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            emit(f"\nGráfico guardado en: {output_path}")
        if self.show_plot:
            plt.show()
        else:
            plt.close(fig)

        # ─────────── DIAGNÓSTICO ───────────
        emit("\n" + "=" * 60)
        emit("  DIAGNÓSTICO DE COHERENCIA")
        emit("=" * 60)

        total_px = gray.size
        if cv_shadow_clip is not None:
            shadow_clip_px = np.sum(gray <= cv_shadow_clip)
        else:
            shadow_clip_px = 0
        highlight_equiv = np.sum(gray > cv_highlight_clip)
        if cv_shadow_clip is not None:
            main_region_mask = (gray > cv_shadow_clip) & (gray <= cv_highlight_clip)
        else:
            main_region_mask = gray <= cv_highlight_clip
        in_main_region = np.sum(main_region_mask)

        if cv_shadow_clip is not None:
            emit(
                f"\n  Sombras en región de contraste subumbral "
                f"(<=CV {cv_shadow_clip:.1f}): "
                f"{shadow_clip_px:>10,} "
                f"({100*shadow_clip_px/total_px:.1f}%)"
            )
            emit(
                f"  Umbral sombras: CV={cv_shadow_clip:.1f}, "
                f"D≈{d_shadow_clip:.3f}, "
                f"transición={shadow_clip_transition}, "
                f"modelo={contrast_model_description}"
            )
        else:
            emit("\n  No se detectó una región continua de contraste subumbral en sombras.")
        emit(
            f"  Altas luces por encima del límite superior del papel "
            f"(>CV {cv_highlight_clip:.1f}): "
            f"{highlight_equiv:>10,} ({100*highlight_equiv/total_px:.1f}%)"
        )
        emit(
            f"  Umbral altas luces: CV={cv_highlight_clip:.1f}, "
            f"Dmin papel={d_highlight_clip:.3f}, "
            f"límite superior útil=CV≤{cv_highlight_clip:.1f}"
        )
        emit(f"  Zona principal no marcada:   "
              f"{in_main_region:>10,} ({100*in_main_region/total_px:.1f}%)")

        # Contraste por zonas
        tonal_start = int_shadow_clip if int_shadow_clip is not None else int(np.clip(round(cv_min_sent), 0, 255))
        rng = int_white - tonal_start
        if args.relative_luminance_y:
            zone_quantity = slope_relative_luminance
            zone_label = "|d(10^-D)/dCV|"
        elif args.contrast_model == "dicom-jnd":
            zone_quantity = slope_jnd
            zone_label = "|dJND/dCV|"
        else:
            zone_quantity = slope_d
            zone_label = "|dD_VIS/dCV|"
        emit(f"\n  Contraste de la curva por zona ({zone_label}):")
        if rng > 3:
            t = rng / 3
            zs = slice(tonal_start, int(tonal_start + t))
            zm = slice(int(tonal_start + t), int(tonal_start + 2 * t))
            zh = slice(int(tonal_start + 2 * t), int_white)

            s_s = np.mean(zone_quantity[zs]); s_m = np.mean(zone_quantity[zm]); s_h = np.mean(zone_quantity[zh])
            emit(f"    Sombras      (CV {tonal_start}–{tonal_start+int(t)}):   {s_s:.5f}")
            emit(f"    Medios tonos (CV {tonal_start+int(t)}–{tonal_start+int(2*t)}): {s_m:.5f}")
            emit(f"    Altas luces  (CV {tonal_start+int(2*t)}–{int_white}):  {s_h:.5f}")

            if s_s < s_m < s_h:
                emit("    Pendiente creciente: sombras < medios < luces")
        else:
            emit("    No hay rango tonal útil suficiente entre la zona de sombras y el blanco equivalente.")

        test_cv = 122
        pred_d = full_d_abs_raw[int(round(test_cv))]
        emit(f"\n  Verificación predictiva:")
        emit(f"    CV enviado = {test_cv} → D_VIS predicha = {pred_d:.3f}")

        test_d = 0.73
        valid_mask = np.isfinite(full_d_abs_raw)
        if np.any(valid_mask):
            pred_cv = np.interp(
                test_d,
                full_d_abs_raw[valid_mask][::-1],
                full_x[valid_mask][::-1]
            )
            emit(f"    D_VIS = {test_d:.3f} → CV enviado estimado = {pred_cv:.1f}")


        self.curve_data = {
            "names": names,
            "d_ref": d_ref,
            "cv_in": cv_in,
            "d_abs": d_abs,
            "d_target_abs": d_target_abs,
            "lstar": lstar,
            "full_x": full_x,
            "full_d_abs_raw": full_d_abs_raw,
            "full_d_abs_display": full_d_abs_display,
            "full_lstar": full_lstar,
            "slope_d": slope_d,
            "slope_jnd": slope_jnd,
            "full_relative_luminance": full_relative_luminance,
            "slope_relative_luminance": slope_relative_luminance,
            "heatmap_slope": heatmap_slope,
            "slope_norm": slope_norm,
            "heatmap_label": heatmap_label,
            "v_max": v_max,
            "plot_y_mode": "relative-luminance" if relative_luminance_y else "density",
            "plot_y_values": y_curve_plot,
        }
        self.image_data = {
            "gray": gray,
            "hist": hist,
            "total_pixels": total_px,
        }
        self.result = CurveVsHistoResult(
            plot_output_path=output_path if self.save_plot else None,
            false_color_output_path=false_color_result.output_path if false_color_result else None,
            cgats_path=cgats_full,
            image_path=image_full,
            cv_shadow_clip=cv_shadow_clip,
            d_shadow_clip=d_shadow_clip,
            shadow_clip_transition=shadow_clip_transition,
            cv_highlight_clip=cv_highlight_clip,
            d_highlight_clip=d_highlight_clip,
            dmax_practical=dmax_practical,
            dmax_practical_source=dmax_practical_source,
            contrast_model_description=contrast_model_description,
            shadow_clipped_pixels=int(shadow_clip_px),
            highlight_clipped_pixels=int(highlight_equiv),
            main_region_pixels=int(in_main_region),
            false_color_result=false_color_result,
            curve_data=self.curve_data,
            image_data=self.image_data,
        )
        return self.result


# CLI
def main(argv=None):
    configure_cli_encoding()
    args = parse_args(argv)
    verifier = CurveVsHistoVerifier.from_args(args)
    try:
        verifier.run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
