#!/usr/bin/env python3
"""Build an interactive intermediate-layer explorer for a local DINOv3 image."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "dinov3"
WEIGHTS = ROOT / "checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
IMAGE = ROOT / "smash-sample.jpg"
FEATURES_OUT = ROOT / "outputs/smash-sample.dinov3_vits16plus.layers.pt"
PNG_OUT = ROOT / "outputs/smash-sample.dino-layer-similarity.png"

WIDTH = 768
HEIGHT = 432
PROJECTION_DIM = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=IMAGE)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--features-out", type=Path, default=FEATURES_OUT)
    parser.add_argument("--png", type=Path, default=PNG_OUT)
    parser.add_argument("--social-png", type=Path)
    parser.add_argument("--square-png", type=Path)
    parser.add_argument("--query-x", type=int, default=21)
    parser.add_argument("--query-y", type=int, default=14)
    parser.add_argument("--query-label", default="Central fighter")
    return parser.parse_args()


def image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0)


def jpeg_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=86, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def aligned_pca_maps(features: torch.Tensor) -> np.ndarray:
    """Return per-layer PCA coordinates aligned to the final layer's axes."""
    layer_scores: list[torch.Tensor] = []
    for layer in features:
        centered = layer - layer.mean(dim=0, keepdim=True)
        _, _, axes = torch.pca_lowrank(centered, q=3, center=False)
        layer_scores.append(centered @ axes)

    reference = layer_scores[-1]
    aligned: list[torch.Tensor] = []
    permutations = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    reference = (reference - reference.mean(0)) / reference.std(0).clamp_min(1e-6)
    for scores in layer_scores:
        standardized = (scores - scores.mean(0)) / scores.std(0).clamp_min(1e-6)
        correlation = standardized.T @ reference / standardized.shape[0]
        permutation = max(
            permutations,
            key=lambda p: sum(abs(float(correlation[p[i], i])) for i in range(3)),
        )
        reordered = scores[:, permutation]
        signs = torch.tensor(
            [1.0 if correlation[permutation[i], i] >= 0 else -1.0 for i in range(3)]
        )
        aligned.append(reordered * signs)

    stacked = torch.stack(aligned)
    output = torch.empty_like(stacked)
    for layer_index in range(stacked.shape[0]):
        layer = stacked[layer_index]
        low = torch.quantile(layer, 0.02, dim=0)
        high = torch.quantile(layer, 0.98, dim=0)
        output[layer_index] = ((layer - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    return (output * 255).round().to(torch.uint8).numpy()


def heat_colors(values: np.ndarray) -> np.ndarray:
    stops = np.array([0.0, 0.35, 0.7, 1.0], dtype=np.float32)
    colors = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    return np.stack(
        [np.interp(values, stops, colors[:, channel]) for channel in range(3)],
        axis=-1,
    )


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["Arial Bold.ttf", "Arial.ttf"] if bold else ["Arial.ttf", "Arial Bold.ttf"]
    for name in names:
        path = Path("/System/Library/Fonts/Supplemental") / name
        if path.exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


def similarity_panel(
    source: Image.Image,
    flat: torch.Tensor,
    *,
    layer_index: int,
    query_x: int,
    query_y: int,
    width: int,
    height: int,
) -> tuple[Image.Image, float, float]:
    patch_height, patch_width = 27, 48
    source_resized = source.resize((width, height), Image.Resampling.BICUBIC)
    source_array = np.asarray(source_resized, dtype=np.float32)
    layer = F.normalize(flat[layer_index], dim=-1)
    query_index = query_y * patch_width + query_x
    similarities = (layer @ layer[query_index]).reshape(patch_height, patch_width)
    low = torch.quantile(similarities, 0.08)
    high = torch.quantile(similarities, 0.98)
    scaled = ((similarities - low) / (high - low).clamp_min(1e-6)).clamp(0, 1).numpy()
    heat = Image.fromarray(heat_colors(scaled).round().astype(np.uint8)).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    heat_array = np.asarray(heat, dtype=np.float32)
    alpha = 0.18 + 0.72 * np.power(
        np.asarray(
            Image.fromarray((scaled * 255).astype(np.uint8)).resize(
                (width, height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0,
        1.5,
    )
    panel_array = source_array * (1 - alpha[..., None]) + heat_array * alpha[..., None]
    return (
        Image.fromarray(panel_array.clip(0, 255).astype(np.uint8)),
        float(low),
        float(high),
    )


def draw_query_box(
    image: Image.Image,
    *,
    query_x: int,
    query_y: int,
    width: int = 4,
) -> None:
    patch_height, patch_width = 27, 48
    left = round(query_x * image.width / patch_width)
    top = round(query_y * image.height / patch_height)
    right = round((query_x + 1) * image.width / patch_width)
    bottom = round((query_y + 1) * image.height / patch_height)
    draw = ImageDraw.Draw(image)
    draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), outline=(20, 20, 20), width=width + 3)
    draw.rectangle((left, top, right, bottom), outline=(255, 255, 255), width=width)


def render_social_sheet(
    image: Image.Image,
    flat: torch.Tensor,
    path: Path,
    *,
    query_x: int,
    query_y: int,
    query_label: str,
) -> None:
    canvas = Image.new("RGB", (1080, 1350), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(44, bold=True)
    subtitle_font = load_font(25)
    layer_font = load_font(34, bold=True)
    legend_font = load_font(21)

    draw.text((50, 26), "DINOv3 FEATURES ACROSS DEPTH", font=title_font, fill=(248, 248, 248))
    draw.text(
        (52, 79),
        f"Query: {query_label} (white box)",
        font=subtitle_font,
        fill=(205, 205, 205),
    )

    source_width, source_height = 900, 506
    source = image.resize((source_width, source_height), Image.Resampling.BICUBIC)
    draw_query_box(source, query_x=query_x, query_y=query_y, width=5)
    canvas.paste(source, (90, 118))

    bar_left, bar_top, bar_width, bar_height = 50, 659, 330, 18
    gradient = np.repeat(np.linspace(0, 1, bar_width, dtype=np.float32)[None, :], bar_height, axis=0)
    canvas.paste(
        Image.fromarray(heat_colors(gradient).round().astype(np.uint8)),
        (bar_left, bar_top),
    )
    draw.text((50, 632), "LOW", font=legend_font, fill=(205, 205, 205))
    high_label = "HIGH"
    high_box = draw.textbbox((0, 0), high_label, font=legend_font)
    draw.text(
        (bar_left + bar_width - (high_box[2] - high_box[0]), 632),
        high_label,
        font=legend_font,
        fill=(205, 205, 205),
    )
    draw.multiline_text(
        (430, 638),
        "Brighter patches are more similar.\nEach layer uses its own color scale.",
        font=legend_font,
        fill=(205, 205, 205),
        spacing=3,
    )

    layers = [("LAYER 1", 0), ("LAYER 5", 4), ("LAYER 9", 8), ("LAYER 12", 11)]
    tile_width, tile_height = 470, 264
    positions = [(50, 744), (560, 744), (50, 1064), (560, 1064)]
    for (label, layer_index), (left, top) in zip(layers, positions, strict=True):
        panel, _, _ = similarity_panel(
            image,
            flat,
            layer_index=layer_index,
            query_x=query_x,
            query_y=query_y,
            width=tile_width,
            height=tile_height,
        )
        draw_query_box(panel, query_x=query_x, query_y=query_y, width=3)
        label_top = top - 43
        draw.text((left, label_top), label, font=layer_font, fill=(248, 248, 248))
        canvas.paste(panel, (left, top))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def render_square_sheet(
    image: Image.Image,
    flat: torch.Tensor,
    path: Path,
    *,
    query_x: int,
    query_y: int,
    query_label: str,
) -> None:
    canvas = Image.new("RGB", (1080, 1080), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(42, bold=True)
    subtitle_font = load_font(24)
    panel_font = load_font(36, bold=True)
    legend_font = load_font(22)
    footer_font = load_font(24, bold=True)

    draw.text((30, 24), "DINOv3 FEATURES ACROSS DEPTH", font=title_font, fill=(248, 248, 248))
    draw.text(
        (32, 75),
        f"Query: {query_label} (white box)",
        font=subtitle_font,
        fill=(205, 205, 205),
    )

    tile_width, tile_height = 510, 287
    panels = [
        ("SOURCE + QUERY", None, (20, 170)),
        ("LAYER 1", 0, (550, 170)),
        ("LAYER 9", 8, (20, 540)),
        ("LAYER 12", 11, (550, 540)),
    ]
    for label, layer_index, (left, top) in panels:
        draw.text((left, top - 45), label, font=panel_font, fill=(248, 248, 248))
        if layer_index is None:
            panel = image.resize((tile_width, tile_height), Image.Resampling.BICUBIC)
        else:
            panel, _, _ = similarity_panel(
                image,
                flat,
                layer_index=layer_index,
                query_x=query_x,
                query_y=query_y,
                width=tile_width,
                height=tile_height,
            )
        draw_query_box(panel, query_x=query_x, query_y=query_y, width=4)
        canvas.paste(panel, (left, top))

    bar_left, bar_top, bar_width, bar_height = 30, 898, 350, 20
    gradient = np.repeat(np.linspace(0, 1, bar_width, dtype=np.float32)[None, :], bar_height, axis=0)
    canvas.paste(
        Image.fromarray(heat_colors(gradient).round().astype(np.uint8)),
        (bar_left, bar_top),
    )
    draw.text((bar_left, 866), "LOW", font=legend_font, fill=(205, 205, 205))
    high_label = "HIGH"
    high_box = draw.textbbox((0, 0), high_label, font=legend_font)
    draw.text(
        (bar_left + bar_width - (high_box[2] - high_box[0]), 866),
        high_label,
        font=legend_font,
        fill=(205, 205, 205),
    )
    draw.multiline_text(
        (430, 866),
        "Brighter patches are more similar to the query.\nEach layer uses its own color scale.",
        font=legend_font,
        fill=(205, 205, 205),
        spacing=4,
    )
    draw.text(
        (30, 986),
        "COSINE SIMILARITY OF PATCH FEATURES",
        font=footer_font,
        fill=(175, 175, 175),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def render_similarity_sheet(
    image: Image.Image,
    flat: torch.Tensor,
    path: Path,
    *,
    query_x: int,
    query_y: int,
) -> None:
    patch_height, patch_width = 27, 48
    panels = [
        ("Source / query", None),
        ("Layer 1", 0),
        ("Layer 3", 2),
        ("Layer 5", 4),
        ("Layer 7", 6),
        ("Layer 9", 8),
        ("Layer 11", 10),
        ("Layer 12", 11),
    ]
    columns = 4
    rows = 2
    tile_width = 360
    tile_height = 203
    label_height = 30
    margin = 16
    legend_height = 62
    sheet_width = columns * tile_width + (columns + 1) * margin
    sheet_height = rows * (label_height + tile_height) + (rows + 1) * margin + legend_height
    sheet = Image.new("RGB", (sheet_width, sheet_height), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    small_font = ImageFont.load_default(size=16)
    source = image.resize((tile_width, tile_height), Image.Resampling.BICUBIC)
    source_array = np.asarray(source, dtype=np.float32)
    normalized = F.normalize(flat, dim=-1)
    query_index = query_y * patch_width + query_x

    for panel_index, (label, layer_index) in enumerate(panels):
        row, column = divmod(panel_index, columns)
        left = margin + column * (tile_width + margin)
        top = margin + row * (label_height + tile_height + margin)
        draw.text((left, top + 3), label, font=font, fill=(245, 245, 245))

        if layer_index is None:
            panel = source.copy()
        else:
            layer = normalized[layer_index]
            similarities = (layer @ layer[query_index]).reshape(patch_height, patch_width)
            low = torch.quantile(similarities, 0.08)
            high = torch.quantile(similarities, 0.98)
            range_label = f"p08 {float(low):.2f}  p98 {float(high):.2f}"
            range_box = draw.textbbox((0, 0), range_label, font=small_font)
            range_width = range_box[2] - range_box[0]
            draw.text(
                (left + tile_width - range_width, top + 5),
                range_label,
                font=small_font,
                fill=(190, 190, 190),
            )
            scaled = ((similarities - low) / (high - low).clamp_min(1e-6)).clamp(0, 1).numpy()
            heat = Image.fromarray(heat_colors(scaled).round().astype(np.uint8)).resize(
                (tile_width, tile_height), Image.Resampling.BILINEAR
            )
            heat_array = np.asarray(heat, dtype=np.float32)
            alpha = 0.18 + 0.72 * np.power(
                np.asarray(Image.fromarray((scaled * 255).astype(np.uint8)).resize(
                    (tile_width, tile_height), Image.Resampling.BILINEAR
                ), dtype=np.float32) / 255.0,
                1.5,
            )
            panel_array = source_array * (1 - alpha[..., None]) + heat_array * alpha[..., None]
            panel = Image.fromarray(panel_array.clip(0, 255).astype(np.uint8))

        panel_draw = ImageDraw.Draw(panel)
        patch_left = round(query_x * tile_width / patch_width)
        patch_top = round(query_y * tile_height / patch_height)
        patch_right = round((query_x + 1) * tile_width / patch_width)
        patch_bottom = round((query_y + 1) * tile_height / patch_height)
        panel_draw.rectangle(
            (patch_left, patch_top, patch_right, patch_bottom),
            outline=(255, 255, 255),
            width=2,
        )
        sheet.paste(panel, (left, top + label_height))

    legend_top = sheet_height - legend_height + 5
    bar_left = margin
    bar_top = legend_top + 18
    bar_width = 300
    bar_height = 12
    gradient = np.linspace(0, 1, bar_width, dtype=np.float32)[None, :]
    gradient = np.repeat(gradient, bar_height, axis=0)
    bar = Image.fromarray(heat_colors(gradient).round().astype(np.uint8))
    sheet.paste(bar, (bar_left, bar_top))
    draw.text((bar_left, legend_top - 2), "Relative cosine similarity", font=small_font, fill=(220, 220, 220))
    draw.text((bar_left + bar_width + 12, bar_top - 3), "low  ->  high", font=small_font, fill=(220, 220, 220))
    draw.text(
        (bar_left + 470, legend_top - 2),
        "Each layer is independently rescaled from its p08 to p98 cosine range.",
        font=small_font,
        fill=(190, 190, 190),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=True)


def build_fragment(
    *,
    image_url: str,
    projected_features: np.ndarray,
    pca_maps: np.ndarray,
    layers: int,
    patch_height: int,
    patch_width: int,
    query_x: int,
    query_y: int,
    query_label: str,
) -> str:
    feature_b64 = base64.b64encode(projected_features.tobytes()).decode("ascii")
    pca_b64 = base64.b64encode(pca_maps.tobytes()).decode("ascii")
    metadata = json.dumps(
        {
            "layers": layers,
            "patchHeight": patch_height,
            "patchWidth": patch_width,
            "projectionDim": PROJECTION_DIM,
            "imageWidth": WIDTH,
            "imageHeight": HEIGHT,
        },
        separators=(",", ":"),
    )

    return f'''<div id="dino-layer-explorer">
  <style>
    #dino-layer-explorer {{
      display: grid;
      gap: 14px;
      color: var(--foreground);
    }}
    #dino-layer-explorer .dino-controls {{
      justify-content: space-between;
      align-items: end;
    }}
    #dino-layer-explorer .dino-mode {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    #dino-layer-explorer .dino-layer-control {{
      min-width: min(100%, 260px);
      flex: 1 1 260px;
    }}
    #dino-layer-explorer .dino-stage {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    #dino-layer-explorer .dino-view {{
      min-width: 0;
      display: grid;
      gap: 6px;
      align-content: start;
    }}
    #dino-layer-explorer .dino-view-label {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
    }}
    #dino-layer-explorer canvas {{
      width: 100%;
      aspect-ratio: 16 / 9;
      display: block;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--muted);
    }}
    #dino-layer-explorer #dino-source {{
      cursor: crosshair;
    }}
    #dino-layer-explorer .dino-legend {{
      display: flex;
      gap: 10px;
      align-items: center;
      min-height: 20px;
      flex-wrap: wrap;
    }}
    #dino-layer-explorer [hidden] {{
      display: none !important;
    }}
    #dino-layer-explorer .dino-ramp {{
      width: min(180px, 46%);
      height: 8px;
      border-radius: 4px;
      background: linear-gradient(90deg, color-mix(in srgb, var(--viz-series-1) 4%, transparent), var(--viz-series-1));
    }}
    #dino-layer-explorer .dino-swatches {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    #dino-layer-explorer .dino-swatch {{
      display: inline-flex;
      gap: 5px;
      align-items: center;
    }}
    #dino-layer-explorer .dino-swatch::before {{
      content: "";
      width: 10px;
      height: 10px;
      border-radius: 2px;
      background: var(--swatch);
    }}
    @media (max-width: 560px) {{
      #dino-layer-explorer .dino-stage {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>

  <div class="viz-controls dino-controls">
    <div class="dino-mode" role="group" aria-label="Visualization mode">
      <button type="button" class="btn" data-mode="similarity" aria-pressed="true">Query similarity</button>
      <button type="button" class="btn" data-mode="pca" aria-pressed="false">PCA structure</button>
    </div>
    <label class="form-label">
      Query
      <select class="form-select" id="dino-query">
        <option value="{query_x},{query_y}">{query_label}</option>
        <option value="custom">Custom click</option>
      </select>
    </label>
    <label class="form-label dino-layer-control">
      Layer <output id="dino-layer-value">1 / {layers}</output>
      <input class="form-range" id="dino-layer" type="range" min="0" max="{layers - 1}" value="0" step="1">
    </label>
  </div>

  <div class="dino-stage">
    <div class="dino-view">
      <div class="dino-view-label">
        <span>Source image</span>
        <span class="text-small text-muted" id="dino-query-value">patch {query_x}, {query_y}</span>
      </div>
      <canvas id="dino-source" width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="Source image; click to choose a query patch"></canvas>
    </div>
    <div class="dino-view">
      <div class="dino-view-label">
        <span id="dino-result-label">Layer 1 similarity</span>
        <span class="text-small text-muted">{patch_height} × {patch_width} patch grid</span>
      </div>
      <canvas id="dino-result" width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="Layer 1 query-patch similarity map"></canvas>
    </div>
  </div>

  <div class="dino-legend text-small" id="dino-similarity-legend">
    <span class="text-muted">less similar</span>
    <span class="dino-ramp" aria-hidden="true"></span>
    <span class="text-muted">more similar</span>
    <span class="text-muted" id="dino-range"></span>
  </div>
  <div class="dino-legend dino-swatches text-small" id="dino-pca-legend" hidden>
    <span class="text-muted">Arbitrary PCA axes:</span>
    <span class="dino-swatch" style="--swatch: var(--viz-series-1)">PC1</span>
    <span class="dino-swatch" style="--swatch: var(--viz-series-2)">PC2</span>
    <span class="dino-swatch" style="--swatch: var(--viz-series-3)">PC3</span>
  </div>

  <script>
    (() => {{
      const root = document.getElementById("dino-layer-explorer");
      const meta = {metadata};
      const imageUrl = {json.dumps(image_url)};
      const featurePayload = {json.dumps(feature_b64)};
      const pcaPayload = {json.dumps(pca_b64)};

      const decodeBytes = (payload, Type) => {{
        const binary = atob(payload);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
        return new Type(bytes.buffer);
      }};

      const features = decodeBytes(featurePayload, Int8Array);
      const pca = decodeBytes(pcaPayload, Uint8Array);
      const source = root.querySelector("#dino-source");
      const result = root.querySelector("#dino-result");
      const sourceCtx = source.getContext("2d");
      const resultCtx = result.getContext("2d");
      const layerInput = root.querySelector("#dino-layer");
      const layerValue = root.querySelector("#dino-layer-value");
      const querySelect = root.querySelector("#dino-query");
      const queryValue = root.querySelector("#dino-query-value");
      const resultLabel = root.querySelector("#dino-result-label");
      const rangeLabel = root.querySelector("#dino-range");
      const similarityLegend = root.querySelector("#dino-similarity-legend");
      const pcaLegend = root.querySelector("#dino-pca-legend");
      const modeButtons = [...root.querySelectorAll("[data-mode]")];
      const tokenCount = meta.patchHeight * meta.patchWidth;
      let layer = 0;
      let queryX = {query_x};
      let queryY = {query_y};
      let mode = "similarity";
      let image = null;

      const resolvedThemeCss = (name) => {{
        const probe = document.createElement("span");
        probe.style.color = `var(${{name}})`;
        root.appendChild(probe);
        const value = getComputedStyle(probe).color;
        probe.remove();
        return value;
      }};

      const colorRgb = (cssValue) => {{
        const probe = document.createElement("canvas");
        probe.width = 1;
        probe.height = 1;
        const context = probe.getContext("2d");
        context.fillStyle = cssValue.trim();
        context.fillRect(0, 0, 1, 1);
        return context.getImageData(0, 0, 1, 1).data.slice(0, 3);
      }};

      const themeColor = (name) => colorRgb(resolvedThemeCss(name));

      const drawSource = () => {{
        if (!image) return;
        sourceCtx.clearRect(0, 0, source.width, source.height);
        sourceCtx.drawImage(image, 0, 0, source.width, source.height);
        const patchW = source.width / meta.patchWidth;
        const patchH = source.height / meta.patchHeight;
        sourceCtx.strokeStyle = resolvedThemeCss("--primary");
        sourceCtx.lineWidth = 3;
        sourceCtx.strokeRect(queryX * patchW + 1.5, queryY * patchH + 1.5, patchW - 3, patchH - 3);
        sourceCtx.beginPath();
        sourceCtx.moveTo((queryX + 0.5) * patchW - 8, (queryY + 0.5) * patchH);
        sourceCtx.lineTo((queryX + 0.5) * patchW + 8, (queryY + 0.5) * patchH);
        sourceCtx.moveTo((queryX + 0.5) * patchW, (queryY + 0.5) * patchH - 8);
        sourceCtx.lineTo((queryX + 0.5) * patchW, (queryY + 0.5) * patchH + 8);
        sourceCtx.stroke();
      }};

      const vectorOffset = (layerIndex, tokenIndex) =>
        (layerIndex * tokenCount + tokenIndex) * meta.projectionDim;

      const drawSimilarity = () => {{
        const queryIndex = queryY * meta.patchWidth + queryX;
        const queryOffset = vectorOffset(layer, queryIndex);
        const values = new Float32Array(tokenCount);
        let queryNorm = 0;
        for (let d = 0; d < meta.projectionDim; d += 1) {{
          const value = features[queryOffset + d];
          queryNorm += value * value;
        }}
        queryNorm = Math.sqrt(queryNorm);

        for (let token = 0; token < tokenCount; token += 1) {{
          const offset = vectorOffset(layer, token);
          let dot = 0;
          let norm = 0;
          for (let d = 0; d < meta.projectionDim; d += 1) {{
            const value = features[offset + d];
            dot += value * features[queryOffset + d];
            norm += value * value;
          }}
          values[token] = dot / Math.max(1, Math.sqrt(norm) * queryNorm);
        }}

        const sorted = [...values].sort((a, b) => a - b);
        const low = sorted[Math.floor(sorted.length * 0.08)];
        const high = sorted[Math.floor(sorted.length * 0.98)];
        const highlight = themeColor("--viz-series-1");
        const heat = document.createElement("canvas");
        heat.width = meta.patchWidth;
        heat.height = meta.patchHeight;
        const heatCtx = heat.getContext("2d");
        const pixels = heatCtx.createImageData(meta.patchWidth, meta.patchHeight);
        for (let token = 0; token < tokenCount; token += 1) {{
          const normalized = Math.max(0, Math.min(1, (values[token] - low) / Math.max(1e-5, high - low)));
          const index = token * 4;
          pixels.data[index] = highlight[0];
          pixels.data[index + 1] = highlight[1];
          pixels.data[index + 2] = highlight[2];
          pixels.data[index + 3] = Math.round(225 * Math.pow(normalized, 1.7));
        }}
        heatCtx.putImageData(pixels, 0, 0);

        resultCtx.clearRect(0, 0, result.width, result.height);
        resultCtx.globalAlpha = 0.48;
        resultCtx.drawImage(image, 0, 0, result.width, result.height);
        resultCtx.globalAlpha = 1;
        resultCtx.imageSmoothingEnabled = true;
        resultCtx.drawImage(heat, 0, 0, result.width, result.height);
        rangeLabel.textContent = `relative scale ${{low.toFixed(2)}}–${{high.toFixed(2)}} cosine`;
        result.setAttribute("aria-label", `Layer ${{layer + 1}} cosine-similarity map for query patch ${{queryX}}, ${{queryY}}`);
      }};

      const drawPca = () => {{
        const colors = [themeColor("--viz-series-1"), themeColor("--viz-series-2"), themeColor("--viz-series-3")];
        const background = themeColor("--background");
        const map = document.createElement("canvas");
        map.width = meta.patchWidth;
        map.height = meta.patchHeight;
        const mapCtx = map.getContext("2d");
        const pixels = mapCtx.createImageData(meta.patchWidth, meta.patchHeight);
        const layerOffset = layer * tokenCount * 3;
        for (let token = 0; token < tokenCount; token += 1) {{
          const coordinates = [
            pca[layerOffset + token * 3] / 255,
            pca[layerOffset + token * 3 + 1] / 255,
            pca[layerOffset + token * 3 + 2] / 255,
          ];
          const weights = coordinates.map((value) => Math.exp((value - 0.5) * 3.5));
          const total = weights[0] + weights[1] + weights[2];
          const index = token * 4;
          for (let channel = 0; channel < 3; channel += 1) {{
            pixels.data[index + channel] = Math.round(
              background[channel] * 0.12 + 0.88 *
                (colors[0][channel] * weights[0] +
                  colors[1][channel] * weights[1] +
                  colors[2][channel] * weights[2]) / total
            );
          }}
          pixels.data[index + 3] = 255;
        }}
        mapCtx.putImageData(pixels, 0, 0);
        resultCtx.clearRect(0, 0, result.width, result.height);
        resultCtx.imageSmoothingEnabled = true;
        resultCtx.drawImage(map, 0, 0, result.width, result.height);
        result.setAttribute("aria-label", `Layer ${{layer + 1}} three-component PCA feature map`);
      }};

      const render = () => {{
        layerValue.textContent = `${{layer + 1}} / ${{meta.layers}}`;
        queryValue.textContent = `patch ${{queryX}}, ${{queryY}}`;
        resultLabel.textContent = mode === "similarity"
          ? `Layer ${{layer + 1}} similarity`
          : `Layer ${{layer + 1}} PCA structure`;
        similarityLegend.hidden = mode !== "similarity";
        pcaLegend.hidden = mode !== "pca";
        querySelect.disabled = mode !== "similarity";
        drawSource();
        if (!image) return;
        if (mode === "similarity") drawSimilarity();
        else drawPca();
      }};

      layerInput.addEventListener("input", () => {{
        layer = Number(layerInput.value);
        render();
      }});

      querySelect.addEventListener("change", () => {{
        if (querySelect.value === "custom") return;
        [queryX, queryY] = querySelect.value.split(",").map(Number);
        render();
      }});

      source.addEventListener("click", (event) => {{
        if (mode !== "similarity") return;
        const bounds = source.getBoundingClientRect();
        queryX = Math.max(0, Math.min(meta.patchWidth - 1, Math.floor((event.clientX - bounds.left) / bounds.width * meta.patchWidth)));
        queryY = Math.max(0, Math.min(meta.patchHeight - 1, Math.floor((event.clientY - bounds.top) / bounds.height * meta.patchHeight)));
        querySelect.value = "custom";
        render();
      }});

      modeButtons.forEach((button) => {{
        button.addEventListener("click", () => {{
          mode = button.dataset.mode;
          modeButtons.forEach((candidate) => candidate.setAttribute("aria-pressed", String(candidate === button)));
          render();
        }});
      }});

      const loadedImage = new Image();
      loadedImage.onload = () => {{
        image = loadedImage;
        render();
      }};
      loadedImage.src = imageUrl;
    }})();
  </script>
</div>
'''


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO))
    from dinov3.hub.backbones import dinov3_vits16plus

    image = Image.open(args.image).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
    model = dinov3_vits16plus(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()

    with torch.inference_mode():
        outputs = model.get_intermediate_layers(
            image_tensor(image),
            n=list(range(len(model.blocks))),
            reshape=True,
            norm=True,
        )

    # L, H, W, D keeps neighboring patches contiguous for browser-side queries.
    features = torch.stack([output[0].permute(1, 2, 0) for output in outputs]).float()
    layers, patch_height, patch_width, feature_dim = features.shape
    flat = features.reshape(layers, patch_height * patch_width, feature_dim)

    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features.half(),
            "layers": list(range(layers)),
            "image_size": (HEIGHT, WIDTH),
            "patch_grid": (patch_height, patch_width),
            "model": "dinov3_vits16plus",
        },
        args.features_out,
    )

    generator = torch.Generator().manual_seed(7)
    random_matrix = torch.randn(feature_dim, PROJECTION_DIM, generator=generator)
    projection, _ = torch.linalg.qr(random_matrix, mode="reduced")
    projected = F.normalize(F.normalize(flat, dim=-1) @ projection, dim=-1)
    quantized = (projected * 127).round().clamp(-127, 127).to(torch.int8).numpy()
    pca_maps = aligned_pca_maps(flat)
    render_similarity_sheet(
        image,
        flat,
        args.png,
        query_x=args.query_x,
        query_y=args.query_y,
    )
    if args.social_png is not None:
        render_social_sheet(
            image,
            flat,
            args.social_png,
            query_x=args.query_x,
            query_y=args.query_y,
            query_label=args.query_label,
        )
    if args.square_png is not None:
        render_square_sheet(
            image,
            flat,
            args.square_png,
            query_x=args.query_x,
            query_y=args.query_y,
            query_label=args.query_label,
        )

    fragment = build_fragment(
        image_url=jpeg_data_url(image),
        projected_features=quantized,
        pca_maps=pca_maps,
        layers=layers,
        patch_height=patch_height,
        patch_width=patch_width,
        query_x=args.query_x,
        query_y=args.query_y,
        query_label=args.query_label,
    )
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(fragment)

    print(f"model: dinov3_vits16plus ({layers} blocks, {feature_dim} features)")
    print(f"image: {WIDTH}x{HEIGHT}; patch grid: {patch_width}x{patch_height}")
    print(f"saved features: {args.features_out}")
    print(f"saved similarity sheet: {args.png}")
    if args.social_png is not None:
        print(f"saved social sheet: {args.social_png}")
    if args.square_png is not None:
        print(f"saved square sheet: {args.square_png}")
    print(f"saved explorer: {args.html} ({args.html.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
