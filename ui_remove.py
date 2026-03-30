import os
import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting.utils import download_model, prepare_img_and_mask
from transformers import (
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
    SamModel,
    SamProcessor,
)


@dataclass(frozen=True)
class Models:
    dino_processor: GroundingDinoProcessor
    dino_model: GroundingDinoForObjectDetection
    sam_processor: SamProcessor
    sam_model: SamModel
    lama: "LamaTorchscript"
    device: torch.device


DEFAULT_PROMPT = "hud, ui, button, icon, minimap, health bar, ammo, score, menu, text"
DEFAULT_INPUT = "input"
DEFAULT_OUTPUT = "output"
DEFAULT_MODE = "ui"

LAMA_MODEL_URL_DEFAULT = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
PROJECT_CACHE_DIR = Path(".cache")


def _setup_local_caches() -> None:
    os.environ.setdefault("HF_HOME", str(PROJECT_CACHE_DIR / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_CACHE_DIR / "huggingface" / "transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # меньше шума в консоли
    os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE_DIR / "torch"))

    (PROJECT_CACHE_DIR / "huggingface").mkdir(parents=True, exist_ok=True)
    (PROJECT_CACHE_DIR / "huggingface" / "transformers").mkdir(parents=True, exist_ok=True)
    (PROJECT_CACHE_DIR / "torch").mkdir(parents=True, exist_ok=True)


class LamaTorchscript:

    def __init__(self, *, device: torch.device, model_path: str | None = None) -> None:
        url = os.environ.get("LAMA_MODEL_URL", LAMA_MODEL_URL_DEFAULT)
        path = model_path or os.environ.get("LAMA_MODEL")
        if path:
            model_path_resolved = path
        else:
            model_path_resolved = download_model(url)

        self.model = torch.jit.load(model_path_resolved, map_location="cpu")
        self.model.eval()

        self.device = device
        try:
            self.model.to(self.device)
        except Exception:
            self.device = torch.device("cpu")
            self.model.to(self.device)

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        orig_w, orig_h = image.size
        image_t, mask_t = prepare_img_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image_t, mask_t)

        cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
        cur_res = cur_res[:orig_h, :orig_w]
        return Image.fromarray(cur_res)


def _poisson_blend(
    *,
    original_rgb: Image.Image,
    inpainted_rgb: Image.Image,
    mask_u8: np.ndarray,
) -> Image.Image:
    o = np.array(original_rgb)  # RGB
    r = np.array(inpainted_rgb)  # RGB
    if o.shape != r.shape:
        return inpainted_rgb

    m = mask_u8
    if m.ndim != 2:
        m = np.squeeze(m)
    if m.ndim != 2 or m.shape[0] != o.shape[0] or m.shape[1] != o.shape[1]:
        return inpainted_rgb

    m = (m > 0).astype(np.uint8) * 255
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return inpainted_rgb

    center = (int(xs.mean()), int(ys.mean()))
    # seamlessClone работает в BGR
    src = r[:, :, ::-1]
    dst = o[:, :, ::-1]
    try:
        blended = cv2.seamlessClone(src, dst, m, center, cv2.NORMAL_CLONE)
        return Image.fromarray(blended[:, :, ::-1])
    except Exception:
        return inpainted_rgb


def _enhance_same_size(
    image_rgb: Image.Image,
    *,
    method: str,
    amount: float,
    radius: float,
    threshold: int,
) -> Image.Image:
    method = (method or "none").strip().lower()
    if method in {"none", "off", "0"}:
        return image_rgb

    def unsharp(pil: Image.Image) -> Image.Image:
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=max(0.1, float(radius)))
        sharp = cv2.addWeighted(bgr, 1.0 + float(amount), blur, -float(amount), 0)
        if threshold > 0:
            diff = cv2.absdiff(bgr, blur)
            mask = (diff.max(axis=2) > threshold).astype(np.uint8)[:, :, None]
            sharp = (sharp * mask + bgr * (1 - mask)).astype(np.uint8)
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB))

    if method == "sharp":
        return unsharp(image_rgb)

    if method == "sharp2x":
        w, h = image_rgb.size
        up = image_rgb.resize((w * 2, h * 2), resample=Image.LANCZOS)
        up_sharp = unsharp(up)
        return up_sharp.resize((w, h), resample=Image.LANCZOS)

    raise ValueError("enhance должен быть одним из: none, sharp, sharp2x")


def _ensure_ssl_certs() -> None:
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi  # type: ignore

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        return


def _device() -> torch.device:
    # Приоритет: CUDA (Windows/Linux) → MPS (Apple Silicon) → CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _parse_device(name: str) -> torch.device:
    name = (name or "auto").strip().lower()
    if name == "auto":
        return _device()
    if name in {"cuda", "gpu"} or name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна (PyTorch без CUDA или нет NVIDIA GPU/драйверов).")
        # torch.device("cuda") корректно для "cuda:0" и т.п.
        return torch.device(name if name.startswith("cuda:") else "cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS недоступен в этой сборке PyTorch/на этом Mac.")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device должен быть одним из: auto, cpu, cuda, mps")


def _parse_lama_device(name: str) -> torch.device:
    """
    Устройство для LaMa TorchScript.
    На некоторых сборках/девайсах модель может не поддерживать перенос на GPU/MPS —
    тогда в LamaTorchscript есть безопасный fallback на CPU.
    """
    name = (name or "auto").strip().lower()
    if name == "auto":
        return torch.device("cpu")
    if name in {"cuda", "gpu"} or name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна (PyTorch без CUDA или нет NVIDIA GPU/драйверов).")
        return torch.device(name if name.startswith("cuda:") else "cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS недоступен в этой сборке PyTorch/на этом Mac.")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("lama-device должен быть одним из: auto, cpu, cuda, mps")


def _as_dino_query(prompt: str) -> str:
    parts = [p.strip() for p in prompt.replace(";", ",").split(",") if p.strip()]
    if not parts:
        parts = [DEFAULT_PROMPT]
    return ". ".join(parts) + "."


def load_models(
    *,
    dino_model_id: str,
    sam_model_id: str,
    lama_model_path: str | None,
    lama_device: torch.device,
    device: torch.device,
    offline: bool,
) -> Models:
    _ensure_ssl_certs()
    _setup_local_caches()

    dino_processor = GroundingDinoProcessor.from_pretrained(dino_model_id, local_files_only=offline)
    dino_model = GroundingDinoForObjectDetection.from_pretrained(dino_model_id, local_files_only=offline).to(device)
    dino_model.eval()

    sam_processor = SamProcessor.from_pretrained(sam_model_id, local_files_only=offline)
    sam_model = SamModel.from_pretrained(sam_model_id, local_files_only=offline).to(device)
    sam_model.eval()

    try:
        lama = LamaTorchscript(device=lama_device, model_path=lama_model_path)
    except Exception as e:
        msg = str(e)
        if "CERTIFICATE_VERIFY_FAILED" in msg or "certificate verify failed" in msg:
            raise RuntimeError(
            ) from e
        raise

    return Models(
        dino_processor=dino_processor,
        dino_model=dino_model,
        sam_processor=sam_processor,
        sam_model=sam_model,
        lama=lama,
        device=device,
    )


def _resize_keep_aspect(pil: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    if max_side <= 0:
        return pil, 1.0

    w, h = pil.size
    m = max(w, h)
    if m <= max_side:
        return pil, 1.0

    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil.resize((new_w, new_h), resample=Image.BICUBIC), scale


def dino_detect_xyxy(
    models: Models,
    image_rgb: Image.Image,
    *,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    max_box_area: float,
) -> np.ndarray:
    query = _as_dino_query(prompt)
    inputs = models.dino_processor(images=image_rgb, text=query, return_tensors="pt").to(models.device)

    with torch.no_grad():
        outputs = models.dino_model(**inputs)


    target_sizes = torch.tensor([image_rgb.size[::-1]], device=models.device)  # (h, w)

    results = models.dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )[0]

    boxes = results["boxes"]  # (N, 4) xyxy в пикселях
    if boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)

    b = boxes.detach().float().cpu().numpy()
    w, h = image_rgb.size

    b[:, 0] = np.clip(b[:, 0], 0, w - 1)
    b[:, 2] = np.clip(b[:, 2], 0, w - 1)
    b[:, 1] = np.clip(b[:, 1], 0, h - 1)
    b[:, 3] = np.clip(b[:, 3], 0, h - 1)

    bw = np.maximum(0.0, b[:, 2] - b[:, 0])
    bh = np.maximum(0.0, b[:, 3] - b[:, 1])
    area = bw * bh
    img_area = float(w * h)
    keep = (bw >= 2.0) & (bh >= 2.0)
    if max_box_area > 0:
        keep = keep & ((area / img_area) <= float(max_box_area))

    b = b[keep]
    if b.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return b


def _mask_from_boxes_rects(
    *,
    image_size: tuple[int, int],
    boxes_xyxy: np.ndarray,
    pad: int = 0,
) -> np.ndarray:
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    if boxes_xyxy.shape[0] == 0:
        return mask
    p = max(0, int(pad))
    for x1, y1, x2, y2 in boxes_xyxy.astype(np.int32):
        x1 = max(0, x1 - p)
        y1 = max(0, y1 - p)
        x2 = min(w - 1, x2 + p)
        y2 = min(h - 1, y2 + p)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def _ocr_text_mask(
    image_rgb: Image.Image,
    *,
    reader: Any,
    langs: list[str],
    conf: float,
    min_size: int,
    pad: int,
    pad_ratio: float,
    rect_pad: int,
    rect_pad_ratio: float,
    close: int,
) -> np.ndarray:
    img = np.array(image_rgb)  # RGB
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    def scaled_pad(max_pad: int, ratio: float, bw: int, bh: int) -> int:

        m = max(bw, bh)
        rp = int(round(max(0.0, float(ratio)) * float(m)))
        rp = max(0, rp)
        if max_pad > 0:
            rp = min(int(max_pad), rp)
        return rp

    results = reader.readtext(img, detail=1, paragraph=False)
    for item in results:

        bbox, _text, score = item[0], item[1], float(item[2])
        if score < float(conf):
            continue
        pts = np.array(bbox, dtype=np.int32)
        x_min = int(np.min(pts[:, 0]))
        x_max = int(np.max(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        y_max = int(np.max(pts[:, 1]))
        bw = int(x_max - x_min)
        bh = int(y_max - y_min)
        if bw < int(min_size) or bh < int(min_size):
            continue

        local = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(local, [pts], 255)


        rp = scaled_pad(int(rect_pad), float(rect_pad_ratio), bw, bh)
        if rp > 0:
            rx1 = max(0, x_min - rp)
            ry1 = max(0, y_min - rp)
            rx2 = min(w - 1, x_max + rp)
            ry2 = min(h - 1, y_max + rp)
            cv2.rectangle(local, (rx1, ry1), (rx2, ry2), 255, thickness=-1)

        dp = scaled_pad(int(pad), float(pad_ratio), bw, bh)
        cp = max(0, int(close))
        margin = max(dp, rp, cp, 0) + 2
        x1 = max(0, x_min - margin)
        y1 = max(0, y_min - margin)
        x2 = min(w, x_max + margin + 1)
        y2 = min(h, y_max + margin + 1)
        roi = local[y1:y2, x1:x2]

        if dp > 0:
            k = dp if (dp % 2 == 1) else (dp + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            roi = cv2.dilate(roi, kernel, iterations=1)

        if cp > 0:
            k = cp if (cp % 2 == 1) else (cp + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

        local[y1:y2, x1:x2] = roi
        mask = np.maximum(mask, local)

    return mask


_OCR_READERS: dict[tuple[tuple[str, ...], bool], Any] = {}


def _get_ocr_reader(langs: list[str], *, gpu: bool) -> Any:
    key = (tuple(langs), bool(gpu))
    r = _OCR_READERS.get(key)
    if r is not None:
        return r
    try:
        import easyocr  # type: ignore
    except Exception as e:
        raise RuntimeError("pip install easyocr") from e
    r = easyocr.Reader(list(key[0]), gpu=bool(gpu))
    _OCR_READERS[key] = r
    return r


def _union_masks(*masks: np.ndarray) -> np.ndarray:
    out = None
    for m in masks:
        if out is None:
            out = m.copy()
        else:
            out = np.maximum(out, m)
    return out if out is not None else None


def _prepare_ocr_views(image_rgb: Image.Image, passes: str) -> list[np.ndarray]:
    pset = {p.strip().lower() for p in (passes or "rgb,clahe").split(",") if p.strip()}
    base = np.array(image_rgb)  # RGB
    views: list[np.ndarray] = []

    if "rgb" in pset:
        views.append(base)

    if "clahe" in pset:
        bgr = base[:, :, ::-1]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        views.append(bgr2[:, :, ::-1])


    uniq: list[np.ndarray] = []
    seen: set[int] = set()
    for v in views:
        hsh = hash(v.tobytes())
        if hsh not in seen:
            uniq.append(v)
            seen.add(hsh)
    return uniq


def ocr_text_mask_multipass(
    image_rgb: Image.Image,
    *,
    reader: Any,
    conf: float,
    min_size: int,
    pad: int,
    pad_ratio: float,
    rect_pad: int,
    rect_pad_ratio: float,
    close: int,
    passes: str,
) -> np.ndarray:
    views = _prepare_ocr_views(image_rgb, passes)
    masks: list[np.ndarray] = []
    for v in views:
        pil = Image.fromarray(v, mode="RGB")
        masks.append(
            _ocr_text_mask(
                pil,
                reader=reader,
                langs=[],
                conf=conf,
                min_size=min_size,
                pad=pad,
                pad_ratio=pad_ratio,
                rect_pad=rect_pad,
                rect_pad_ratio=rect_pad_ratio,
                close=close,
            )
        )
    return np.maximum.reduce(masks) if masks else np.zeros((image_rgb.height, image_rgb.width), dtype=np.uint8)


def sam_masks_from_boxes(
    models: Models,
    image_rgb: Image.Image,
    boxes_xyxy: np.ndarray,
) -> list[np.ndarray]:
    if boxes_xyxy.shape[0] == 0:
        return []


    inputs = models.sam_processor(
        image_rgb,
        input_boxes=[boxes_xyxy.tolist()],
        return_tensors="pt",
    )

    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
            inputs[k] = v.float()

    inputs = inputs.to(models.device)

    with torch.no_grad():
        outputs = models.sam_model(**inputs)

    processed_masks = models.sam_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )[0]  # (num_boxes, H, W)

    iou_scores = getattr(outputs, "iou_scores", None)
    if iou_scores is not None:

        iou_scores = iou_scores[0].detach().float().cpu()

    masks = []
    for i in range(processed_masks.shape[0]):
        pm = processed_masks[i]

        if pm.ndim == 3:
            if iou_scores is not None and i < iou_scores.shape[0]:
                best_j = int(torch.argmax(iou_scores[i]).item())
            else:
                best_j = 0
            pm = pm[best_j]

        m = pm.detach().cpu().numpy()
        m = np.squeeze(m)
        if m.ndim != 2:
            raise RuntimeError(f"Неожиданная форма маски SAM после выбора: {m.shape}")

        masks.append((m > 0.0).astype(np.uint8))
    return masks


def merge_and_refine_masks(
    masks01: list[np.ndarray],
    *,
    dilate: int,
) -> np.ndarray:
    if not masks01:
        raise ValueError("Empty mask list")

    base = np.squeeze(masks01[0])
    if base.ndim != 2:
        raise RuntimeError(f"Неожиданная форма базовой маски: {base.shape}")
    m = np.zeros_like(base, dtype=np.uint8)
    for x in masks01:
        x2 = np.squeeze(x)
        if x2.ndim != 2:
            raise RuntimeError(f"Неожиданная форма маски: {x2.shape}")
        m = np.maximum(m, (x2 > 0).astype(np.uint8) * 255)

    if dilate > 0:
        k = dilate if (dilate % 2 == 1) else (dilate + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = np.ascontiguousarray(m)
        m = cv2.dilate(m, kernel, iterations=1)

    return m


def remove_ui(
    models: Models,
    image_rgb: Image.Image,
    *,
    mode: str,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    dilate: int,
    max_box_area: float,
    max_mask_area: float,
    tighten_steps: int,
    ocr_langs: list[str],
    ocr_gpu: bool,
    ocr_conf: float,
    ocr_min_size: int,
    ocr_pad: int,
    ocr_rect_pad: int,
    ocr_close: int,
    ocr_passes: str,
    ocr_pad_ratio: float,
    ocr_rect_pad_ratio: float,
    poisson_blend: bool,
) -> tuple[Image.Image, Image.Image]:
    h, w = image_rgb.height, image_rgb.width
    empty = Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L")

    mode = (mode or DEFAULT_MODE).strip().lower()
    if mode not in {"ui", "text", "ui+text"}:
        raise ValueError("mode должен быть одним из: ui, text, ui+text")


    if mode == "text":
        reader = _get_ocr_reader(ocr_langs, gpu=bool(ocr_gpu))
        mask_u8 = ocr_text_mask_multipass(
            image_rgb,
            reader=reader,
            conf=ocr_conf,
            min_size=ocr_min_size,
            pad=ocr_pad,
            pad_ratio=ocr_pad_ratio,
            rect_pad=ocr_rect_pad,
            rect_pad_ratio=ocr_rect_pad_ratio,
            close=ocr_close,
            passes=ocr_passes,
        )
        if np.count_nonzero(mask_u8) == 0:
            return image_rgb, empty
        mask_pil = Image.fromarray(mask_u8, mode="L")
        out = models.lama(image_rgb, mask_pil)
        if poisson_blend:
            out = _poisson_blend(original_rgb=image_rgb, inpainted_rgb=out, mask_u8=mask_u8)
        return out, mask_pil


    ocr_mask = None
    if mode == "ui+text":
        reader = _get_ocr_reader(ocr_langs, gpu=bool(ocr_gpu))
        ocr_mask = ocr_text_mask_multipass(
            image_rgb,
            reader=reader,
            conf=ocr_conf,
            min_size=ocr_min_size,
            pad=ocr_pad,
            pad_ratio=ocr_pad_ratio,
            rect_pad=ocr_rect_pad,
            rect_pad_ratio=ocr_rect_pad_ratio,
            close=ocr_close,
            passes=ocr_passes,
        )

    thr = float(box_threshold)
    tthr = float(text_threshold)
    steps = max(1, int(tighten_steps))

    for _ in range(steps):
        dino_prompt = prompt

        boxes = dino_detect_xyxy(
            models,
            image_rgb,
            prompt=dino_prompt,
            box_threshold=thr,
            text_threshold=tthr,
            max_box_area=max_box_area,
        )

        masks01 = sam_masks_from_boxes(models, image_rgb, boxes) if boxes.shape[0] else []
        dino_mask_u8 = merge_and_refine_masks(masks01, dilate=dilate) if masks01 else np.zeros((h, w), dtype=np.uint8)


        if ocr_mask is not None:
            mask_u8 = np.maximum(dino_mask_u8, ocr_mask)
        else:
            mask_u8 = dino_mask_u8

        if np.count_nonzero(mask_u8) == 0:
            thr = min(0.95, thr * 1.25)
            tthr = min(0.95, tthr * 1.25)
            continue

        ratio = float(np.count_nonzero(mask_u8)) / float(h * w)


        if max_mask_area > 0 and ratio > float(max_mask_area):
            thr = min(0.95, thr * 1.25)
            tthr = min(0.95, tthr * 1.25)
            continue

        mask_pil = Image.fromarray(mask_u8, mode="L")
        out = models.lama(image_rgb, mask_pil)
        if poisson_blend:
            out = _poisson_blend(original_rgb=image_rgb, inpainted_rgb=out, mask_u8=mask_u8)
        return out, mask_pil


    return image_rgb, empty


def _is_image_path(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _iter_images(input_dir: Path, *, exclude_dirs: list[Path] | None = None) -> list[Path]:
    exclude_dirs = exclude_dirs or []
    images: list[Path] = []
    for p in input_dir.rglob("*"):
        if not _is_image_path(p):
            continue
        if any(_is_relative_to(p, ex) for ex in exclude_dirs):
            continue
        images.append(p)
    return sorted(images, key=lambda x: str(x.relative_to(input_dir)).lower())


def _save_image_like_input(image_rgb: Image.Image, out_path: Path, *, input_suffix: str) -> None:
    """
    Если out_path уже содержит расширение — используем его.
    Если out_path без расширения — используем input_suffix (как у входного файла).
    """
    suffix = out_path.suffix.lower()
    if not suffix:
        suffix = (input_suffix or ".png").lower()
        out_path = out_path.with_suffix(suffix)

    fmt = None
    save_kwargs: dict[str, object] = {}

    if suffix in {".jpg", ".jpeg"}:
        fmt = "JPEG"
        save_kwargs = {"quality": 95, "subsampling": 0, "optimize": True}
    elif suffix == ".png":
        fmt = "PNG"
        save_kwargs = {"optimize": True}
    elif suffix == ".webp":
        fmt = "WEBP"
        save_kwargs = {"quality": 95, "method": 6}
    elif suffix in {".tif", ".tiff"}:
        fmt = "TIFF"
    elif suffix == ".bmp":
        fmt = "BMP"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb.save(out_path, format=fmt, **save_kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Удаление UI/текста: GroundingDINO → SAM → LaMa (локально)")
    ap.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        type=str,
        help="Путь к файлу ИЛИ папке с изображениями (по умолчанию: input/)",
    )
    ap.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=str,
        help="Путь к файлу ИЛИ папке для результатов (по умолчанию: output/)",
    )
    ap.add_argument(
        "--mask-out",
        default=None,
        type=str,
        help="Для файла: путь к маске. Для папки: папка для масок (например output/masks)",
    )
    ap.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        type=str,
        help="ui|text|ui+text. text = удалять только текст (паук/паутина остаются).",
    )
    ap.add_argument(
        "--text-only",
        action="store_true",
        help="Удалять только текст (OCR), не трогая UI. Эквивалентно --mode text, но имеет приоритет.",
    )
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, type=str, help="Что искать (через запятую)")
    ap.add_argument("--box-threshold", default=0.25, type=float, help="Порог bbox (меньше → больше находок)")
    ap.add_argument("--text-threshold", default=0.25, type=float, help="Порог текста (меньше → больше находок)")
    ap.add_argument("--dilate", default=15, type=int, help="Расширение маски (px), чтобы убрать каймы")
    ap.add_argument(
        "--max-box-area",
        default=0.45,
        type=float,
        help="Макс. доля площади одного bbox (0..1). Большие bbox часто ломают маску. 0 = выкл.",
    )
    ap.add_argument(
        "--max-mask-area",
        default=0.35,
        type=float,
        help="Макс. доля площади итоговой маски (0..1). Если больше — ужесточаем пороги. 0 = выкл.",
    )
    ap.add_argument(
        "--tighten-steps",
        default=4,
        type=int,
        help="Сколько раз ужесточать пороги, если маска слишком большая",
    )
    ap.add_argument(
        "--ocr-langs",
        default="en",
        type=str,
        help="Языки easyocr через запятую (например: en или en,ru). Используется в mode=text/ui+text.",
    )
    ap.add_argument(
        "--ocr-gpu",
        default="auto",
        type=str,
        help="Использовать GPU для OCR: auto|true|false. auto = включить, если доступна CUDA.",
    )
    ap.add_argument(
        "--ocr-conf",
        default=0.15,
        type=float,
        help="Порог уверенности OCR (0..1). Используется в mode=text/ui+text.",
    )
    ap.add_argument(
        "--ocr-min-size",
        default=6,
        type=int,
        help="Минимальный размер bbox текста (px). Используется в mode=text/ui+text.",
    )
    ap.add_argument(
        "--ocr-pad",
        default=15,
        type=int,
        help="Расширение маски OCR (px). Используется в mode=text/ui+text.",
    )
    ap.add_argument(
        "--ocr-pad-ratio",
        default=0.22,
        type=float,
        help="Адаптивный паддинг OCR (доля от размера bbox). Снижает захват соседнего UI.",
    )
    ap.add_argument(
        "--ocr-rect-pad",
        default=18,
        type=int,
        help="Доп. запас вокруг OCR bbox прямоугольником (px). Сильно помогает со шрифтами/обводкой.",
    )
    ap.add_argument(
        "--ocr-rect-pad-ratio",
        default=0.28,
        type=float,
        help="Адаптивный rect-pad (доля от bbox). Делает маску точнее на мелком тексте.",
    )
    ap.add_argument(
        "--ocr-close",
        default=9,
        type=int,
        help="Морфологическое закрытие OCR маски (px). Заполняет дырки/соединяет фрагменты.",
    )
    ap.add_argument(
        "--ocr-passes",
        default="rgb,clahe",
        type=str,
        help="OCR проходы: rgb,clahe (через запятую). Помогает ловить текст с низким контрастом.",
    )
    ap.add_argument(
        "--tight-text-mask",
        action="store_true",
        help="Сделать маску строго по контурам текста (без расширения/склейки). Полезно, чтобы не задевать UI рядом.",
    )
    ap.add_argument(
        "--tight-mask-pad",
        default=5,
        type=int,
        help="Небольшое расширение (px) для --tight-text-mask, чтобы лучше восстанавливался фон по краям текста.",
    )
    ap.add_argument(
        "--poisson-blend",
        action="store_true",
        help="Смешать инпейнт с оригиналом по маске (меньше швов/артефактов).",
    )
    ap.add_argument(
        "--enhance",
        default="none",
        type=str,
        help="Пост-улучшение без смены размера: none|sharp|sharp2x",
    )
    ap.add_argument("--enhance-amount", default=0.6, type=float, help="Сила unsharp (примерно 0.3..1.2)")
    ap.add_argument("--enhance-radius", default=1.2, type=float, help="Радиус unsharp (sigma)")
    ap.add_argument("--enhance-threshold", default=3, type=int, help="Порог (0 = везде, иначе только по контрасту)")
    ap.add_argument("--max-side", default=1600, type=int, help="Уменьшить картинку по большей стороне (0 = не менять)")
    ap.add_argument(
        "--open-viewer",
        action="store_true",
        help="После обработки открыть UI‑просмотрщик `ui_viewer.py` для папки output/ (удобно для быстрого просмотра/дочистки кистью).",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Не качать модели (только из кэша). Используй после первого успешного запуска.",
    )
    ap.add_argument("--device", default="auto", type=str, help="auto|cuda|mps|cpu")
    ap.add_argument(
        "--lama-device",
        default="auto",
        type=str,
        help="auto|cuda|mps|cpu (у TorchScript могут быть ограничения; при проблемах будет fallback на cpu)",
    )
    ap.add_argument("--dino-model", default="IDEA-Research/grounding-dino-base", type=str)
    ap.add_argument("--sam-model", default="facebook/sam-vit-base", type=str)
    ap.add_argument(
        "--lama-model",
        default=None,
        type=str,
        help="Путь к локальному файлу big-lama.pt (если не хочешь/не можешь скачивать автоматически)",
    )
    args = ap.parse_args()


    warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))


    if bool(args.text_only):
        args.mode = "text"

    device = _parse_device(args.device)
    lama_device = _parse_lama_device(str(args.lama_device))
    models = load_models(
        dino_model_id=args.dino_model,
        sam_model_id=args.sam_model,
        lama_model_path=args.lama_model,
        lama_device=lama_device,
        device=device,
        offline=bool(args.offline),
    )

    out_path = Path(args.output)
    mask_out = Path(args.mask_out) if args.mask_out else None

    def _launch_viewer(view_dir: Path) -> None:
        viewer = Path(__file__).with_name("ui_viewer.py")
        if not viewer.exists():
            _log(f"[yellow]Не найден ui_viewer.py рядом со скриптом: {viewer}[/yellow]")
            return
        try:
            subprocess.run([sys.executable, str(viewer), "--dir", str(view_dir)], check=False)
        except Exception as e:
            _log(f"[yellow]Не удалось запустить просмотрщик: {e}[/yellow]")

    def _log(msg: str) -> None:
        try:
            from rich.console import Console  # type: ignore

            Console().print(msg)
        except Exception:
            print(msg)

    def _progress_iter(items: list[Path]):
        try:
            from rich.progress import (
                Progress,
                BarColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )  # type: ignore

            with Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Обработка", total=len(items))
                for it in items:
                    yield it
                    progress.advance(task)
        except Exception:
            for it in items:
                yield it

    def process_one(image_path: Path, out_file: Path, mask_file: Path | None) -> None:
        t0 = time.time()
        image = Image.open(image_path).convert("RGB")
        work_image, scale = _resize_keep_aspect(image, int(args.max_side))

        ocr_langs = [x.strip() for x in str(args.ocr_langs).split(",") if x.strip()]
        if not ocr_langs:
            ocr_langs = ["en"]

        # OCR GPU: auto = включить при доступной CUDA
        ocr_gpu_raw = str(getattr(args, "ocr_gpu", "auto")).strip().lower()
        if ocr_gpu_raw in {"1", "true", "yes", "y", "on"}:
            ocr_gpu = True
        elif ocr_gpu_raw in {"0", "false", "no", "n", "off"}:
            ocr_gpu = False
        elif ocr_gpu_raw == "auto":
            ocr_gpu = torch.cuda.is_available()
        else:
            raise ValueError("--ocr-gpu должен быть: auto|true|false")

        # Флаг “строго по тексту”: отключаем все расширения маски.
        if bool(args.tight_text_mask):
            # Оставляем только контур текста + небольшой dilate (чтобы убрать антиалиасинг/обводку)
            ocr_pad = max(0, int(args.tight_mask_pad))
            ocr_pad_ratio = 0.0
            ocr_rect_pad = 0
            ocr_rect_pad_ratio = 0.0
            ocr_close = 0
            ocr_passes = "rgb"
        else:
            ocr_pad = int(args.ocr_pad)
            ocr_pad_ratio = float(args.ocr_pad_ratio)
            ocr_rect_pad = int(args.ocr_rect_pad)
            ocr_rect_pad_ratio = float(args.ocr_rect_pad_ratio)
            ocr_close = int(args.ocr_close)
            ocr_passes = str(args.ocr_passes)

        out_work, mask_work = remove_ui(
            models,
            work_image,
            mode=str(args.mode),
            prompt=args.prompt,
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
            dilate=int(args.dilate),
            max_box_area=float(args.max_box_area),
            max_mask_area=float(args.max_mask_area),
            tighten_steps=int(args.tighten_steps),
            ocr_langs=ocr_langs,
            ocr_gpu=bool(ocr_gpu),
            ocr_conf=float(args.ocr_conf),
            ocr_min_size=int(args.ocr_min_size),
            ocr_pad=ocr_pad,
            ocr_rect_pad=ocr_rect_pad,
            ocr_close=ocr_close,
            ocr_passes=ocr_passes,
            ocr_pad_ratio=ocr_pad_ratio,
            ocr_rect_pad_ratio=ocr_rect_pad_ratio,
            poisson_blend=bool(args.poisson_blend),
        )

        # Если уменьшали для ускорения — переносим маску на исходное разрешение и инпейтим уже в нем.
        if scale != 1.0:
            mask_full = mask_work.resize(image.size, resample=Image.NEAREST)
            out_full = models.lama(image, mask_full)
        else:
            mask_full = mask_work
            out_full = out_work

        # Пост-улучшение "без смены размера"
        out_full = _enhance_same_size(
            out_full,
            method=str(args.enhance),
            amount=float(args.enhance_amount),
            radius=float(args.enhance_radius),
            threshold=int(args.enhance_threshold),
        )

        _save_image_like_input(out_full, out_file, input_suffix=image_path.suffix)

        if mask_file is not None:
            mask_file.parent.mkdir(parents=True, exist_ok=True)
            mask_full.save(mask_file)

        dt = time.time() - t0
        _log(f"[green]OK[/green] {image_path.name} → {out_file.name}  [dim]({dt:.2f}s)[/dim]")

    # Режим "папка → папка"
    if in_path.is_dir():
        out_dir = out_path
        out_dir.mkdir(parents=True, exist_ok=True)

        mask_dir = mask_out
        if mask_dir is None and args.mask_out is not None:
            mask_dir = Path(args.mask_out)
        # если маски включены, но путь не задан — кладем в output/masks
        if mask_dir is None and args.mask_out is None:
            mask_dir = None

        if mask_dir is not None:
            mask_dir.mkdir(parents=True, exist_ok=True)

        exclude_dirs: list[Path] = []
        if _is_relative_to(out_dir, in_path):
            exclude_dirs.append(out_dir)
        if mask_dir is not None and _is_relative_to(mask_dir, in_path):
            exclude_dirs.append(mask_dir)

        images = _iter_images(in_path, exclude_dirs=exclude_dirs)
        if not images:
            raise FileNotFoundError(f"В папке {in_path} нет изображений (png/jpg/webp/...)")

        _log(
            f"[bold]Найдено:[/bold] {len(images)}  |  "
            f"[bold]mode:[/bold] {args.mode}  |  "
            f"[bold]device:[/bold] {device}  |  "
            f"[bold]lama:[/bold] {models.lama.device}  |  "
            f"[bold]offline:[/bold] {bool(args.offline)}"
        )

        for p in _progress_iter(images):
            rel = p.relative_to(in_path)
            # Сохраняем в том же формате и в той же иерархии, что и вход
            out_file = out_dir / rel
            mask_file = (mask_dir / rel.parent / f"{p.stem}_mask.png") if (mask_dir is not None) else None
            process_one(p, out_file, mask_file)

        if bool(args.open_viewer):
            _launch_viewer(out_dir)
        return 0

    # Режим "файл → файл/папка"
    if not _is_image_path(in_path):
        raise ValueError(f"Неподдерживаемый файл: {in_path}")

    if out_path.suffix == "" or out_path.is_dir():
        out_file = (out_path if out_path.suffix == "" else out_path) / f"{in_path.stem}{in_path.suffix}"
    else:
        out_file = out_path

    if mask_out is None:
        mask_file = None
    else:
        # Если дали папку — кладем стандартное имя
        if mask_out.suffix == "" or mask_out.is_dir():
            mask_file = (mask_out if mask_out.suffix == "" else mask_out) / f"{in_path.stem}_mask.png"
        else:
            mask_file = mask_out

    process_one(in_path, out_file, mask_file)

    if bool(args.open_viewer):
        _launch_viewer(out_file.parent)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
