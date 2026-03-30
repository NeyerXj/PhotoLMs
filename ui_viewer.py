import argparse
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# Переиспользуем LaMa и сохранение из основного скрипта
from ui_remove import (  # type: ignore
    LamaTorchscript,
    _ensure_ssl_certs,
    _parse_lama_device,
    _save_image_like_input,
    _setup_local_caches,
)


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _is_image_path(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS


def _is_viewable_result_image(p: Path, root_dir: Path) -> bool:
    if not _is_image_path(p):
        return False
    try:
        rel = p.relative_to(root_dir)
    except Exception:
        rel = p
    parts_lower = {part.lower() for part in rel.parts[:-1]}
    if "masks" in parts_lower or "_backup_originals" in parts_lower or "_backups_originals" in parts_lower:
        return False
    if p.stem.lower().endswith("_mask"):
        return False
    return True


def _iter_images(dir_path: Path) -> list[Path]:
    images = [p for p in dir_path.rglob("*") if _is_viewable_result_image(p, dir_path)]
    return sorted(images, key=lambda x: str(x.relative_to(dir_path)).lower())


def _bgr_to_pil_rgb(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")


def _mask_to_pil_l(mask_u8: np.ndarray) -> Image.Image:
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    return Image.fromarray(mask_u8, mode="L")


def _pil_rgb_to_bgr(pil: Image.Image) -> np.ndarray:
    rgb = np.array(pil, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _masked_bbox(mask_u8: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _run_lama_on_roi(
    *,
    lama: LamaTorchscript,
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    roi_pad: int,
    max_side: int,
) -> np.ndarray:
    bbox = _masked_bbox(mask_u8)
    if bbox is None:
        return image_bgr.copy()

    h, w = mask_u8.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = max(0, int(roi_pad))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    roi_img = image_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
    roi_mask_full = mask_u8[y1 : y2 + 1, x1 : x2 + 1].copy()

    src_h, src_w = roi_img.shape[:2]
    work_img = roi_img
    work_mask = roi_mask_full

    m = max(src_h, src_w)
    if int(max_side) > 0 and m > int(max_side):
        scale = float(max_side) / float(m)
        dst_w = max(1, int(round(src_w * scale)))
        dst_h = max(1, int(round(src_h * scale)))
        work_img = cv2.resize(roi_img, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        work_mask = cv2.resize(roi_mask_full, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)

    out_roi = _pil_rgb_to_bgr(lama(_bgr_to_pil_rgb(work_img), _mask_to_pil_l(work_mask)))

    if out_roi.shape[1] != src_w or out_roi.shape[0] != src_h:
        out_roi = cv2.resize(out_roi, (src_w, src_h), interpolation=cv2.INTER_LINEAR)

    result = image_bgr.copy()
    result[y1 : y2 + 1, x1 : x2 + 1] = out_roi
    return result


@dataclass
class ViewerState:
    images: list[Path]
    idx: int
    brush: int
    alpha: float
    backup_dir: Path
    lama: LamaTorchscript
    lama_roi_pad: int
    lama_max_side: int

    path: Optional[Path] = None
    base_bgr: Optional[np.ndarray] = None  # текущее изображение (может быть изменено)
    orig_bgr: Optional[np.ndarray] = None  # как на диске (после последнего save/reload)
    mask: Optional[np.ndarray] = None
    dirty: bool = False
    drawing: bool = False
    erasing: bool = False
    last_xy: Optional[tuple[int, int]] = None
    status_msg: str = ""
    show_overlay: bool = True
    # view mapping (display -> original image)
    view_scale: float = 1.0
    view_offx: int = 0
    view_offy: int = 0
    view_disp_w: int = 0
    view_disp_h: int = 0

    # async processing (so UI doesn't freeze)
    processing: bool = False
    _worker: Optional[threading.Thread] = None
    _worker_out: Optional[np.ndarray] = None
    _worker_err: Optional[str] = None
    _processing_started_at: float = 0.0

    def load_current(self) -> None:
        self.path = self.images[self.idx]
        bgr = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Не удалось прочитать файл: {self.path}")
        self.base_bgr = bgr
        self.orig_bgr = bgr.copy()
        self.mask = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.uint8)
        self.dirty = False
        self.status_msg = ""
        self.processing = False
        self._worker = None
        self._worker_out = None
        self._worker_err = None

    def clear_mask(self) -> None:
        if self.mask is not None:
            self.mask.fill(0)

    def reload_from_disk(self) -> None:
        if self.path is None:
            return
        bgr = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Не удалось прочитать файл: {self.path}")
        self.base_bgr = bgr
        self.orig_bgr = bgr.copy()
        self.clear_mask()
        self.dirty = False
        self.status_msg = "Перезагружено (изменения сброшены)"

    def apply_inpaint(self) -> None:
        if self.base_bgr is None or self.mask is None or self.path is None:
            return
        if self.processing:
            self.status_msg = "Уже идёт обработка…"
            return
        if int(np.count_nonzero(self.mask)) == 0:
            self.status_msg = "Маска пустая"
            return
        # Запускаем LaMa в фоне, чтобы окно не "залагивало"
        img_bgr = self.base_bgr.copy()
        mask_u8 = self.mask.copy()
        self.processing = True
        self._worker_out = None
        self._worker_err = None
        self._processing_started_at = time.time()
        self.status_msg = "Processing…"

        def _run() -> None:
            try:
                self._worker_out = _run_lama_on_roi(
                    lama=self.lama,
                    image_bgr=img_bgr,
                    mask_u8=mask_u8,
                    roi_pad=self.lama_roi_pad,
                    max_side=self.lama_max_side,
                )
            except Exception as e:
                self._worker_err = str(e)

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def _ensure_backup(self) -> None:
        if self.path is None:
            return
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        try:
            base_dir = self.backup_dir.parent
            rel = self.path.relative_to(base_dir)
            backup_path = self.backup_dir / rel
        except Exception:
            backup_path = self.backup_dir / self.path.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if backup_path.exists():
            return
        shutil.copy2(self.path, backup_path)

    def save_to_disk(self) -> None:
        if self.path is None or self.base_bgr is None:
            return
        self._ensure_backup()
        pil = _bgr_to_pil_rgb(self.base_bgr)
        _save_image_like_input(pil, self.path, input_suffix=self.path.suffix)
        self.orig_bgr = self.base_bgr.copy()
        self.dirty = False
        self.status_msg = "Сохранено (оригинал в _backup_originals)"


def _render(state: ViewerState) -> np.ndarray:
    assert state.base_bgr is not None and state.mask is not None and state.path is not None
    # Вырезаем/масштабируем под размер окна с сохранением пропорций
    try:
        _x, _y, win_w, win_h = cv2.getWindowImageRect("PhotoLM Viewer")
        win_w = int(win_w)
        win_h = int(win_h)
        if win_w <= 0 or win_h <= 0:
            raise RuntimeError("bad window rect")
    except Exception:
        # fallback
        win_w, win_h = 1280, 720

    base = state.base_bgr
    h0, w0 = base.shape[:2]
    # небольшой отступ под текст сверху, чтобы он не перекрывал картинку
    top_pad = 74
    avail_w = max(64, win_w)
    avail_h = max(64, win_h - top_pad)
    scale = min(avail_w / float(w0), avail_h / float(h0))
    scale = float(max(1e-6, scale))
    disp_w = max(1, int(round(w0 * scale)))
    disp_h = max(1, int(round(h0 * scale)))
    offx = int((win_w - disp_w) // 2)
    offy = int(top_pad + (avail_h - disp_h) // 2)

    state.view_scale = scale
    state.view_offx = offx
    state.view_offy = offy
    state.view_disp_w = disp_w
    state.view_disp_h = disp_h

    # Собираем превью: накладка маски на оригинал, затем resize
    img = base.copy()

    # Накладка маски (красным)
    m = state.mask
    if int(np.count_nonzero(m)) > 0:
        alpha = float(np.clip(state.alpha, 0.0, 1.0))
        red = np.array([0, 0, 255], dtype=np.float32)
        sel = m > 0
        base = img[sel].astype(np.float32)
        img[sel] = np.clip(base * (1.0 - alpha) + red * alpha, 0, 255).astype(np.uint8)

    # resize под окно
    if disp_w != w0 or disp_h != h0:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(img, (disp_w, disp_h), interpolation=interp)

    # фон/канва
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)
    x1 = max(0, offx)
    y1 = max(0, offy)
    x2 = min(win_w, offx + disp_w)
    y2 = min(win_h, offy + disp_h)
    if x2 > x1 and y2 > y1:
        roi = canvas[y1:y2, x1:x2]
        src = img[(y1 - offy) : (y2 - offy), (x1 - offx) : (x2 - offx)]
        if src.shape[:2] == roi.shape[:2]:
            roi[:] = src

    # Строка статуса
    idx = state.idx + 1
    total = len(state.images)
    dirty = "DIRTY" if state.dirty else "OK"
    mask_px = int(np.count_nonzero(state.mask))
    proc = "  [PROCESSING]" if state.processing else ""
    text = f"{idx}/{total}  {state.path.name}  brush={state.brush}px  mask={mask_px}px  {dirty}{proc}"
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    if state.status_msg:
        msg = state.status_msg
        if state.processing:
            # простая анимация точек
            dt = time.time() - float(state._processing_started_at)
            dots = "." * (1 + (int(dt * 3) % 3))
            msg = "Processing" + dots
        cv2.putText(canvas, msg, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, msg, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    if state.show_overlay:
        lines = [
            "Hotkeys: N/Right=next  P/Left=prev",
            "Brush: [ ] or Up/Down (screen px)",
            "Paint mask: LMB   Erase: RMB",
            "Apply (inpaint): E/Enter   Save: S   Reset: X",
            "Clear mask: C   Toggle overlay: H   Quit: Q/Esc",
        ]
        y = canvas.shape[0] - 12 - (len(lines) - 1) * 22
        y = max(90, y)
        for i, ln in enumerate(lines):
            yy = y + i * 22
            cv2.putText(canvas, ln, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, ln, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def _mouse_cb(event, x, y, flags, userdata) -> None:  # type: ignore
    state: ViewerState = userdata
    if state.mask is None:
        return
    if state.processing:
        return

    # перевод координат из окна в координаты исходного изображения
    scale = float(max(1e-6, state.view_scale))
    ix = int(round((x - int(state.view_offx)) / scale))
    iy = int(round((y - int(state.view_offy)) / scale))
    if state.base_bgr is None:
        return
    h0, w0 = state.base_bgr.shape[:2]
    if ix < 0 or iy < 0 or ix >= w0 or iy >= h0:
        # клик не по картинке (в полях)
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.erasing = False
        state.last_xy = (ix, iy)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.erasing = True
        state.drawing = False
        state.last_xy = (ix, iy)
    elif event in {cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP}:
        state.drawing = False
        state.erasing = False
        state.last_xy = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if not (state.drawing or state.erasing):
            return
        val = 255 if state.drawing else 0
        # brush задан в "экранных" пикселях, переводим в пиксели исходника
        r = max(1, int(round(float(state.brush) / scale)))
        if state.last_xy is None:
            state.last_xy = (ix, iy)
        x0, y0 = state.last_xy
        cv2.line(state.mask, (x0, y0), (ix, iy), int(val), thickness=r * 2, lineType=cv2.LINE_AA)
        cv2.circle(state.mask, (ix, iy), r, int(val), thickness=-1, lineType=cv2.LINE_AA)
        state.last_xy = (ix, iy)


def _print_help() -> None:
    print(
        "\nУправление:\n"
        "  ЛКМ — рисовать маску (удалить область)\n"
        "  ПКМ — стирать маску\n"
        "  колесо/скролл не используется\n"
        "\n"
        "Клавиши:\n"
        "  N / →  : следующее фото (если DIRTY — автосейв)\n"
        "  P / ←  : предыдущее фото (если DIRTY — автосейв)\n"
        "  [ / ] или ↑/↓ : размер кисти -/+\n"
        "  C      : очистить маску\n"
        "  E/Enter: применить LaMa-инпейтинг по маске (в память)\n"
        "  S      : сохранить (перезаписать файл; оригинал в _backup_originals)\n"
        "  X      : сбросить изменения (перечитать файл)\n"
        "  H      : показать подсказку в консоли / вкл-выкл оверлей\n"
        "  Q/Esc  : выход (если DIRTY — автосейв)\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Просмотр output/ по одной картинке + кисть удаления (LaMa).")
    ap.add_argument("--dir", default="output", type=str, help="Папка с изображениями (по умолчанию: output/)")
    ap.add_argument("--start", default=None, type=str, help="Старт: индекс (0..N-1) или имя файла")
    ap.add_argument("--brush", default=28, type=int, help="Размер кисти (px)")
    ap.add_argument("--alpha", default=0.45, type=float, help="Прозрачность маски (0..1)")
    ap.add_argument(
        "--backup-dir",
        default=None,
        type=str,
        help="Куда складывать оригиналы перед перезаписью (по умолчанию: <dir>/_backup_originals)",
    )
    ap.add_argument("--lama-model", default=None, type=str, help="Путь к big-lama.pt (если не качать автоматически)")
    ap.add_argument("--lama-device", default="auto", type=str, help="auto|cuda|mps|cpu")
    ap.add_argument("--lama-roi-pad", default=128, type=int, help="Запас вокруг маски перед инпейтингом (px)")
    ap.add_argument("--lama-max-side", default=1600, type=int, help="Ограничение большей стороны ROI (0 = без ограничения)")
    args = ap.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(str(dir_path))

    images = _iter_images(dir_path)
    if not images:
        raise FileNotFoundError(f"В папке {dir_path} нет изображений (png/jpg/webp/...)")

    start_idx = 0
    if args.start is not None:
        s = str(args.start).strip()
        if s.isdigit():
            start_idx = int(s)
        else:
            # поиск по имени
            lower = s.lower()
            for i, p in enumerate(images):
                if p.name.lower() == lower:
                    start_idx = i
                    break
    start_idx = int(np.clip(start_idx, 0, max(0, len(images) - 1)))

    _ensure_ssl_certs()
    _setup_local_caches()
    lama_device = _parse_lama_device(str(args.lama_device))
    lama = LamaTorchscript(device=lama_device, model_path=str(args.lama_model) if args.lama_model else None)

    backup_dir = Path(args.backup_dir) if args.backup_dir else (dir_path / "_backup_originals")

    state = ViewerState(
        images=images,
        idx=start_idx,
        brush=max(1, int(args.brush)),
        alpha=float(args.alpha),
        backup_dir=backup_dir,
        lama=lama,
        lama_roi_pad=max(0, int(args.lama_roi_pad)),
        lama_max_side=max(0, int(args.lama_max_side)),
    )
    state.load_current()

    win = "PhotoLM Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_cb, state)
    _print_help()

    # Коды стрелок для cv2.waitKeyEx (Windows / X11 / legacy)
    KEY_LEFT = {2424832, 65361, 81}
    KEY_RIGHT = {2555904, 65363, 83}
    KEY_UP = {2490368, 65362, 82}
    KEY_DOWN = {2621440, 65364, 84}

    while True:
        frame = _render(state)
        cv2.imshow(win, frame)

        # если фоновая обработка закончилась — применяем результат
        if state.processing:
            if state._worker_err is not None:
                state.processing = False
                state.status_msg = f"Ошибка: {state._worker_err}"
                state._worker = None
                state._worker_err = None
            elif state._worker_out is not None:
                state.base_bgr = state._worker_out
                state._worker_out = None
                state.processing = False
                state.clear_mask()
                state.dirty = True
                state.status_msg = "Инпейтинг применён (нажми S чтобы сохранить)"
                state._worker = None

        key = cv2.waitKeyEx(16)
        if key == -1:
            continue

        # выход
        if key in {ord("q"), ord("Q"), 27}:  # q or Esc
            if state.dirty:
                state.save_to_disk()
            break

        if key in {ord("h"), ord("H")}:
            _print_help()
            state.show_overlay = not state.show_overlay
            state.status_msg = "Подсказка: консоль / оверлей переключен"
            continue

        # навигация (автосейв если DIRTY)
        if key in {ord("n"), ord("N")} or (key in KEY_RIGHT):  # n or Right
            if state.dirty:
                state.save_to_disk()
            state.idx = min(len(state.images) - 1, state.idx + 1)
            state.load_current()
            continue
        if key in {ord("p"), ord("P")} or (key in KEY_LEFT):  # p or Left
            if state.dirty:
                state.save_to_disk()
            state.idx = max(0, state.idx - 1)
            state.load_current()
            continue

        # кисть
        if key == ord("["):
            state.brush = max(1, int(state.brush) - 2)
            state.status_msg = f"Кисть: {state.brush}px"
            continue
        if key == ord("]"):
            state.brush = min(512, int(state.brush) + 2)
            state.status_msg = f"Кисть: {state.brush}px"
            continue
        if key in KEY_UP:
            state.brush = min(512, int(state.brush) + 2)
            state.status_msg = f"Кисть: {state.brush}px"
            continue
        if key in KEY_DOWN:
            state.brush = max(1, int(state.brush) - 2)
            state.status_msg = f"Кисть: {state.brush}px"
            continue

        # маска
        if key in {ord("c"), ord("C")}:
            state.clear_mask()
            state.status_msg = "Маска очищена"
            continue

        # применить инпейтинг
        if key in {ord("e"), ord("E"), 13}:  # e or Enter
            state.apply_inpaint()
            continue

        # сохранить
        if key in {ord("s"), ord("S")}:
            state.save_to_disk()
            continue

        # сбросить изменения
        if key in {ord("x"), ord("X")}:
            state.reload_from_disk()
            continue

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

