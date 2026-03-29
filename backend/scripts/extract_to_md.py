# Dependencies (Python 3.10+):
#   pip install pdfplumber pandas google-genai Pillow PyYAML

from __future__ import annotations

import io
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
import yaml
from google import genai
from google.genai import types
from PIL import Image

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

GEMINI_MODEL = "gemini-2.5-flash"

CSV_COLUMNS = ("order_id", "date", "shop_name", "amount", "currency", "notes")

VISION_PROMPT = """请读取这张图片中的文字与版式信息，将内容整理为 Markdown。

要求：
1. 保留购买商品、金额、日期等信息的结构与层次（可用标题、列表、表格）。
2. 输出语言与图片中主要文字一致；若为多语言，以中文摘要优先并保留关键原文。
3. 不要添加图片中不存在的猜测性内容；无法识别的部分用「（无法识别）」标注。
4. 只输出 Markdown 正文，不要代码块包裹全文。"""


_LOGGING_CONFIGURED = False


def _setup_logging() -> None:
    global _LOGGING_CONFIGURED
    log_path = OUT_DIR / "extract_to_md.log"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if _LOGGING_CONFIGURED:
        return
    _LOGGING_CONFIGURED = True
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)


def build_markdown_front_matter(source_name: str, extra: dict[str, Any] | None = None) -> str:
    meta: dict[str, Any] = {
        "source_file": source_name,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "processor": "extract_to_md",
    }
    if extra:
        meta.update(extra)
    dumped = yaml.safe_dump(
        meta,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    return f"---\n{dumped}\n---\n\n"


def _escape_md_cell(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("|", "\\|")


def _table_to_markdown(table: list[list[Any | None]]) -> str:
    if not table or not any(any(c is not None and str(c).strip() for c in row) for row in table):
        return ""
    rows: list[list[str]] = []
    for row in table:
        rows.append([_escape_md_cell(c) for c in row])
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    header = norm[0]
    body = norm[1:] if len(norm) > 1 else []
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def process_pdf(path: Path) -> str:
    parts: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text(layout=True) or ""
                    text = text.strip()
                    if text:
                        parts.append(text)
                    tables = page.extract_tables() or []
                    for t in tables:
                        try:
                            md = _table_to_markdown(t)
                            if md:
                                parts.append(md)
                        except Exception:
                            LOGGER.exception("PDF table conversion failed: %s page %s", path, i + 1)
                except Exception:
                    LOGGER.exception("PDF page extract failed: %s page %s", path, i + 1)
    except Exception:
        LOGGER.exception("Failed to open or read PDF: %s", path)
        return ""
    return "\n\n".join(parts).strip()


def _read_csv_with_fallback(path: Path) -> pd.DataFrame | None:
    encodings = ("utf-8-sig", "utf-8", "cp932", "latin-1")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            LOGGER.exception("CSV read failed (%s encoding): %s", enc, path)
            return None
    try:
        return pd.read_csv(path, encoding_errors="replace")
    except Exception:
        LOGGER.exception("CSV read failed after fallbacks: %s", path)
        return None


def process_csv(path: Path) -> str:
    df = _read_csv_with_fallback(path)
    if df is None or df.empty:
        return ""
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        LOGGER.warning("CSV missing columns %s in %s — using empty placeholders", missing, path)
    chunks: list[str] = []
    for _, row in df.iterrows():
        def g(col: str) -> str:
            if col not in df.columns:
                return "N/A"
            v = row[col]
            if pd.isna(v):
                return "N/A"
            return str(v).strip() or "N/A"

        block = (
            f"# 注文ID: {g('order_id')}\n"
            f"- 日付: {g('date')}\n"
            f"- 店舗: {g('shop_name')}\n"
            f"- 金額: {g('amount')} {g('currency')}\n"
            f"- 備考: {g('notes')}\n"
        )
        chunks.append(block.strip())
    return "\n\n---\n\n".join(chunks).strip()


def process_image(path: Path) -> str:
    if not os.environ.get("GEMINI_API_KEY"):
        LOGGER.error("GEMINI_API_KEY is not set; skipping image: %s", path)
        return ""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        with genai.Client() as client:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    VISION_PROMPT,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ],
            )
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            LOGGER.warning("Gemini returned no candidates (possibly blocked): %s", path)
            return ""
        try:
            text = (response.text or "").strip()
        except ValueError as e:
            LOGGER.warning("Gemini response has no text (blocked or empty): %s — %s", path, e)
            return ""
        return text
    except Exception:
        LOGGER.exception("Image / Gemini processing failed: %s", path)
        return ""


def write_markdown_output(source_path: Path, body: str) -> None:
    if not body:
        LOGGER.warning("No content extracted; skipping output for %s", source_path)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fm = build_markdown_front_matter(source_path.name)
    out_path = OUT_DIR / f"{source_path.stem}.md"
    try:
        out_path.write_text(fm + body + "\n", encoding="utf-8")
        LOGGER.info("Wrote %s", out_path)
    except OSError:
        LOGGER.exception("Failed to write %s", out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging()
    if not RAW_DIR.is_dir():
        LOGGER.error("Raw directory does not exist: %s", RAW_DIR)
        return
    for entry in sorted(RAW_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file():
            continue
        suf = entry.suffix.lower()
        try:
            if suf == ".pdf":
                body = process_pdf(entry)
            elif suf == ".csv":
                body = process_csv(entry)
            elif suf in IMAGE_EXTENSIONS:
                body = process_image(entry)
            else:
                LOGGER.debug("Skipping unsupported file: %s", entry.name)
                continue
            write_markdown_output(entry, body)
        except Exception:
            LOGGER.exception("Unexpected error processing %s", entry)


if __name__ == "__main__":
    main()
