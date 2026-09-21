"""Synthetic document corpus generation with OCR noise and layout variants."""

from .corpus import Document, FieldSpan, generate_corpus
from .ocr_noise import apply_ocr_noise
from .layouts import render_layout

__all__ = ["Document", "FieldSpan", "generate_corpus", "apply_ocr_noise", "render_layout"]
