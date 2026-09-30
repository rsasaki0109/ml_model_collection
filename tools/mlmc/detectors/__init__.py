"""Detector implementations shared by several models.

A model's ``detector.py`` re-exports one of these when its ONNX graph has the
same interface as another model's; otherwise it implements its own.
"""
