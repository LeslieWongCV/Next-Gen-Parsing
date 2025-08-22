# utils.py
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
import os
import json
from datetime import datetime

def pdf_to_images(pdf_path):
    """Convert PDF to list of images (one per page)"""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(np.array(img))
    return images

import hashlib

def files_are_identical_hash(file1, file2):
    """
    hash
    """
    def get_hash(filepath):
        hash_sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    return get_hash(file1) == get_hash(file2)

import os

def files_are_identical_fast(file1, file2):
    """
    compare size and hash
    """
    if os.path.getsize(file1) != os.path.getsize(file2):
        return False

    return files_are_identical(file1, file2)  