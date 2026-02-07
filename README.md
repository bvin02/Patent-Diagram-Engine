# Sketch to SVG Pipeline

Converts hand-drawn pencil sketches into clean, editable SVG line art.

## Stage 1: Binarization

Transforms photos of pencil drawings into pure black-and-white images by separating lines from paper.

### Algorithm

1. **Denoising** - FastNlMeansDenoising removes paper texture and camera noise
2. **Adaptive Thresholding** - Gaussian-weighted local thresholding handles uneven lighting
3. **Morphological Cleanup** - Close/open operations remove artifacts and strengthen lines

### Usage

```bash
# Default conversion
python binarization.py sketch.png

# Compare multiple threshold settings
python binarization.py --compare sketch.png

# Custom parameters
python binarization.py sketch.png --block-size 15 --c-value 12 --denoise 10
```

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--block-size` | 21 | Neighborhood size for threshold calculation. Smaller = more detail, larger = smoother |
| `--c-value` | 10 | Threshold offset. Higher = more white (aggressive background removal) |
| `--denoise` | 10 | Noise reduction strength. 0 disables |
| `--morph` | 1 | Morphological kernel size. 0 disables |
| `--invert` | false | Output white lines on black |

### Presets (via --compare)

- **soft** - Preserves gray tones, minimal cleanup
- **medium** - Balanced for typical pencil sketches
- **crisp** - Clean lines, aggressive noise removal
- **aggressive** - Maximum contrast, may lose fine detail

### Dependencies

```bash
pip install opencv-python numpy
```

### Output

Input: `sketch.png` -> Output: `sketch_binary.png`

Pure 1-bit black/white PNG ready for vectorization.
