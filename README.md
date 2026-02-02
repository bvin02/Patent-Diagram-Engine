# Sketch to SVG Pipeline

Converts hand-drawn pencil sketches into clean, editable SVG line art.

### Dependencies

```bash
pip install opencv-python numpy
```

# Stage 1: Binarization

Transforms photos of pencil drawings into pure black-and-white images by separating lines from paper.

### Algorithm

1. **Denoising** - [FastNlMeansDenoising](https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html) removes paper texture and camera noise
2. **Adaptive Thresholding** - Gaussian-weighted local thresholding handles uneven lighting
3. **Morphological Cleanup** - Close/open operations remove artifacts and strengthen lines

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--block-size` | 21 | Neighborhood size for threshold calculation. Smaller = more detail, larger = smoother |
| `--c-value` | 10 | Threshold offset. Higher = more white (aggressive background removal) |
| `--denoise` | 10 | Noise reduction strength. 0 disables |
| `--morph` | 1 | Morphological kernel size. 0 disables |
| `--invert` | false | Output white lines on black |

### Usage

```bash
# Default conversion
python main.py sketch.png

# Compare multiple threshold settings
python main.py --compare sketch.png

# Custom parameters
python main.py sketch.png --block-size 15 --c-value 12 --denoise 10
```

Presets (via --compare)
- **soft** - Preserves gray tones, minimal cleanup
- **medium** - Balanced for typical pencil sketches
- **crisp** - Clean lines, aggressive noise removal
- **aggressive** - Maximum contrast, may lose fine detail


### Output

Input: `sketch.png` -> Output: `sketch_binary.png`

Pure 1-bit black/white PNG ready for vectorization.

# Stage 2: Vectorization
Transforms the seperated line and paper, black-and-white image into a vector line format.
```bash
[Convert pixel line/arc -> vector line/curve] + [Group connected paths together (components)]
```

**Possible approaches:**
- [Image Inpainting](https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html) techniques to generate line masks
  - for line in mask, line is classified into straight or arced
    - straight: endpoints identified; stroke normalized
    - arc: bezier points identified; stroke normalized
  - close masks (< 5-10px) are joined (connect edgepoints)

- Corner Detection techniques ([Harris](https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html), [Shi-Tomasi](https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html)) to detect all junctions/corners
  - straight line paths formed by joining corner points
  - if arc, bezier points identified using pixel approximations
