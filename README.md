# PCA teaching demos

Eight synthetic examples for teaching what principal component analysis (PCA) preserves and what it can miss. The gallery displays each example in stages, side by side: observations without PC arrows, fitted PC directions, and lower-dimensional projections.

## View and use

Open `gallery.html` in a browser after downloading or cloning the repository. No build step or server is required. On narrow screens, scroll each row horizontally.

Each figure is a separate file for lecture-slide assembly. Static images are exported at 300 DPI. Each 3D GIF rotates smoothly once every 20 seconds at 20 frames per second, with and without PC1/PC2 arrows.

- Static panels have a high-resolution PNG copy button.
- Animated panels offer original GIF copying when the browser supports it; otherwise the button offers a GIF download. No PNG is substituted for the animation.
- To preserve animation in Google Slides, download the GIF and use **Insert → Image → Upload from computer**.
- Browser clipboard permissions and the destination application can affect pasting. Direct animated-GIF paste into Google Slides has not been verified.

## Examples

| Example | Teaching point |
| --- | --- |
| Elongated cloud | Centered PCA finds a useful one-dimensional direction, even away from the origin. |
| Circle in 3D | One PC collapses the circle to a line (50% variance); two PCs recover the circle (100%). |
| Class signal in PC2 | High explained variance does not guarantee retention of class information. |
| Outlier sensitivity | One distant observation can rotate the leading component. Both axes use equal scales. |
| Measurement units | Changing a feature's units changes covariance PCA. |
| Tilted plane | Two PCs recover a nearly planar structure embedded in 3D. |
| Swiss roll | A linear projection does not unroll a curved surface. |
| Spherical surface | Two PCs superimpose different parts of the surface. |

The data are synthetic, not neuroscience measurements. PCA is mean-centered without feature standardization. One-dimensional projections have no vertical jitter. PC arrows start at the mean; their lengths are enlarged for visibility and do not encode explained variance. Component signs are arbitrary. For the exact circle, the first two eigenvalues are equal, so any orthonormal basis in its plane is valid.

## Reproduce

Create a Python virtual environment, then run:

```sh
python -m pip install -r requirements-lock.txt
python generate_examples.py
```

To regenerate PNGs and the gallery while retaining the included GIFs:

```sh
python generate_examples.py --no-gifs
```

The random seed is `20260930`. The included dependency list records the generation environment. The plotting-style helper is included; no personal skill installation is needed.

## Contents

- `generate_examples.py`: synthetic data, PCA fits, plots, GIFs, captions, and gallery generation.
- `gallery.html`: ready-to-open review gallery.
- `gallery_template.html`, `gallery_ui.js`: gallery layout and clipboard behavior.
- `figures/`: individual PNGs, GIFs, detached legends, and explanatory captions.
- `data/`: source observations, PC scores, and color values in CSV format.
- `results.json`: PCA means, components, variance fractions, and teaching notes. For the circle, the gallery shows both one- and two-PC projections.
- `clipboard-assets/`: generated GIF payload scripts, loaded only when copying a GIF. These duplicate GIF bytes to support local-file galleries without local-file fetch restrictions.
- `style/`: portable plotting-style snapshot used by the generator.
- `README.txt`: generated technical and teaching notes.
