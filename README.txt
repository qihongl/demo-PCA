PCA TEACHING EXAMPLES

Run: pip install numpy scipy scikit-learn matplotlib seaborn pillow
python generate_examples.py    # includes data-only and PC-arrow GIFs
python generate_examples.py --no-gifs   # PNG previews only

Seed: 20260930. Centered PCA; no standardization. All examples are synthetic.
The circle and sphere have nonlinear intrinsic dimension lower than their ambient dimension;
PCA estimates linear subspaces and does not promise a nonlinear coordinate chart.
A high explained-variance fraction does not ensure retention of task-relevant information.
GIFs: 400 frames, 50 ms/frame, 20 seconds/rotation, infinite loop; only camera changes.
PNGs: 300 DPI, 3.5 inches high; separate panels and legends.
Gallery: all stages are visible in one row per example. Static panels copy the original high-resolution PNG.
Animated panels copy the original GIF when the browser supports image/gif clipboard writes; otherwise the button offers Download GIF. No PNG substitution. Insert the downloaded GIF into Google Slides to preserve animation.
If the browser blocks clipboard access, use Open PNG and the native Copy image action or download.
Continuous colors identify the generating variable named in the PCA colorbar.
Group A=blue, Group B=orange in the low-variance example. Class labels never enter PCA.
Each gallery row shows data-only figures, PC arrows, and projections together; individual files remain separate for slide assembly.
The 3D circle includes both 1D (50% variance) and 2D (100% variance) PCA projections.
In 3D, PC1 is blue and PC2 is orange; both start at the mean.
PC1 arrows start at the mean and follow the fitted loading direction (blue, or gray on the class plot). Arrow lengths are enlarged for visibility, not a variance scale. Outlier is magenta.
One-dimensional score plots place every point exactly on the PC1 axis; identical scores overlap.
Each figure has a companion caption; results.json records means, components and variance.
PC signs are arbitrary. With equal eigenvalues, individual component directions are nonunique.

1. Elongated cloud (n=120)
PCA works well when observations lie near a straight line. The principal axis passes through the sample mean, even though the cloud is far from the origin.
Explained variance: 99.52%, 0.48%

2. Circle in 3D (n=100)
A circle in a tilted 3D plane has one intrinsic parameter (angle), but needs two linear coordinates. One PC retains 50% of variance and folds the circle onto a line. Two PCs retain 100% and recover the circle exactly. The first two eigenvalues are equal: any orthonormal pair in the circle plane is a valid PC1/PC2 basis.
Explained variance: 50.00%, 50.00%, 0.00%

3. Class signal in PC2 (n=72)
PCA maximizes variance, not class separation. Here the two classes have identical PC1 scores, so their separation is lost even though PC1 explains most variance. Classes are used only for coloring, not fitting PCA.
Explained variance: 97.21%, 2.79%

4. Outlier sensitivity (n=81)
Squared deviations give a distant outlier a large influence. Adding one off-axis point rotates PC1 away from the main horizontal cloud. The dashed gray arrow is PC1 fitted without the outlier.
Explained variance: 73.04%, 26.96%

5. Original units (n=100)
Changing the numerical units of one variable can change PCA. Both panels show the same points in original units; the second PC1 axis is mapped back from data where x1 was multiplied by 100. Scaling must reflect the scientific question, not be chosen automatically.
Explained variance: 89.66%, 10.34%

6. Tilted plane (n=220)
A nearly flat two-dimensional sheet can be recovered by two PCs, regardless of its orientation in 3D. The small third component mainly captures the added thickness.
Explained variance: 54.58%, 45.39%, 0.03%

7. Swiss roll (n=405)
This is an intrinsically two-dimensional sheet, but it is curved. Two PCs make a linear shadow rather than unrolling the sheet; separated parts of the sheet overlap in the projection. Color tracks position along the roll.
Explained variance: 50.25%, 27.65%, 22.09%

8. Spherical surface (n=320)
A sphere has a two-dimensional surface, but no single plane can represent it without losing information. Projection overlays the front and back. The three variance fractions are almost equal, so there is no clear low-variance direction to discard.
Explained variance: 33.35%, 33.33%, 33.32%
