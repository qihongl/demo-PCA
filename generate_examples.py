"""Reproducible synthetic PCA lecture figures. Run with --no-gifs for quick previews.
Install: pip install numpy scipy scikit-learn matplotlib seaborn pillow
"""
import argparse
import base64
from html import escape
import struct
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'style' / 'scripts'))
from plot_style import plot_context, save_panel

OUT = ROOT / 'figures'
DATA = ROOT / 'data'
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260930)
BLUE, ORANGE, MAGENTA = '#0072B2', '#D55E00', '#CC79A7'
UNCERTAINTY = 'Raw synthetic observations; no inferential confidence intervals apply.'
SPECS = []


def add(key, title, X, keep, lesson, kind='black', values=None, color_label='', **extra):
    model = PCA().fit(X)
    Z = model.transform(X)
    ratios = model.explained_variance_ratio_
    s = dict(key=key, title=title, X=X, model=model, Z=Z, keep=keep, lesson=lesson,
             kind=kind, values=values, color_label=color_label, ratios=ratios, **extra)
    SPECS.append(s)
    return s


def build_data():
    # 1: a translated, elongated cloud; standard PCA is centered at its mean.
    angle = np.deg2rad(35)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    latent = RNG.normal(size=(120, 2)) * [2.5, .20]
    X = latent @ rot.T + [6, 4]
    add('elongated_cloud', 'Elongated cloud', X, 1,
        'PCA works well when observations lie near a straight line. The principal axis passes through the sample mean, even though the cloud is far from the origin.')

    # 2: an exact circle in an oblique 3D plane, with two equal-variance PCs.
    theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
    normal = np.array([1., -1., 1.]); normal /= np.linalg.norm(normal)
    u = np.array([1., 1., 0.]); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    X = 2.5 * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    add('circle', 'Circle in 3D', X, 1,
        'A circle in a tilted 3D plane has one intrinsic parameter (angle), but needs two linear coordinates. One PC retains 50% of variance and folds the circle onto a line. Two PCs retain 100% and recover the circle exactly. The first two eigenvalues are equal: any orthonormal pair in the circle plane is a valid PC1/PC2 basis.',
        kind='cyclic', values=theta, color_label='Angle (radians)')

    # 3: identical horizontal variability in two classes; class identity is PC2.
    t = np.linspace(-4.5, 4.5, 36)
    small = RNG.normal(0, .055, len(t))
    small -= small.mean()
    small -= t * (t @ small) / (t @ t)  # exactly remove horizontal covariance
    X = np.vstack((np.column_stack((t, small + .45)), np.column_stack((t, small - .45))))
    labels = np.repeat([0, 1], len(t))
    add('low_variance_signal', 'Class signal in PC2', X, 1,
        'PCA maximizes variance, not class separation. Here the two classes have identical PC1 scores, so their separation is lost even though PC1 explains most variance. Classes are used only for coloring, not fitting PCA.',
        kind='classes', values=labels)

    # 4: one off-axis observation dominates the covariance matrix.
    t = RNG.normal(0, 2, 80)
    y = RNG.normal(0, .15, 80)
    inliers = np.column_stack((t, y))
    inliers -= inliers.mean(axis=0)
    X = np.vstack((inliers, [[0., 28.]]))
    add('outlier', 'Outlier sensitivity', X, 1,
        'Squared deviations give a distant outlier a large influence. Adding one off-axis point rotates PC1 away from the main horizontal cloud. The dashed gray arrow is PC1 fitted without the outlier.',
        kind='outlier', values=np.r_[np.zeros(80), 1], clean_model=PCA().fit(inliers))

    # 5: the same physical observations; change the units of x1 by a factor of 100.
    X = RNG.normal(size=(100, 2)) * [.55, 1.8]
    unit = add('measurement_units', 'Original units', X, 1,
        'Changing the numerical units of one variable can change PCA. Both panels show the same points in original units; the second PC1 axis is mapped back from data where x1 was multiplied by 100. Scaling must reflect the scientific question, not be chosen automatically.')
    unit['scaled_model'] = PCA().fit(X * [100, 1])

    # 6: a two-dimensional sheet embedded obliquely in three dimensions.
    normal = np.array([1., -1., 1.]); normal /= np.linalg.norm(normal)
    u = np.array([1., 1., 0.]); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    uv = RNG.uniform(-3, 3, (220, 2))
    X = uv[:, :1] * u + uv[:, 1:] * v + RNG.normal(0, .045, (220, 1)) * normal
    add('tilted_plane', 'Tilted plane', X, 2,
        'A nearly flat two-dimensional sheet can be recovered by two PCs, regardless of its orientation in 3D. The small third component mainly captures the added thickness.',
        kind='continuous', values=uv[:, 0], color_label='Position along sheet')

    # 7: a sheet rolled into a nonlinear manifold. Height variance ensures that
    # one retained direction tracks height, leaving only one axis for the curl.
    t, h = np.meshgrid(np.linspace(1.5*np.pi, 4.5*np.pi, 45), np.linspace(-15, 15, 9))
    t, h = t.ravel(), h.ravel()
    X = np.column_stack((t*np.cos(t), t*np.sin(t), h))
    X += RNG.normal(0, .065, X.shape)
    add('swiss_roll', 'Swiss roll', X, 2,
        'This is an intrinsically two-dimensional sheet, but it is curved. Two PCs make a linear shadow rather than unrolling the sheet; separated parts of the sheet overlap in the projection. Color tracks position along the roll.',
        kind='continuous', values=t, color_label='Position along roll')

    # 8: a spherical surface, sampled approximately uniformly.
    n = 320
    z = 1 - 2*(np.arange(n)+.5)/n
    phi = np.arange(n) * np.pi * (3-np.sqrt(5))
    r = np.sqrt(1-z*z)
    X = 3*np.column_stack((r*np.cos(phi), r*np.sin(phi), z))
    add('sphere', 'Spherical surface', X, 2,
        'A sphere has a two-dimensional surface, but no single plane can represent it without losing information. Projection overlays the front and back. The three variance fractions are almost equal, so there is no clear low-variance direction to discard.',
        kind='continuous', values=X[:, 2], color_label='Original height')


def rgba(s):
    if s['kind'] == 'classes':
        return np.array([BLUE if x == 0 else ORANGE for x in s['values']])
    if s['kind'] == 'outlier':
        return np.array(['#333333' if x == 0 else MAGENTA for x in s['values']])
    if s['kind'] in ['continuous', 'cyclic']:
        norm = Normalize(0, 2*np.pi) if s['kind'] == 'cyclic' else Normalize(np.min(s['values']), np.max(s['values']))
        cmap = plt.get_cmap('twilight' if s['kind'] == 'cyclic' else 'viridis')
        return cmap(norm(s['values']))
    return '#333333'


def colorbar(f, ax, s):
    if s['kind'] not in ['continuous', 'cyclic']:
        return
    norm = Normalize(0, 2*np.pi) if s['kind'] == 'cyclic' else Normalize(np.min(s['values']), np.max(s['values']))
    m = plt.cm.ScalarMappable(norm=norm, cmap='twilight' if s['kind'] == 'cyclic' else 'viridis')
    cb = f.colorbar(m, ax=ax, fraction=.045, pad=.08, shrink=.75)
    cb.set_label(s['color_label'], fontsize=12)
    cb.ax.tick_params(labelsize=10)
    cb.locator = MaxNLocator(3); cb.update_ticks()


def style2d(ax, X):
    span = max(np.ptp(X, axis=0))
    for j, setter in enumerate([ax.set_xlim, ax.set_ylim]):
        mid = (X[:, j].min()+X[:, j].max())/2
        setter(mid-.64*span, mid+.64*span)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')


def pc_arrow(ax, center, direction, length, color='#0072B2', linestyle='-', label='PC1'):
    # Preserve the fitted loading direction and sign; enlarge its unit length
    # for legibility in data coordinates, as explained in the captions.
    direction = direction / np.linalg.norm(direction)
    tip = center + length * direction
    arrow = FancyArrowPatch(center, tip, arrowstyle='-|>', mutation_scale=18,
                            shrinkA=0, shrinkB=0, color=color, linewidth=2,
                            linestyle=linestyle, zorder=4, label=label)
    ax.add_patch(arrow)
    return arrow


def legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return FancyArrowPatch((-xdescent, height/2-ydescent),
                           (width-xdescent, height/2-ydescent),
                           arrowstyle='-|>', mutation_scale=fontsize,
                           shrinkA=0, shrinkB=0)


def caption(s, what):
    ratios = ', '.join(f'{x:.2%}' for x in s['ratios'])
    details = ''
    if s['kind']=='classes': details = 'Group A and B each have 36 observations; shared legend is low_variance_signal_original-legend.png. '
    if s['kind']=='outlier': details = 'Main cloud n=80, outlier n=1; shared legend is outlier_original-legend.png. '
    return (f"{s['title']}. {what}. n={len(s['X'])} synthetic observations; feature units arbitrary except for the explicitly demonstrated unit conversion. "
            + details + f"Centered covariance PCA; no variance standardization. Explained-variance fractions: {ratios}. {s['lesson']}")


def save(f, s, suffix, what, **kw):
    save_panel(f, OUT / f"{s['key']}_{suffix}.png", caption=caption(s, what), uncertainty=UNCERTAINTY,
               legend_kwargs={'handler_map': {FancyArrowPatch: HandlerPatch(patch_func=legend_arrow)}}, **kw)
    plt.close(f)


def raw2d(s, show_pcs=True):
    f, ax = plt.subplots(figsize=(4.3, 3.5))
    X = s['X']
    if s['kind'] in ['classes', 'outlier']:
        for value in [0, 1]:
            selected = s['values'] == value
            label = (['Group A', 'Group B'] if s['kind']=='classes' else ['Main cloud', 'Outlier'])[value]
            ax.scatter(*X[selected].T, c=rgba(s)[selected], s=22, alpha=.85, linewidths=0, zorder=2, label=label)
    else:
        ax.scatter(*X.T, c=rgba(s), s=22, alpha=.85, linewidths=0, zorder=2)
    if show_pcs:
        length = max(np.ptp(X, axis=0)) * .52
        pc_arrow(ax, s['model'].mean_, s['model'].components_[0], length,
                color='#555555' if s['kind'] == 'classes' else BLUE,
                label='PC1 (all points)' if s['kind'] == 'outlier' else 'PC1')
        if s['kind'] == 'outlier':
            pc_arrow(ax, s['clean_model'].mean_, s['clean_model'].components_[0], np.ptp(X[:-1, 0])*.48,
                    color='#888888', linestyle='--', label='PC1 (without outlier)')
    title = s['title'] if show_pcs else question_title(s)
    ax.set_title(title, fontsize=17, pad=10)
    style2d(ax, X)
    colorbar(f, ax, s)
    extra = 'Equal scale per unit and equal numerical ranges on both axes.'
    what = ('Original coordinates with an arrow from the sample mean along the fitted PC1 loading vector. Arrow length is enlarged for visibility and does not encode variance; the component sign is arbitrary. '
            if show_pcs else 'Original observations without principal-component arrows, for a prediction exercise. Points, colors, and axis limits match the version with arrows. ')
    save(f, s, 'original' if show_pcs else 'data', what+extra)


def projection(s, keep=None, suffix='pca'):
    keep = s['keep'] if keep is None else keep
    f, ax = plt.subplots(figsize=(4.3, 3.5))
    Z = s['Z']
    if keep == 1:
        # A one-dimensional representation: every point lies exactly on PC1.
        ax.scatter(Z[:, 0], np.zeros(len(Z)), c=rgba(s), s=25, alpha=.8,
                   linewidths=0, zorder=3)
        ax.set_ylim(-.4, .4); ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['bottom'].set_color('#777777')
        ax.set_xlabel('PC1 score'); ax.set_ylabel('')
        what = 'One-PC scores on a single horizontal axis. All vertical coordinates are exactly zero; observations with identical scores overlap.'
    else:
        ax.scatter(Z[:, 0], Z[:, 1], c=rgba(s), s=14, alpha=.9, linewidths=0)
        style2d(ax, Z[:, :2]); ax.set_xlabel('PC1 score'); ax.set_ylabel('PC2 score')
        what = 'Projection onto the first two principal components; the same observations and colors as in the 3D view.'
    ax.xaxis.set_major_locator(MaxNLocator(3))
    kept = np.sum(s['ratios'][:keep])
    ax.set_title(f"{keep}D PCA: {kept:.2%} variance", fontsize=17, pad=10)
    colorbar(f, ax, s)
    save(f, s, suffix, what, legend=False)


def units_after(s, show_pcs=True):
    f, ax = plt.subplots(figsize=(4.3, 3.5))
    X, model = s['X'], s['scaled_model']
    ax.scatter(*X.T, c='#333333', s=22, alpha=.85, linewidths=0, zorder=2)
    direction = model.components_[0] / [100, 1]
    if show_pcs:
        pc_arrow(ax, X.mean(0), direction, np.max(np.ptp(X, axis=0))*.52, label='PC1 after unit change')
    style2d(ax, X)
    ax.set_title('Feature 1 × 100', fontsize=17, pad=10)
    if not show_pcs:
        save(f, s, 'rescaled_data', 'Observations without a PC arrow. Displayed coordinates remain in original units, matching the solution panel; PCA will be fitted after multiplying feature 1 by 100.', legend=False)
        return
    save(f, s, 'rescaled', 'Same points shown in original units; the arrow starts at the mean and follows PC1 fitted in converted units then mapped back for comparison. Arrow length is enlarged for visibility and does not encode variance; the component sign is arbitrary. Converted-data PC1 variance fraction: '+f"{model.explained_variance_ratio_[0]:.2%}")


def question_title(s):
    return 'Two classes' if s['key'] == 'low_variance_signal' else s['title']


def make3d(s, show_pcs=False):
    f = plt.figure(figsize=(4.3, 3.5))
    ax = f.add_subplot(111, projection='3d')
    X = s['X']
    ax.scatter(*X.T, c=rgba(s), s=13, alpha=.95, linewidths=0, depthshade=False)
    spans = np.ptp(X, axis=0)
    if show_pcs:
        center = s['model'].mean_
        for i, color in enumerate([BLUE, ORANGE]):
            vector = s['model'].components_[i] * max(spans) * .38
            ax.quiver(*center, *vector, color=color, linewidth=2,
                      arrow_length_ratio=.18, normalize=False)
            tip = center + 1.10 * vector
            ax.text(*tip, f'PC{i+1}', color=color, fontsize=12)
    for j, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        setter(X[:,j].min()-.10*spans[j], X[:,j].max()+.10*spans[j])
    ax.set_box_aspect(spans)
    ax.set_xlabel('X', labelpad=-10, fontsize=16); ax.set_ylabel('Y', labelpad=-10, fontsize=16); ax.set_zlabel('Z', labelpad=-10, fontsize=16)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticks([])
        axis.line.set_linewidth(1.2)
        axis.pane.fill = False
        axis.pane.set_edgecolor('white')
    ax.grid(False)
    ax.set_title(s['title'], fontsize=17, pad=4)
    ax.view_init(elev=24, azim=30)
    f.tight_layout(pad=.7)
    return f, ax


def rotation(s, skip, show_pcs=False):
    f, ax = make3d(s, show_pcs)
    stem = s['key'] + ('_components' if show_pcs else '')
    note = ('PC1 (blue) and PC2 (orange) arrows start at the mean; lengths are enlarged for visibility, not a variance scale. Signs are arbitrary. ' if show_pcs else 'Data only, without PC arrows, for a prediction exercise. ')
    save_panel(f, OUT / f"{stem}_poster.png", caption=caption(s, note+'Static preview of the 3D data; colors match the PCA projections.'), uncertainty=UNCERTAINTY, legend=False)
    if not skip:
        # 400 frames at 20 fps: a 20-second, constant-speed seamless revolution.
        # Quantize every frame with one shared palette to avoid palette flicker.
        f.set_dpi(160)
        frames = []
        palette = None
        for i in range(400):
            ax.view_init(elev=24, azim=30 + i*360/400)
            f.canvas.draw()
            rgb = Image.fromarray(np.asarray(f.canvas.buffer_rgba())[:,:,:3].copy())
            if palette is None:
                palette = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            frames.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
        frames[0].save(OUT / f"{stem}_rotation.gif", save_all=True,
                       append_images=frames[1:], duration=50, loop=0, optimize=False, disposal=2)
        (OUT / f"{stem}_rotation.caption.md").write_text(
            caption(s, note+'Fixed dataset; only the camera rotates, 360 degrees in 20 seconds at 20 fps. Constant camera elevation and axis limits. Color matches the PCA panel.')+'\n\nUncertainty: '+UNCERTAINTY+'\n')
        print('GIF finished:', stem, flush=True)
    plt.close(f)


def exports(skip):
    rows = []
    for s in SPECS:
        X, Z = s['X'], s['Z']
        cols = [f'x{i+1}' for i in range(X.shape[1])] + [f'pc{i+1}' for i in range(X.shape[1])]
        arr = np.column_stack((X, Z))
        if s['values'] is not None:
            cols.append('color_value'); arr = np.column_stack((arr, s['values']))
        np.savetxt(DATA / f"{s['key']}.csv", arr, delimiter=',', header=','.join(cols), comments='')
        if s['key'] == 'measurement_units':
            np.savetxt(DATA / 'measurement_units_converted.csv', X*[100, 1], delimiter=',', header='x1_times_100,x2', comments='')
        rows.append(dict(example=s['key'], title=s['title'], n=len(X), dimensions=X.shape[1], retained=s['keep'],
                         variance_fractions=s['ratios'].tolist(), mean=s['model'].mean_.tolist(),
                         components=s['model'].components_.tolist(), lesson=s['lesson']))
        if X.shape[1] == 2:
            raw2d(s, show_pcs=False)
            raw2d(s)
            if s['key']=='measurement_units':
                units_after(s, show_pcs=False)
                units_after(s)
            else: projection(s)
        else:
            rotation(s, skip)
            rotation(s, skip, show_pcs=True)
            projection(s)
            if s['key']=='circle': projection(s, keep=2, suffix='pca_2d')
        print('Completed:', s['key'], flush=True)
    (ROOT / 'results.json').write_text(json.dumps(rows, indent=2))
    make_gallery(rows)


def make_gallery(rows):
    cards = []
    assets = {}

    def panel(display_file, png_file, label, title, animation=False):
        name = display_file if animation else png_file
        raw = (OUT / name).read_bytes()
        if animation:
            if raw[:6] not in [b'GIF87a', b'GIF89a']:
                raise ValueError(f'Not a GIF: {name}')
            width, height = struct.unpack('<HH', raw[6:10])
            # Load GIF clipboard bytes only if requested. Local script assets also
            # work for file:// galleries without fetching local files via CORS.
            asset_dir = ROOT / 'clipboard-assets'
            asset_dir.mkdir(exist_ok=True)
            script_name = name + '.js'
            encoded = base64.b64encode(raw).decode('ascii')
            (asset_dir / script_name).write_text('window.registerGalleryGif(' + json.dumps(name) + ',' + json.dumps(encoded) + ');')
            assets[name] = dict(mime='image/gif', script='clipboard-assets/'+script_name, width=width, height=height)
        else:
            if raw[:8] != b'\x89PNG\r\n\x1a\n':
                raise ValueError(f'Not a PNG: {name}')
            width, height = struct.unpack('>II', raw[16:24])
            assets[name] = dict(mime='image/png', base64=base64.b64encode(raw).decode('ascii'), width=width, height=height)
        status_id = 'status-' + name.replace('.', '-')
        button_label = 'Copy GIF' if animation else 'Copy hi-res PNG'
        file_type = 'GIF' if animation else 'PNG'
        still_link = f'<a href="figures/{escape(png_file)}" download>Download hi-res still</a>' if animation else ''
        return (f'<figure><img src="figures/{escape(display_file)}" alt="{escape(title)} — {escape(label)}">'
                f'<figcaption>{escape(label)}<span class="resolution">{width} × {height} {file_type}</span></figcaption>'
                f'<div class="actions"><button type="button" data-copy-image="{escape(name)}" '
                f'aria-label="{button_label}: {escape(title)} — {escape(label)}" aria-describedby="{status_id}">{button_label}</button>'
                f'<a href="figures/{escape(name)}" target="_blank" rel="noopener">Open {file_type}</a>'
                f'<a href="figures/{escape(name)}" download>Download {file_type}</a>{still_link}</div>'
                f'<p id="{status_id}" class="copy-status" role="status" aria-live="polite"></p></figure>')

    for r in rows:
        key, title = r['example'], r['title']
        is3d = r['dimensions'] == 3
        if is3d:
            question = panel(f'{key}_rotation.gif', f'{key}_poster.png', 'Data only · guess PC1 and PC2', title, True)
            answer = panel(f'{key}_components_rotation.gif', f'{key}_components_poster.png', 'Fitted PC1 and PC2 · slow rotation', title, True)
        else:
            question = panel(f'{key}_data.png', f'{key}_data.png', 'Data only · guess PC1', title)
            answer = panel(f'{key}_original.png', f'{key}_original.png', 'Original data + fitted PC1', title)
        if key == 'measurement_units':
            question += panel(f'{key}_rescaled_data.png', f'{key}_rescaled_data.png', 'Guess PC1 after changing units · displayed in original coordinates', title)
            answer += panel(f'{key}_rescaled.png', f'{key}_rescaled.png', 'PC1 after changing units', title)
        elif key != 'circle':
            answer += panel(f'{key}_pca.png', f'{key}_pca.png', 'PCA projection', title)
        extra = ''
        if key == 'circle':
            extra = panel('circle_pca.png', 'circle_pca.png', '1D PCA · 50% variance · circle collapses to a line', title) + panel('circle_pca_2d.png', 'circle_pca_2d.png', '2D PCA · 100% variance · circle preserved', title)
        cards.append(f'<section><h2>{escape(title)}</h2><p>{escape(r["lesson"])}</p><div class="pair">'
                     + question + answer + extra + '</div>'
                     + f'<p class="small">{r["n"]} synthetic points · <a href="data/{key}.csv">Data CSV</a></p></section>')
    html = (ROOT / 'gallery_template.html').read_text()
    html = html.replace('__CARDS__', ''.join(cards))
    html = html.replace('__IMAGE_ASSETS__', json.dumps(assets, separators=(',', ':')))
    html = html.replace('__GALLERY_SCRIPT__', (ROOT / 'gallery_ui.js').read_text())
    (ROOT / 'gallery.html').write_text(html)
    text=['PCA TEACHING EXAMPLES','', 'Run: pip install numpy scipy scikit-learn matplotlib seaborn pillow',
          'python generate_examples.py    # includes data-only and PC-arrow GIFs', 'python generate_examples.py --no-gifs   # PNG previews only','',
          'Seed: 20260930. Centered PCA; no standardization. All examples are synthetic.',
          'The circle and sphere have nonlinear intrinsic dimension lower than their ambient dimension;',
          'PCA estimates linear subspaces and does not promise a nonlinear coordinate chart.',
          'A high explained-variance fraction does not ensure retention of task-relevant information.',
          'GIFs: 400 frames, 50 ms/frame, 20 seconds/rotation, infinite loop; only camera changes.',
          'PNGs: 300 DPI, 3.5 inches high; separate panels and legends.',
          'Gallery: all stages are visible in one row per example. Static panels copy the original high-resolution PNG.',
          'Animated panels copy the original GIF when the browser supports image/gif clipboard writes; otherwise the button offers Download GIF. No PNG substitution. Insert the downloaded GIF into Google Slides to preserve animation.',
          'If the browser blocks clipboard access, use Open PNG and the native Copy image action or download.',
          'Continuous colors identify the generating variable named in the PCA colorbar.',
          'Group A=blue, Group B=orange in the low-variance example. Class labels never enter PCA.',
          'Each gallery row shows data-only figures, PC arrows, and projections together; individual files remain separate for slide assembly.',
          'The 3D circle includes both 1D (50% variance) and 2D (100% variance) PCA projections.',
          'In 3D, PC1 is blue and PC2 is orange; both start at the mean.',
          'PC1 arrows start at the mean and follow the fitted loading direction (blue, or gray on the class plot). Arrow lengths are enlarged for visibility, not a variance scale. Outlier is magenta.',
          'One-dimensional score plots place every point exactly on the PC1 axis; identical scores overlap.',
          'Each figure has a companion caption; results.json records means, components and variance.',
          'PC signs are arbitrary. With equal eigenvalues, individual component directions are nonunique.','']
    for i,r in enumerate(rows,1):
        text += [f'{i}. {r["title"]} (n={r["n"]})',r['lesson'],
                 'Explained variance: '+', '.join(f'{v:.2%}' for v in r['variance_fractions']),'']
    (ROOT/'README.txt').write_text('\n'.join(text))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-gifs',action='store_true')
    args=parser.parse_args()
    build_data()
    with plot_context(): exports(args.no_gifs)
