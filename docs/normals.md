# Normal orientation and checking it in ParaView

> **Experimental — for expert users.** The orientation heuristics have not been validated
> across datasets yet, and the automatic decision is wrong for some components by
> construction (see below). Always inspect the result before using oriented surfaces for
> analysis, and expect the interface to change.

`morphometrics flip_normals` orients each connected component so its normals point
away from the estimated inside of the organelle. Because the sign of every curvature
measurement follows the normal, this is what makes curvature comparable **between**
tomograms: without it, whether a bulge reads as positively or negatively curved is an
accident of how the mesh happened to be generated.

Because you cannot know which components the automatic rule gets wrong until you have
looked at the result, the workflow is **orient → inspect → correct**:

```bash
morphometrics flip_normals config.yml                             # 1. orient everything
morphometrics flip_normals config.yml --graph TE1_IMM.AVV_rh9.gt --dry-run   # (preview)
# 2. inspect TE1_IMM.AVV_rh9_oriented.vtp in ParaView (below)
morphometrics manual_flip TE1_IMM.AVV_rh9_oriented.gt --labels 3,7  # 3. correct
```

Run it after `pycurv` (or after `accept_refinement`, if you refine). It labels connected
components itself, so it does not need `label_components` first. Outputs are written
alongside the input as `<base>_oriented.gt` / `.vtp` / `.csv`, carrying two extra
per-triangle properties: `component_number` and `normals_flipped` (1 where flipped).

## What gets flipped

Flipping a normal is not just negating it — everything that depends on the surface
having a chosen side has to move with it. The principal curvatures **swap as well as
negate**, because pycurv keeps `kappa_1` as the maximum and `kappa_2` as the minimum:

| Property | Under a flip |
|---|---|
| `kappa_1`, `kappa_2` | swap and negate: κ1′ = −κ2, κ2′ = −κ1 |
| `min_curvature`, `max_curvature` | swap and negate |
| `mean_curvature_VV`, `mean_curvature` | negate |
| `shape_index_VV`, `shape_index_cat` | negate |
| `n_v`, `normal`, `avg_normals` | negate |
| `t_1`, `t_2` | swap (each stays paired with its curvature) |
| `gauss_curvature_VV`, `gauss_curvature` | **unchanged** — a product of two negated values |
| `curvedness_VV` | **unchanged** — depends only on magnitudes |
| `area`, `xyz`, `thickness`, distances | unchanged |

The swap is what preserves `kappa_1 >= kappa_2`; negating alone would invert that
invariant everywhere. Gaussian curvature and curvedness are even in the normal and must
*not* be negated.

## The heuristic, and where it fails

Each component is flipped so its **area-weighted mean `mean_curvature_VV` is positive**.
For a closed-ish compartment — OMM, the IMM boundary, an ER sheet — that correctly puts
the normals on the outside.

It is wrong for **disconnected tubular cristae**, and the distinction matters: a crista
still attached to the IMM is part of that component and simply inherits the parent's
orientation, which is correct. An isolated crista fragment becomes its own component and
is oriented by its own tube geometry, which points it the *opposite* way from the
connected ones. So the components to watch are the small, disconnected, tubular ones —
not cristae in general.

The command cannot tell these apart from geometry alone, which is why you inspect the
result before relying on it.

## Two criteria, and why disagreement is the useful signal

There are two independent ways to ask which way is out, and the command reports both:

- **`--criterion curvature`** (default) — flip until area-weighted `mean_curvature_VV` is
  positive. Purely local, so it works on open sheets and fragments, and needs no
  assumption about the component being closed. It is uninformative when a component's
  curvature cancels out.
- **`--criterion centroid`** — flip until normals point away from the component's centre
  of mass. This assumes the component is *star-shaped* (every point visible from the
  centroid). Decisive for closed blobs; meaningless for an open sheet, where the centroid
  lies on the surface and every normal is perpendicular to the radial direction.

They **systematically disagree on tubular geometry**. Measured on a real cristae-rich
IMM, they disagreed on 24 of 28 components (on a closed OMM, 2 of 5) — and not randomly:
every component with `mean_H > 0` had `centroid < 0`. For a crista tube, normals pointing
out of the lumen give a *positive* centroid alignment but a *negative* mean curvature, so
the two rules point them opposite ways. Which you want is a biology question — note that
a crista lumen is continuous with the intermembrane space, so "away from the matrix"
means *into* the crista lumen.

Because their failure modes are independent, **disagreement is a better flag than either
confidence alone**. Each component is reported with both, plus a `confidence` — the ratio
of net mean curvature to curvedness, i.e. how much of the available curvature signal
actually points one way:

```
  id  triangles     mean_H   conf  centroid   decision
   1      39526   +0.00179   0.27    +0.360   -> kept
   2      22154   +0.00048   0.09    +0.168   -> kept
   3      21769   -0.00370   0.59    +0.757 ? -> FLIP
   4      11564   -0.00056   0.12    -0.170   -> FLIP
   5       4984   -0.00530   0.45    +0.603 ? -> FLIP
```

A component is marked `?` when its confidence is below 0.05 (curvature cancels out, so
the decision is near-arbitrary) **or** when the two criteria disagree. Those are listed in
a note at the end. **Check them first** — component 3 above has decisive centroid
alignment (+0.76) pointing the opposite way from the curvature rule, which is exactly the
case worth a look.

## Checking normal direction in ParaView

Open `<base>_oriented.vtp`. Two views are useful; the glyph view answers "which way do
they point", the scalar view answers "which component is that".

### 1. Glyphs — see the actual direction

1. Select the surface in the Pipeline Browser, then **Filters → Common → Glyph**.
2. In Properties set:
   - **Glyph Type**: `Arrow`
   - **Orientation Array**: `n_v` (the voted normal; `normal` is the raw mesh normal)
   - **Scale Array**: `No scale array`
   - **Scale Factor**: start around 5 (in nm — a few times your triangle size) and adjust
   - **Masking → Glyph Mode**: `Every Nth Point`, **Stride** ~50 for a large surface,
     otherwise the arrows are an unreadable thicket
3. Click **Apply**, then color the glyphs by `component_number`.
4. Clip through the organelle (**Filters → Common → Clip**) so you are looking at a cross
   section. Arrows should point *out* of the compartment — away from the matrix for the
   IMM boundary, away from the cytosol-facing side for the OMM.

Wherever the arrows point *into* a lumen, note that component's `component_number`.

### 2. Scalar colouring — a fast first pass

Colour the surface by `mean_curvature_VV` with a diverging colormap (**Coolwarm**) and
rescale symmetrically about zero (**Rescale to Custom Range**, e.g. −0.05 to 0.05). After
a successful run most of each component reads warm/positive. A component that is
predominantly cool/negative was either excluded or is one of the ambiguous ones.

Colouring by `normals_flipped` (0/1) shows exactly which triangles the command changed,
and by `component_number` shows the labelling the ids refer to.

> To compare against the original, load the pre-flip `.vtp` alongside it. `curvedness_VV`
> and `gauss_curvature_VV` should look identical in both — if they do not, something other
> than orientation changed.

## Correcting components: `manual_flip`

You cannot know which components need correcting until you have seen the result, so the
correction step works on the file you just inspected:

```bash
morphometrics manual_flip TE1_IMM.AVV_rh9_oriented.gt --labels 3,7
```

This flips components 3 and 7 outright, **in either direction** — it fixes a component
the automatic pass flipped wrongly *and* one it should have flipped but didn't. The ids
come from the `component_number` property stored in the file, so they are exactly the
ids reported by `flip_normals` and the ones you coloured by in ParaView; nothing is
re-labelled and nothing can renumber underneath you.

It overwrites `.gt`/`.vtp`/`.csv` in place (pass `--output-dir` to write elsewhere), so
you can re-open the same file in ParaView and iterate. Flipping is a **toggle**: running
the same command twice returns the component exactly to where it started (verified
bit-exact across every property).

### `--exclude-labels`, and when to use it instead

`flip_normals --exclude-labels` suppresses the automatic decision at orientation time:

```bash
morphometrics flip_normals config.yml --graph TE1_IMM.AVV_rh9.gt --exclude-labels 3,7
morphometrics flip_normals config.yml --exclude-labels TE1_IMM:3,7 --exclude-labels TF1_IMM:2
```

It is the weaker tool — it can only *prevent* a flip, never cause one, and it requires
re-running from the original graph. Its use is scripted reproducibility: once you know
the answer for a dataset, recording the exclusions makes a single `flip_normals` call
reproduce the corrected result from scratch. For interactive work, prefer `manual_flip`.

A bare list applies to every surface; prefix it with a surface name to scope it (the
prefix matches if it appears in the graph's base name, so `TE1_IMM` selects
`TE1_IMM.AVV_rh9.gt`).

Component ids are assigned by descending size (1 = largest) and are **stable for a given
input graph**, but re-running `pycurv` or accepting a different refinement iteration can
renumber them. Re-check the ids after any change upstream.

> Component ids come from the same numbering as `morphometrics label_components`, so a
> `_components` run and an `_oriented` run of the same graph agree on which id is which.

## Doing this from a GUI

This inspect-and-correct loop is intended to become a GUI plugin, where you click a
component to toggle its orientation — `manual_flip` is deliberately shaped the same way
(a toggle on named components of an existing file), so the plugin can call it directly.
Until then the ParaView pass above plus `manual_flip` is the supported workflow.
