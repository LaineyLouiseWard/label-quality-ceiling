#!/usr/bin/env python3
"""
scripts/figures/study_area.py

Study-area map for the Biodiversity dataset (Section~\\ref{sec:biodiversity}).

The dataset is Ireland-only: 2,143 Pleiades tiles (0.5 m GSD) from three sites under
a single operator (ODOS Technologies). The licensed regions are not confidential, so
the figure shows their true locations and footprints. It motivates the
geographic-concentration limitation: ~91% of tiles come from one ~6x7 km inland block,
and the minority semi-natural grassland class is almost entirely confined to the two
small southwest sites.

Two panels:
  (a) Context -- the island of Ireland (Natural Earth 10 m, shipped with Cartopy) with
      the three sites at true location; the inland site is visually dominant.
  (b) Footprints to scale -- each site's real tile-coverage polygon (union of the raw
      GeoTIFF tile extents) drawn at a common kilometre scale, so the 91%/6%/3% size
      contrast is honest. The two southwest sites are flagged as the semi-natural source.

Geo-referencing (read from the raw GeoTIFFs, nothing hard-coded):
  data/biodiversity_raw/images/biodiversity_*.tif  -> EPSG:32629 (UTM 29N), georeferenced
  data/biodiversity_raw/images/ireland1_*.tif      -> EPSG:4326 (WGS84)
  data/biodiversity_raw/images/ireland2_*.tif      -> EPSG:4326 (WGS84)
The UTM tiles are reprojected to WGS84 with pyproj. Run this in an environment with a
working PROJ (e.g. conda env `S2S_AI`); the `ClassImbalance` env's proj.db is broken.

Per-site tile counts are counted from the raw tiles and equal the dataset split totals
(biodiversity 1553+204+195=1952; ireland1 54+4+6=64; ireland2 99+11+17=127; total 2143).

Output:
  figures/study_area.pdf (+ .png for QC), copied to manuscript/Figures/study_area.pdf

Computer Modern via text.usetex (mathtext-cm fallback), matching every other paper figure.
"""

from __future__ import annotations

import argparse
import glob
import math
import shutil
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

import rasterio
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)

# --- site definitions ------------------------------------------------------
# prefix -> (display label, short region, kind)  kind: "inland" | "sw"
SITES = {
    "biodiversity_": ("Inland midlands", r"Cos.\ Limerick \& Tipperary", "inland"),
    "ireland2_":     ("Southwest",        r"Co.\ Kerry",                   "sw"),
    "ireland1_":     ("Southwest",        r"West Cork",                    "sw"),
}

# One distinct colour per site, so each footprint is keyed to its own dot on the
# locator. The colour is an identity tag, not a land-cover or country category.
SITE_COLOUR = {
    "biodiversity_": "#3C6E9E",  # inland midlands -- blue
    "ireland2_":     "#E0843A",  # SW, Co. Kerry  -- orange
    "ireland1_":     "#4F9D8E",  # SW, West Cork  -- teal
}

# Documented per-site CRS, used as a fallback when rasterio/PROJ cannot resolve the tile's
# EPSG code (e.g. a stale proj.db: to_epsg() returns None even though the projection math works).
KNOWN_EPSG = {"biodiversity_": 32629, "ireland2_": 4326, "ireland1_": 4326}


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(10):
        if (p / "data").is_dir() and (p / "scripts").is_dir():
            return p
        if p == p.parent:
            break
        p = p.parent
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not find repo root")


def setup_font(use_tex: bool):
    rc = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
    }
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)


def site_geometry(root: Path):
    """For each site return its WGS84 tile boxes, centroid, count, and a local-km
    coverage polygon (union of tile footprints, origin at the site's SW corner)."""
    t29 = Transformer.from_crs("EPSG:32629", "EPSG:4326", always_xy=True)
    sites = {}
    for prefix in SITES:
        boxes = []
        for f in sorted(glob.glob(str(root / f"data/biodiversity_raw/images/{prefix}*.tif"))):
            with rasterio.open(f) as src:
                l, b, r, t = src.bounds
                try:
                    epsg = src.crs.to_epsg() if src.crs else None
                except Exception:
                    epsg = None
            if epsg is None:
                epsg = KNOWN_EPSG[prefix]   # stale proj.db fallback (see KNOWN_EPSG)
            if epsg == 32629:
                xs, ys = t29.transform([l, r, r, l], [b, b, t, t])
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
            elif epsg == 4326:
                boxes.append((l, b, r, t))
            else:
                raise RuntimeError(f"{f}: unexpected CRS {epsg}")
        if not boxes:
            raise FileNotFoundError(f"no raw tiles for {prefix}")
        a = np.array(boxes)
        lon0, lat0, lon1, lat1 = a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()
        clat = 0.5 * (lat0 + lat1)
        kx, ky = 111.0 * math.cos(math.radians(clat)), 111.0  # deg -> km, local tangent
        polys_km = [box((x0 - lon0) * kx, (y0 - lat0) * ky, (x1 - lon0) * kx, (y1 - lat0) * ky)
                    for x0, y0, x1, y1 in boxes]
        cover_km = unary_union(polys_km)
        cover_wgs = unary_union([box(*b) for b in boxes])  # true footprint in lon/lat
        sites[prefix] = {
            "centroid": (0.5 * (lon0 + lon1), 0.5 * (lat0 + lat1)),
            "bbox": (lon0, lat0, lon1, lat1),
            "count": len(boxes),
            "cover_km": cover_km,
            "cover_wgs": cover_wgs,
            "extent_km": (cover_km.bounds[2] - cover_km.bounds[0],
                          cover_km.bounds[3] - cover_km.bounds[1]),
        }
    total = sum(s["count"] for s in sites.values())
    for s in sites.values():
        s["share"] = 100.0 * s["count"] / total
    sites["_total"] = total
    return sites


def add_geom(ax, geom, dx=0.0, dy=0.0, **kw):
    """Add a shapely (Multi)Polygon to a plain axes, translated by (dx, dy)."""
    geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for g in geoms:
        verts, codes = [], []
        for ring in [g.exterior, *g.interiors]:
            xy = list(ring.coords)
            verts += [(x + dx, y + dy) for x, y in xy]
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), **kw))


def footprints_row(ax, sites, use_tex):
    """The three real site footprints in a row at a COMMON km scale (so their
    relative sizes are honest), each labelled only with its true W x H km; the
    fill colour keys every footprint to its dot on the locator (panel b)."""
    ax.set_aspect("equal")
    ax.axis("off")
    times = r"$\times$" if use_tex else "x"
    order = ["biodiversity_", "ireland2_", "ireland1_"]
    gap = 1.6   # enough that the two small SW sites' dimension labels do not collide
    x = 0.0
    hmax = max(sites[p]["extent_km"][1] for p in order)
    for prefix in order:
        s = sites[prefix]
        w, h = s["extent_km"]
        add_geom(ax, s["cover_km"], x, 0, facecolor=SITE_COLOUR[prefix], edgecolor="black",
                 linewidth=0.9, alpha=0.97, zorder=3)
        cx = x + w / 2
        # tile count centred ON the footprint (count only; the region is given in the caption)
        cc = s["cover_km"].centroid
        dy_lab = -1.5 if prefix == "biodiversity_" else 0.0  # nudge inland label off a coverage hole
        ax.text(cc.x + x, cc.y + dy_lab, f"{s['count']:,}\ntiles", ha="center", va="center",
                fontsize=13, color="white", fontweight="bold", linespacing=0.95, zorder=6)
        ax.text(cx, -0.5, f"{w:.1f} {times} {h:.1f} km", ha="center", va="top",
                fontsize=12.5, color="#333333", zorder=5)
        x += w + gap
    total_w = x - gap

    # vertical 2 km scale bar (one bar suffices -- common scale)
    xb = -1.0
    ax.plot([xb, xb], [0, 2], color="black", lw=1.5, zorder=5)
    for yy in (0, 2):
        ax.plot([xb - 0.18, xb + 0.18], [yy, yy], color="black", lw=1.5, zorder=5)
    ax.text(xb - 0.32, 1, "2 km", ha="right", va="center", rotation=90, fontsize=12, zorder=5)

    ax.set_xlim(xb - 0.7, total_w + 0.3)
    ax.set_ylim(-1.5, hmax + 0.4)


LOC_EXTENT = [-10.9, -5.0, 51.2, 55.6]  # island of Ireland, with even ocean margin on all sides


def locator(ax, sites, use_tex):
    """Small Ireland locator with degree graticule: a coloured dot per site."""
    ax.set_extent(LOC_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#DCE9F2", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#EDE9E0", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.5,
                   edgecolor="#5a5a5a", zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="#bbbbbb",
                      linestyle=":", zorder=1)
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-10, -8, -6])
    gl.ylocator = mticker.FixedLocator([52, 54])
    deg = r"$^{\circ}$" if use_tex else "°"
    gl.xformatter = LongitudeFormatter(degree_symbol=deg)
    gl.yformatter = LatitudeFormatter(degree_symbol=deg)
    gl.xlabel_style = gl.ylabel_style = {"size": 11.5, "color": "#444444"}

    pc = ccrs.PlateCarree()
    for prefix in ("biodiversity_", "ireland2_", "ireland1_"):
        lon, lat = sites[prefix]["centroid"]
        ax.scatter(lon, lat, s=75, marker="o", facecolors=SITE_COLOUR[prefix],
                   edgecolors="black", linewidths=0.7, zorder=5, transform=pc)


def _merc_aspect(ext):
    """Width/height aspect of the Mercator view of a lon/lat extent, using the
    same projection the locator axes uses so the map fills its box exactly."""
    merc = ccrs.Mercator()
    x0, y0 = merc.transform_point(ext[0], ext[2], ccrs.PlateCarree())
    x1, y1 = merc.transform_point(ext[1], ext[3], ccrs.PlateCarree())
    return (x1 - x0) / (y1 - y0)


def render(root: Path, out_dir: Path, use_tex: bool):
    setup_font(use_tex)
    sites = site_geometry(root)

    W, H = 8.6, 4.3
    fig = plt.figure(figsize=(W, H))
    ax_fp = fig.add_axes([0.045, 0.06, 0.56, 0.90])
    footprints_row(ax_fp, sites, use_tex)
    fig.canvas.draw()  # finalise the equal-aspect layout before measuring it

    # Align the locator: its top to the inland (blue) box top, its bottom to the
    # bottom of the dimension labels -- both read back from the rendered panel.
    inv = fig.transFigure.inverted()
    rend = fig.canvas.get_renderer()
    w_in, h_in = sites["biodiversity_"]["extent_km"]
    y_top = inv.transform(ax_fp.transData.transform((w_in / 2, h_in)))[1]
    y_bot = min(inv.transform((t.get_window_extent(renderer=rend).x0,
                               t.get_window_extent(renderer=rend).y0))[1]
                for t in ax_fp.texts)

    h_loc = y_top - y_bot
    w_loc = _merc_aspect(LOC_EXTENT) * h_loc * (H / W)
    loc_left = ax_fp.get_position().x1 + 0.075
    ax_loc = fig.add_axes([loc_left, y_bot, w_loc, h_loc], projection=ccrs.Mercator())
    locator(ax_loc, sites, use_tex)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "study_area.pdf"
    png = out_dir / "study_area.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # mirror into the submission tree
    sub = root / "manuscript/Figures/study_area.pdf"
    if sub.parent.is_dir():
        shutil.copyfile(pdf, sub)

    print(f"[study_area] {sites['_total']} tiles across 3 sites")
    for prefix, (name, region, _) in SITES.items():
        s = sites[prefix]
        ew, eh = s["extent_km"]
        print(f"  {name:<16} {region:<28} {s['count']:>5,} ({s['share']:4.1f}%)  "
              f"{ew:.1f} x {eh:.1f} km  centroid=({s['centroid'][0]:.3f}, {s['centroid'][1]:.3f})")
    print(f"  -> {pdf}\n  -> {png}")
    if sub.parent.is_dir():
        print(f"  -> {sub}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--no-tex", action="store_true")
    args = ap.parse_args()
    root = find_repo_root()
    out_dir = (root / args.out_dir).resolve()
    use_tex = not args.no_tex
    try:
        render(root, out_dir, use_tex)
    except Exception as e:
        if not use_tex:
            raise
        print(f"[study_area] usetex failed ({e}); retrying with mathtext")
        render(root, out_dir, use_tex=False)


if __name__ == "__main__":
    main()
