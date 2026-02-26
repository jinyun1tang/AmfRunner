#!/usr/bin/env python3
"""sitedata_writer.py — Extract single-point ELM domain and surface data files.

Overview
--------
Reads global ELM inputdata files (domain + surface dataset) and writes
single-point files for an AmeriFlux site.

Domain file (domain.lnd.*.nc)
------------------------------
  Dims  : ni=1, nj=1, nv=4
  Vars  : xc [deg E], yc [deg N], xv, yv (4 corner vertices),
           area [radian²], frac=1.0, mask=1

Surface data file (surfdata*.nc)
---------------------------------
  Dims  : lsmlon=1, lsmlat=1  (all other dims kept as-is)
  Key single-point overrides applied after copying nearest cell:
    LONGXY / LATIXY          ← exact site coordinates
    LANDFRAC_PFT = 1.0
    PFTDATA_MASK = 1
    PCT_NATVEG   = 100.0     (100 % natural vegetation)
    PCT_CROP     = 0.0
    PCT_WETLAND  = 0.0
    PCT_LAKE     = 0.0
    PCT_GLACIER  = 0.0
    PCT_URBAN    = 0.0       (all 3 density classes)
    PCT_NAT_PFT              ← from AmeriFlux_pftdata.txt if site known
    PCT_SAND / PCT_CLAY      ← from AmeriFlux_soildata.txt if site known
                               (applied uniformly to all soil layers)

PTCLM data
----------
  Bundled in data/ directory (from elm-olmt/inputdata/PTCLM):
    AmeriFlux_sitedata.txt   site coordinates and period
    AmeriFlux_pftdata.txt    dominant PFT fractions (up to 5 PFTs)
    AmeriFlux_soildata.txt   bulk sand/clay %

  ELM natural PFT codes (lsmpft dim, 0-indexed):
    0  = bare ground
    1  = needleleaf evergreen temperate tree
    2  = needleleaf evergreen boreal tree
    3  = needleleaf deciduous boreal tree
    4  = broadleaf evergreen tropical tree
    5  = broadleaf evergreen temperate tree
    6  = broadleaf deciduous tropical tree
    7  = broadleaf deciduous temperate tree
    8  = broadleaf deciduous boreal tree
    9  = broadleaf evergreen shrub
    10 = broadleaf deciduous temperate shrub
    11 = broadleaf deciduous boreal shrub
    12 = arctic C3 grass
    13 = C3 grass
    14 = C4 grass
    15 = C3 crop (legacy)
    16 = C4 crop (legacy)
"""

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import netCDF4 as nc


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / 'data'


# ---------------------------------------------------------------------------
# PTCLM lookup tables
# ---------------------------------------------------------------------------

def load_ptclm_pfts() -> dict:
    """Load AmeriFlux_pftdata.txt.

    Returns
    -------
    dict mapping site_code -> list of (frac_pct, pft_code) tuples
    """
    pfts = {}
    with open(_DATA_DIR / 'AmeriFlux_pftdata.txt') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            code = row['site_code'].strip()
            pairs = []
            for k in range(1, 6):
                try:
                    frac = float(row[f'pft_f{k}'])
                    pftc = int(float(row[f'pft_c{k}']))
                    if frac > 0 and pftc > 0:
                        pairs.append((frac, pftc))
                except (ValueError, KeyError):
                    pass
            if pairs:
                pfts[code] = pairs
    return pfts


def load_ptclm_soils() -> dict:
    """Load AmeriFlux_soildata.txt.

    Returns
    -------
    dict mapping site_code -> (sand_pct, clay_pct)  or None if no data
    """
    soils = {}
    with open(_DATA_DIR / 'AmeriFlux_soildata.txt') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            code = row['site_code'].strip()
            try:
                sand = float(row['layer_sand%'])
                clay = float(row['layer_clay%'])
                if sand >= 0 and clay >= 0:
                    soils[code] = (sand, clay)
            except (ValueError, KeyError):
                pass
    return soils


# ---------------------------------------------------------------------------
# Nearest-cell search
# ---------------------------------------------------------------------------

def find_nearest_ji(lons: np.ndarray, lats: np.ndarray,
                    site_lon: float, site_lat: float) -> tuple[int, int]:
    """Return (j, i) index of the nearest cell in 2-D lon/lat arrays.

    Uses squared-degree distance; wraps longitude differences to [-180, 180].
    """
    dlat = lats - site_lat
    dlon = lons - site_lon
    dlon = (dlon + 180.0) % 360.0 - 180.0   # wrap to [-180, 180]
    dist2 = dlat ** 2 + dlon ** 2
    idx = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(idx[0]), int(idx[1])


# ---------------------------------------------------------------------------
# Global file discovery
# ---------------------------------------------------------------------------

def _find_file(inputdata_dir: Path, subdirs: list[str],
               patterns: list[str]) -> Path:
    """Search for a file matching patterns under candidate subdirs, then globally."""
    # Try explicit subdirs first
    for sd in subdirs:
        root = inputdata_dir / sd
        if root.is_dir():
            for pat in patterns:
                hits = sorted(root.glob(pat))
                if hits:
                    return hits[-1]
    # Recursive search from inputdata_dir
    for pat in patterns:
        hits = sorted(inputdata_dir.glob(f'**/{pat}'))
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        f"Could not find a file matching {patterns} under {inputdata_dir}.\n"
        "Pass the explicit path via domain_file= or surfdata_file=."
    )


def find_domain_file(inputdata_dir: Path) -> Path:
    """Auto-discover the global ELM land domain file."""
    return _find_file(
        inputdata_dir,
        subdirs=['share/domains/domain.clm', 'share/domains'],
        patterns=['domain.lnd.*.nc', 'domain.clm.*.nc', 'domain.*.nc'],
    )


def find_surfdata_file(inputdata_dir: Path) -> Path:
    """Auto-discover the global ELM surface dataset."""
    return _find_file(
        inputdata_dir,
        subdirs=['lnd/clm2/surfdata_map', 'lnd/clm2'],
        patterns=['surfdata*.nc', 'surfdata_*.nc'],
    )


# ---------------------------------------------------------------------------
# Domain file writer
# ---------------------------------------------------------------------------

def write_domain(
    src_path: Path,
    out_path: Path,
    site_lon: float,
    site_lat: float,
    site_id: str,
) -> None:
    """Write a 1×1 ELM domain file from the global domain file.

    The global domain file is SCRIP-style with dims (nj, ni) and vars
    xc[nj,ni], yc[nj,ni], xv[nj,ni,nv], yv[nj,ni,nv], area[nj,ni],
    frac[nj,ni], mask[nj,ni].

    The output has nj=ni=1, nv=4 with site coordinates and frac=1, mask=1.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(str(src_path), 'r') as src:
        src.set_auto_mask(False)

        # Determine coordinate variable names
        if 'xc' in src.variables:
            lons = src.variables['xc'][:]
            lats = src.variables['yc'][:]
            lon_var, lat_var = 'xc', 'yc'
            area_var = 'area'
        else:
            lons = src.variables['LONGXY'][:]
            lats = src.variables['LATIXY'][:]
            lon_var, lat_var = 'LONGXY', 'LATIXY'
            area_var = 'AREA'

        j, i = find_nearest_ji(lons, lats, site_lon, site_lat)
        print(
            f"  Domain   : nearest cell at "
            f"({float(lats[j, i]):.4f}°N, {float(lons[j, i]):.4f}°E), "
            f"index ({j},{i})",
            flush=True,
        )

        # Cell area
        if area_var in src.variables:
            area_val = float(src.variables[area_var][j, i])
        else:
            # Approximate: 0.5° × 0.5° cell in radian²
            deg = 0.5 * math.pi / 180.0
            area_val = deg ** 2 * math.cos(math.radians(site_lat))

    # Half-resolution for vertex offsets (assume 0.5° grid)
    dx = 0.25

    with nc.Dataset(str(out_path), 'w', format='NETCDF4_CLASSIC') as dst:
        dst.title     = f"ELM domain for AmeriFlux site {site_id}"
        dst.source    = "Extracted from global domain file by make-elm-sitedata"
        dst.created   = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        dst.site_id   = site_id
        dst.latitude  = float(site_lat)
        dst.longitude = float(site_lon)

        dst.createDimension('ni', 1)
        dst.createDimension('nj', 1)
        dst.createDimension('nv', 4)

        def _var(name, dtype, dims, long_name, units=''):
            v = dst.createVariable(name, dtype, dims)
            v.long_name = long_name
            if units:
                v.units = units
            return v

        v = _var('xc', 'f8', ('nj', 'ni'), 'longitude of grid cell center', 'degrees_east')
        v[0, 0] = site_lon

        v = _var('yc', 'f8', ('nj', 'ni'), 'latitude of grid cell center', 'degrees_north')
        v[0, 0] = site_lat

        # Corners: SW, SE, NE, NW (CCW convention)
        v = _var('xv', 'f8', ('nj', 'ni', 'nv'), 'longitude of grid cell vertices', 'degrees_east')
        v[0, 0, :] = [site_lon - dx, site_lon + dx,
                      site_lon + dx, site_lon - dx]

        v = _var('yv', 'f8', ('nj', 'ni', 'nv'), 'latitude of grid cell vertices', 'degrees_north')
        v[0, 0, :] = [site_lat - dx, site_lat - dx,
                      site_lat + dx, site_lat + dx]

        v = _var('area', 'f8', ('nj', 'ni'), 'area of grid cell', 'radian^2')
        v[0, 0] = area_val

        v = _var('frac', 'f8', ('nj', 'ni'), 'fraction of grid cell that is active', 'unitless')
        v[0, 0] = 1.0

        v = _var('mask', 'i4', ('nj', 'ni'), 'domain mask (1=land, 0=ocean)')
        v[0, 0] = 1


# ---------------------------------------------------------------------------
# Surface data helpers
# ---------------------------------------------------------------------------

def _extract_at_cell(src_var, dims_in: tuple, j: int, i: int) -> np.ndarray:
    """Extract one (j, i) cell from a variable with any combination of dims.

    For lsmlat/lsmlon dims: slice [j:j+1] / [i:i+1] → size 1.
    All other dims: keep as-is with slice(None).
    """
    slices = []
    for d in dims_in:
        if d == 'lsmlat':
            slices.append(slice(j, j + 1))
        elif d == 'lsmlon':
            slices.append(slice(i, i + 1))
        else:
            slices.append(slice(None))
    return src_var[tuple(slices)]


def _set_var(dst, vname: str, val) -> None:
    """Assign val to dst.variables[vname][:] if it exists."""
    if vname in dst.variables:
        dst.variables[vname][:] = val


def _override_singlepoint(
    dst,
    site_lon: float,
    site_lat: float,
    pft_fracs: list | None,
    soil_sand: float | None,
    soil_clay: float | None,
) -> None:
    """Apply single-point overrides to the surface data file."""
    # Coordinates
    _set_var(dst, 'LONGXY', site_lon)
    _set_var(dst, 'LATIXY', site_lat)

    # Grid fractions and mask
    _set_var(dst, 'LANDFRAC_PFT', 1.0)
    _set_var(dst, 'PFTDATA_MASK', 1)
    _set_var(dst, 'LANDMASK',     1)

    # Land cover: 100 % natural vegetation
    _set_var(dst, 'PCT_NATVEG',  100.0)
    _set_var(dst, 'PCT_CROP',      0.0)
    _set_var(dst, 'PCT_WETLAND',   0.0)
    _set_var(dst, 'PCT_LAKE',      0.0)
    _set_var(dst, 'PCT_GLACIER',   0.0)

    # Urban: multi-density array (numurbl, lsmlat, lsmlon) or (lsmlat, lsmlon)
    if 'PCT_URBAN' in dst.variables:
        dst.variables['PCT_URBAN'][:] = 0.0

    # PCT_NAT_PFT: shape can be (lsmlat, lsmlon, lsmpft) or (lsmpft, lsmlat, lsmlon)
    if pft_fracs and 'PCT_NAT_PFT' in dst.variables:
        v = dst.variables['PCT_NAT_PFT']
        shape = v.shape
        vdims = v.dimensions
        try:
            pft_axis = list(vdims).index('lsmpft')
        except ValueError:
            pft_axis = None

        if pft_axis is not None:
            npft = shape[pft_axis]
            arr = np.zeros(shape, dtype=np.float32)
            total = sum(f for f, _ in pft_fracs)
            for frac, code in pft_fracs:
                if 0 <= code < npft:
                    # Build index tuple: 0 for lsmlat/lsmlon, code for lsmpft
                    idx = [0] * len(shape)
                    idx[pft_axis] = code
                    arr[tuple(idx)] = frac * 100.0 / total if total > 0 else frac
            v[:] = arr

    # Soil texture: broadcast to all soil layers
    if soil_sand is not None:
        _set_var(dst, 'PCT_SAND', soil_sand)
    if soil_clay is not None:
        _set_var(dst, 'PCT_CLAY', soil_clay)


# ---------------------------------------------------------------------------
# Surface data file writer
# ---------------------------------------------------------------------------

# Dims we know how to handle (slice lsmlat/lsmlon; keep others)
_KNOWN_DIMS = {
    'lsmlat', 'lsmlon', 'lsmpft', 'nlevsoi', 'nlevsoilc',
    'numurbl', 'time', 'nchar', 'nglcec', 'nglcecp1', 'nglcnec',
}


def write_surfdata(
    src_path: Path,
    out_path: Path,
    site_lon: float,
    site_lat: float,
    site_id: str,
    pft_fracs: list | None,
    soil_sand: float | None,
    soil_clay: float | None,
) -> None:
    """Write a 1×1 ELM surface data file from the global surfdata file.

    Copies every variable from the global file, extracting data at the
    nearest (j, i) gridcell.  Key land-cover and soil variables are then
    overridden for a single-point natural-vegetation site.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(str(src_path), 'r') as src:
        src.set_auto_mask(False)

        lons = src.variables['LONGXY'][:]
        lats = src.variables['LATIXY'][:]
        j, i = find_nearest_ji(lons, lats, site_lon, site_lat)
        print(
            f"  Surfdata : nearest cell at "
            f"({float(lats[j, i]):.4f}°N, {float(lons[j, i]):.4f}°E), "
            f"index ({j},{i})",
            flush=True,
        )

        with nc.Dataset(str(out_path), 'w', format='NETCDF4_CLASSIC') as dst:
            # Global attributes
            for attr in src.ncattrs():
                try:
                    setattr(dst, attr, getattr(src, attr))
                except Exception:
                    pass
            dst.title   = f"ELM surface data for AmeriFlux site {site_id}"
            dst.source  = "Extracted from global surfdata by make-elm-sitedata"
            dst.created = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
            dst.site_id = site_id

            # Dimensions: lsmlat=lsmlon=1, all others as in source
            for dname, dim in src.dimensions.items():
                if dname in ('lsmlon', 'lsmlat'):
                    dst.createDimension(dname, 1)
                elif dim.isunlimited():
                    dst.createDimension(dname, None)
                else:
                    dst.createDimension(dname, len(dim))

            # Variables
            for vname, src_var in src.variables.items():
                dims_in = src_var.dimensions
                dtype   = src_var.dtype

                # Skip variables whose dims are unknown (can't slice reliably)
                unknown = [d for d in dims_in if d not in _KNOWN_DIMS
                           and d not in src.dimensions]
                if unknown:
                    print(f"  Skipping {vname}: unknown dims {unknown}", flush=True)
                    continue

                # Create variable with same dims in destination
                fv = getattr(src_var, '_FillValue', None)
                create_kwargs = dict(zlib=True, complevel=4)
                if fv is not None:
                    create_kwargs['fill_value'] = fv

                try:
                    dst_var = dst.createVariable(vname, dtype, dims_in,
                                                 **create_kwargs)
                except Exception as exc:
                    print(f"  Warning: could not create {vname}: {exc}", flush=True)
                    continue

                # Copy variable attributes
                for attr in src_var.ncattrs():
                    if attr == '_FillValue':
                        continue
                    try:
                        setattr(dst_var, attr, getattr(src_var, attr))
                    except Exception:
                        pass

                # Extract and write data
                try:
                    data = _extract_at_cell(src_var, dims_in, j, i)
                    dst_var[:] = data
                except Exception as exc:
                    print(f"  Warning: could not copy {vname}: {exc}", flush=True)

            # Apply single-point overrides
            _override_singlepoint(dst, site_lon, site_lat,
                                  pft_fracs, soil_sand, soil_clay)

    print(
        f"  Surfdata : written with PCT_NATVEG=100, "
        f"sand={soil_sand}%, clay={soil_clay}%",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_sitedata(
    site_id: str,
    lat: float,
    lon: float,
    inputdata_dir: str | Path,
    out_dir: str | Path,
    domain_file: str | Path | None = None,
    surfdata_file: str | Path | None = None,
) -> dict:
    """Create ELM domain and surface data files for a single AmeriFlux site.

    Parameters
    ----------
    site_id       : AmeriFlux site ID (e.g. 'US-NR1')
    lat, lon      : Site coordinates in decimal degrees
    inputdata_dir : Root directory of downloaded ELM inputdata
    out_dir       : Output directory; two files are written here
    domain_file   : Optional explicit path to global domain file
    surfdata_file : Optional explicit path to global surfdata file

    Returns
    -------
    dict with keys domain_file, surfdata_file, status
    """
    inputdata_dir = Path(inputdata_dir)
    out_dir       = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PTCLM lookup
    pfts_db  = load_ptclm_pfts()
    soils_db = load_ptclm_soils()

    pft_fracs  = pfts_db.get(site_id)
    soil_entry = soils_db.get(site_id)
    soil_sand  = soil_entry[0] if soil_entry else None
    soil_clay  = soil_entry[1] if soil_entry else None

    if pft_fracs:
        pft_str = ', '.join(f'{c}:{f:.1f}%' for f, c in pft_fracs)
        print(f"  PFT data : {pft_str}", flush=True)
    else:
        print(f"  PFT data : site {site_id} not in PTCLM table; "
              "keeping global nearest-cell values", flush=True)

    if soil_sand is not None:
        print(f"  Soil     : sand={soil_sand}%, clay={soil_clay}%", flush=True)
    else:
        print(f"  Soil     : site {site_id} not in PTCLM table; "
              "keeping global nearest-cell values", flush=True)

    # Locate global input files
    if domain_file is None:
        src_domain = find_domain_file(inputdata_dir)
        print(f"  Domain src : {src_domain}", flush=True)
    else:
        src_domain = Path(domain_file)

    if surfdata_file is None:
        src_surf = find_surfdata_file(inputdata_dir)
        print(f"  Surfdata src : {src_surf}", flush=True)
    else:
        src_surf = Path(surfdata_file)

    # Output filenames
    tag       = site_id.replace('-', '')
    dom_out   = out_dir / f'domain.lnd.1x1pt_{tag}.nc'
    surf_out  = out_dir / f'surfdata_1x1pt_{tag}.nc'

    # Write domain
    print(f"Writing domain → {dom_out}", flush=True)
    write_domain(src_domain, dom_out, lon, lat, site_id)

    # Write surfdata
    print(f"Writing surfdata → {surf_out}", flush=True)
    write_surfdata(src_surf, surf_out, lon, lat, site_id,
                   pft_fracs, soil_sand, soil_clay)

    return {
        'domain_file':   str(dom_out.resolve()),
        'surfdata_file': str(surf_out.resolve()),
        'status':        'success',
    }
