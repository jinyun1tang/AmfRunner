#!/usr/bin/env python3
"""elmwriter.py — Convert AmeriFlux FLUXNET ERA5 HH CSV to ELM cpl_bypass forcing.

ELM cpl_bypass 'site' type format  (all_hourly.nc)
===================================================
Source: components/elm/src/cpl/lnd_import_export.F90

File layout
-----------
  Dimensions : DTIME (unlimited, no Feb-29), n=1 (single gridcell)
  Time var   : DTIME  — fractional days since Jan-1 of start_year UTC,
               starting at 0.0, step = timestep_s / 86400
  Scalar vars: start_year, end_year  (int32; read by ELM to find year range)
  Met vars   : shape (DTIME, n), float32 with scale_factor=1 / add_offset=0
               TBOT      air temperature                  [K]
               PSRF      surface pressure                 [Pa]
               RH        relative humidity                [%]  <- NOT QBOT
               FSDS      downward shortwave radiation     [W/m2]
               PRECTmms  precipitation rate               [mm/s]
               WIND      wind speed                       [m/s]
               FLDS      downward longwave radiation      [W/m2]

Input data
----------
  Reads the ERA5_HH CSV from the FLUXNET zip archive or a plain directory.
  Columns used:
    TIMESTAMP_START, TIMESTAMP_END
    TA_ERA    [°C]          → TBOT (+273.15)
    SW_IN_ERA [W/m2]        → FSDS (direct, ≥ 0)
    LW_IN_ERA [W/m2]        → FLDS (direct, ≥ 0)
    PA_ERA    [kPa]         → PSRF (×1000)
    P_ERA     [mm/interval] → PRECTmms (÷ timestep_s, ≥ 0)
    WS_ERA    [m/s]         → WIND (direct, ≥ 0)
    RH_ERA    [%]           → RH  (direct)          — preferred
    VPD_ERA   [hPa]         → RH  (derived)         — fallback when RH_ERA absent

  RH from VPD:
    esat [hPa] = 6.112 × exp(17.67 × T_C / (T_C + 243.5))
    RH   [%]   = (1 − VPD / esat) × 100,  clipped to [0, 100]

Time-zone convention
--------------------
  AmeriFlux TIMESTAMP_START is in LOCAL STANDARD TIME (LST), no DST.
  ELM needs UTC.   UTC = LST − utc_offset_h
  For western sites utc_offset_h < 0:   e.g. MST (UTC−7) → utc_offset_h = −7
    00:00 LST Jan-1  →  07:00 UTC Jan-1

No-leap calendar
----------------
  ELM indexes forcing as (yr − start_year) × 365 × steps_per_day.
  Feb-29 rows are removed so every year has exactly 365 days.
"""

import io
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc


# ---------------------------------------------------------------------------
# ELM variable order — must match lnd_import_export.F90 metvars array
# ---------------------------------------------------------------------------
ELM_VARS = [
    ('TBOT',     'atmospheric air temperature',               'K'),
    ('PSRF',     'surface pressure at the lowest atm level', 'Pa'),
    ('RH',       'atmospheric relative humidity',            '%'),
    ('FSDS',     'atmospheric incident solar radiation',     'W/m2'),
    ('PRECTmms', 'precipitation',                           'mm/s'),
    ('WIND',     'atmospheric wind velocity magnitude',      'm/s'),
    ('FLDS',     'atmospheric longwave radiation',           'W/m2'),
]

# Required ERA5 data columns — checked after TIMESTAMP_START becomes the index
AMF_ERA5_REQUIRED = ['TA_ERA', 'SW_IN_ERA', 'LW_IN_ERA', 'PA_ERA', 'P_ERA', 'WS_ERA']
_FILL = np.float32(1.0e36)


# ---------------------------------------------------------------------------
# Data discovery — zip archive or plain directory
# ---------------------------------------------------------------------------

def _era5_hh_pattern(site_id: str) -> list[str]:
    """Candidate CSV name patterns for the ERA5 HH file, most-specific first."""
    return [
        f"AMF_{site_id}_FLUXNET_ERA5_HH_*.csv",
        f"AMF_{site_id}_BASE_HH_*.csv",
        f"AMF_{site_id}_BASE_HR_*.csv",
        f"*ERA5_HH*.csv",
        f"*ERA5_HH*",
        "*.csv",
    ]


def find_amf_data(data_dir: Path, site_id: str) -> tuple[Path, str | None]:
    """Locate the ERA5 HH file.

    Returns
    -------
    (source_path, csv_name_in_zip)
      If the file is inside a zip:   source_path = zip file,  csv_name_in_zip = member name
      If the file is a plain CSV:    source_path = csv file,  csv_name_in_zip = None
    """
    # 1. Check for a zip archive whose name contains the site_id
    zip_candidates = sorted(data_dir.glob(f"*{site_id}*.zip"))
    if not zip_candidates:
        zip_candidates = sorted(data_dir.glob("*.zip"))

    for zpath in zip_candidates:
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            for pattern in _era5_hh_pattern(site_id):
                # simple glob-style match on the basename
                import fnmatch
                matches = [n for n in names if fnmatch.fnmatch(Path(n).name, pattern)
                           or fnmatch.fnmatch(n, pattern)]
                if matches:
                    return zpath, sorted(matches)[-1]

    # 2. Fall back to plain files on disk
    for pattern in _era5_hh_pattern(site_id):
        matches = sorted(data_dir.glob(pattern))
        if matches:
            return matches[-1], None

    raise FileNotFoundError(
        f"ERA5 HH forcing file not found for '{site_id}' in {data_dir}.\n"
        "Expected a zip archive or CSV whose name contains 'ERA5_HH'."
    )


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_amf_csv(source: Path, csv_name: str | None) -> pd.DataFrame:
    """Load the ERA5 HH CSV from a zip archive or a plain file."""
    if csv_name is not None:
        with zipfile.ZipFile(source) as zf:
            with zf.open(csv_name) as f:
                raw = io.TextIOWrapper(f, encoding='utf-8')
                df = pd.read_csv(raw, comment='#',
                                 na_values=['-9999', '-9999.0'],
                                 low_memory=False)
    else:
        df = pd.read_csv(source, comment='#',
                         na_values=['-9999', '-9999.0'],
                         low_memory=False)

    for col in ('TIMESTAMP_START', 'TIMESTAMP_END'):
        df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d%H%M')
    df = df.set_index('TIMESTAMP_START').sort_index()
    return df


def detect_timestep_s(df: pd.DataFrame) -> int:
    """Return the dominant timestep in seconds (1800 for HH, 3600 for HR)."""
    diffs = df.index.to_series().diff().dropna()
    return int(diffs.mode()[0].total_seconds())


# ---------------------------------------------------------------------------
# Time-zone and calendar transformations
# ---------------------------------------------------------------------------

def lst_to_utc(df: pd.DataFrame, utc_offset_h: float) -> pd.DataFrame:
    """Shift index from LST to UTC.  UTC = LST − utc_offset_h."""
    df = df.copy()
    df.index = df.index - pd.Timedelta(hours=utc_offset_h)
    return df


def remove_feb29(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where month=2, day=29 (ELM no-leap calendar)."""
    mask = ~((df.index.month == 2) & (df.index.day == 29))
    n_removed = int((~mask).sum())
    if n_removed:
        print(f"  Removed {n_removed} Feb-29 rows (no-leap calendar)", flush=True)
    return df.loc[mask]


# ---------------------------------------------------------------------------
# Humidity: RH from VPD when RH_ERA is absent
# ---------------------------------------------------------------------------

def vpd_to_rh(vpd_hpa: np.ndarray, t_c: np.ndarray) -> np.ndarray:
    """Derive relative humidity [%] from VPD [hPa] and air temperature [°C].

    esat [hPa] = 6.112 × exp(17.67 × T / (T + 243.5))    (Magnus formula)
    RH   [%]   = (1 − VPD / esat) × 100
    """
    esat = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    rh   = (1.0 - vpd_hpa / esat) * 100.0
    return np.clip(rh, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Variable construction
# ---------------------------------------------------------------------------

def build_elm_arrays(df: pd.DataFrame, timestep_s: int) -> dict:
    """Compute the 7 ELM forcing variables from AmeriFlux ERA5 columns."""
    ta_c  = df['TA_ERA'].values.astype(np.float64)
    pa_pa = df['PA_ERA'].values.astype(np.float64) * 1000.0   # kPa → Pa

    # Relative humidity: prefer RH_ERA, fall back to VPD_ERA
    if 'RH_ERA' in df.columns:
        rh = np.clip(df['RH_ERA'].values.astype(np.float64), 0.0, 100.0)
        print("  Humidity : using RH_ERA [%]", flush=True)
    elif 'VPD_ERA' in df.columns:
        rh = vpd_to_rh(df['VPD_ERA'].values.astype(np.float64), ta_c)
        print("  Humidity : derived RH from VPD_ERA [hPa] + TA_ERA", flush=True)
    else:
        raise RuntimeError(
            "Neither RH_ERA nor VPD_ERA found in CSV. "
            "Cannot compute relative humidity for ELM."
        )

    return {
        'TBOT':     ta_c + 273.15,
        'PSRF':     pa_pa,
        'RH':       rh,
        'FSDS':     np.clip(df['SW_IN_ERA'].values.astype(np.float64), 0.0, None),
        'PRECTmms': np.clip(df['P_ERA'].values.astype(np.float64) / timestep_s,
                            0.0, None),
        'WIND':     np.clip(df['WS_ERA'].values.astype(np.float64), 0.0, None),
        'FLDS':     np.clip(df['LW_IN_ERA'].values.astype(np.float64), 0.0, None),
    }


# ---------------------------------------------------------------------------
# NetCDF writer
# ---------------------------------------------------------------------------

def write_all_hourly_nc(
    df: pd.DataFrame,
    elm_data: dict,
    out_path: Path,
    site_id: str,
    lat: float,
    lon: float,
    timestep_s: int,
) -> None:
    """Write ELM all_hourly.nc in cpl_bypass site format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dt_days    = timestep_s / 86400.0
    start_year = int(df.index.year.min())
    end_year   = int(df.index.year.max())
    ntimes     = len(df)

    # DTIME: 0.0, dt_days, 2*dt_days, … (elapsed days from midnight UTC Jan-1 start_year)
    dtime_vals = np.arange(ntimes, dtype=np.float64) * dt_days

    with nc.Dataset(str(out_path), 'w', format='NETCDF4_CLASSIC') as ds:
        ds.title       = f"ELM cpl_bypass site forcing for {site_id}"
        ds.source      = "AmeriFlux FLUXNET ERA5 HH product"
        ds.created     = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        ds.site_id     = site_id
        ds.latitude    = float(lat)
        ds.longitude   = float(lon)
        ds.calendar    = "noleap"
        ds.time_zone   = "UTC"

        ds.createDimension('DTIME', None)
        ds.createDimension('n', 1)

        vt           = ds.createVariable('DTIME', 'f8', ('DTIME',))
        vt.long_name = (f"days since {start_year}-01-01 00:00:00 UTC "
                        "(noleap, Feb-29 removed)")
        vt.units     = f"days since {start_year}-01-01 00:00:00"
        vt.calendar  = "noleap"
        vt[:]        = dtime_vals

        v_sy           = ds.createVariable('start_year', 'i4', ())
        v_sy.long_name = "first calendar year of forcing data"
        v_sy[()]       = start_year

        v_ey           = ds.createVariable('end_year', 'i4', ())
        v_ey.long_name = "last calendar year of forcing data"
        v_ey[()]       = end_year

        for vname, long_name, units in ELM_VARS:
            data = elm_data[vname].astype(np.float32)
            data[~np.isfinite(data)] = _FILL

            var              = ds.createVariable(
                vname, 'f4', ('DTIME', 'n'),
                fill_value=_FILL, zlib=True, complevel=4,
            )
            var.long_name    = long_name
            var.units        = units
            var.scale_factor = np.float32(1.0)
            var.add_offset   = np.float32(0.0)
            var[:, 0]        = data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert(
    site_id: str,
    data_dir: str | Path,
    out_dir: str | Path,
    lat: float,
    lon: float,
    utc_offset_h: float | None = None,
) -> str:
    """Convert AmeriFlux FLUXNET ERA5 HH CSV to ELM all_hourly.nc.

    Parameters
    ----------
    site_id      : AmeriFlux site ID, e.g. 'US-NR1'
    data_dir     : Directory containing the site zip archive (or plain CSV)
    out_dir      : Output directory; all_hourly.nc is written here
    lat, lon     : Site coordinates (decimal degrees; lon < 0 west)
    utc_offset_h : LST offset from UTC in hours (e.g. -7 for MST).
                   Estimated as round(lon / 15) if None.

    Returns
    -------
    Absolute path of the written all_hourly.nc.
    """
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)

    if utc_offset_h is None:
        utc_offset_h = round(lon / 15.0)
        print(f"  UTC offset estimated from longitude ({lon:.3f}°): "
              f"{utc_offset_h:+d} h", flush=True)

    # 1. Find the ERA5 HH file (inside zip or plain CSV)
    source, csv_name = find_amf_data(data_dir, site_id)
    if csv_name:
        print(f"Loading {source.name} / {csv_name}", flush=True)
    else:
        print(f"Loading {source}", flush=True)

    df = load_amf_csv(source, csv_name)

    missing = [c for c in AMF_ERA5_REQUIRED if c not in df.columns]
    if missing:
        raise RuntimeError(f"Required columns not found in CSV: {missing}")

    timestep_s = detect_timestep_s(df)
    print(f"  Timestep : {timestep_s} s ({timestep_s / 3600:.4g} h)", flush=True)
    print(f"  Raw span : {df.index[0]} – {df.index[-1]} LST "
          f"({len(df)} records)", flush=True)

    # 2. LST → UTC
    df = lst_to_utc(df, utc_offset_h)
    print(f"  UTC span : {df.index[0]} – {df.index[-1]} UTC "
          f"(shifted {utc_offset_h:+.1f} h)", flush=True)

    # 3. Remove Feb-29
    df = remove_feb29(df)
    years = sorted(df.index.year.unique())
    print(f"  Years    : {years[0]}–{years[-1]}  "
          f"({len(years)} years, {len(df)} records)", flush=True)

    # 4. Build ELM arrays
    elm_data = build_elm_arrays(df, timestep_s)

    # 5. Write NetCDF
    out_path = out_dir / 'all_hourly.nc'
    write_all_hourly_nc(df, elm_data, out_path, site_id, lat, lon, timestep_s)
    print(f"Written  → {out_path.resolve()}", flush=True)

    return str(out_path.resolve())
