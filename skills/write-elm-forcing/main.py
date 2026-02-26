#!/usr/bin/env python3
"""write-elm-forcing skill — entry point.

Reads JSON from STDIN (or first CLI argument) and calls elmwriter.convert().

Input JSON
----------
{
    "site_id"      : "US-NR1",      // required
    "lat"          : 40.033,        // required
    "lon"          : -105.546,      // required
    "data_dir"     : "project/ameriflux/US-NR1",  // optional
    "out_dir"      : "project/forcing/US-NR1",    // optional
    "utc_offset_h" : -7             // optional; estimated from lon if omitted
}

Output JSON
-----------
{
    "site_id"      : "US-NR1",
    "output_file"  : "/abs/path/to/all_hourly.nc",
    "out_dir"      : "/abs/path/to/project/forcing/US-NR1",
    "status"       : "success"
}
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import elmwriter


def _load_input() -> dict:
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            return json.loads(raw)
    if len(sys.argv) > 1:
        try:
            return json.loads(sys.argv[1])
        except json.JSONDecodeError:
            pass
    return {}


def main() -> None:
    payload = _load_input()

    site_id = payload.get('site_id')
    if not site_id:
        raise RuntimeError("'site_id' is required (e.g. 'US-NR1')")

    lat = payload.get('lat')
    lon = payload.get('lon')
    if lat is None or lon is None:
        raise RuntimeError("'lat' and 'lon' are required (site coordinates in degrees)")

    data_dir     = payload.get('data_dir') or f"project/ameriflux/{site_id}"
    out_dir      = payload.get('out_dir')  or f"project/forcing/{site_id}"
    utc_offset_h = payload.get('utc_offset_h')

    out_file = elmwriter.convert(
        site_id=site_id,
        data_dir=data_dir,
        out_dir=out_dir,
        lat=float(lat),
        lon=float(lon),
        utc_offset_h=float(utc_offset_h) if utc_offset_h is not None else None,
    )

    result = {
        'site_id':     site_id,
        'output_file': out_file,
        'out_dir':     str(Path(out_dir).resolve()),
        'status':      'success',
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
