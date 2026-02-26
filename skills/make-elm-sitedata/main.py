#!/usr/bin/env python3
"""make-elm-sitedata skill — entry point.

Reads JSON from STDIN (or first CLI argument) and calls sitedata_writer.make_sitedata().

Input JSON
----------
{
    "site_id"       : "US-NR1",              // required
    "lat"           : 40.0329,               // required
    "lon"           : -105.5464,             // required
    "inputdata_dir" : "/path/to/inputdata",  // required
    "out_dir"       : "project/sitedata/US-NR1",  // optional
    "domain_file"   : "/explicit/path/domain.lnd.nc",   // optional
    "surfdata_file" : "/explicit/path/surfdata.nc"       // optional
}

Output JSON
-----------
{
    "site_id"      : "US-NR1",
    "domain_file"  : "/abs/path/domain.lnd.1x1pt_USNR1.nc",
    "surfdata_file": "/abs/path/surfdata_1x1pt_USNR1.nc",
    "out_dir"      : "/abs/path/to/out_dir",
    "status"       : "success"
}
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sitedata_writer


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

    inputdata_dir = payload.get('inputdata_dir')
    if not inputdata_dir:
        raise RuntimeError(
            "'inputdata_dir' is required: root directory of downloaded ELM inputdata"
        )

    out_dir      = payload.get('out_dir') or f'project/sitedata/{site_id}'
    domain_file  = payload.get('domain_file')
    surfdata_file = payload.get('surfdata_file')

    result = sitedata_writer.make_sitedata(
        site_id=site_id,
        lat=float(lat),
        lon=float(lon),
        inputdata_dir=inputdata_dir,
        out_dir=out_dir,
        domain_file=domain_file,
        surfdata_file=surfdata_file,
    )

    result['site_id'] = site_id
    result['out_dir'] = str(Path(out_dir).resolve())

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
