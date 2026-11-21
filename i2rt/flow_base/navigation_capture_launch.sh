#!/bin/sh
# Prepared launcher; install on Pi only after deployment approval.
set -eu
case "${SSH_ORIGINAL_COMMAND:-check}" in
  check|start) ;;
  *) echo 'Only check or start is allowed.' >&2; exit 2 ;;
esac
/usr/bin/python3 - "${SSH_ORIGINAL_COMMAND:-check}" <<'PY'
from pathlib import Path
import json,subprocess,sys
active=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:args=p.read_bytes().split(b'\0')
    except OSError:continue
    if any(a and Path(a.decode(errors='replace')).name in ('head_tilt_controller.py','flow_base_controller.py') for a in args):active.append(p.parent.name)
synced=subprocess.check_output(['timedatectl','show','-p','NTPSynchronized','--value'],text=True).strip()
if sys.argv[1]=='start':
    if active:raise SystemExit('Controller already running; refusing duplicate')
    if synced!='yes':raise SystemExit('Wait for Pi time synchronization')
else:print(json.dumps({'controller_pids':active,'time_synchronized':synced}))
PY
if test "${SSH_ORIGINAL_COMMAND:-check}" = start; then
    echo 'Starting HeadTiltVehicle: may home rail and move head. Ctrl+C stops controller.'
    cd /home/i2rt/i2rt
    export PYTHONPATH=/home/i2rt/i2rt
    exec .venv/bin/python i2rt/flow_base/head_tilt_controller.py --channel can0
fi
