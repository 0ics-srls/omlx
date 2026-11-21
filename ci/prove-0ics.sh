#!/usr/bin/env bash
# 0ics: oMLX's own tests, the command of upstream's ci.yml (Python 3.11, the unit tests without the slow and integration
# ones, three xdist workers grouped as upstream groups them), one script for our release (.github/workflows/rilascio-0ics.yml)
# and for ubuntu-ai's esegui on the Mac Studio.
#   ci/prove-0ics.sh <venv> [pytest arguments...]
#     <venv>  the environment, created with uv when missing and reused (--allow-existing)
# The tests already red on the intact upstream tag v0.7.3 are listed in ci/non-verdi-v0.7.3.txt (one pytest id per line)
# and deselected; any other red stops the run.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
[ $# -ge 1 ] || { echo "usage: ci/prove-0ics.sh <venv> [pytest arguments...]" >&2; exit 2; }
venv="$1"; shift
cd "$(dirname "$0")/.."

uv venv -q --allow-existing --python 3.11 "$venv"
# the native custom kernels, built as the release wheel builds them (OMLX_WITH_CUSTOM_KERNEL=1): without them the tests
# exercise another oMLX than the one in service (on a NAX GPU the fused sparse decode paths need dsa_topk_indices);
# --reinstall-package rebuilds them from this tree every run
OMLX_WITH_CUSTOM_KERNEL=1 uv pip install -q --python "$venv/bin/python" --reinstall-package omlx -e ".[mcp]" \
  "pytest<9.2" pytest-asyncio pytest-xdist

gia_rossi=()
if [ -f ci/non-verdi-v0.7.3.txt ]; then
  while IFS= read -r prova; do
    case "$prova" in ''|'#'*) ;; *) gia_rossi+=("--deselect=$prova") ;; esac
  done < ci/non-verdi-v0.7.3.txt
fi

"$venv/bin/python" -m pytest tests/ -m "not slow and not integration" -n 3 --dist loadgroup --no-loadscope-reorder \
  -p no:cacheprovider ${gia_rossi[@]+"${gia_rossi[@]}"} "$@"
