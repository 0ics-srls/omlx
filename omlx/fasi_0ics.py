"""0ics: OMLX_0ICS_PHASE_TIMING=1 logs where a request's time goes before its prefill (server, VLM, scheduler)."""
import logging
import os
import time

ATTIVO = os.environ.get("OMLX_0ICS_PHASE_TIMING") == "1"
_log = logging.getLogger("omlx.fasi_0ics")


def ora() -> float:
    return time.perf_counter()


def fase(msg: str, *args) -> None:
    if ATTIVO:
        _log.info("[fasi] " + msg, *args)
