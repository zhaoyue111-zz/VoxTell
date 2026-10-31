"""Compatibility entrypoint for the full-volume Oracle Support diagnostic."""

from voxtell.inference.oracle_support_full_volume import *  # noqa: F401,F403
from voxtell.inference.oracle_support_full_volume import main


if __name__ == "__main__":
    raise SystemExit(main())
