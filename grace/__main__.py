"""Allow ``python -m grace`` to launch a run."""

from grace.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
