"""Exceptions specific to truman."""


class StoppedEarly(Exception):
    """Raised when a run is forced to stop before the environment has finished."""
