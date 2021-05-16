"""Init and create global env registry."""

from gym.envs import registration

from truman import history  # noqa
from truman._version import get_versions
from truman.run.interface import run  # noqa


# Global registry
registry = registration.EnvRegistry()

# import all env modules, these contain the `register` calls which register envs to registry
from truman import time_period_step  # noqa


__version__ = get_versions()["version"]
del get_versions
