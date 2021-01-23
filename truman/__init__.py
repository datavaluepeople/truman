from truman import history  # noqa

from gym.envs import registration

# Global registry
registry = registration.EnvRegistry()

# import all env modules, these contain the `register` calls which register envs to registry
from truman import time_period_step  # noqa
