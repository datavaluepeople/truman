"""Registration of agents.

Could be eventually be split out into a stand-alone package to use for registering and managing
agents.
"""
from typing import Optional

import re
import importlib
import logging

from gym import Env

from truman.typing import Agent


logger = logging.getLogger(__name__)

# This format is true today, but it's *not* an official spec.
# [username/](agent-name)-v(version)    agent-name is group 1, version is group 2
agent_id_re = re.compile(r"^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$")


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class AgentSpec:
    """A specification for a particular instance of an agent.

    Used to register agent and parameters full specification for official evaluations.

    Args:
        id (str): The official agent ID
        entry_point (Optional[str]): The Python entrypoint of the agent class
            (e.g.module.name:factory_func, or module.name:Class)
        nondeterministic (bool): Whether this environment is non-deterministic even after seeding
        kwargs (dict): The kwargs to pass to the agent class
    """

    def __init__(
        self,
        id: str,
        entry_point: Optional[str] = None,
        nondeterministic: bool = False,
        kwargs: Optional[dict] = None,
    ):
        self.id = id
        self.entry_point = entry_point
        self.nondeterministic = nondeterministic
        self._kwargs = {} if kwargs is None else kwargs

        match = agent_id_re.search(id)
        if not match:
            raise ValueError(
                f"Attempted to register malformed agent ID: {id}. "
                f"(Currently all IDs must be of the form {agent_id_re.pattern}.)"
            )
        self._agent_name = match.group(1)

    def make(self, env: Optional[Env] = None, **kwargs) -> Agent:
        """Instantiates an instance of the agent compatible with given env."""
        if self.entry_point is None:
            raise ValueError(
                f"Attempting to make deprecated agent {self.id}. "
                "(HINT: is there a newer registered version of this agent?)"
            )
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        factory = load(self.entry_point)
        agent = factory(env, **_kwargs)

        return agent

    def __repr__(self):
        return "AgentSpec({})".format(self.id)


class AgentRegistry:
    """Register an agent by ID.

    IDs should remain stable over time and should be guaranteed to resolve to the same agent
    dynamics (or be desupported). The goal is that results of a particular agent should always be
    comparable, and not depend on the version of the code that was running.
    """

    def __init__(self):
        self.agent_specs = {}

    def make(self, id: str, env: Optional[Env] = None, **kwargs) -> Agent:
        """Instantiate an instance of an agent of the given id compatible with the given env."""
        logging.info(f"Making new agent: {id} ({kwargs})")
        try:
            return self.agent_specs[id].make(env, **kwargs)
        except KeyError:
            raise KeyError(f"No registered agent with id {id}")

    def all(self):
        """Return all the agents in the registry."""
        return self.agent_specs.values()

    def register(
        self,
        id: str,
        entry_point: Optional[str] = None,
        nondeterministic: bool = False,
        kwargs: Optional[dict] = None,
    ):
        """Register an agent.

        Args:
            id (str): The official agent ID
            entry_point (Optional[str]): The Python entrypoint of the agent class
                (e.g.module.name:factory_func, or module.name:Class)
            nondeterministic (bool): Whether this environment is non-deterministic even after
                seeding
            kwargs (dict): The kwargs to pass to the agent class
        """
        if id in self.agent_specs:
            raise ValueError(f"Cannot re-register id {id}")
        self.agent_specs[id] = AgentSpec(id, entry_point, nondeterministic, kwargs)
