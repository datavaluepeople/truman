[![PyPI version](https://img.shields.io/pypi/v/truman.svg)](https://pypi.org/project/truman/)
![lint-test status](https://github.com/datavaluepeople/truman/actions/workflows/lint-test.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/datavaluepeople/truman/branch/main/graph/badge.svg?token=3W8T5OSRZZ)](https://codecov.io/gh/datavaluepeople/truman)

# truman: dynamic complex-system simulations for one-shot optimal decision-making agents

## What is it?

**truman** is a package that implements suites of _environments_ (system
simulations) exhibiting behaviours of real world large scale systems, e.g.
changes in online consumer cohort conversion as a function of changes in
product price.

truman is _not_ an environment for training reinforcement learning agents, but
aims to be an effective way to _develop and validate one-shot optimal
decision-making agents_ that perform well on unique systems that can’t be
reliably simulated and that have a high cost of experimentation.

## Main Features

- Environments that are compatible (built on) OpenAI's
  [Gym](https://github.com/openai/gym) interface
- Various suites of environments that exhibit common behaviours of real world
  dynamic systems
- `truman.agent_registion` interface for managing agents and their
  hyperparameters
- `truman.run` interface for running suites of agents on suites of environments
  and storing performance summaries and full histories

## Installation

To get started, you'll need to have Python 3.7+ installed. Then:

```
pip install truman
```

### Editable install from source

You can also clone the truman Git repository directly. This is useful when
you're working on adding new environments or modifying truman itself. Clone and
install in editable mode using:

```
git clone https://github.com/datavaluepeople/truman
cd truman
pip install -e .
```

## Background: Why truman?

The base framework that environments are built upon is OpenAI’s
[Gym](https://github.com/openai/gym).  Gym is a powerful framework for building
environments and developing reinforcement learning algorithms - but Gym's
environments are mostly directed towards _training_ agents on problems that can
be simulated exactly, e.g. playing an Atari game.  Our work at
[datavaluepeople](https://datavaluepeople.com/) is often developing
reinforcement learning algorithms for making a massive number of optimal
decisions simultaneously on high noise and changing environments, e.g. pricing
100,000s of travel products daily, or health intervention decisions for
1,000,000s of humans daily.  In such environments, agents should be able to
learn quickly and adapt to novel behaviours, since the price of testing
algorithms is very high.

Thus the suites of environments in truman are directed towards the goal of
_large scale optimised decision making on complex systems_, and only allow
agents a single episode to both learn and optimize on simultaneously.

## License

[MIT](LICENSE.txt)

