# Truman: simulation system to enable development of RL algos

See [Problem Spec doc](https://docs.google.com/document/d/1kHmvkw4ok7knxq1hOK_XbKnm2sc-bL-7bU-tz53rC0A/edit#heading=h.ttm0ptnazbea) for what
we are aiming for.


## Possible frameworks to build on

Searching for "reinforcement-learning simulation python" mainly brings up stuff to simulate physical systems or computer game systems, which is not
really what we are after (e.g. https://github.com/google/dopamine).

* [OpenAI gym](https://github.com/openai/gym)
  * This most likely The One, ?
  * [Bandits implementation](https://github.com/JKCooper2/gym-bandits)
* [Mesa](https://github.com/projectmesa/mesa)
  * This is probably highest on a score of most general and most well developed and maintained.
  * It is aiming at *agent-based* modelling though. We are kind doing agent-based modelling (definitely in the case of simple multi-armed bandits),
but once we get to the larger scale and higher dim simulations, the 'agent-based' parts of the package might become too much over head, and all we
really want to do is define stochastic processes (I guess a single process can be a single agent then).
* [ReAgent](https://github.com/facebookresearch/ReAgent) from Facebook.
  * Looks like a framework for actually building the RL algorithms :/
  * Here is their [simulation for an e-commerce site](https://github.com/facebookresearch/ReAgent/blob/master/serving/examples/ecommerce/customer_simulator.py)

