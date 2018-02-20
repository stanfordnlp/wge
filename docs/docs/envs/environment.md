# Environment


All environments implement the `Environment` interface. A policy interacts
with the environment by calling the environment's `step` method and passing in
actions.

Note that an environment object is _batched_. It actually represents a batch
of environments, each running in parallel (so that we can train faster).

Thus far, we have been mostly using `MiniWoBEnvironment`.

