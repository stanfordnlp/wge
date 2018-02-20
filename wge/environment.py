import abc


class Environment(object):
    """An environment is an interface between the policy and the world.
    Imitates Gym / Universe API. Can run multiple instances in parallel.

    (Equivalent to Executor in strongsup.)
    """

    @classmethod
    def make(cls, domain, subdomain):
        """Creates a new Environment for a particular domain.
        Also starts up necessary resources for execution.

        Args:
            domain (str): (e.g., "scone" or "miniwob")
            subdomain (str): (e.g., "alchemy" or "click-test")

        Returns:
            Environment
        """
        if domain == 'miniwob':
            from wge.miniwob.environment import MiniWoBEnvironment
            return MiniWoBEnvironment(subdomain)
        elif domain == 'formwob':
            from wge.formwob.environment import FormWoBEnvironment
            return FormWoBEnvironment(subdomain)
        raise ValueError('Unknown domain name {}'.format(domain))

    @abc.abstractmethod
    def configure(self, num_instances=1, **kwargs):
        """Configures the number of instances.

        Args:
            num_instances (int): number of instances
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Initiates new instances. Must be called at the beginning of an
        episode.

        Returns:
            states (list[State]): initial state for each
                instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, actions):
        """Applies an action on each instance and returns the results.

        Args:
            actions (list[Action or None]): action for each instance.
                None can be used if no operation should be performed,
                or if the instance is already `done`.

        Returns:
            tuple (states, rewards, dones, info)
            states (list[State]): state for each instance.
            rewards (list[float]): rewards for the last time step
            dones (list[bool]): once `done` is True, further actions on that
                instance will give undefined results.
            info (dict): additional debug information.
                Global debug information is directly in the root level
                Local information for instance i is in info['n'][i]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """Clean up the resources."""
        raise NotImplementedError

    @abc.abstractproperty
    def num_instances(self):
        raise NotImplementedError
