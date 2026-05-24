class BaseDiscreteActionEnv:
    """Small local base needed by the vendored LaMer Sokoban wrapper."""

    def __init__(self):
        self.reward = 0
        self._actions = []
        self._actions_valid = []
        self._actions_effective = []

    def _reset_tracking_variables(self):
        self.reward = 0
        self._actions = []
        self._actions_valid = []
        self._actions_effective = []
