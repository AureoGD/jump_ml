class SwitchRule:
    def __init__(self, thresholds, initial_mode=1):
        """
        thresholds: [T_switch] → step threshold to switch to mode 2
        initial_mode: 0 or 1 → which mode to use before switching to 2
        """
        assert initial_mode in [0, 1], "Initial mode must be 0 or 1"
        self.threshold = thresholds[0]  # single switch threshold
        self.initial_mode = initial_mode

    def evaluate(self, step_count):
        if step_count < self.threshold:
            return self.initial_mode
        else:
            return 2
