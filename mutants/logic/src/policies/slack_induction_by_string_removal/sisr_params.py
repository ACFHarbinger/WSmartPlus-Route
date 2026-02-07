class SISRParams:
    """
    Configuration parameters for the SISR solver.
    """

    def __init__(
        self,
        time_limit: float = 10.0,
        max_iterations: int = 1000,
        start_temp: float = 100.0,
        cooling_rate: float = 0.995,
        max_string_len: int = 10,
        avg_string_len: float = 3.0,
        blink_rate: float = 0.01,
        destroy_ratio: float = 0.2,
    ):
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.start_temp = start_temp
        self.cooling_rate = cooling_rate
        self.max_string_len = max_string_len
        self.avg_string_len = avg_string_len
        self.blink_rate = blink_rate
        self.destroy_ratio = destroy_ratio
