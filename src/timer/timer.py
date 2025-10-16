from loguru import logger
from time import time, perf_counter
from typing import Optional, Any


class Timer:
    """
    A robust context manager for timing code execution.

    Features:
    - Uses high-precision perf_counter() for accurate timing
    - Handles exceptions gracefully
    - Provides multiple output formats
    - Optional custom naming and logging levels
    - Returns elapsed time for programmatic use
    """

    def __init__(
        self,
        name: Optional[str] = None,
        log_level: str = "success",
        use_perf_counter: bool = True,
        silent: bool = False,
    ) -> None:
        """
        Initialize the Timer.

        Args:
            name: Optional name for the timer (displayed in logs)
            log_level: Logging level ('trace', 'debug', 'info', 'warning', 'error', 'success')
            use_perf_counter: Use high-precision perf_counter vs time()
            silent: If True, suppress automatic logging
        """
        self.name = name
        self.log_level = log_level.lower()
        self.timer_func = perf_counter if use_perf_counter else time
        self.silent = silent
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def _log(self, message: str, level: str) -> None:
        """Log message at specified level."""
        log_func = getattr(logger, level, logger.info)
        log_func(message)

    def __enter__(self) -> "Timer":
        """Start timing and return self for access to elapsed time."""
        self.start_time = self.timer_func()
        if not self.silent:
            start_msg = "⏱️ Starting timer"
            if self.name:
                start_msg += f" <{self.name}>"
            self._log(start_msg, "debug")
        return self

    def __exit__(
        self,
        exception_type: Optional[type],
        exception_value: Optional[Exception],
        exception_traceback: Optional[Any],
    ) -> None:
        """Stop timing and log results."""
        self.end_time = self.timer_func()

        if self.start_time is None:
            logger.error("Timer was not properly started")
            return

        if self.silent:
            return

        self.elapsed = self.end_time - self.start_time

        # Format time with appropriate precision
        time_str = self._format_time(self.elapsed)

        # Create the log message
        msg = "⏱️ Computation time"
        if self.name:
            msg += f" for '{self.name}'"
        msg += f": {time_str}"

        # Add exception info if an error occurred
        if exception_type is not None:
            msg += f" (completed with {exception_type.__name__}: {exception_value})"
            log_level = "warning"
        else:
            log_level = self.log_level

        self._log(msg, log_level)

    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        if seconds < 1e-6:
            return f"{seconds * 1e9:.2f} ns"
        elif seconds < 1e-3:
            return f"{seconds * 1e6:.2f} μs"
        elif seconds < 1:
            return f"{seconds * 1e3:.2f} ms"
        elif seconds < 60:
            return f"{seconds:.3f} s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"
