"""
Memory monitoring and OOM detection utilities.
Tracks system memory usage and triggers cleanup or graceful shutdown when needed.
"""
import os
import gc
import logging
import threading
import signal
from typing import Optional, Callable
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Current memory usage statistics."""
    total_mb: float
    used_mb: float
    available_mb: float
    percent_used: float
    gpu_total_mb: Optional[float] = None
    gpu_used_mb: Optional[float] = None
    gpu_free_mb: Optional[float] = None


class MemoryMonitor:
    """
    Monitor system and GPU memory usage.
    Provides alerts and callbacks when memory thresholds are exceeded.
    """

    def __init__(
        self,
        warning_threshold: float = 80.0,
        critical_threshold: float = 90.0,
        check_interval: int = 30,
        on_critical: Optional[Callable] = None
    ):
        """
        Initialize memory monitor.

        Args:
            warning_threshold: Memory usage % for warning alerts (default: 80%)
            critical_threshold: Memory usage % for critical alerts (default: 90%)
            check_interval: Seconds between memory checks (default: 30)
            on_critical: Callback function when critical threshold is exceeded
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.on_critical = on_critical

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        # System memory
        mem = psutil.virtual_memory()
        stats = MemoryStats(
            total_mb=mem.total / (1024 * 1024),
            used_mb=mem.used / (1024 * 1024),
            available_mb=mem.available / (1024 * 1024),
            percent_used=mem.percent
        )

        # GPU memory (if available)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.mem_get_info()
                stats.gpu_total_mb = gpu_mem[1] / (1024 * 1024)
                stats.gpu_free_mb = gpu_mem[0] / (1024 * 1024)
                stats.gpu_used_mb = stats.gpu_total_mb - stats.gpu_free_mb
        except Exception:
            pass

        return stats

    def is_near_oom(self, stats: Optional[MemoryStats] = None) -> bool:
        """Check if system is near OOM (memory usage exceeds critical threshold)."""
        if stats is None:
            stats = self.get_memory_stats()
        return stats.percent_used >= self.critical_threshold

    def is_memory_high(self, stats: Optional[MemoryStats] = None) -> bool:
        """Check if memory usage is high (exceeds warning threshold)."""
        if stats is None:
            stats = self.get_memory_stats()
        return stats.percent_used >= self.warning_threshold

    def force_cleanup(self):
        """Force garbage collection and cleanup."""
        logger.info("Forcing memory cleanup...")
        gc.collect()

        # Try to clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache: {e}")

    def start_monitoring(self):
        """Start background memory monitoring thread."""
        if self._monitoring:
            logger.warning("Memory monitoring already active")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="MemoryMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started memory monitoring (warning: {self.warning_threshold}%, "
                   f"critical: {self.critical_threshold}%)")

    def stop_monitoring(self):
        """Stop background memory monitoring thread."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        logger.info("Stopped memory monitoring")

    def _monitor_loop(self):
        """Background thread that monitors memory usage."""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()

                if stats.percent_used >= self.critical_threshold:
                    logger.critical(
                        f"CRITICAL: Memory usage at {stats.percent_used:.1f}% "
                        f"({stats.used_mb:.0f}MB / {stats.total_mb:.0f}MB)"
                    )

                    # Force cleanup
                    self.force_cleanup()

                    # Check again after cleanup
                    stats_after = self.get_memory_stats()
                    if stats_after.percent_used >= self.critical_threshold:
                        logger.error(
                            f"Still at critical memory level after cleanup: "
                            f"{stats_after.percent_used:.1f}%"
                        )

                        # Call critical callback if provided
                        if self.on_critical:
                            self.on_critical(stats_after)

                elif stats.percent_used >= self.warning_threshold:
                    logger.warning(
                        f"WARNING: Memory usage at {stats.percent_used:.1f}% "
                        f"({stats.used_mb:.0f}MB / {stats.total_mb:.0f}MB)"
                    )

            except Exception as e:
                logger.error(f"Error in memory monitor loop: {e}")

            # Wait before next check
            threading.Event().wait(self.check_interval)


# Global memory monitor instance
_global_monitor: Optional[MemoryMonitor] = None


def get_monitor() -> Optional[MemoryMonitor]:
    """Get the global memory monitor instance."""
    return _global_monitor


def init_monitor(
    warning_threshold: float = 80.0,
    critical_threshold: float = 90.0,
    check_interval: int = 30,
    on_critical: Optional[Callable] = None
) -> MemoryMonitor:
    """
    Initialize and start the global memory monitor.

    Args:
        warning_threshold: Memory usage % for warning alerts
        critical_threshold: Memory usage % for critical alerts
        check_interval: Seconds between memory checks
        on_critical: Callback when critical threshold exceeded

    Returns:
        The initialized MemoryMonitor instance
    """
    global _global_monitor

    if _global_monitor is not None:
        logger.warning("Memory monitor already initialized, returning existing instance")
        return _global_monitor

    _global_monitor = MemoryMonitor(
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        check_interval=check_interval,
        on_critical=on_critical
    )

    _global_monitor.start_monitoring()
    return _global_monitor


def stop_monitor():
    """Stop the global memory monitor."""
    global _global_monitor
    if _global_monitor is not None:
        _global_monitor.stop_monitoring()
        _global_monitor = None
