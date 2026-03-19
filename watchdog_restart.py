#!/usr/bin/env python3
"""
Watchdog script for managing the DeepSeek OCR service with automatic restart on failure.

This script monitors the service and restarts it if it crashes or exits due to OOM.
It can be run directly or used as a supervisord/systemd service.

Usage:
    python watchdog_restart.py [--max-restarts MAX] [--restart-delay SECONDS]
"""
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('watchdog.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ServiceWatchdog:
    """Monitor and restart the service on failure."""

    def __init__(
        self,
        command: list[str],
        max_restarts: int = -1,  # -1 = unlimited
        restart_delay: int = 10,
        working_dir: Optional[Path] = None
    ):
        """
        Initialize the watchdog.

        Args:
            command: Command to run (as a list)
            max_restarts: Maximum number of restarts before giving up (-1 for unlimited)
            restart_delay: Seconds to wait before restart
            working_dir: Working directory for the service
        """
        self.command = command
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.working_dir = working_dir or Path(__file__).parent

        self.restart_count = 0
        self.process: Optional[subprocess.Popen] = None
        self.shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_requested = True
        if self.process:
            self.process.terminate()
        sys.exit(0)

    def _log_restart(self, exit_code: int):
        """Log restart event."""
        timestamp = datetime.now().isoformat()

        # Determine restart reason
        if exit_code == 137:
            reason = "OOM (exit code 137)"
        elif exit_code < 0:
            reason = f"Signal {-exit_code}"
        else:
            reason = f"Exit code {exit_code}"

        logger.critical(
            f"[{timestamp}] Service exited ({reason}). "
            f"Restart {self.restart_count + 1}/{self.max_restarts if self.max_restarts > 0 else '∞'}"
        )

    def _run_service(self) -> int:
        """
        Run the service and wait for it to complete.

        Returns:
            Exit code of the service
        """
        logger.info(f"Starting service: {' '.join(self.command)}")

        self.process = subprocess.Popen(
            self.command,
            cwd=self.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for process to complete
        exit_code = self.process.wait()
        self.process = None

        return exit_code

    def run(self):
        """Main watchdog loop."""
        logger.info("Watchdog started")
        logger.info(f"Max restarts: {self.max_restarts if self.max_restarts > 0 else 'unlimited'}")
        logger.info(f"Restart delay: {self.restart_delay}s")

        while not self.shutdown_requested:
            # Run the service
            exit_code = self._run_service()

            # Check if shutdown was requested
            if self.shutdown_requested:
                logger.info("Shutdown requested, not restarting")
                break

            # Check if we've exceeded max restarts
            if self.max_restarts > 0 and self.restart_count >= self.max_restarts:
                logger.error(
                    f"Exceeded maximum restarts ({self.max_restarts}), giving up"
                )
                break

            # Check if service exited cleanly (exit code 0)
            if exit_code == 0:
                logger.info("Service exited cleanly, not restarting")
                break

            # Log the restart
            self._log_restart(exit_code)
            self.restart_count += 1

            # Wait before restart
            if not self.shutdown_requested:
                logger.info(f"Waiting {self.restart_delay}s before restart...")
                time.sleep(self.restart_delay)

        logger.info("Watchdog stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Watchdog for DeepSeek OCR service"
    )
    parser.add_argument(
        '--max-restarts',
        type=int,
        default=-1,
        help='Maximum number of restarts before giving up (-1 for unlimited)'
    )
    parser.add_argument(
        '--restart-delay',
        type=int,
        default=10,
        help='Seconds to wait before restart (default: 10)'
    )
    parser.add_argument(
        '--command',
        type=str,
        default='python serve_pdf.py',
        help='Command to run the service (default: "python serve_pdf.py")'
    )

    args = parser.parse_args()

    # Parse command
    command = args.command.split()

    # Create and run watchdog
    watchdog = ServiceWatchdog(
        command=command,
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay
    )

    try:
        watchdog.run()
    except KeyboardInterrupt:
        logger.info("Watchdog interrupted")
        sys.exit(0)


if __name__ == '__main__':
    main()
