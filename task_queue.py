"""
In-memory task queue for sequential PDF processing
"""
import asyncio
import threading
from queue import Queue, Empty
from typing import Callable, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TaskQueue:
    """
    In-memory FIFO queue for managing tasks sequentially.
    Thread-safe implementation for background task processing.
    """

    def __init__(self, max_workers: int = 1):
        """
        Initialize the task queue

        Args:
            max_workers: Number of concurrent workers (default: 1 for sequential processing)
        """
        self.queue = Queue()
        self.max_workers = max_workers
        self.workers = []
        self.running = False
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self):
        """Start the queue workers"""
        if self.running:
            logger.warning("Queue workers already running")
            return

        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker,
                name=f"TaskQueueWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {self.max_workers} queue worker(s)")

    def stop(self):
        """Stop the queue workers"""
        self.running = False
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        self.workers.clear()
        logger.info("Stopped queue workers")

    def add_task(self, task_id: str, task_func: Callable, *args, **kwargs):
        """
        Add a task to the queue

        Args:
            task_id: Unique identifier for the task
            task_func: Function to execute
            *args: Positional arguments for task_func
            **kwargs: Keyword arguments for task_func
        """
        task = {
            'task_id': task_id,
            'func': task_func,
            'args': args,
            'kwargs': kwargs,
            'added_at': datetime.now().isoformat()
        }

        with self._lock:
            self.active_tasks[task_id] = {
                'status': 'queued',
                'added_at': task['added_at'],
                'started_at': None,
                'completed_at': None
            }

        self.queue.put(task)
        logger.info(f"Task {task_id} added to queue. Queue size: {self.queue.qsize()}")

    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a task

        Args:
            task_id: Task identifier

        Returns:
            Task information dict or None if not found
        """
        with self._lock:
            return self.active_tasks.get(task_id)

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all tasks"""
        with self._lock:
            return self.active_tasks.copy()

    def _worker(self):
        """Worker thread that processes tasks from the queue"""
        logger.info(f"Worker {threading.current_thread().name} started")

        while self.running:
            try:
                # Get task from queue with timeout to allow checking running flag
                task = self.queue.get(timeout=1)

                task_id = task['task_id']
                task_func = task['func']
                args = task['args']
                kwargs = task['kwargs']

                # Update task status to processing
                with self._lock:
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id]['status'] = 'processing'
                        self.active_tasks[task_id]['started_at'] = datetime.now().isoformat()

                logger.info(f"Processing task {task_id}")

                try:
                    # Execute the task
                    task_func(*args, **kwargs)

                    # Update task status to completed
                    with self._lock:
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id]['status'] = 'completed'
                            self.active_tasks[task_id]['completed_at'] = datetime.now().isoformat()

                    logger.info(f"Task {task_id} completed successfully")

                except Exception as e:
                    # Update task status to failed
                    with self._lock:
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id]['status'] = 'failed'
                            self.active_tasks[task_id]['error'] = str(e)
                            self.active_tasks[task_id]['completed_at'] = datetime.now().isoformat()

                    logger.error(f"Task {task_id} failed: {e}", exc_info=True)

                finally:
                    self.queue.task_done()

            except Empty:
                # No task available, continue waiting
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)

        logger.info(f"Worker {threading.current_thread().name} stopped")

    def remove_task_info(self, task_id: str):
        """
        Remove task information from active tasks

        Args:
            task_id: Task identifier
        """
        with self._lock:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                logger.info(f"Removed task info for {task_id}")


# Global queue instance
_global_queue: Optional[TaskQueue] = None


def get_queue(max_workers: int = 1) -> TaskQueue:
    """
    Get or create the global task queue instance

    Args:
        max_workers: Number of concurrent workers (only used on first call)

    Returns:
        TaskQueue instance
    """
    global _global_queue
    if _global_queue is None:
        _global_queue = TaskQueue(max_workers=max_workers)
        _global_queue.start()
    return _global_queue


def shutdown_queue():
    """Shutdown the global queue"""
    global _global_queue
    if _global_queue is not None:
        _global_queue.stop()
        _global_queue = None
