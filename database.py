import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Database path
raw_path = os.getenv("SQLITE_PATH", "task_status.db")
DB_PATH = Path(raw_path)

# Handle case where user provides a directory path
if DB_PATH.is_dir() or raw_path.endswith('/') or raw_path.endswith('\\'):
    DB_PATH = DB_PATH / "task_status.db"

print(f"Using Database Path: {DB_PATH.absolute()}")

# Thread-local storage for database connections
_thread_local = threading.local()


def get_connection():
    """Get thread-local database connection"""
    if not hasattr(_thread_local, "connection"):
        _thread_local.connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _thread_local.connection.row_factory = sqlite3.Row
    return _thread_local.connection


@contextmanager
def get_db():
    """Context manager for database operations"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_database():
    """Initialize the database with required tables"""
    # Ensure the parent directory exists
    try:
        if not DB_PATH.parent.exists():
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to create database directory {DB_PATH.parent}: {e}")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_message TEXT,
                filename TEXT,
                total_pages INTEGER,
                processed_pages INTEGER,
                file_hash TEXT
            )
        """)

        # Migration: Add file_hash column if it doesn't exist (for existing databases)
        cursor.execute("PRAGMA table_info(tasks)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'file_hash' not in columns:
            cursor.execute("ALTER TABLE tasks ADD COLUMN file_hash TEXT")

        # Create index on status for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)
        """)

        # Create index on created_at for cleanup queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON tasks(created_at)
        """)

        # Create index on file_hash for duplicate detection
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_hash ON tasks(file_hash)
        """)

        conn.commit()


def create_task(job_id: str, filename: str, file_hash: Optional[str] = None) -> bool:
    """
    Create a new task in the database

    Args:
        job_id: Unique job identifier
        filename: Original filename
        file_hash: SHA-256 hash of the file content

    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            cursor.execute("""
                INSERT INTO tasks (job_id, status, created_at, updated_at, filename, processed_pages, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (job_id, "pending", now, now, filename, 0, file_hash))
            return True
    except Exception as e:
        print(f"Error creating task: {e}")
        return False


def update_task_status(
    job_id: str,
    status: str,
    error_message: Optional[str] = None,
    total_pages: Optional[int] = None,
    processed_pages: Optional[int] = None
) -> bool:
    """
    Update task status

    Args:
        job_id: Job identifier
        status: New status (pending, processing, completed, failed)
        error_message: Error message if failed
        total_pages: Total number of pages
        processed_pages: Number of processed pages

    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()

            # Build update query dynamically
            updates = ["status = ?", "updated_at = ?"]
            params = [status, now]

            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)

            if total_pages is not None:
                updates.append("total_pages = ?")
                params.append(total_pages)

            if processed_pages is not None:
                updates.append("processed_pages = ?")
                params.append(processed_pages)

            params.append(job_id)

            query = f"UPDATE tasks SET {', '.join(updates)} WHERE job_id = ?"
            cursor.execute(query, params)
            return True
    except Exception as e:
        print(f"Error updating task status: {e}")
        return False


def get_task_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get task status by job ID

    Args:
        job_id: Job identifier

    Returns:
        Dictionary with task information or None if not found
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT job_id, status, created_at, updated_at, error_message,
                       filename, total_pages, processed_pages, file_hash
                FROM tasks
                WHERE job_id = ?
            """, (job_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    except Exception as e:
        print(f"Error getting task status: {e}")
        return None


def get_completed_task_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """
    Find a completed task with the given file hash

    Args:
        file_hash: SHA-256 hash of file content

    Returns:
        Task dictionary or None
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            # Find the most recent completed task with this hash
            cursor.execute("""
                SELECT job_id, status, created_at, updated_at, error_message,
                       filename, total_pages, processed_pages, file_hash
                FROM tasks
                WHERE file_hash = ? AND status = 'completed'
                ORDER BY created_at DESC
                LIMIT 1
            """, (file_hash,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    except Exception as e:
        print(f"Error getting completed task by hash: {e}")
        return None


def delete_task(job_id: str) -> bool:
    """
    Delete a task from the database

    Args:
        job_id: Job identifier

    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tasks WHERE job_id = ?", (job_id,))
            return True
    except Exception as e:
        print(f"Error deleting task: {e}")
        return False


def get_all_tasks(status: Optional[str] = None) -> list[Dict[str, Any]]:
    """
    Get all tasks, optionally filtered by status

    Args:
        status: Optional status filter

    Returns:
        List of task dictionaries
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("""
                    SELECT job_id, status, created_at, updated_at, error_message,
                           filename, total_pages, processed_pages, file_hash
                    FROM tasks
                    WHERE status = ?
                    ORDER BY created_at DESC
                """, (status,))
            else:
                cursor.execute("""
                    SELECT job_id, status, created_at, updated_at, error_message,
                           filename, total_pages, processed_pages, file_hash
                    FROM tasks
                    ORDER BY created_at DESC
                """)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"Error getting all tasks: {e}")
        return []


def cleanup_old_tasks(days: int = 7) -> int:
    """
    Clean up tasks older than specified days

    Args:
        days: Number of days to keep

    Returns:
        Number of tasks deleted
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)

            cursor.execute("""
                DELETE FROM tasks
                WHERE created_at < ?
            """, (cutoff_date.isoformat(),))

            return cursor.rowcount
    except Exception as e:
        print(f"Error cleaning up old tasks: {e}")
        return 0

