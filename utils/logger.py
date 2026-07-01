"""
Thread-safe logging system

Design goals:
1. Singleton pattern - ensure only one log manager globally
2. Thread-local storage - use threading.local() for task context
3. Thread safety - protect all shared resources with locks
4. Idempotent initialization - setup_logging can be safely called multiple times
5. Append mode - log files use append mode to avoid overwriting
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

_thread_local = threading.local()
_init_lock = threading.Lock()
_log_manager: Optional['LogManager'] = None
_initialized = False


class LogManager:
    """Thread-safe log manager"""

    def __init__(self, log_dir: str, level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.level = getattr(logging, level.upper())

        self._handlers: Dict[str, logging.FileHandler] = {}
        self._handler_lock = threading.Lock()
        self._batch_log_file: Optional[Path] = None

        self.formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] - %(message)s',
            datefmt='%H:%M:%S'
        )

    def get_or_create_handler(self, log_key: str) -> logging.FileHandler:
        """Get or create file handler for the specified key (thread-safe)"""
        with self._handler_lock:
            if log_key not in self._handlers:
                if log_key == 'batch':
                    if self._batch_log_file is None:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        self._batch_log_file = self.log_dir / f"batch_{timestamp}.log"
                    log_file = self._batch_log_file
                else:
                    log_file = self.log_dir / f"{log_key}.log"

                handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
                handler.setFormatter(self.formatter)
                handler.setLevel(self.level)
                self._handlers[log_key] = handler

            return self._handlers[log_key]

    def close_all(self):
        """Close all handlers"""
        with self._handler_lock:
            for handler in self._handlers.values():
                handler.close()
            self._handlers.clear()


class SmartLogHandler(logging.Handler):
    """
    Smart log routing handler

    Routes logs to the correct file based on logger name and thread context:
    - logger named 'batch_runner' -> batch log file
    - other loggers -> route to task log file based on thread's task_id
    """

    def __init__(self, log_manager: LogManager):
        super().__init__()
        self.log_manager = log_manager

    def emit(self, record):
        """Route log to the correct file"""
        try:
            if record.name == 'batch_runner':
                log_key = 'batch'
            else:
                task_id = getattr(_thread_local, 'task_id', None)
                if not task_id:
                    return
                log_key = task_id

            handler = self.log_manager.get_or_create_handler(log_key)
            handler.emit(record)

        except Exception:
            self.handleError(record)


def setup_logging(log_dir: str, level: str = "INFO", console: bool = True):
    """
    Initialize logging system (idempotent, safe to call multiple times)

    Args:
        log_dir: Log directory path
        level: Log level (DEBUG/INFO/WARNING/ERROR)
        console: Whether to output to console (only effective on first call)
    """
    global _log_manager, _initialized

    with _init_lock:
        if _initialized:
            return

        _log_manager = LogManager(log_dir, level)

        root = logging.getLogger()
        root.setLevel(getattr(logging, level.upper()))
        root.handlers.clear()

        smart_handler = SmartLogHandler(_log_manager)
        smart_handler.setLevel(getattr(logging, level.upper()))
        root.addHandler(smart_handler)

        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                fmt='%(asctime)s [%(levelname)s] - %(message)s',
                datefmt='%H:%M:%S'
            ))
            root.addHandler(console_handler)

        _initialized = True


def set_task_context(task_id: str):
    """Set task context for current thread"""
    _thread_local.task_id = task_id


def clear_task_context():
    """Clear task context for current thread"""
    if hasattr(_thread_local, 'task_id'):
        del _thread_local.task_id


def remove_console():
    """Remove console handler"""
    root = logging.getLogger()
    console_handlers = [
        h for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and not isinstance(h, SmartLogHandler)
    ]
    for handler in console_handlers:
        root.removeHandler(handler)


def add_console():
    """Add console handler"""
    root = logging.getLogger()
    has_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and not isinstance(h, SmartLogHandler)
        for h in root.handlers
    )
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] - %(message)s',
            datefmt='%H:%M:%S'
        ))
        root.addHandler(console_handler)


def reset_logging():
    """Reset logging system (mainly for testing)"""
    global _log_manager, _initialized

    with _init_lock:
        if _log_manager:
            _log_manager.close_all()
            _log_manager = None

        root = logging.getLogger()
        root.handlers.clear()

        _initialized = False
