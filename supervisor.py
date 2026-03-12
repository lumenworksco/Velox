"""Task supervisor for async mode — restarts failed coroutines with backoff."""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# Maximum restart delay (seconds)
_MAX_BACKOFF = 300
# Initial restart delay (seconds)
_INITIAL_BACKOFF = 5


class TaskSupervisor:
    """Supervises async tasks with automatic restart on failure.

    Each supervised task is restarted on crash with exponential backoff
    (5s → 10s → 20s → ... → 300s cap).
    """

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._restart_counts: dict[str, int] = {}
        self._running = True
        self._notify_fn = None  # Optional async notification callback

    def set_notify(self, fn):
        """Set async callback for crash alerts: fn(name, error, restart_count)."""
        self._notify_fn = fn

    async def supervised_task(self, name: str, coro_fn, *args, **kwargs):
        """Run *coro_fn(*args, **kwargs)* forever, restarting on failure.

        Parameters
        ----------
        name : str
            Human-readable task name for logging.
        coro_fn : callable
            Async callable that returns a coroutine.
        """
        self._restart_counts[name] = 0

        while self._running:
            try:
                logger.info(f"[supervisor] Starting task: {name}")
                await coro_fn(*args, **kwargs)
                # If the coroutine returns normally, don't restart
                logger.info(f"[supervisor] Task {name} completed normally")
                break
            except asyncio.CancelledError:
                logger.info(f"[supervisor] Task {name} cancelled")
                break
            except Exception as exc:
                self._restart_counts[name] += 1
                count = self._restart_counts[name]
                delay = min(_INITIAL_BACKOFF * (2 ** (count - 1)), _MAX_BACKOFF)

                logger.error(
                    f"[supervisor] Task {name} crashed (restart #{count}): {exc}",
                    exc_info=True,
                )

                # Send notification if available
                if self._notify_fn:
                    try:
                        await self._notify_fn(name, exc, count)
                    except Exception:
                        pass

                logger.info(f"[supervisor] Restarting {name} in {delay}s...")
                await asyncio.sleep(delay)

    async def launch(self, name: str, coro_fn, *args, **kwargs) -> asyncio.Task:
        """Launch a supervised task and track it."""
        task = asyncio.create_task(
            self.supervised_task(name, coro_fn, *args, **kwargs),
            name=f"supervised-{name}",
        )
        self._tasks[name] = task
        return task

    async def stop_all(self):
        """Cancel all supervised tasks."""
        self._running = False
        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"[supervisor] Cancelled task: {name}")
        # Wait for all tasks to finish
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    def get_status(self) -> dict:
        """Return status of all supervised tasks."""
        return {
            name: {
                "running": not task.done(),
                "restart_count": self._restart_counts.get(name, 0),
            }
            for name, task in self._tasks.items()
        }
