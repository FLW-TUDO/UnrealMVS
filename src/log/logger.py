import logging
import os
import queue
import shutil
import threading


def _input_with_timeout(prompt: str, timeout: float) -> str | None:
    """Read a line from stdin with a timeout. Returns None on timeout or non-interactive stdin."""
    import sys
    if not sys.stdin.isatty():
        return None

    result: queue.Queue[str] = queue.Queue()

    def _read():
        try:
            result.put(input())
        except EOFError:
            result.put("")

    print(prompt, end="", flush=True)
    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    return result.get_nowait()


class Logger:
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir

        if os.path.exists(log_dir):
            answer = _input_with_timeout(f"Directory '{log_dir}' exists. Overwrite? (y/N): ", timeout=5)
            if answer is None:
                answer = "n"
                print("\nNo response — keeping existing directory.")

            if answer == "y":
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
                print(f"Recreated '{log_dir}'.")
            else:
                print(f"Using existing '{log_dir}'.")
        else:
            os.makedirs(log_dir)

        logging.basicConfig(
            filename=os.path.join(log_dir, "training.log"),
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    @staticmethod
    def log(message: str, level: str = "info"):
        lvl = level.lower()
        getattr(logging, lvl, logging.info)(message)
        print(f"[{lvl.upper()}] {message}")

    @staticmethod
    def log_to_file(message: str, level: str = "info"):
        Logger.log(message, level)

    def save_copy(self, file_path: str):
        shutil.copy(file_path, self.log_dir)
