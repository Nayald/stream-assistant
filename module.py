import abc
import threading


class Module(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self.thread: threading.Thread | None = None
        self.is_running = threading.Event()

    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    def start(self) -> None:
        if not self.is_running.is_set():
            self.is_running.set()
            self.thread = threading.Thread(target=self.run, name=self.name, daemon=True)
            self.thread.start()
            print(f"[{self.name}] started.")

    def stop(self) -> None:
        if self.is_running.is_set() and self.thread:
            self.is_running.clear()
            self.thread.join()
            self.thread = None
            print(f"[{self.name}] stopped.")
