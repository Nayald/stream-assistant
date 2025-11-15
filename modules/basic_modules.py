from collections import defaultdict
from collections.abc import Callable
from typing import Any

import tqdm
from messaging import Message, MessageType, Producer, Sink, Source
from modules.module import Module


class BasicSink(Module, Sink):
    def __init__(self, name: str, transform: Callable[[MessageType, Any], None]) -> None:
        Module.__init__(self, name=name)
        Sink.__init__(self, name=name)
        self.transform = transform

    def run(self) -> None:
        while self.is_running.is_set():
            try:
                _, _, message = self.sink_queue.get(timeout=0.1)
                if message.producer in self.last_ids:
                    if message.id < self.last_ids[message.producer]:
                        tqdm.tqdm.write(f"[{self.name}] drop old message from [{message.producer.name}]")
                        continue

                    self.last_ids[message.producer] = message.id
            except Exception:
                continue

            self.transform(message.type, message.content)


class BasicProxy(Module, Sink, Source):
    def __init__(self, name: str, transform: Callable[[MessageType, Any], list[tuple[MessageType, Any]]], priority: int = 0) -> None:
        Module.__init__(self, name=name)
        Sink.__init__(self, name=name)
        Source.__init__(self, name=name, priority=priority)
        self.transform = transform

    def run(self) -> None:
        while self.is_running.is_set():
            try:
                _, _, message = self.sink_queue.get(timeout=0.1)
                if message.producer in self.last_ids:
                    if message.id < self.last_ids[message.producer]:
                        tqdm.tqdm.write(f"[{self.name}] drop old message from [{message.producer.name}]")
                        continue

                    self.last_ids[message.producer] = message.id
            except Exception:
                continue

            for type, content in self.transform(message.type, message.content):
                self.send_message(type, content)


class BasicBroker(Module, Sink, Producer):
    def __init__(self, name: str, priority: int = 0) -> None:
        Module.__init__(self, name=name)
        Sink.__init__(self, name=name)
        Producer.__init__(self, name=name, priority=priority)
        self.sinks: dict[Producer, dict[Sink, Callable[[MessageType, Any], list[tuple[MessageType, Any]]]]] = defaultdict(dict)

    def register_route(self, source: Producer, sink: Sink, transform: Callable[[MessageType, Any], list[tuple[MessageType, Any]]]) -> None:
        self.sinks[source][sink] = transform

    def unregister_route(self, source: Producer, sink: Sink) -> None:
        self.sinks[source].pop(sink, None)

    def run(self) -> None:
        while self.is_running.is_set():
            try:
                _, _, message = self.sink_queue.get(timeout=0.1)
                for sink in self.sinks[message.producer]:
                    result = self.sinks[message.producer][sink](message.type, message.content)
                    for type, content in result:
                        if type != MessageType.NONE:
                            sink.receive_message(Message(self, next(self.msg_counter), type, content))
            except Exception:
                continue
