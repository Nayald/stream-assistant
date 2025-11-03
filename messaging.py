from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, auto, unique
from itertools import count
from queue import PriorityQueue
from typing import Any

from module import Module


@dataclass
class Producer:
    name: str
    priority: int


@unique
class MessageType(IntEnum):
    NONE = auto()
    TEXT = auto()
    AUDIO = auto()
    IMAGE = auto()


@dataclass
class Message:
    producer: Producer
    message_type: MessageType
    content: Any


class Sink:
    _counter = count()

    def __init__(self, name: str) -> None:
        self.name = name
        self.sink_queue: PriorityQueue[tuple[int, int, Message]] = PriorityQueue()

    def receive_message(self, message: Message) -> None:
        self.sink_queue.put((message.producer.priority, next(self._counter), message))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Source(Producer):
    def __init__(self, name: str, priority: int = 0) -> None:
        super().__init__(name=name, priority=priority)
        self.sinks: dict[MessageType, set[Sink]] = defaultdict(set)

    def register_sink(self, sink: Sink, message_type: MessageType) -> None:
        if sink not in self.sinks[message_type]:
            self.sinks[message_type].add(sink)

    def unregister_sink(self, sink: Sink, message_type: MessageType) -> None:
        if sink in self.sinks[message_type]:
            self.sinks[message_type].discard(sink)

    def send_message(self, message_type: MessageType, content: Any) -> None:
        for sink in self.sinks[message_type]:
            sink.receive_message(Message(self, message_type, content))


# class Broker(Sink, Producer):
#    def __init__(self, name: str, priority: int = 0) -> None:
#        Sink.__init__(self, name=name)
#        Producer.__init__(self, name=name, priority=priority)
#        self.sinks: dict[Producer, dict[Sink, Callable[[Message], Message]]] = defaultdict(dict)
#
#    def register_route(self, source: Producer, sink: Sink, transform: Callable[[Message], Message] | None = None) -> None:
#        self.sinks[source][sink] = transform if transform else lambda msg: msg
#
#    def unregister_route(self, source: Producer, sink: Sink) -> None:
#        if sink in self.sinks[source]:
#            del self.sinks[source][sink]
#
#    def send_message(self, source: Producer, message_type: MessageType, content: Any) -> None:
#        for sink in self.sinks[source]:
#            sink.receive_message(Message(self, message_type, content))


class Broker(Module, Sink, Producer):
    def __init__(self, name: str, priority: int = 0) -> None:
        Module.__init__(self, name=name)
        Sink.__init__(self, name=name)
        Producer.__init__(self, name=name, priority=priority)
        self.sinks: dict[Producer, dict[Sink, Callable[[Message], tuple[MessageType, Any]]]] = defaultdict(dict)

    def register_route(self, source: Producer, sink: Sink, transform: Callable[[Message], tuple[MessageType, Any]] | None = None) -> None:
        self.sinks[source][sink] = transform if transform else lambda msg: (msg.message_type, msg.content)

    def unregister_route(self, source: Producer, sink: Sink) -> None:
        if sink in self.sinks[source]:
            del self.sinks[source][sink]

    def run(self) -> None:
        print(f"[{self.name}] Broker started.")
        while self.is_running.is_set():
            try:
                _, _, message = self.sink_queue.get(timeout=0.1)
                for sink in self.sinks[message.producer]:
                    message_type, content = self.sinks[message.producer][sink](message)
                    if message_type != MessageType.NONE:
                        sink.receive_message(Message(self, message_type, content))
            except Exception:
                continue

        print(f"[{self.name}] Broker stopped.")
