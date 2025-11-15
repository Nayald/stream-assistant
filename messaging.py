from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto, unique
from itertools import count
from queue import PriorityQueue
from typing import Any

global_uid = count()


def new_uid() -> int:
    return next(global_uid)


@unique
class MessageType(IntEnum):
    NONE = auto()
    TEXT = auto()
    AUDIO = auto()
    IMAGE = auto()
    INFO = auto()


@dataclass
class MessagingNode:
    uid: int = field(default_factory=new_uid, init=False)
    name: str = field(default="")

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MessagingNode) and self.uid == other.uid


@dataclass(eq=False)
class Producer(MessagingNode):
    priority: int = field(default=0)
    msg_counter: count = field(default_factory=count, init=False)


@dataclass
class Message:
    producer: Producer
    id: int
    type: MessageType
    content: Any


class Sink(MessagingNode):
    def __init__(self, name: str) -> None:
        MessagingNode.__init__(self, name=name)
        self.sink_queue: PriorityQueue[tuple[int, int, Message]] = PriorityQueue()
        self.queue_counter = count()
        self.last_ids: dict[Producer, int] = {}

    def receive_message(self, message: Message) -> None:
        self.sink_queue.put((message.producer.priority, next(self.queue_counter), message))

    def keep_last_from(self, producer: Producer) -> None:
        self.last_ids[producer] = -1

    def keep_all_from(self, producer: Producer) -> None:
        if producer in self.last_ids:
            del self.last_ids[producer]

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sink) and self.uid == other.uid


class Source(Producer):
    def __init__(self, name: str, priority: int = 0) -> None:
        Producer.__init__(self, name=name, priority=priority)
        self.sinks: dict[MessageType, set[Sink]] = defaultdict(set)

    def register_sink(self, sink: Sink, message_type: MessageType) -> None:
        self.sinks[message_type].add(sink)

    def unregister_sink(self, sink: Sink, message_type: MessageType) -> None:
        self.sinks[message_type].discard(sink)

    def send_message(self, message_type: MessageType, content: Any) -> None:
        msg = Message(self, next(self.msg_counter), message_type, content)
        for sink in self.sinks[message_type]:
            sink.receive_message(msg)
