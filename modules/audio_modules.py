import time
from enum import IntEnum, auto, unique

import keyboard
import numpy as np
import sounddevice as sd
import tqdm
from messaging import MessageType, Sink, Source
from modules.module import Module


@unique
class InputDeviceMode(IntEnum):
    KEY = auto()
    THRESHOLD = auto()


class InputDevice(Module, Source):
    def __init__(
        self,
        name: str,
        samplerate: int = 16000,
        channels: int = 1,
        blocksize: int = 1024,
        silence_threshold_db: float = -40,
        silence_max_duration: float = 1.0,
        key_record: str = "²",
        key_mode_switch: str = "$",
    ) -> None:
        Module.__init__(self, name=name)
        Source.__init__(self, name=name)
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.silence_threshold_db = silence_threshold_db
        self.key_record = key_record
        self.key_mode_switch = key_mode_switch

        self.mode = InputDeviceMode.KEY
        self.max_silence_chunks = int(silence_max_duration * samplerate / blocksize)
        self.blocks_since_silence = self.max_silence_chunks
        self.buffer = []

    def run(self) -> None:
        print(f"[{self.name}] mode {'threshold' if self.mode == InputDeviceMode.THRESHOLD else 'key'}")
        last_mode_switch = 0

        with sd.InputStream(samplerate=self.samplerate, channels=self.channels, blocksize=self.blocksize) as stream:
            while self.is_running.is_set():
                # handle mode switch
                if keyboard.is_pressed(self.key_mode_switch):
                    now = time.time()
                    if now - last_mode_switch > 0.5:  # debounce
                        self.mode = InputDeviceMode.THRESHOLD if self.mode == InputDeviceMode.KEY else InputDeviceMode.KEY
                        print(f"[{self.name}] Mode switched to {'threshold' if self.mode == InputDeviceMode.THRESHOLD else 'key'}")
                        last_mode_switch = now

                # read a block
                block, _ = stream.read(self.blocksize)

                # handle recording
                if self.mode == InputDeviceMode.KEY and keyboard.is_pressed(self.key_record):
                    self.buffer.append(block)
                elif self.mode == InputDeviceMode.THRESHOLD and (
                    (is_above := 20 * np.log10(np.sqrt(np.mean(block**2))) > self.silence_threshold_db) or self.blocks_since_silence < self.max_silence_chunks
                ):
                    self.buffer.append(block)
                    self.blocks_since_silence = 0 if is_above else (self.blocks_since_silence + 1)
                elif len(self.buffer) > 0:
                    audio_data = np.concatenate(self.buffer, axis=0)
                    audio_data = np.squeeze(audio_data)
                    print(audio_data.shape)
                    self.send_message(message_type=MessageType.AUDIO, content=(audio_data, self.samplerate))
                    self.buffer = []


class OutputDevice(Module, Sink):
    def __init__(self, name: str) -> None:
        Module.__init__(self, name=name)
        Sink.__init__(self, name=name)

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

            if message.type != MessageType.AUDIO:
                continue

            sd.play(message.content, samplerate=24000)
            sd.wait()
