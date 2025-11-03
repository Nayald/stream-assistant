import time
from enum import StrEnum, auto, unique

import keyboard
import nemo.collections.asr as nemo_asr
import numpy as np
import sounddevice as sd
import torch
from kokoro import KPipeline
from messaging import MessageType, Sink, Source
from module import Module
from nemo.utils import logging as nemo_logging


@unique
class InputCaptureMode(StrEnum):
    KEY = auto()
    THRESHOLD = auto()


class InputCapture(Module, Source):
    def __init__(
        self,
        name: str,
        samplerate: int = 16000,
        channels: int = 1,
        blocksize: int = 1024,
        silence_threshold_db: float = -40,
        silence_max_duration: float = 1.0,
        key_record: str = "$",
        key_mode_switch: str = "²",
    ) -> None:
        Module.__init__(self, name=name)
        Source.__init__(self, name=name)
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.silence_threshold_db = silence_threshold_db
        self.key_record = key_record
        self.key_mode_switch = key_mode_switch

        self.mode = InputCaptureMode.KEY
        self.max_silence_chunks = int(silence_max_duration * samplerate / blocksize)
        self.blocks_since_silence = self.max_silence_chunks
        self.buffer = []

    def run(self) -> None:
        print(f"[{self.name}] AudioModule started. Mode: {self.mode}")
        last_mode_switch = 0

        with sd.InputStream(samplerate=self.samplerate, channels=self.channels, blocksize=self.blocksize) as stream:
            while self.is_running.is_set():
                # handle mode switch
                if keyboard.is_pressed(self.key_mode_switch):
                    now = time.time()
                    if now - last_mode_switch > 0.5:  # debounce
                        self.mode = InputCaptureMode.THRESHOLD if self.mode == InputCaptureMode.KEY else InputCaptureMode.KEY
                        print(f"[{self.name}] Mode switched to: {self.mode}")
                        last_mode_switch = now

                # read a block
                block, _ = stream.read(self.blocksize)

                # handle recording
                if self.mode == "key" and keyboard.is_pressed(self.key_record):
                    self.buffer.append(block)
                elif self.mode == "threshold" and 20 * np.log10(np.sqrt(np.mean(block**2)) + 1e-8) > self.silence_threshold_db:
                    self.buffer.append(block)
                    self.blocks_since_silence = 0
                elif self.mode == "threshold" and self.blocks_since_silence < self.max_silence_chunks:
                    self.buffer.append(block)
                    self.blocks_since_silence += 1
                elif len(self.buffer) > 0:
                    audio_data = np.concatenate(self.buffer, axis=0)
                    audio_data = np.squeeze(audio_data)
                    print(audio_data.shape)
                    self.send_message(message_type=MessageType.AUDIO, content=(audio_data, self.samplerate))
                    self.buffer = []

        print(f"[{self.name}] AudioModule stopped.")


nemo_logging.set_verbosity(nemo_logging.ERROR)


class ParakeetConverter(Module, Sink, Source):
    def __init__(self, name: str, device: str = "cpu"):
        Module.__init__(self, name=name)
        Sink.__init__(self, name=name)
        Source.__init__(self, name=name)
        self.model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location=torch.device(device))
        self.model.eval()
        self.model = torch.compile(self.model)

    def run(self) -> None:
        print(f"[{self.name}] Broker started.")
        while self.is_running.is_set():
            try:
                _, _, message = self.sink_queue.get(timeout=0.1)
            except Exception:
                continue

            if message.message_type != MessageType.AUDIO:
                continue

            print("[start reading]")
            with torch.inference_mode():
                transcription = self.model.transcribe([message.content[0]], batch_size=1)[0].text

            print(f"[{self.name}] (transcription)", transcription)
            if transcription:
                self.send_message(MessageType.TEXT, transcription)

        print(f"[{self.name}] AudioModule stopped.")


class KokoroOutput(Module, Sink):
    def __init__(self, name: str, voice: str, device: str = "cpu") -> None:
        Module.__init__(self, name=name)
        Sink.__init__(self, name=name)
        self.pipeline = KPipeline(lang_code="f", device=device)
        self.voice = voice

    def run(self) -> None:
        print(f"[{self.name}] Broker started.")
        while self.is_running.is_set():
            try:
                _, _, message = self.sink_queue.get(timeout=0.1)
            except Exception:
                continue

            if message.message_type != MessageType.TEXT:
                continue

            print("[start reading]")
            with torch.inference_mode():
                for gs, ps, audio in self.pipeline(message.content, voice=self.voice):
                    print(gs)
                    sd.play(audio, samplerate=24000)
                    sd.wait()

            print("[end reading]")


if __name__ == "__main__":
    audio_module = InputCapture("AudioCapture", silence_threshold_db=-35)
    audio_module.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        audio_module.stop()

    print("done.")
