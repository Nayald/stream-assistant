from typing import Any

import torch
from kokoro import KPipeline
from messaging import MessageType


class Kokoro82M:
    def __init__(self, voice: str, device: str = "cpu") -> None:
        self.pipeline = KPipeline(lang_code="f", device=device)
        self.voice = voice

    def transcribe(self, type: MessageType, content: Any) -> list[tuple[MessageType, Any]]:
        if type != MessageType.TEXT:
            return []

        result = []
        with torch.inference_mode():
            for gs, ps, audio in self.pipeline(content, voice=self.voice):
                result.append((MessageType.TEXT, gs))
                result.append((MessageType.AUDIO, (audio, 24000)))

        return result
