from typing import Any

import nemo.collections.asr as nemo_asr
import torch
from messaging import Message, MessageType
from nemo.utils import logging as nemo_logging

nemo_logging.set_verbosity(nemo_logging.ERROR)


class Parakeet:
    def __init__(self, device: str = "cpu"):
        self.model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location=torch.device(device))
        # self.model.eval()
        # self.model = torch.compile(self.model)

    def transcribe(self, type: MessageType, content: Any) -> list[tuple[MessageType, Any]]:
        if type != MessageType.AUDIO:
            return [(MessageType.NONE, None)]

        with torch.inference_mode():
            transcription = self.model.transcribe([content[0]], batch_size=1)[0].text

        if not transcription:
            return [(MessageType.NONE, None)]

        print("(transcription)", transcription)
        return [(MessageType.TEXT, transcription)]
