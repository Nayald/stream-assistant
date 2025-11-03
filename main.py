import os
import time

import psutil
import torch
from audio_modules import InputCapture, KokoroOutput, ParakeetConverter
from messaging import Broker, MessageType, Sink
from models import Phi4, Phi4Multimodal
from module import Module


class TextOutput(Module, Sink):
    def __init__(self, name: str):
        Module.__init__(self, name=name)
        Sink.__init__(self, name=name)

    def run(self) -> None:
        while self.is_running.is_set():
            try:
                _, _, message = self.sink_queue.get(timeout=0.1)
                if message.message_type == MessageType.TEXT:
                    print(message.content)
            except Exception:
                continue

        print(f"[{self.name}] Broker stopped.")


if __name__ == "__main__":
    affinities = list(range(16, 32, 2))
    p = psutil.Process(os.getpid())
    p.cpu_affinity(affinities)
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(len(affinities))
    torch.set_num_interop_threads(2)

    system_prompt_summary = """
    Ta tâche consiste à résumer notre conversation, en essayant de le faire de manière concise tout en conservant les détails les plus importants.
    Tu ne dois surtout pas répondre de nouveau à ce qui est dit dans la conversation, tu dois te focaliser uniquement sur le contexte de la conversation.
    La conversation peut etre précédée d'un prompt système, celui-ci représente le précédent résumé que tu m'as fourni.
    Si le précédent résumé est présent, essaies d'en tenir compte surtout si il contient des éléments pertinant pout la conversation en cours.
    Rédiges le résumé de manière à ce qu'il puisse être utilisé en tant que prompt système pour pouvoir reprendre facilement la conversation.
    Tu dois faire précéder le résumé d'une mention telle que « Résumé de la conversation : » correspondant à la langue utilisée.
    Il est très important que tu utilisies la même langue que dans la conversation.
    Évite d'écrire des commentaires ou des questions sur la conversation.
    """

    input_capture = InputCapture("input_capture", silence_threshold_db=-35)
    parakeet_converter = ParakeetConverter("parakeet_converter")
    input_capture.register_sink(parakeet_converter, MessageType.AUDIO)
    with open("assistant/system_prompt.txt") as f:
        phi4_multimodal = Phi4(system_prompt_host=f.read(), context_size_limit=16384, system_prompt_summary=system_prompt_summary, device="cpu")

    phi4_broker = Broker("phi4_broker")
    parakeet_converter.register_sink(phi4_broker, MessageType.TEXT)
    kokoro_output = KokoroOutput("kokoro_output", "ff_siwis")
    phi4_broker.register_route(parakeet_converter, kokoro_output, phi4_multimodal.process_host)

    kokoro_output.start()
    phi4_broker.start()
    parakeet_converter.start()
    input_capture.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        input_capture.stop()
        parakeet_converter.stop()
        phi4_broker.stop()
        kokoro_output.stop()

    print("done.")
