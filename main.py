import os
import time
from typing import Any

import psutil
import sounddevice as sd
import torch
from messaging import MessageType
from models.asr.parakeet import Parakeet
from models.llm.phi4 import Phi4
from models.tts.kokoro82M import Kokoro82M
from modules.audio_modules import InputDevice, OutputDevice
from modules.basic_modules import BasicBroker, BasicProxy, BasicSink


def simple_tts_output(type: MessageType, content: Any) -> None:
    match type:
        case MessageType.TEXT:
            print("(Reading)", content)
        case MessageType.AUDIO:
            sd.play(content[0], content[1])
            sd.wait()
        case _:
            pass


if __name__ == "__main__":
    affinities = list(range(16, 32, 2))
    p = psutil.Process(os.getpid())
    p.cpu_affinity(affinities)
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(len(affinities))
    torch.set_num_interop_threads(2)
    # torch.cuda.set_stream(torch.cuda.Stream(priority=10))

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

    asr = Parakeet()
    with open("assistant/system_prompt.txt") as f:
        llm = Phi4(system_prompt_host=f.read(), max_new_tokens=512, context_size_limit=16384, system_prompt_summary=system_prompt_summary, device="cpu")

    tts = Kokoro82M(voice="ff_siwis")

    input_device = InputDevice("input_device", silence_threshold_db=-35)
    parakeet_converter = BasicProxy("parakeet_converter", asr.transcribe)
    input_device.register_sink(parakeet_converter, MessageType.AUDIO)

    phi4_broker = BasicBroker("phi4_broker")
    parakeet_converter.register_sink(phi4_broker, MessageType.TEXT)

    kokoro_converter = BasicProxy("kokoro_converter", tts.transcribe)
    phi4_broker.register_route(parakeet_converter, kokoro_converter, llm.process_host)

    output_device = BasicSink("output_device", simple_tts_output)
    kokoro_converter.register_sink(output_device, MessageType.TEXT)
    kokoro_converter.register_sink(output_device, MessageType.AUDIO)

    output_device.start()
    kokoro_converter.start()
    phi4_broker.start()
    parakeet_converter.start()
    input_device.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        input_device.stop()
        parakeet_converter.stop()
        phi4_broker.stop()
        kokoro_converter.stop()
        output_device.stop()

    print("done.")
