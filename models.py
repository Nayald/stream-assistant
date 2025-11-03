import threading
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import tqdm
from messaging import Message, MessageType
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, GenerationConfig, TextIteratorStreamer


class Phi4:
    def __init__(
        self,
        model_path: str = "microsoft/Phi-4-mini-instruct",
        system_prompt_host: str = "",
        system_prompt_chat: str = "",
        context_size_limit: int = 100000,
        system_prompt_summary: str = "",
        device: str = "auto",
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            attn_implementation="sdpa" if device == "cpu" else "flash_attention_2",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        self.history_host: list[dict[str, str]] = [{"role": "system", "content": system_prompt_host}]
        self.history_chat: dict[str, list[dict[str, str]]] = defaultdict(lambda: [{"role": "system", "content": system_prompt_chat}])
        self.context_size_limit = context_size_limit
        self.system_prompt_summary = system_prompt_summary

    @staticmethod
    def progress(streamer: TextIteratorStreamer, description: str, max_new_tokens: int) -> None:
        for _ in tqdm.tqdm(streamer, desc=description, total=max_new_tokens + 1):
            pass

    def summurize(self, history: list[dict[str, str]]) -> str:
        summary = [{"role": "system", "content": self.system_prompt_summary}] + history
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        streamer_thread = threading.Thread(target=Phi4.progress, args=(streamer, "Summurizing", 512))
        streamer_thread.start()
        inputs = self.tokenizer.apply_chat_template(summary, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=1024, generation_config=self.generation_config, streamer=streamer)
        streamer_thread.join()
        text = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
        print(text, "\n")
        return text

    def process_host(self, message: Message) -> tuple[MessageType, Any]:
        if message.message_type != MessageType.TEXT:
            return MessageType.NONE, None

        self.history_host.append({"role": "user", "content": message.content})
        try:
            with torch.inference_mode():
                inputs = self.tokenizer.apply_chat_template(self.history_host, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(
                    self.model.device
                )
                if len(inputs["input_ids"][0]) > self.context_size_limit:
                    has_summary = self.history_host[1]["role"] == "system"
                    self.history_host = [
                        self.history_host[0],
                        {"role": "system", "content": self.summurize(self.history_host[1 : max(3 + has_summary, len(self.history_host) - 7)])},
                        *self.history_host[max(3 + has_summary, len(self.history_host) - 7) :],
                    ]
                    inputs = self.tokenizer.apply_chat_template(self.history_host, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(
                        self.model.device
                    )

                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                streamer_thread = threading.Thread(target=Phi4.progress, args=(streamer, "Generating", 512))
                streamer_thread.start()
                out = self.model.generate(
                    **inputs, max_new_tokens=512, generation_config=self.generation_config, do_sample=True, temperature=0.7, streamer=streamer
                )
                streamer_thread.join()
                text = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
                self.history_host.append({"role": "assistant", "content": text})
                return MessageType.TEXT, text
        except Exception as e:
            print("error:", e)
            return MessageType.NONE, None

    def process_chat(self, message: Message) -> tuple[MessageType, Any]:
        if message.message_type != MessageType.TEXT:
            return MessageType.NONE, None

        user, comment = message.content.split(":")
        self.history_chat[user].append({"role": "user", "content": comment.strip()})
        try:
            with torch.inference_mode():
                inputs = self.tokenizer.apply_chat_template(self.history_host, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(
                    self.model.device
                )
                out = self.model.generate(**inputs, max_new_tokens=256, generation_config=self.generation_config, do_sample=True, temperature=0.7)
                text = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
        except Exception as e:
            print("error:", e)
            return MessageType.NONE, None

        self.history_host.append({"role": "assistant", "content": text})
        return MessageType.TEXT, text


class Phi4Multimodal(Phi4):
    def __init__(
        self,
        model_path: str = "Lexius/Phi-4-multimodal-instruct",
        system_prompt_host: str = "",
        system_prompt_chat: str = "",
        context_size_limit: int = -1,
        system_prompt_summary: str = "",
        device: str = "auto",
    ) -> None:
        Phi4.__init__(self, model_path, system_prompt_host, system_prompt_chat, context_size_limit, system_prompt_summary, device)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        self.audios: list[tuple[np.ndarray, int]] = []

    def process_host(self, message: Message) -> tuple[MessageType, Any]:
        match message.message_type:
            case MessageType.AUDIO:
                self.audios.append(message.content)
                self.history_host.append({"role": "user", "content": f"<|audio_{len(self.audios)}|>"})
            case MessageType.TEXT:
                self.history_host.append({"role": "user", "content": message.content})
            case _:
                return MessageType.NONE, None

        try:
            with torch.inference_mode():
                text = self.tokenizer.apply_chat_template(self.history_host, add_generation_prompt=True, tokenize=False)
                if self.audios:
                    inputs = self.processor(text=text, audios=self.audios, return_tensors="pt").to(self.model.device)
                else:
                    inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)

                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                streamer_thread = threading.Thread(target=Phi4.progress, args=(streamer, "Generating", 512))
                streamer_thread.start()
                out = self.model.generate(
                    **inputs, max_new_tokens=512, generation_config=self.generation_config, do_sample=True, temperature=0.7, streamer=streamer
                )
                streamer_thread.join()
                text = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
        except Exception as e:
            print("error:", e)
            return MessageType.NONE, None

        self.history_host.append({"role": "assistant", "content": text})
        return MessageType.TEXT, text
