#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地模型 LLM 提供者 (HuggingFace Transformers)
仅在需要时导入 torch/transformers
"""

import os
from typing import Dict, List

from llm_provider import BaseLLMProvider, LLMResponse
from logger import get_logger

logger = get_logger()


class LocalModelProvider(BaseLLMProvider):
    """
    基于 HuggingFace Transformers 的本地大语言模型提供者
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_length: int = 4096,
    ):
        """
        初始化本地模型提供者实例

        Args:
            model_path: 本地模型路径或 HuggingFace 模型 ID
            device: 推理设备，默认为 "auto"
            load_in_8bit: 是否以 8-bit 量化加载
            load_in_4bit: 是否以 4-bit 量化加载
            max_length: 上下文最大长度
        """
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """延迟加载模型与分词器（单例模式）"""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading local model from: {self.model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs = {
            "pretrained_model_name_or_path": self.model_path,
            "trust_remote_code": True,
            "device_map": self.device,
        }

        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True
        elif self.load_in_8bit:
            kwargs["load_in_8bit"] = True
        else:
            kwargs["torch_dtype"] = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(**kwargs)
        logger.info(f"Local model loaded successfully: {self.model_path}")

    @property
    def model(self):
        """获取并按需加载底层模型实例"""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """获取并按需加载分词器实例"""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def generate(
        self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """
        使用本地模型生成单轮文本回复

        Args:
            prompt: 输入的提示词文本
            max_tokens: 最大生成的 token 数量
            temperature: 采样温度 (0.0 - 2.0)
            **kwargs: 其他支持的底层模型参数

        Returns:
            LLMResponse 实例，包含生成的文本内容及元数据
        """
        import torch

        assert self.tokenizer is not None
        assert self.model is not None

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu" and hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # 修复类型检查报错：如果返回的是对象（包含 sequences），则提取出纯 Tensor
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs.sequences

        input_len = inputs["input_ids"].shape[1]
        output_len = outputs.shape[1] - input_len
        generated_text = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        return LLMResponse(
            content=generated_text,
            model=self.model_path,
            usage={
                "prompt_tokens": input_len,
                "completion_tokens": output_len,
                "total_tokens": outputs.shape[1],
            },
            finish_reason="stop" if output_len < max_tokens else "length",
        )

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        使用本地模型生成多轮对话回复

        Args:
            messages: 包含角色和内容的对话历史列表
            max_tokens: 最大生成的 token 数量
            temperature: 采样温度 (0.0 - 2.0)
            **kwargs: 其他支持的底层模型参数

        Returns:
            LLMResponse 实例，包含生成的文本内容及元数据
        """

        assert self.tokenizer is not None

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                "\n".join(
                    [f"{msg['role'].upper()}: {msg['content']}" for msg in messages]
                )
                + "\nASSISTANT:"
            )

        return self.generate(prompt, max_tokens, temperature, **kwargs)

    def is_available(self) -> bool:
        """
        检查模型文件是否存在

        Returns:
            bool 值，存在返回 True，否则返回 False
        """
        return os.path.exists(self.model_path)
