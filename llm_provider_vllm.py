#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 高性能推理提供者
仅在需要时导入 vllm
"""

import os

from llm_provider import BaseLLMProvider, LLMResponse
from logger import get_logger

logger = get_logger()


class VLLMProvider(BaseLLMProvider):
    """
    基于 vLLM 引擎的高性能本地模型提供者
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
    ):
        """
        初始化 vLLM 提供者实例

        Args:
            model_path: 模型绝对路径或名称
            tensor_parallel_size: 张量并行的 GPU 数量
            gpu_memory_utilization: GPU 显存占用比例限制
            trust_remote_code: 是否信任远程代码
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self._llm = None

    def _load_model(self):
        """延迟初始化 vLLM 引擎"""
        if self._llm is not None:
            return

        from vllm import LLM

        logger.info(f"Loading vLLM model from: {self.model_path}")

        self._llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
        )
        logger.info(f"vLLM model loaded successfully: {self.model_path}")

    @property
    def llm(self):
        """获取并按需加载 vLLM 实例"""
        if self._llm is None:
            self._load_model()
        return self._llm

    def generate(
        self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """
        使用 vLLM 生成单轮文本回复

        Args:
            prompt: 输入的提示词文本
            max_tokens: 最大生成的 token 数量
            temperature: 采样温度 (0.0 - 2.0)
            **kwargs: vLLM 支持的其他采样参数 (SamplingParams)

        Returns:
            LLMResponse 实例，包含生成的文本内容及元数据
        """
        from vllm import SamplingParams

        assert self.llm is not None

        sampling_params = SamplingParams(
            max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        outputs = self.llm.generate(
            [prompt], sampling_params, use_tqdm=False
        )  # 禁用单次调用的进度条输出
        output = outputs[0]

        prompt_len = len(output.prompt_token_ids if output.prompt_token_ids else [])
        comp_len = len(output.outputs[0].token_ids)

        return LLMResponse(
            content=output.outputs[0].text,
            model=self.model_path,
            usage={
                "prompt_tokens": prompt_len,
                "completion_tokens": comp_len,
                "total_tokens": prompt_len + comp_len,
            },
            finish_reason=output.outputs[0].finish_reason,
        )

    def generate_chat(
        self,
        messages: list,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        使用 vLLM 生成多轮对话回复

        Args:
            messages: 包含角色和内容的对话历史列表
            max_tokens: 最大生成的 token 数量
            temperature: 采样温度 (0.0 - 2.0)
            **kwargs: vLLM 支持的其他采样参数

        Returns:
            LLMResponse 实例，包含生成的文本内容及元数据
        """
        assert self.llm is not None

        prompt = self.llm.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return self.generate(str(prompt), max_tokens, temperature, **kwargs)

    def is_available(self) -> bool:
        """
        检查模型文件是否存在

        Returns:
            bool 值，存在返回 True，否则返回 False
        """
        return os.path.exists(self.model_path)
