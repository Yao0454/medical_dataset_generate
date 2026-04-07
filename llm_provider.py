#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 提供者模块 - 支持多种 LLM 后端
- OpenAI API (默认)
- 本地模型 (HuggingFace Transformers) - 需要安装 torch/transformers
- vLLM (高性能推理) - 需要安装 vllm
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from openai import OpenAI

from logger import get_logger

logger = get_logger()


@dataclass
class LLMResponse:
    """
    统一的 LLM 响应数据类
    """

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


class BaseLLMProvider(ABC):
    """
    LLM 提供者抽象基类
    """

    @abstractmethod
    def generate(
        self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """
        生成单轮文本回复

        Args:
            prompt: 输入的提示词文本
            max_tokens: 最大生成的 token 数量
            temperature: 采样温度 (0.0 - 2.0)
            **kwargs: 其他支持的底层模型参数

        Returns:
            LLMResponse 实例，包含生成的文本内容及元数据
        """
        pass

    @abstractmethod
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        生成多轮对话回复

        Args:
            messages: 包含角色和内容的对话历史列表 (例如[{"role": "user", "content": "你好"}])
            max_tokens: 最大生成的 token 数量
            temperature: 采样温度 (0.0 - 2.0)
            **kwargs: 其他支持的底层模型参数

        Returns:
            LLMResponse 实例，包含生成的文本内容及元数据
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查当前 LLM 提供者是否已准备就绪且可用

        Returns:
            bool 值，如果提供者配置正确且可用则返回 True，否则返回 False
        """
        pass


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI 接口提供者，使用最新 Responses API
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
    ):
        """
        初始化 OpenAI 提供者实例

        Args:
            api_key: OpenAI API 密钥，若为空则尝试从环境变量 OPENAI_API_KEY 获取
            base_url: 自定义 API 地址，若为空则尝试从环境变量 OPENAI_BASE_URL 获取
            model: 默认使用的模型名称 (如 "gpt-4o")
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        self._client = None

    @property
    def client(self) -> OpenAI:
        """
        懒加载获取 OpenAI 客户端实例

        Returns:
            OpenAI 客户端对象
        """
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _call_responses_api(
        self,
        input_data: Union[str, List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> LLMResponse:
        """
        统一的内部请求调用方法，兼容单轮文本和多轮对话结构

        Args:
            input_data: 纯文本提示词或消息字典列表
            max_tokens: 最大生成的 token 数量
            temperature: 采样温度
            **kwargs: 传递给 openai 的其他参数

        Returns:
            LLMResponse 实例，封装了 API 响应内容
        """
        response = self.client.responses.create(
            model=self.model,
            input=input_data,
            max_output_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        usage_data = None
        if hasattr(response, "usage") and response.usage:
            in_tokens = getattr(
                response.usage,
                "input_tokens",
                getattr(response.usage, "prompt_tokens", 0),
            )
            out_tokens = getattr(
                response.usage,
                "output_tokens",
                getattr(response.usage, "completion_tokens", 0),
            )
            usage_data = {
                "prompt_tokens": in_tokens,
                "completion_tokens": out_tokens,
                "total_tokens": getattr(
                    response.usage, "total_tokens", in_tokens + out_tokens
                ),
            }

        return LLMResponse(
            content=response.output_text,
            model=response.model,
            usage=usage_data,
            finish_reason=getattr(
                response, "status", getattr(response, "finish_reason", "completed")
            ),
        )

    def generate(
        self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """
        生成单轮文本回复

        Args:
            prompt: 输入的提示词文本
            max_tokens: 最大生成的 token 数量
            temperature: 采样温度 (0.0 - 2.0)
            **kwargs: 其他支持的底层模型参数

        Returns:
            LLMResponse 实例，包含生成的文本内容及元数据
        """
        return self._call_responses_api(prompt, max_tokens, temperature, **kwargs)

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        生成多轮对话回复

        Args:
            messages: 包含角色和内容的对话历史列表
            max_tokens: 最大生成的 token 数量
            temperature: 采样温度 (0.0 - 2.0)
            **kwargs: 其他支持的底层模型参数

        Returns:
            LLMResponse 实例，包含生成的文本内容及元数据
        """
        return self._call_responses_api(messages, max_tokens, temperature, **kwargs)

    def is_available(self) -> bool:
        """
        检查 API Key 是否已配置

        Returns:
            bool 值，配置了 API Key 返回 True，否则返回 False
        """
        return bool(self.api_key)


def create_llm_provider(provider_type: str = "openai", **kwargs) -> BaseLLMProvider:
    """
    创建 LLM 提供者实例

    Args:
        provider_type: 提供者类型 ("openai", "local", "vllm")
        **kwargs: 提供者特定参数 (如 api_key, model 等)

    Returns:
        BaseLLMProvider 实例
    """
    if provider_type == "openai":
        provider = OpenAIProvider(**kwargs)
    elif provider_type == "local":
        try:
            from llm_provider_local import LocalModelProvider

            provider = LocalModelProvider(**kwargs)
        except ImportError as e:
            raise RuntimeError(
                f"Failed to load local model provider: {e}. Run: pip install torch transformers"
            )
    elif provider_type == "vllm":
        try:
            from llm_provider_vllm import VLLMProvider

            provider = VLLMProvider(**kwargs)
        except ImportError as e:
            raise RuntimeError(
                f"vLLM not installed or failed to load: {e}. Run: pip install vllm"
            )
    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. Available:['openai', 'local', 'vllm']"
        )

    logger.info(f"Created LLM provider: {provider_type}")
    return provider
