#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 高性能推理提供者
仅在需要时导入 vllm
"""

import os
from typing import Dict, List, Any, Optional

from llm_provider import BaseLLMProvider, LLMResponse
from logger import get_logger

logger = get_logger()


class VLLMProvider(BaseLLMProvider):
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self._llm = None
    
    def _load_model(self):
        if self._llm is not None:
            return
        
        from vllm import LLM
        
        logger.info(f"Loading vLLM model from: {self.model_path}")
        
        self._llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code
        )
        
        logger.info(f"vLLM model loaded successfully: {self.model_path}")
    
    @property
    def llm(self):
        if self._llm is None:
            self._load_model()
        return self._llm
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> LLMResponse:
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, **kwargs)
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        return LLMResponse(
            content=output.outputs[0].text,
            model=self.model_path,
            usage={
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            }
        )
    
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> LLMResponse:
        prompt = self.llm.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.generate(prompt, max_tokens, temperature, **kwargs)
    
    def is_available(self) -> bool:
        return os.path.exists(self.model_path)
