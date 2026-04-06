#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地模型 LLM 提供者 (HuggingFace Transformers)
仅在需要时导入 torch/transformers
"""

import os
import time
from typing import Dict, List, Any, Optional

from llm_provider import BaseLLMProvider, LLMResponse
from logger import get_logger

logger = get_logger()


class LocalModelProvider(BaseLLMProvider):
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_length: int = 4096
    ):
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    def _load_model(self):
        if self._model is not None:
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        logger.info(f"Loading local model from: {self.model_path}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
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
        
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_length=self.max_length,
            device_map=self.device
        )
        
        logger.info(f"Local model loaded successfully: {self.model_path}")
    
    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    @property
    def pipeline(self):
        if self._pipeline is None:
            self._load_model()
        return self._pipeline
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> LLMResponse:
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return LLMResponse(
            content=generated_text,
            model=self.model_path,
            usage={
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
                "total_tokens": outputs.shape[1]
            }
        )
    
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> LLMResponse:
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages]) + "\nASSISTANT:"
        
        return self.generate(prompt, max_tokens, temperature, **kwargs)
    
    def is_available(self) -> bool:
        return os.path.exists(self.model_path)
