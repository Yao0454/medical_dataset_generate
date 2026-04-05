#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据模型定义
论文1+2核心数据结构
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class DoMLevel(Enum):
    ZERO_ORDER = 0
    FIRST_ORDER = 1


class ToMErrorType(Enum):
    TYPE_A_OVER_MENTALIZING = "over_mentalizing"
    TYPE_B_UNDER_MENTALIZING = "under_mentalizing"
    TYPE_C_REASONING_ERROR = "reasoning_error"


@dataclass
class MentalState:
    beliefs: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    intentions: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)


@dataclass
class CausalEvent:
    trigger_event: str = ""
    mental_state_before: MentalState = field(default_factory=MentalState)
    mental_state_after: MentalState = field(default_factory=MentalState)
    change_description: str = ""


@dataclass
class ToMErrorRecord:
    error_type: ToMErrorType
    error_description: str
    detected_at_turn: int
    correction_applied: str
    corrected: bool = False


@dataclass
class TemporalMentalTrajectory:
    turn_number: int = 0
    timestamp: str = ""
    mental_state: MentalState = field(default_factory=MentalState)
    causal_event: Optional[CausalEvent] = None
    changes_from_previous: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ToMReasoning:
    should_invoke_tom: bool = False
    dom_level: int = 0
    step1_decision_reason: str = ""
    doctor_known_info: List[str] = field(default_factory=list)
    doctor_unknown_info: List[str] = field(default_factory=list)
    patient_known_info: List[str] = field(default_factory=list)
    patient_knowledge_gaps: List[str] = field(default_factory=list)
    patient_potential_intentions: List[str] = field(default_factory=list)
    patient_mental_state: MentalState = field(default_factory=MentalState)
    next_action_strategy: str = ""
    temporal_trajectory: TemporalMentalTrajectory = field(default_factory=TemporalMentalTrajectory)
    tom_errors_detected: List[ToMErrorRecord] = field(default_factory=list)
    chain_reasoning_trace: List[Dict] = field(default_factory=list)


@dataclass
class DialogueTurn:
    content: str
    role: str
    turn_number: int = 0
    tom_reasoning: Optional[ToMReasoning] = None


@dataclass
class TargetFormat:
    data_source: str
    topic: str
    department: str
    subdepartment: str
    disease: str
    prompt: List[Dict[str, Any]]
    ability: str
    reward_model: Dict[str, str]
    tom_annotations: List[Dict[str, Any]] = field(default_factory=list)
