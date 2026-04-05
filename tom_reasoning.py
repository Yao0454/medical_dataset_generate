#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToM推理模块
论文1核心：双步骤ToM推理
- Step1：自主判断是否需要ToM推理（可返回False）
- Step2：仅在需要时执行信念-情绪-意图推理
论文2核心：时序链式推理、因果触发链
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple

from openai import OpenAI

from tom_models import (
    ToMReasoning,
    MentalState,
    CausalEvent,
    TemporalMentalTrajectory,
    DialogueTurn
)
from tom_error_detector import ToMErrorDetector


class ToMReasoningModule:
    
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.error_detector = ToMErrorDetector()
    
    def _determine_adaptive_dom(self, context: Dict[str, Any], dialogue_history: List[DialogueTurn]) -> int:
        """
        论文1要求：自适应DoM选择
        医疗问诊=合作场景→自动选择0阶或1阶，禁止高阶
        """
        if len(dialogue_history) <= 1:
            return 0
        
        patient_utterances = [t.content for t in dialogue_history if t.role == "user"]
        if not patient_utterances:
            return 0
        
        last_utterance = patient_utterances[-1].lower()
        
        complex_signals = [
            'but', 'however', 'worried', 'concerned', 'afraid',
            '但是', '不过', '担心', '害怕', '顾虑'
        ]
        
        has_complex_signal = any(signal in last_utterance for signal in complex_signals)
        
        if has_complex_signal:
            return 1
        
        question_patterns = ['?', 'what', 'why', 'how', '什么', '为什么', '怎么']
        has_question = any(p in last_utterance for p in question_patterns)
        
        if has_question:
            return 1
        
        return 0
    
    def step1_tom_invocation_decision(
        self,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        task_type: str
    ) -> Tuple[bool, int, str]:
        """
        论文1核心：Step1自主决策是否调用ToM
        必须真实判断，可返回should_invoke_tom=False
        """
        if len(dialogue_history) == 0:
            return True, 0, "Initial consultation requires ToM to establish patient baseline"
        
        patient_utterances = [t.content for t in dialogue_history if t.role == "user"]
        if not patient_utterances:
            return True, 0, "No patient input yet, need ToM for initial assessment"
        
        last_patient_utterance = patient_utterances[-1]
        
        simple_acknowledgment_patterns = [
            r'^(ok|okay|yes|no|sure|好的|是的|没有|行).?$',
            r'^(thank you|thanks|谢谢).?$',
            r'^.{1,5}$'
        ]
        
        is_simple_acknowledgment = any(
            re.match(pattern, last_patient_utterance.strip(), re.IGNORECASE)
            for pattern in simple_acknowledgment_patterns
        )
        
        if is_simple_acknowledgment and len(patient_utterances) > 3:
            return False, 0, "Simple acknowledgment detected, no ToM needed for this turn"
        
        dom_level = self._determine_adaptive_dom(context, dialogue_history)
        
        needs_tom_signals = [
            '?', 'worried', 'concerned', 'afraid', 'confused', 'don\'t know',
            'but', 'however', 'maybe', 'think', 'feel',
            '？', '担心', '害怕', '困惑', '不知道', '但是', '觉得', '感觉'
        ]
        
        needs_tom = any(signal in last_patient_utterance.lower() for signal in needs_tom_signals)
        
        if needs_tom:
            return True, dom_level, f"Patient utterance contains signals requiring ToM analysis"
        
        if dom_level == 1:
            return True, 1, "Complex patient response requires first-order ToM"
        
        return True, 0, "Standard information exchange, using zero-order ToM"
    
    def step2_mental_state_inference(
        self,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        dom_level: int,
        task_type: str,
        previous_trajectory: Optional[TemporalMentalTrajectory]
    ) -> ToMReasoning:
        """
        论文1+2核心：Step2心理状态推理
        - 严格心智边界隔离
        - 时序链式推理
        - 因果触发链
        """
        
        previous_state_summary = ""
        if previous_trajectory and previous_trajectory.mental_state:
            previous_state_summary = f"""
PREVIOUS MENTAL STATE (Turn {previous_trajectory.turn_number}):
- Beliefs: {previous_trajectory.mental_state.beliefs}
- Emotions: {previous_trajectory.mental_state.emotions}
- Intentions: {previous_trajectory.mental_state.intentions}
- Knowledge Gaps: {previous_trajectory.mental_state.knowledge_gaps}
"""
        
        prompt = f"""You are performing Theory of Mind reasoning for a medical consultation.

CRITICAL RULES:
1. DoM Level: {dom_level} (0=direct observation, 1=patient's perspective)
2. STRICTLY SEPARATE mental boundaries:
   - DOCTOR's knowledge vs DOCTOR's unknowns
   - PATIENT's knowledge vs PATIENT's knowledge gaps
3. Use TEMPORAL CHAIN REASONING: Connect current state to previous state
4. Identify CAUSAL TRIGGERS: What event caused mental state changes?

TASK: {task_type}
CHIEF COMPLAINT: {context.get('chief_complaint', 'Unknown')}
{previous_state_summary}

DIALOGUE HISTORY:
{self._format_dialogue_history(dialogue_history)}

PATIENT INFORMATION:
{json.dumps(context.get('patient_info', {}), indent=2, ensure_ascii=False)}

Perform STRUCTURED ToM REASONING:

1. DOCTOR'S KNOWN INFORMATION (What I, as doctor, know for certain):
   - List ONLY facts confirmed through dialogue or records
   - Be strict: if not confirmed, it goes to unknown

2. DOCTOR'S UNKNOWN INFORMATION (What I need to find out):
   - Diagnostic questions still needed
   - Missing history elements
   - Tests or examinations needed

3. PATIENT'S KNOWN INFORMATION (What patient understands):
   - Patient's confirmed understanding of their condition
   - Information patient has explicitly shared

4. PATIENT'S KNOWLEDGE GAPS (What patient doesn't understand):
   - Medical concepts patient may not grasp
   - Information that needs explanation

5. PATIENT'S POTENTIAL INTENTIONS (at DoM level {dom_level}):
   - What does patient want from this consultation?
   - What are patient's concerns or fears?
   - What is patient trying to achieve?

6. PATIENT'S CURRENT MENTAL STATE:
   - Beliefs: What does patient believe about their condition?
   - Emotions: What emotions is patient experiencing NOW?
   - Intentions: What does patient intend to do?
   - Knowledge Gaps: What doesn't patient understand?

7. TEMPORAL CHANGES (Compare to previous state):
   - What changed in patient's mental state?
   - What triggered the change?
   - Causal chain of events

8. CHAIN REASONING TRACE (Step-by-step deduction):
   - Show how you arrived at each conclusion
   - Link evidence to inferences

9. NEXT ACTION STRATEGY:
   - Based on ToM analysis, what should I do?
   - How to address knowledge gaps?
   - How to respond to patient's intentions?

OUTPUT FORMAT (JSON):
{{
    "doctor_known_info": ["confirmed fact 1", "confirmed fact 2"],
    "doctor_unknown_info": ["needed info 1", "needed info 2"],
    "patient_known_info": ["patient knows 1", "patient knows 2"],
    "patient_knowledge_gaps": ["gap 1", "gap 2"],
    "patient_potential_intentions": ["intention 1", "intention 2"],
    "patient_mental_state": {{
        "beliefs": ["belief 1", "belief 2"],
        "emotions": ["emotion 1", "emotion 2"],
        "intentions": ["intention 1", "intention 2"],
        "knowledge_gaps": ["gap 1", "gap 2"]
    }},
    "temporal_changes": {{
        "beliefs_changed": ["what changed"],
        "emotions_changed": ["what changed"],
        "intentions_changed": ["what changed"],
        "trigger_event": "what caused these changes"
    }},
    "chain_reasoning_trace": [
        {{"step": 1, "observation": "what I observed", "inference": "what I concluded"}},
        {{"step": 2, "observation": "...", "inference": "..."}}
    ],
    "next_action_strategy": "detailed strategy"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            mental_state = MentalState(
                beliefs=result.get("patient_mental_state", {}).get("beliefs", []),
                emotions=result.get("patient_mental_state", {}).get("emotions", []),
                intentions=result.get("patient_mental_state", {}).get("intentions", []),
                knowledge_gaps=result.get("patient_mental_state", {}).get("knowledge_gaps", [])
            )
            
            patient_utterance = ""
            for turn in reversed(dialogue_history):
                if turn.role == "user":
                    patient_utterance = turn.content
                    break
            
            errors, corrected_state, corrected_intentions = self.error_detector.detect_and_correct_errors(
                patient_utterance=patient_utterance,
                mental_state=mental_state,
                intentions=result.get("patient_potential_intentions", []),
                dialogue_history=dialogue_history,
                patient_info=context.get('patient_info', {}),
                turn_number=len(dialogue_history)
            )
            
            temporal_changes = result.get("temporal_changes", {})
            causal_event = None
            if temporal_changes.get("trigger_event"):
                causal_event = CausalEvent(
                    trigger_event=temporal_changes.get("trigger_event", ""),
                    mental_state_before=previous_trajectory.mental_state if previous_trajectory else MentalState(),
                    mental_state_after=corrected_state,
                    change_description=f"Beliefs: {temporal_changes.get('beliefs_changed', [])}, Emotions: {temporal_changes.get('emotions_changed', [])}"
                )
            
            trajectory = TemporalMentalTrajectory(
                turn_number=len(dialogue_history),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                mental_state=corrected_state,
                causal_event=causal_event,
                changes_from_previous={
                    "beliefs": temporal_changes.get("beliefs_changed", []),
                    "emotions": temporal_changes.get("emotions_changed", []),
                    "intentions": temporal_changes.get("intentions_changed", [])
                }
            )
            
            return ToMReasoning(
                should_invoke_tom=True,
                dom_level=dom_level,
                step1_decision_reason="ToM invoked based on Step1 decision",
                doctor_known_info=result.get("doctor_known_info", []),
                doctor_unknown_info=result.get("doctor_unknown_info", []),
                patient_known_info=result.get("patient_known_info", []),
                patient_knowledge_gaps=result.get("patient_knowledge_gaps", []),
                patient_potential_intentions=corrected_intentions,
                patient_mental_state=corrected_state,
                next_action_strategy=result.get("next_action_strategy", ""),
                temporal_trajectory=trajectory,
                tom_errors_detected=errors,
                chain_reasoning_trace=result.get("chain_reasoning_trace", [])
            )
            
        except Exception as e:
            print(f"Mental state inference error: {e}")
            return ToMReasoning(
                should_invoke_tom=True,
                dom_level=dom_level,
                step1_decision_reason=f"Error occurred: {str(e)}"
            )
    
    def _format_dialogue_history(self, dialogue_history: List[DialogueTurn]) -> str:
        formatted = []
        for turn in dialogue_history:
            formatted.append(f"[Turn {turn.turn_number}] {turn.role.upper()}: {turn.content}")
        return "\n".join(formatted)
