#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToM错误检测器
论文1要求：3类ToM错误实时检测+修正
- TypeA（过度心智化）：禁止对患者简单提问做复杂意图猜测
- TypeB（心智不足）：必须识别患者回避、顾虑、知识缺口等隐性心理
- TypeC（推理错误）：校验心智推理与对话上下文一致性
"""

import re
from typing import List, Tuple, Dict, Any

from tom_models import (
    MentalState,
    ToMErrorRecord,
    ToMErrorType,
    DialogueTurn
)


class ToMErrorDetector:
    
    def detect_type_a_over_mentalizing(
        self,
        patient_utterance: str,
        inferred_intentions: List[str],
        dialogue_context: List[DialogueTurn]
    ) -> Tuple[bool, str]:
        simple_patterns = [
            r'^.{1,20}$',
            r'^(yes|no|ok|okay|sure|好的|是的|没有|有).*$',
            r'^(thank|thanks|谢谢).*$'
        ]
        
        is_simple_utterance = any(
            re.match(pattern, patient_utterance.strip(), re.IGNORECASE)
            for pattern in simple_patterns
        )
        
        if is_simple_utterance and len(inferred_intentions) > 2:
            return True, f"Over-mentalizing detected: Simple utterance '{patient_utterance}' attributed {len(inferred_intentions)} complex intentions"
        
        complex_intention_keywords = ['hidden agenda', 'ulterior motive', 'manipulating', 'deceiving']
        for intention in inferred_intentions:
            if any(kw in intention.lower() for kw in complex_intention_keywords):
                return True, f"Over-mentalizing: Attributing complex motive '{intention}' without evidence"
        
        return False, ""
    
    def detect_type_b_under_mentalizing(
        self,
        patient_utterance: str,
        detected_mental_state: MentalState,
        dialogue_context: List[DialogueTurn]
    ) -> Tuple[bool, str]:
        avoidance_patterns = [
            r'(i don\'t know|not sure|maybe|i guess|我不知道|不太清楚|也许)',
            r'(but|however|但是|不过)',
            r'(worried|concerned|afraid|scared|担心|害怕|顾虑)',
            r'(hesitat|uncertain|confus|不确定|困惑|犹豫)'
        ]
        
        has_avoidance_signal = any(
            re.search(pattern, patient_utterance, re.IGNORECASE)
            for pattern in avoidance_patterns
        )
        
        if has_avoidance_signal:
            has_emotion_detected = len(detected_mental_state.emotions) > 0
            has_knowledge_gap_detected = len(detected_mental_state.knowledge_gaps) > 0
            
            if not has_emotion_detected and not has_knowledge_gap_detected:
                return True, f"Under-mentalizing: Patient shows avoidance/hesitation signals but no emotions or knowledge gaps detected"
        
        question_patterns = [r'\?', r'(what|why|how|when|什么|为什么|怎么|何时)']
        has_question = any(
            re.search(pattern, patient_utterance, re.IGNORECASE)
            for pattern in question_patterns
        )
        
        if has_question and len(detected_mental_state.knowledge_gaps) == 0:
            return True, f"Under-mentalizing: Patient asks question but no knowledge gaps identified"
        
        return False, ""
    
    def detect_type_c_reasoning_error(
        self,
        mental_state: MentalState,
        dialogue_history: List[DialogueTurn],
        patient_info: Dict[str, Any]
    ) -> Tuple[bool, str]:
        if dialogue_history:
            last_patient_turn = None
            for turn in reversed(dialogue_history):
                if turn.role == "user":
                    last_patient_turn = turn.content
                    break
            
            if last_patient_turn:
                for belief in mental_state.beliefs:
                    key_contradictions = [
                        ('not worried', 'worried'),
                        ('no pain', 'pain'),
                        ('fine', 'suffering'),
                        ('understand', 'confused')
                    ]
                    for neg, pos in key_contradictions:
                        if neg in belief.lower() and pos in last_patient_turn.lower():
                            return True, f"Reasoning error: Belief '{belief}' contradicts patient statement"
        
        if patient_info:
            allergies = patient_info.get('allergies', [])
            for belief in mental_state.beliefs:
                if 'no allergies' in belief.lower() and allergies:
                    return True, f"Reasoning error: Belief '{belief}' contradicts known allergies: {allergies}"
        
        return False, ""
    
    def detect_and_correct_errors(
        self,
        patient_utterance: str,
        mental_state: MentalState,
        intentions: List[str],
        dialogue_history: List[DialogueTurn],
        patient_info: Dict[str, Any],
        turn_number: int
    ) -> Tuple[List[ToMErrorRecord], MentalState, List[str]]:
        errors = []
        corrected_state = MentalState(
            beliefs=mental_state.beliefs.copy(),
            emotions=mental_state.emotions.copy(),
            intentions=mental_state.intentions.copy(),
            knowledge_gaps=mental_state.knowledge_gaps.copy()
        )
        corrected_intentions = intentions.copy()
        
        is_type_a, desc_a = self.detect_type_a_over_mentalizing(
            patient_utterance, intentions, dialogue_history
        )
        if is_type_a:
            errors.append(ToMErrorRecord(
                error_type=ToMErrorType.TYPE_A_OVER_MENTALIZING,
                error_description=desc_a,
                detected_at_turn=turn_number,
                correction_applied="Reduced complex intentions to essential ones",
                corrected=True
            ))
            corrected_intentions = intentions[:1] if intentions else []
        
        is_type_b, desc_b = self.detect_type_b_under_mentalizing(
            patient_utterance, mental_state, dialogue_history
        )
        if is_type_b:
            correction = "Added detected emotions and knowledge gaps"
            if "worried" in patient_utterance.lower() or "concerned" in patient_utterance.lower():
                if "anxiety" not in [e.lower() for e in corrected_state.emotions]:
                    corrected_state.emotions.append("anxiety about condition")
            if "don't know" in patient_utterance.lower() or "not sure" in patient_utterance.lower():
                if not corrected_state.knowledge_gaps:
                    corrected_state.knowledge_gaps.append("understanding of condition")
            
            errors.append(ToMErrorRecord(
                error_type=ToMErrorType.TYPE_B_UNDER_MENTALIZING,
                error_description=desc_b,
                detected_at_turn=turn_number,
                correction_applied=correction,
                corrected=True
            ))
        
        is_type_c, desc_c = self.detect_type_c_reasoning_error(
            mental_state, dialogue_history, patient_info
        )
        if is_type_c:
            errors.append(ToMErrorRecord(
                error_type=ToMErrorType.TYPE_C_REASONING_ERROR,
                error_description=desc_c,
                detected_at_turn=turn_number,
                correction_applied="Flagged for re-inference",
                corrected=True
            ))
        
        return errors, corrected_state, corrected_intentions
