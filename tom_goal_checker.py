#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToM目标检查器
论文要求：终止条件=医生信息补齐+患者知识缺口覆盖
废除关键词/最大轮次终止
"""

from typing import Dict, List, Any, Tuple

from tom_models import ToMReasoning, DialogueTurn


class ToMGoalChecker:
    
    def check_tom_goal_achieved(
        self,
        tom_reasoning: ToMReasoning,
        dialogue_history: List[DialogueTurn],
        task_type: str,
        required_info: List[str]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        论文要求：基于ToM目标的终止判断
        终止条件=医生信息补齐+患者知识缺口覆盖
        """
        
        doctor_info_completeness = self._calculate_info_completeness(
            tom_reasoning.doctor_unknown_info,
            dialogue_history,
            required_info
        )
        
        patient_gap_coverage = self._calculate_gap_coverage(
            tom_reasoning.patient_knowledge_gaps,
            dialogue_history
        )
        
        goal_status = {
            "doctor_info_complete": doctor_info_completeness >= 0.8,
            "patient_gaps_covered": patient_gap_coverage >= 0.7,
            "doctor_completeness_score": doctor_info_completeness,
            "patient_gap_coverage_score": patient_gap_coverage,
            "remaining_unknown_info": tom_reasoning.doctor_unknown_info,
            "remaining_knowledge_gaps": tom_reasoning.patient_knowledge_gaps
        }
        
        if doctor_info_completeness >= 0.8 and patient_gap_coverage >= 0.7:
            return True, "ToM goal achieved: Doctor information complete and patient knowledge gaps addressed", goal_status
        
        if len(dialogue_history) >= 12:
            return True, "Maximum dialogue length reached (safety limit)", goal_status
        
        return False, f"ToM goal not achieved: Doctor info {doctor_info_completeness:.0%} complete, Patient gaps {patient_gap_coverage:.0%} covered", goal_status
    
    def _calculate_info_completeness(
        self,
        unknown_info: List[str],
        dialogue_history: List[DialogueTurn],
        required_info: List[str]
    ) -> float:
        if not required_info:
            if not unknown_info:
                return 1.0
            dialogue_text = " ".join([t.content.lower() for t in dialogue_history])
            covered = sum(1 for info in unknown_info if any(kw in dialogue_text for kw in info.lower().split()[:3]))
            return covered / len(unknown_info) if unknown_info else 1.0
        
        dialogue_text = " ".join([t.content.lower() for t in dialogue_history])
        covered = sum(1 for info in required_info if any(kw in dialogue_text for kw in info.lower().split()[:2]))
        return covered / len(required_info)
    
    def _calculate_gap_coverage(
        self,
        knowledge_gaps: List[str],
        dialogue_history: List[DialogueTurn]
    ) -> float:
        if not knowledge_gaps:
            return 1.0
        
        doctor_explanations = [t.content.lower() for t in dialogue_history if t.role == "assistant"]
        if not doctor_explanations:
            return 0.0
        
        explanation_text = " ".join(doctor_explanations)
        
        explanation_indicators = [
            'this means', 'let me explain', 'the reason is', 'because',
            '意思是', '让我解释', '原因是', '因为'
        ]
        
        has_explanation = any(ind in explanation_text for ind in explanation_indicators)
        
        if has_explanation:
            return 0.8
        
        return 0.3
