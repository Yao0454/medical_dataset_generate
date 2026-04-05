#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
患者心智模拟器
论文2核心：患者心智模拟器
回复必须完全由动态时序心理状态驱动
"""

import json
from typing import Dict, List, Any, Optional

from openai import OpenAI

from tom_models import (
    ToMReasoning,
    TemporalMentalTrajectory,
    DialogueTurn
)


class PatientMindSimulator:
    
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
    
    def generate_patient_response(
        self,
        tom_reasoning: ToMReasoning,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        task_type: str,
        previous_trajectory: Optional[TemporalMentalTrajectory]
    ) -> str:
        """
        论文2要求：患者回复完全由动态时序心理状态驱动
        体现情绪/意图/知识缺口的连续变化
        """
        
        trajectory_info = ""
        if tom_reasoning.temporal_trajectory:
            traj = tom_reasoning.temporal_trajectory
            trajectory_info = f"""
CURRENT TEMPORAL TRAJECTORY:
- Turn: {traj.turn_number}
- Mental State Changes: {traj.changes_from_previous}
- Causal Trigger: {traj.causal_event.trigger_event if traj.causal_event else 'N/A'}
"""
        
        previous_context = ""
        if previous_trajectory and previous_trajectory.mental_state:
            previous_context = f"""
PREVIOUS STATE (for continuity):
- Previous Emotions: {previous_trajectory.mental_state.emotions}
- Previous Intentions: {previous_trajectory.mental_state.intentions}
- Previous Knowledge Gaps: {previous_trajectory.mental_state.knowledge_gaps}
"""
        
        prompt = f"""You are simulating a patient's mind in a medical consultation.
Your response MUST be driven by the patient's current mental state.

=== PATIENT'S CURRENT MENTAL STATE (DRIVING YOUR RESPONSE) ===
- Beliefs: {tom_reasoning.patient_mental_state.beliefs}
- Emotions: {tom_reasoning.patient_mental_state.emotions}
- Intentions: {tom_reasoning.patient_mental_state.intentions}
- Knowledge Gaps: {tom_reasoning.patient_mental_state.knowledge_gaps}

=== PATIENT'S POTENTIAL INTENTIONS ===
{json.dumps(tom_reasoning.patient_potential_intentions, indent=2)}

{trajectory_info}
{previous_context}

=== PATIENT BACKGROUND ===
{json.dumps(context.get('patient_info', {}), indent=2, ensure_ascii=False)}

=== DIALOGUE HISTORY ===
{self._format_dialogue_history(dialogue_history)}

=== CRITICAL INSTRUCTIONS ===
Generate a response that:
1. REFLECTS the patient's current emotions (show these emotions naturally)
2. PURSUES the patient's intentions (what patient wants to achieve)
3. REVEALS knowledge gaps naturally (ask questions if confused)
4. MAINTAINS continuity with previous mental state
5. RESPONDS to the doctor's last question/statement

The response must be:
- Emotionally consistent with: {tom_reasoning.patient_mental_state.emotions}
- Intention-driven: patient is trying to {tom_reasoning.patient_mental_state.intentions[:2] if tom_reasoning.patient_mental_state.intentions else 'get help'}
- Knowledge-appropriate: patient has gaps in {tom_reasoning.patient_mental_state.knowledge_gaps[:2] if tom_reasoning.patient_mental_state.knowledge_gaps else 'general understanding'}

OUTPUT: Just the patient's response (natural, conversational, emotion-reflecting)
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Patient simulation error: {e}")
            return "I see. Can you explain more about that?"
    
    def _format_dialogue_history(self, dialogue_history: List[DialogueTurn]) -> str:
        formatted = []
        for turn in dialogue_history:
            formatted.append(f"[Turn {turn.turn_number}] {turn.role.upper()}: {turn.content}")
        return "\n".join(formatted)
