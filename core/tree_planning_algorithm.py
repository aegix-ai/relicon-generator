"""
Tree-Based Planning Algorithm for Millisecond-Precision Video Creation
Implements a hierarchical tree structure that breaks down video creation from concept to atomic millisecond-level execution.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from core.logger import get_logger
from core.universal_brand_intelligence import UniversalBrandIntelligenceGenerator

logger = get_logger(__name__)

class NodeType(Enum):
    """Types of nodes in the planning tree."""
    CONCEPT = "concept"          # Overall concept/story
    ACT = "act"                  # Story acts/sections  
    SCENE = "scene"              # Individual scenes
    SHOT = "shot"                # Camera shots within scenes
    MOMENT = "moment"            # Specific moments (1-3 seconds)
    MILLISECOND = "millisecond"  # Frame-level precision

@dataclass
class TimeInterval:
    """Precise time interval with millisecond accuracy."""
    start_ms: int
    end_ms: int
    duration_ms: int = field(init=False)
    
    def __post_init__(self):
        self.duration_ms = self.end_ms - self.start_ms
    
    def overlaps(self, other: 'TimeInterval') -> bool:
        return not (self.end_ms <= other.start_ms or other.end_ms <= self.start_ms)
    
    def contains(self, timestamp_ms: int) -> bool:
        return self.start_ms <= timestamp_ms <= self.end_ms

@dataclass
class CreativeIntent:
    """Captures the creative intention at each level."""
    emotion: str           # Target emotional response
    purpose: str          # What this achieves in the story
    visual_style: str     # Visual treatment
    pacing: str          # Fast, medium, slow
    emphasis: str        # What to emphasize
    transition: str      # How it connects to next element

class PlanningNode(ABC):
    """Abstract base class for all planning tree nodes."""
    
    def __init__(self, 
                 node_id: str,
                 node_type: NodeType,
                 time_interval: TimeInterval,
                 creative_intent: CreativeIntent,
                 parent: Optional['PlanningNode'] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.time_interval = time_interval
        self.creative_intent = creative_intent
        self.parent = parent
        self.children: List['PlanningNode'] = []
        self.metadata: Dict[str, Any] = {}
        
    def add_child(self, child: 'PlanningNode') -> None:
        """Add a child node and set its parent."""
        child.parent = self
        self.children.append(child)
        
    def get_depth(self) -> int:
        """Get the depth of this node in the tree."""
        if not self.parent:
            return 0
        return self.parent.get_depth() + 1
    
    def get_path(self) -> List[str]:
        """Get the path from root to this node."""
        if not self.parent:
            return [self.node_id]
        return self.parent.get_path() + [self.node_id]
    
    @abstractmethod
    def generate_execution_plan(self) -> Dict[str, Any]:
        """Generate detailed execution plan for this node."""
        pass
    
    @abstractmethod
    def validate_timing(self) -> bool:
        """Validate that timing constraints are met."""
        pass

class ConceptNode(PlanningNode):
    """Root level - overall video concept and story."""
    
    def __init__(self, brand_info: Dict[str, Any], duration_ms: int):
        time_interval = TimeInterval(0, duration_ms)
        creative_intent = CreativeIntent(
            emotion="brand_aligned",
            purpose="communicate_brand_value",
            visual_style="professional_cinematic",
            pacing="dynamic",
            emphasis="brand_benefits",
            transition="seamless_flow"
        )
        super().__init__("concept_root", NodeType.CONCEPT, time_interval, creative_intent)
        self.brand_info = brand_info
        self.story_arc = self._generate_story_arc()
        
    def _generate_story_arc(self) -> Dict[str, Any]:
        """Generate the overall story structure."""
        return {
            "hook": "attention_grabbing_opening",
            "problem": "audience_pain_point",
            "solution": "brand_as_solution", 
            "proof": "credibility_social_proof",
            "action": "clear_call_to_action"
        }
    
    def generate_execution_plan(self) -> Dict[str, Any]:
        return {
            "concept": {
                "brand": self.brand_info.get('brand_name', 'Unknown'),
                "story_arc": self.story_arc,
                "total_duration_ms": self.time_interval.duration_ms,
                "target_emotion": self.creative_intent.emotion
            }
        }
    
    def validate_timing(self) -> bool:
        child_duration = sum(child.time_interval.duration_ms for child in self.children)
        return abs(child_duration - self.time_interval.duration_ms) < 100  # 100ms tolerance

class ActNode(PlanningNode):
    """Story acts/sections (typically 3-5 acts)."""
    
    def __init__(self, act_id: str, act_purpose: str, time_interval: TimeInterval):
        creative_intent = CreativeIntent(
            emotion=self._map_purpose_to_emotion(act_purpose),
            purpose=act_purpose,
            visual_style="purpose_aligned",
            pacing=self._map_purpose_to_pacing(act_purpose),
            emphasis=act_purpose,
            transition="smooth_progression"
        )
        super().__init__(act_id, NodeType.ACT, time_interval, creative_intent)
        self.act_purpose = act_purpose
        
    def _map_purpose_to_emotion(self, purpose: str) -> str:
        mapping = {
            "hook": "curiosity_excitement",
            "problem": "empathy_concern", 
            "solution": "relief_hope",
            "proof": "confidence_trust",
            "action": "motivation_urgency"
        }
        return mapping.get(purpose, "neutral")
    
    def _map_purpose_to_pacing(self, purpose: str) -> str:
        mapping = {
            "hook": "fast",
            "problem": "medium",
            "solution": "medium",
            "proof": "fast", 
            "action": "fast"
        }
        return mapping.get(purpose, "medium")
        
    def generate_execution_plan(self) -> Dict[str, Any]:
        return {
            "act": {
                "purpose": self.act_purpose,
                "emotion": self.creative_intent.emotion,
                "pacing": self.creative_intent.pacing,
                "time_range": [self.time_interval.start_ms, self.time_interval.end_ms],
                "scenes": len(self.children)
            }
        }
    
    def validate_timing(self) -> bool:
        return all(child.validate_timing() for child in self.children)

class SceneNode(PlanningNode):
    """Individual scene with specific visual concept."""
    
    def __init__(self, scene_id: str, visual_concept: str, time_interval: TimeInterval):
        creative_intent = CreativeIntent(
            emotion="scene_appropriate",
            purpose="advance_story",
            visual_style=visual_concept,
            pacing="scene_appropriate",
            emphasis="visual_impact",
            transition="cinematic"
        )
        super().__init__(scene_id, NodeType.SCENE, time_interval, creative_intent)
        self.visual_concept = visual_concept
        self.script_line = ""
        self.camera_movements: List[str] = []
        self.lighting_notes = ""
        
    def set_script(self, script_line: str) -> None:
        self.script_line = script_line
        
    def set_camera_movements(self, movements: List[str]) -> None:
        self.camera_movements = movements
        
    def generate_execution_plan(self) -> Dict[str, Any]:
        return {
            "scene": {
                "visual_concept": self.visual_concept,
                "script": self.script_line,
                "camera_movements": self.camera_movements,
                "time_range": [self.time_interval.start_ms, self.time_interval.end_ms],
                "duration_seconds": self.time_interval.duration_ms / 1000,
                "shots": len(self.children)
            }
        }
    
    def validate_timing(self) -> bool:
        # Scene should be 3-15 seconds typically
        duration_sec = self.time_interval.duration_ms / 1000
        return 3.0 <= duration_sec <= 15.0

class ShotNode(PlanningNode):
    """Camera shot within a scene (1-5 seconds typically)."""
    
    def __init__(self, shot_id: str, shot_type: str, time_interval: TimeInterval):
        creative_intent = CreativeIntent(
            emotion="shot_specific",
            purpose="visual_storytelling",
            visual_style=shot_type,
            pacing="shot_appropriate", 
            emphasis="composition",
            transition="cut_or_smooth"
        )
        super().__init__(shot_id, NodeType.SHOT, time_interval, creative_intent)
        self.shot_type = shot_type  # close-up, wide, medium, etc.
        self.camera_angle = "eye_level"
        self.movement = "static"
        self.focus_subject = ""
        
    def generate_execution_plan(self) -> Dict[str, Any]:
        return {
            "shot": {
                "type": self.shot_type,
                "angle": self.camera_angle,
                "movement": self.movement,
                "focus": self.focus_subject,
                "time_range": [self.time_interval.start_ms, self.time_interval.end_ms],
                "duration_ms": self.time_interval.duration_ms
            }
        }
    
    def validate_timing(self) -> bool:
        # Shots should be 500ms to 8 seconds
        return 500 <= self.time_interval.duration_ms <= 8000

class MomentNode(PlanningNode):
    """Specific moment within a shot (1-3 seconds)."""
    
    def __init__(self, moment_id: str, action: str, time_interval: TimeInterval):
        creative_intent = CreativeIntent(
            emotion="moment_specific",
            purpose="precise_action",
            visual_style="action_focused",
            pacing="precise",
            emphasis="key_action",
            transition="fluid"
        )
        super().__init__(moment_id, NodeType.MOMENT, time_interval, creative_intent)
        self.action = action
        self.key_elements: List[str] = []
        
    def generate_execution_plan(self) -> Dict[str, Any]:
        return {
            "moment": {
                "action": self.action,
                "key_elements": self.key_elements,
                "time_range": [self.time_interval.start_ms, self.time_interval.end_ms],
                "duration_ms": self.time_interval.duration_ms
            }
        }
    
    def validate_timing(self) -> bool:
        return 200 <= self.time_interval.duration_ms <= 3000

class MillisecondNode(PlanningNode):
    """Frame-level precision for specific visual/audio elements."""
    
    def __init__(self, ms_id: str, element_type: str, time_interval: TimeInterval):
        creative_intent = CreativeIntent(
            emotion="precise",
            purpose="frame_perfect",
            visual_style="exact",
            pacing="millisecond",
            emphasis=element_type,
            transition="exact"
        )
        super().__init__(ms_id, NodeType.MILLISECOND, time_interval, creative_intent)
        self.element_type = element_type  # "subtitle", "logo", "transition", etc.
        self.properties: Dict[str, Any] = {}
        
    def generate_execution_plan(self) -> Dict[str, Any]:
        return {
            "millisecond": {
                "element_type": self.element_type,
                "properties": self.properties,
                "exact_timing": [self.time_interval.start_ms, self.time_interval.end_ms]
            }
        }
    
    def validate_timing(self) -> bool:
        return True  # Any duration is valid for millisecond precision

class TreePlanningAlgorithm:
    """Main tree-based planning algorithm that breaks down video creation hierarchically."""
    
    def __init__(self):
        self.root: Optional[ConceptNode] = None
        self.execution_timeline: List[Dict[str, Any]] = []
        self.universal_brand_generator = UniversalBrandIntelligenceGenerator()
        self.professional_standards = self._initialize_professional_standards()
    
    def _initialize_professional_standards(self) -> Dict[str, Any]:
        """Initialize professional video production standards with precision timing."""
        return {
            'minimum_shot_duration_ms': 800,   # 0.8s minimum for dynamic content
            'maximum_shot_duration_ms': 12000, # 12s maximum for establishing shots
            'optimal_scene_duration_ms': 8000, # 8s optimal for commercial pacing
            'precision_scene_control': True,   # Enable second-by-second control
            'frame_rate': 30,                  # 30fps for precise timing
            'keyframe_interval_ms': 500,       # Keyframes every 500ms
            'transition_buffer_ms': 83,        # 2.5 frames buffer (83ms at 30fps)
            'subtitle_sync_tolerance_ms': 33,  # 1 frame tolerance (33ms at 30fps)
            'logo_minimum_visibility_ms': 2000, # 2s minimum brand exposure
            'call_to_action_duration_ms': 4000, # 4s for strong CTA impact
            'professional_pacing': {
                'fast': {'cuts_per_second': 2.0, 'movement_speed': 'dynamic', 'shot_variance': 0.3},
                'medium': {'cuts_per_second': 1.2, 'movement_speed': 'smooth', 'shot_variance': 0.5},
                'slow': {'cuts_per_second': 0.7, 'movement_speed': 'cinematic', 'shot_variance': 0.8}
            },
            'engineered_timing': {
                'hook_peak_attention_ms': 1200,    # Peak attention at 1.2s
                'problem_emotional_peak_ms': 2500,  # Emotional resonance at 2.5s  
                'solution_reveal_timing_ms': 1800,  # Solution reveal timing
                'proof_credibility_window_ms': 3000, # 3s credibility window
                'action_urgency_buildup_ms': 2000   # 2s urgency buildup
            }
        }
        
    def create_planning_tree(self, brand_info: Dict[str, Any], target_duration_ms: int) -> ConceptNode:
        """Create a complete planning tree from concept to millisecond level using universal brand intelligence."""
        logger.info("Creating professional hierarchical planning tree", action="tree.planning.start")
        
        # Step 1: Analyze brand with universal intelligence for professional planning
        enhanced_brand_info = self._analyze_brand_for_planning(brand_info)
        
        # Step 2: Create root concept with enhanced brand intelligence
        self.root = ConceptNode(enhanced_brand_info, target_duration_ms)
        
        # Step 3: Break down into acts (story structure) - professional standards
        self._create_professional_act_structure(self.root)
        
        # Step 4: Break down acts into scenes - brand-aware and professional
        self._create_professional_scene_breakdown()
        
        # Step 5: Break down scenes into shots - cinematically professional
        self._create_professional_shot_breakdown()
        
        # Step 6: Create moment precision - every second planned
        self._create_professional_moment_precision()
        
        # Step 7: Add millisecond-level elements - frame-perfect timing
        self._add_professional_millisecond_elements()
        
        # Step 8: Validate and optimize entire tree
        self._validate_and_optimize_tree()
        
        logger.info("Professional planning tree created with full hierarchy", action="tree.planning.complete")
        return self.root
    
    def _analyze_brand_for_planning(self, brand_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze brand using universal intelligence for professional planning."""
        brand_name = brand_info.get('brand_name', 'Brand')
        company_description = (
            brand_info.get('company_description') or 
            brand_info.get('description') or
            f"A business offering {', '.join(brand_info.get('key_benefits', ['quality solutions']))}"
        )
        
        try:
            # Get universal brand intelligence for professional planning
            universal_intelligence = self.universal_brand_generator.analyze_brand_deeply(
                brand_name, company_description, brand_info
            )
            
            # Enhanced brand info with professional insights
            enhanced_info = brand_info.copy()
            enhanced_info.update({
                'universal_intelligence': universal_intelligence,
                'business_type': universal_intelligence.business_type,
                'industry_category': universal_intelligence.industry_category,
                'authentic_personality': universal_intelligence.brand_essence.get('authentic_personality', []),
                'creative_opportunities': universal_intelligence.creative_opportunities,
                'visual_identity_cues': universal_intelligence.visual_identity_cues,
                'professional_planning': True
            })
            
            logger.info(f"Enhanced brand analysis complete for {brand_name}", 
                       action="brand.analysis.professional.complete")
            return enhanced_info
            
        except Exception as e:
            logger.warning(f"Universal brand analysis failed, using standard planning: {e}")
            enhanced_info = brand_info.copy()
            enhanced_info['professional_planning'] = False
            return enhanced_info
    
    def _create_professional_act_structure(self, root: ConceptNode) -> None:
        """Break down concept into professional story acts based on brand intelligence."""
        total_duration = root.time_interval.duration_ms
        universal_intelligence = root.brand_info.get('universal_intelligence')
        
        # Professional act structure based on brand type and intelligence
        if universal_intelligence:
            business_type = universal_intelligence.business_type
            creative_opportunities = universal_intelligence.creative_opportunities
            
            # Adapt act structure based on business type
            if business_type == 'Product':
                acts = [
                    ("hook", 0.12),      # 12% - product intrigue
                    ("problem", 0.18),   # 18% - need identification
                    ("solution", 0.40),  # 40% - product demonstration
                    ("proof", 0.20),     # 20% - credibility/results
                    ("action", 0.10)     # 10% - purchase CTA
                ]
            elif business_type == 'Service':
                acts = [
                    ("hook", 0.15),      # 15% - expertise display
                    ("problem", 0.22),   # 22% - client pain points
                    ("solution", 0.35),  # 35% - service benefits
                    ("proof", 0.18),     # 18% - case studies/testimonials
                    ("action", 0.10)     # 10% - consultation CTA
                ]
            elif business_type == 'Platform':
                acts = [
                    ("hook", 0.18),      # 18% - ecosystem value
                    ("problem", 0.20),   # 20% - connectivity issues
                    ("solution", 0.32),  # 32% - platform benefits
                    ("proof", 0.20),     # 20% - user success stories
                    ("action", 0.10)     # 10% - sign-up CTA
                ]
            else:  # Hybrid or unknown
                acts = [
                    ("hook", 0.15),      # 15% - comprehensive intro
                    ("problem", 0.20),   # 20% - multifaceted challenges
                    ("solution", 0.35),  # 35% - integrated solution
                    ("proof", 0.20),     # 20% - diverse credibility
                    ("action", 0.10)     # 10% - engagement CTA
                ]
        else:
            # Fallback professional structure
            acts = [
                ("hook", 0.15),      
                ("problem", 0.20),   
                ("solution", 0.35),  
                ("proof", 0.20),     
                ("action", 0.10)     
            ]
        
        # Apply professional timing standards
        current_time = 0
        for act_name, ratio in acts:
            act_duration = int(total_duration * ratio)
            
            # Ensure minimum act duration for professional quality
            min_act_duration = 3000  # 3 seconds minimum
            act_duration = max(act_duration, min_act_duration)
            
            act_interval = TimeInterval(current_time, current_time + act_duration)
            act_node = ActNode(f"act_{act_name}", act_name, act_interval)
            
            # Add professional metadata
            act_node.metadata.update({
                'business_type_optimized': True,
                'professional_timing': True,
                'brand_aligned': universal_intelligence is not None
            })
            
            root.add_child(act_node)
            current_time += act_duration
    
    def _create_professional_scene_breakdown(self) -> None:
        """Break down each act into engineered scenes with precise timing control."""
        for act_node in self.root.children:
            if not isinstance(act_node, ActNode):
                continue
                
            act_duration = act_node.time_interval.duration_ms
            universal_intelligence = self.root.brand_info.get('universal_intelligence')
            
            # Engineered scene timing with psychological impact points
            engineered_timing = self.professional_standards['engineered_timing']
            optimal_scene_duration = self.professional_standards['optimal_scene_duration_ms']
            
            # Calculate engineered scene breakdown based on act purpose
            scene_timings = self._calculate_engineered_scene_timings(act_node.act_purpose, act_duration, engineered_timing)
            
            current_time = act_node.time_interval.start_ms
            for i, scene_duration in enumerate(scene_timings):
                scene_end = min(current_time + scene_duration, act_node.time_interval.end_ms)
                scene_interval = TimeInterval(current_time, scene_end)
                
                # Generate professional visual concept with precise timing awareness
                visual_concept = self._generate_engineered_visual_concept(
                    act_node.act_purpose, i, len(scene_timings), scene_duration, universal_intelligence
                )
                
                scene_node = SceneNode(f"scene_{act_node.node_id}_{i}", visual_concept, scene_interval)
                
                # Add engineered scene elements with timing precision
                self._enhance_scene_with_engineered_elements(scene_node, act_node, universal_intelligence, i, scene_duration)
                
                act_node.add_child(scene_node)
                current_time = scene_end
    
    def _calculate_engineered_scene_timings(self, act_purpose: str, act_duration: int, engineered_timing: Dict[str, int]) -> List[int]:
        """Calculate precise scene timings based on psychological impact engineering."""
        if act_purpose == "hook":
            # Hook timing: Fast initial grab + sustained attention
            peak_attention = engineered_timing['hook_peak_attention_ms']
            if act_duration <= 4000:  # Short hook
                return [peak_attention, act_duration - peak_attention] if act_duration > peak_attention else [act_duration]
            else:  # Extended hook
                return [peak_attention, (act_duration - peak_attention) // 2, act_duration - peak_attention - (act_duration - peak_attention) // 2]
                
        elif act_purpose == "problem":
            # Problem timing: Build-up + emotional peak + reinforcement  
            emotional_peak = engineered_timing['problem_emotional_peak_ms']
            if act_duration <= 5000:  # Short problem
                buildup = min(1500, act_duration // 3)
                return [buildup, act_duration - buildup]
            else:  # Extended problem
                buildup = min(2000, act_duration // 3)
                peak = min(emotional_peak, act_duration // 2)
                return [buildup, peak, act_duration - buildup - peak]
                
        elif act_purpose == "solution":
            # Solution timing: Reveal + demonstration + impact
            reveal_timing = engineered_timing['solution_reveal_timing_ms']
            third = act_duration // 3
            return [reveal_timing, third, act_duration - reveal_timing - third]
            
        elif act_purpose == "proof":
            # Proof timing: Credibility window with multiple evidence points
            credibility_window = engineered_timing['proof_credibility_window_ms']
            if act_duration <= credibility_window:
                return [act_duration]
            else:
                num_evidence_points = min(3, act_duration // 2000)  # Max 3 evidence points
                return [act_duration // num_evidence_points] * num_evidence_points
                
        elif act_purpose == "action":
            # Action timing: Urgency buildup + CTA + reinforcement
            urgency_buildup = engineered_timing['action_urgency_buildup_ms']
            if act_duration <= 3000:
                return [act_duration // 2, act_duration - act_duration // 2]
            else:
                cta_duration = min(2000, act_duration // 2)
                return [urgency_buildup, cta_duration, act_duration - urgency_buildup - cta_duration]
        
        # Fallback: Equal distribution
        optimal_duration = self.professional_standards['optimal_scene_duration_ms']
        num_scenes = max(1, act_duration // optimal_duration)
        return [act_duration // num_scenes] * num_scenes
    
    def _generate_engineered_visual_concept(self, act_purpose: str, scene_index: int, total_scenes: int, 
                                          scene_duration: int, universal_intelligence) -> str:
        """Generate visual concepts engineered for precise timing and psychological impact."""
        duration_seconds = scene_duration / 1000.0
        
        # Timing-aware concept generation
        if universal_intelligence:
            brand_type = universal_intelligence.business_type
            creative_opportunities = universal_intelligence.creative_opportunities[:3] if universal_intelligence.creative_opportunities else []
            
            # Engineered concepts based on timing and psychology
            timing_concepts = {
                "hook": [
                    f"instant_attention_grabbing_{brand_type}_reveal_{duration_seconds:.1f}s",
                    f"curiosity_generating_{creative_opportunities[0] if creative_opportunities else 'mystery'}_opening_{duration_seconds:.1f}s",
                    f"dynamic_brand_introduction_with_{duration_seconds:.1f}s_impact"
                ],
                "problem": [
                    f"emotional_pain_point_visualization_{duration_seconds:.1f}s_buildup",
                    f"relatable_struggle_demonstration_in_{duration_seconds:.1f}s",
                    f"authentic_challenge_showcase_{brand_type}_context_{duration_seconds:.1f}s"
                ],
                "solution": [
                    f"brand_solution_reveal_with_{duration_seconds:.1f}s_demonstration",
                    f"transformation_visual_{creative_opportunities[1] if len(creative_opportunities) > 1 else 'benefit'}_in_{duration_seconds:.1f}s",
                    f"before_after_comparison_{duration_seconds:.1f}s_impact"
                ],
                "proof": [
                    f"credibility_evidence_{scene_index + 1}_with_{duration_seconds:.1f}s_focus",
                    f"testimonial_moment_{brand_type}_success_{duration_seconds:.1f}s",
                    f"results_showcase_scene_{scene_index + 1}_lasting_{duration_seconds:.1f}s"
                ],
                "action": [
                    f"urgency_building_CTA_{duration_seconds:.1f}s_window",
                    f"compelling_next_step_{brand_type}_focused_{duration_seconds:.1f}s",
                    f"conversion_optimized_action_prompt_{duration_seconds:.1f}s"
                ]
            }
            
            concepts = timing_concepts.get(act_purpose, [f"professional_{act_purpose}_scene_{duration_seconds:.1f}s"])
            selected_concept = concepts[min(scene_index, len(concepts) - 1)]
            
            return f"engineered_{selected_concept}_professionally_timed"
            
        else:
            # Fallback engineered concepts without brand intelligence
            return f"engineered_professional_{act_purpose}_scene_{scene_index + 1}_duration_{duration_seconds:.1f}s"
    
    def _enhance_scene_with_engineered_elements(self, scene_node: SceneNode, act_node: ActNode, 
                                              universal_intelligence, scene_index: int, scene_duration: int) -> None:
        """Enhance scene with engineered timing and psychological impact elements."""
        duration_seconds = scene_duration / 1000.0
        
        # Professional timing metadata with engineering precision
        scene_node.metadata.update({
            'engineered_timing': True,
            'precise_duration_ms': scene_duration,
            'psychological_impact_optimized': True,
            'act_purpose': act_node.act_purpose,
            'scene_position': scene_index,
            'timing_precision': 'frame_perfect',
            'second_by_second_control': True,
            'duration_engineered': f"{duration_seconds:.2f}s"
        })
        
        # Add timing-specific enhancements
        if act_node.act_purpose == "hook" and scene_index == 0:
            scene_node.metadata['attention_optimization'] = {
                'peak_attention_timing_ms': 1200,
                'visual_impact_maximized': True,
                'curiosity_generation': 'immediate'
            }
        elif act_node.act_purpose == "problem":
            scene_node.metadata['emotional_engineering'] = {
                'empathy_buildup_ms': min(1500, scene_duration // 2),
                'emotional_peak_timing': 'mathematically_calculated',
                'relatability_maximized': True
            }
        elif act_node.act_purpose == "solution":
            scene_node.metadata['solution_engineering'] = {
                'reveal_timing_optimized': True,
                'transformation_clarity': 'engineered',
                'benefit_demonstration_precise': f"{duration_seconds:.1f}s"
            }
        elif act_node.act_purpose == "proof":
            scene_node.metadata['credibility_engineering'] = {
                'evidence_strength_timed': True,
                'trust_building_optimized': True,
                'proof_clarity_duration': f"{duration_seconds:.1f}s"
            }
        elif act_node.act_purpose == "action":
            scene_node.metadata['conversion_engineering'] = {
                'urgency_buildup_calculated': True,
                'cta_timing_optimized': True,
                'action_motivation_precise': f"{duration_seconds:.1f}s"
            }
        
        # Brand intelligence timing enhancements
        if universal_intelligence:
            scene_node.metadata['brand_timing_intelligence'] = {
                'brand_voice_timing': universal_intelligence.brand_essence.get('brand_voice_characteristics', 'professional'),
                'business_type_optimized': universal_intelligence.business_type,
                'audience_attention_engineered': True,
                'brand_recognition_timing': 'scientifically_calculated'
            }
    
    def _generate_professional_visual_concept(self, act_purpose: str, scene_index: int, 
                                            total_scenes: int, universal_intelligence) -> str:
        """Generate professional visual concepts with brand intelligence."""
        if universal_intelligence:
            # Use creative opportunities from universal intelligence
            creative_opportunities = universal_intelligence.creative_opportunities[:3]
            visual_cues = universal_intelligence.visual_identity_cues
            
            base_concepts = {
                "hook": creative_opportunities[0] if creative_opportunities else "attention_grabbing_visual",
                "problem": f"authentic_challenge_demonstration_in_{universal_intelligence.industry_category}_context",
                "solution": f"{universal_intelligence.brand_name}_natural_solution_integration",
                "proof": f"credible_{universal_intelligence.business_type}_results_showcase",
                "action": f"compelling_{universal_intelligence.business_type}_call_to_action"
            }
            
            # Add visual identity elements
            if visual_cues and visual_cues.get('authentic_environments'):
                environment = visual_cues['authentic_environments'][0]
                concept = f"{base_concepts.get(act_purpose, 'professional_visual')}_{environment}"
            else:
                concept = base_concepts.get(act_purpose, "professional_visual")
                
            return f"professional_{concept}_scene_{scene_index + 1}_of_{total_scenes}"
        else:
            # Fallback professional concepts
            concepts = {
                "hook": f"professional_attention_grabbing_visual_scene_{scene_index + 1}",
                "problem": f"authentic_pain_point_visualization_scene_{scene_index + 1}",
                "solution": f"brand_solution_demonstration_scene_{scene_index + 1}",
                "proof": f"credibility_showcase_scene_{scene_index + 1}",
                "action": f"compelling_call_to_action_scene_{scene_index + 1}"
            }
            return concepts.get(act_purpose, f"professional_visual_scene_{scene_index + 1}")
    
    def _enhance_scene_with_professional_elements(self, scene_node: SceneNode, 
                                                act_node: ActNode, universal_intelligence) -> None:
        """Enhance scene with professional elements based on brand intelligence."""
        # Add professional timing metadata
        scene_node.metadata.update({
            'professional_standards_applied': True,
            'act_purpose': act_node.act_purpose,
            'optimal_duration': True,
            'brand_intelligence_enhanced': universal_intelligence is not None
        })
        
        # Add professional script elements
        if universal_intelligence:
            brand_voice = universal_intelligence.brand_essence.get('brand_voice_characteristics', 'professional')
            scene_node.metadata['brand_voice'] = brand_voice
            scene_node.metadata['authenticity_markers'] = universal_intelligence.authenticity_markers[:2]
        
        # Add professional camera guidance
        scene_node.metadata['professional_cinematography'] = {
            'lighting': 'professional_commercial_lighting',
            'composition': 'rule_of_thirds_with_brand_focus',
            'movement': 'smooth_purposeful_camera_work'
        }
    
    def _create_professional_shot_breakdown(self) -> None:
        """Break down scenes into professional camera shots with precise timing."""
        universal_intelligence = self.root.brand_info.get('universal_intelligence')
        
        for act_node in self.root.children:
            for scene_node in act_node.children:
                if not isinstance(scene_node, SceneNode):
                    continue
                    
                scene_duration = scene_node.time_interval.duration_ms
                
                # Professional shot calculation based on scene duration and purpose
                min_shot_duration = self.professional_standards['minimum_shot_duration_ms']
                max_shot_duration = self.professional_standards['maximum_shot_duration_ms']
                
                # Determine optimal number of shots
                if scene_duration <= 3000:  # 3 seconds or less
                    num_shots = 1
                elif scene_duration <= 6000:  # 3-6 seconds
                    num_shots = 2
                elif scene_duration <= 10000:  # 6-10 seconds
                    num_shots = 3
                else:  # > 10 seconds
                    num_shots = max(3, min(5, scene_duration // 2500))  # Max 5 shots per scene
                
                shot_duration = scene_duration // num_shots
                
                # Ensure each shot meets minimum duration
                if shot_duration < min_shot_duration:
                    num_shots = max(1, scene_duration // min_shot_duration)
                    shot_duration = scene_duration // num_shots
                
                current_time = scene_node.time_interval.start_ms
                
                for i in range(num_shots):
                    shot_end = min(current_time + shot_duration, scene_node.time_interval.end_ms)
                    shot_interval = TimeInterval(current_time, shot_end)
                    
                    # Determine professional shot type
                    shot_type = self._determine_professional_shot_type(
                        scene_node.visual_concept, i, num_shots, universal_intelligence
                    )
                    
                    shot_node = ShotNode(f"shot_{scene_node.node_id}_{i}", shot_type, shot_interval)
                    
                    # Add professional shot elements
                    self._enhance_shot_with_professional_elements(shot_node, scene_node, universal_intelligence)
                    
                    scene_node.add_child(shot_node)
                    current_time = shot_end
    
    def _determine_professional_shot_type(self, visual_concept: str, shot_index: int, 
                                        total_shots: int, universal_intelligence) -> str:
        """Determine professional shot type based on cinematic principles."""
        # Professional shot sequencing
        if total_shots == 1:
            return "hero_medium_shot"  # Single impactful shot
        elif total_shots == 2:
            return "establishing_wide" if shot_index == 0 else "impactful_close_up"
        elif total_shots == 3:
            if shot_index == 0:
                return "establishing_wide"
            elif shot_index == 1:
                return "action_medium"
            else:
                return "emotional_close_up"
        else:  # 4+ shots
            shot_sequence = ["establishing_wide", "action_medium", "detail_close_up", "emotional_reaction", "concluding_wide"]
            return shot_sequence[min(shot_index, len(shot_sequence) - 1)]
    
    def _enhance_shot_with_professional_elements(self, shot_node: ShotNode, 
                                               scene_node: SceneNode, universal_intelligence) -> None:
        """Enhance shot with professional cinematography elements."""
        # Professional camera settings
        shot_node.metadata.update({
            'professional_cinematography': True,
            'camera_settings': {
                'aperture': 'f/2.8_for_cinematic_depth',
                'focal_length': 'appropriate_for_shot_type',
                'frame_rate': '24fps_cinematic_standard',
                'color_grading': 'professional_commercial_grade'
            },
            'lighting_setup': 'three_point_lighting_with_brand_consideration',
            'audio_sync': 'frame_perfect_alignment'
        })
        
        if universal_intelligence:
            # Brand-specific enhancements
            visual_cues = universal_intelligence.visual_identity_cues
            if visual_cues and visual_cues.get('aesthetic_direction'):
                shot_node.metadata['brand_aesthetic'] = visual_cues['aesthetic_direction'][0]
    
    def _create_professional_moment_precision(self) -> None:
        """Create professional moment-level precision within shots - every second planned."""
        universal_intelligence = self.root.brand_info.get('universal_intelligence')
        
        for act_node in self.root.children:
            for scene_node in act_node.children:
                for shot_node in scene_node.children:
                    if not isinstance(shot_node, ShotNode):
                        continue
                        
                    shot_duration = shot_node.time_interval.duration_ms
                    
                    # Professional moment breakdown - every second is planned
                    optimal_moment_duration = 1500  # 1.5 seconds for professional pacing
                    min_moment_duration = 800   # 0.8 seconds minimum
                    max_moment_duration = 2500  # 2.5 seconds maximum
                    
                    # Calculate professional moment breakdown
                    num_moments = max(1, round(shot_duration / optimal_moment_duration))
                    moment_duration = shot_duration // num_moments
                    
                    # Ensure professional timing standards
                    if moment_duration < min_moment_duration:
                        num_moments = max(1, shot_duration // min_moment_duration)
                        moment_duration = shot_duration // num_moments
                    elif moment_duration > max_moment_duration:
                        num_moments = max(1, shot_duration // max_moment_duration)
                        moment_duration = shot_duration // num_moments
                    
                    current_time = shot_node.time_interval.start_ms
                    for i in range(num_moments):
                        moment_end = min(current_time + moment_duration, shot_node.time_interval.end_ms)
                        moment_interval = TimeInterval(current_time, moment_end)
                        
                        # Define professional moment action
                        action = self._define_professional_moment_action(
                            shot_node.shot_type, i, num_moments, universal_intelligence
                        )
                        
                        moment_node = MomentNode(f"moment_{shot_node.node_id}_{i}", action, moment_interval)
                        
                        # Add professional moment metadata
                        self._enhance_moment_with_professional_elements(moment_node, shot_node, universal_intelligence)
                        
                        shot_node.add_child(moment_node)
                        current_time = moment_end
    
    def _define_professional_moment_action(self, shot_type: str, moment_index: int, 
                                         total_moments: int, universal_intelligence) -> str:
        """Define professional moment actions with brand intelligence."""
        # Professional moment sequencing based on shot type
        moment_sequences = {
            "establishing_wide": ["scene_establishment", "environmental_context", "subject_introduction"],
            "hero_medium_shot": ["subject_focus", "brand_integration", "emotional_connection"],
            "action_medium": ["action_initiation", "brand_interaction", "result_demonstration"],
            "detail_close_up": ["detail_focus", "quality_showcase", "brand_element_highlight"],
            "emotional_close_up": ["emotional_capture", "authentic_reaction", "brand_connection"],
            "impactful_close_up": ["impact_moment", "brand_revelation", "emotional_payoff"],
            "concluding_wide": ["resolution_wide", "brand_conclusion", "call_to_action_setup"]
        }
        
        sequence = moment_sequences.get(shot_type, ["generic_action", "brand_moment", "transition"])
        base_action = sequence[min(moment_index, len(sequence) - 1)]
        
        # Enhance with brand intelligence
        if universal_intelligence:
            brand_type = universal_intelligence.business_type.lower()
            return f"{base_action}_{brand_type}_focused"
        else:
            return f"professional_{base_action}"
    
    def _enhance_moment_with_professional_elements(self, moment_node: MomentNode, 
                                                 shot_node: ShotNode, universal_intelligence) -> None:
        """Enhance moment with professional timing and brand elements."""
        moment_node.metadata.update({
            'professional_timing': True,
            'frame_perfect_execution': True,
            'shot_type_context': shot_node.shot_type,
            'duration_optimized': True
        })
        
        if universal_intelligence:
            # Add brand-specific timing considerations
            moment_node.metadata.update({
                'brand_voice_timing': universal_intelligence.brand_essence.get('brand_voice_characteristics'),
                'industry_appropriate': universal_intelligence.industry_category,
                'authenticity_maintained': True
            })
    
    def _add_professional_millisecond_elements(self) -> None:
        """Add professional millisecond-precision elements with brand intelligence."""
        total_duration = self.root.time_interval.duration_ms
        universal_intelligence = self.root.brand_info.get('universal_intelligence')
        
        # Professional logo timing standards
        logo_min_duration = self.professional_standards['logo_minimum_visibility_ms']
        cta_duration = self.professional_standards['call_to_action_duration_ms']
        
        # Brand-aware logo placement
        if universal_intelligence:
            # Early brand introduction for recognition
            logo_start = 800   # 0.8 seconds - after initial hook
            logo_end = min(2500, logo_start + logo_min_duration)  # Ensure minimum visibility
            
            logo_interval = TimeInterval(logo_start, logo_end)
            logo_node = MillisecondNode("professional_logo_intro", "brand_logo", logo_interval)
            
            # Enhanced logo metadata with brand intelligence
            logo_node.properties.update({
                'brand_name': universal_intelligence.brand_name,
                'business_type': universal_intelligence.business_type,
                'placement_style': 'professional_commercial',
                'opacity_animation': 'fade_in_hold_fade_out',
                'size_optimization': 'readability_focused'
            })
            
            self.root.add_child(logo_node)
            
            # Professional call-to-action timing
            cta_start = total_duration - cta_duration
            cta_end = total_duration - 200  # Leave 200ms buffer
            cta_interval = TimeInterval(cta_start, cta_end)
            
            cta_node = MillisecondNode("professional_cta", "call_to_action", cta_interval)
            cta_node.properties.update({
                'cta_type': f"{universal_intelligence.business_type}_focused",
                'brand_aligned': True,
                'urgency_level': 'professional',
                'contact_prominence': 'high_visibility'
            })
            
            self.root.add_child(cta_node)
        else:
            # Standard professional timing
            logo_interval = TimeInterval(600, 2000)
            logo_node = MillisecondNode("standard_logo", "logo", logo_interval)
            self.root.add_child(logo_node)
            
            cta_interval = TimeInterval(total_duration - 3000, total_duration - 400)
            cta_node = MillisecondNode("standard_cta", "call_to_action", cta_interval)
            self.root.add_child(cta_node)
        
        # Add professional subtitle sync points
        self._add_subtitle_sync_points()
    
    def _add_subtitle_sync_points(self) -> None:
        """Add millisecond-perfect subtitle synchronization points."""
        subtitle_tolerance = self.professional_standards['subtitle_sync_tolerance_ms']
        
        # Calculate subtitle timing for each scene
        for act_node in self.root.children:
            for scene_node in act_node.children:
                if isinstance(scene_node, SceneNode):
                    # Professional subtitle timing - start 50ms after scene start
                    subtitle_start = scene_node.time_interval.start_ms + 50
                    subtitle_end = scene_node.time_interval.end_ms - 100  # End 100ms before scene end
                    
                    if subtitle_end > subtitle_start:
                        subtitle_interval = TimeInterval(subtitle_start, subtitle_end)
                        subtitle_node = MillisecondNode(
                            f"subtitle_{scene_node.node_id}", 
                            "subtitle", 
                            subtitle_interval
                        )
                        
                        subtitle_node.properties.update({
                            'sync_precision': f"±{subtitle_tolerance}ms",
                            'scene_alignment': scene_node.node_id,
                            'professional_timing': True,
                            'readability_optimized': True
                        })
                        
                        scene_node.add_child(subtitle_node)
    
    def _validate_and_optimize_tree(self) -> None:
        """Validate and optimize the entire planning tree for professional standards."""
        if not self.root:
            return
            
        logger.info("Validating professional planning tree", action="tree.validation.start")
        
        # Validate timing consistency
        self._validate_timing_consistency()
        
        # Optimize for professional standards
        self._optimize_professional_flow()
        
        # Validate brand alignment
        self._validate_brand_alignment()
        
        logger.info("Professional tree validation complete", action="tree.validation.complete")
    
    def _validate_timing_consistency(self) -> None:
        """Validate that all timing is consistent and professional."""
        total_planned = sum(child.time_interval.duration_ms for child in self.root.children 
                           if isinstance(child, ActNode))
        
        tolerance = 200  # 200ms tolerance
        if abs(total_planned - self.root.time_interval.duration_ms) > tolerance:
            logger.warning(f"Timing inconsistency detected: planned={total_planned}ms, target={self.root.time_interval.duration_ms}ms")
    
    def _optimize_professional_flow(self) -> None:
        """Optimize the tree for professional video flow."""
        # Add transition buffers between scenes
        transition_buffer = self.professional_standards['transition_buffer_ms']
        
        for act_node in self.root.children:
            if isinstance(act_node, ActNode):
                for i, scene_node in enumerate(act_node.children):
                    if isinstance(scene_node, SceneNode) and i > 0:
                        # Add transition buffer metadata
                        scene_node.metadata['transition_buffer_ms'] = transition_buffer
    
    def _validate_brand_alignment(self) -> None:
        """Validate that all elements maintain brand alignment."""
        universal_intelligence = self.root.brand_info.get('universal_intelligence')
        
        if universal_intelligence:
            brand_elements = 0
            total_elements = 0
            
            for act_node in self.root.children:
                for scene_node in act_node.children:
                    total_elements += 1
                    if scene_node.metadata.get('brand_intelligence_enhanced'):
                        brand_elements += 1
            
            alignment_ratio = brand_elements / total_elements if total_elements > 0 else 0
            
            if alignment_ratio < 0.8:  # 80% brand alignment minimum
                logger.warning(f"Low brand alignment: {alignment_ratio:.2%} of elements are brand-aware")
            else:
                logger.info(f"Professional brand alignment achieved: {alignment_ratio:.2%} brand-aware elements")
    
    def _generate_visual_concept(self, act_purpose: str, scene_index: int) -> str:
        """Generate visual concept based on act purpose and scene position."""
        concepts = {
            "hook": [
                "attention_grabbing_visual",
                "dynamic_action_shot", 
                "intriguing_close_up"
            ],
            "problem": [
                "pain_point_visualization",
                "struggle_demonstration",
                "before_state_showing"
            ],
            "solution": [
                "product_hero_shot",
                "transformation_visual",
                "benefit_demonstration"
            ],
            "proof": [
                "credibility_visual",
                "testimonial_moment",
                "result_showcase"
            ],
            "action": [
                "call_to_action_visual",
                "contact_information",
                "urgency_creator"
            ]
        }
        
        act_concepts = concepts.get(act_purpose, ["generic_visual"])
        return act_concepts[min(scene_index, len(act_concepts) - 1)]
    
    def _determine_shot_type(self, visual_concept: str, shot_index: int, total_shots: int) -> str:
        """Determine camera shot type based on visual concept and position."""
        if total_shots == 1:
            return "medium_shot"
        
        # First shot usually establishes, last shot usually closes
        if shot_index == 0:
            return "wide_shot"
        elif shot_index == total_shots - 1:
            return "close_up"
        else:
            return "medium_shot"
    
    def _define_moment_action(self, shot_type: str, moment_index: int) -> str:
        """Define specific action for each moment."""
        actions = {
            "wide_shot": ["establish_scene", "show_context"],
            "medium_shot": ["show_interaction", "demonstrate_feature"],
            "close_up": ["capture_emotion", "show_detail"]
        }
        
        shot_actions = actions.get(shot_type, ["generic_action"])
        return shot_actions[min(moment_index, len(shot_actions) - 1)]
    
    def generate_complete_execution_timeline(self) -> List[Dict[str, Any]]:
        """Generate complete execution timeline with millisecond precision."""
        if not self.root:
            return []
        
        self.execution_timeline = []
        self._traverse_and_generate_timeline(self.root)
        
        # Sort by start time
        self.execution_timeline.sort(key=lambda x: x.get('start_ms', 0))
        
        return self.execution_timeline
    
    def _traverse_and_generate_timeline(self, node: PlanningNode) -> None:
        """Recursively traverse tree and generate timeline entries."""
        execution_plan = node.generate_execution_plan()
        execution_plan['node_id'] = node.node_id
        execution_plan['node_type'] = node.node_type.value
        execution_plan['start_ms'] = node.time_interval.start_ms
        execution_plan['end_ms'] = node.time_interval.end_ms
        execution_plan['duration_ms'] = node.time_interval.duration_ms
        
        self.execution_timeline.append(execution_plan)
        
        # Traverse children
        for child in node.children:
            self._traverse_and_generate_timeline(child)
    
    def export_tree_structure(self) -> Dict[str, Any]:
        """Export the complete tree structure for analysis and debugging."""
        if not self.root:
            return {}
        
        return self._export_node(self.root)
    
    def _export_node(self, node: PlanningNode) -> Dict[str, Any]:
        """Export a single node and its children."""
        return {
            'node_id': node.node_id,
            'node_type': node.node_type.value,
            'time_interval': {
                'start_ms': node.time_interval.start_ms,
                'end_ms': node.time_interval.end_ms,
                'duration_ms': node.time_interval.duration_ms
            },
            'creative_intent': {
                'emotion': node.creative_intent.emotion,
                'purpose': node.creative_intent.purpose,
                'visual_style': node.creative_intent.visual_style,
                'pacing': node.creative_intent.pacing
            },
            'execution_plan': node.generate_execution_plan(),
            'children': [self._export_node(child) for child in node.children]
        }