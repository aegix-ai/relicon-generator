"""
Relicon AI Ad Creator - AI Agents Package
Revolutionary AI agent system with modular architecture
"""

# Import from modular master planner components
from .master_planner import MasterPlannerAgent, master_planner
from .scene_architect import SceneArchitectAgent, scene_architect

# Import specialized planning modules
from .planning import (
    BrandAnalyzer, NarrativeDesigner, SceneBreakdown, TimingCalculator,
    BrandIntegrator, PlanFinalizer
)

# Import scene architecture modules
from .architecture import (
    ComponentCalculator, TimingOptimizer, PromptGenerator, 
    SceneAssembler, TemplateManager
)

__all__ = [
    # Main Agents
    'MasterPlannerAgent', 'master_planner',
    'SceneArchitectAgent', 'scene_architect',
    
    # Planning Modules
    'BrandAnalyzer', 'NarrativeDesigner', 'SceneBreakdown', 
    'TimingCalculator', 'BrandIntegrator', 'PlanFinalizer',
    
    # Architecture Modules
    'ComponentCalculator', 'TimingOptimizer', 'PromptGenerator',
    'SceneAssembler', 'TemplateManager'
]
