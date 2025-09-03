"""
Universal Brand Intelligence Generator
Uses GPT-4o's deep creativity to understand any business, product, or service without assumptions.
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from core.logger import get_logger
from providers.openai import OpenAIProvider

logger = get_logger(__name__)

@dataclass
class UniversalBrandIntelligence:
    """Universal brand intelligence extracted via GPT-4o analysis."""
    brand_name: str
    business_type: str  # Dynamically determined
    industry_category: str  # Discovered, not predetermined
    brand_essence: Dict[str, Any]  # Core identity elements
    customer_profile: Dict[str, Any]  # Real customer understanding
    value_architecture: Dict[str, Any]  # Value delivery structure
    emotional_landscape: Dict[str, Any]  # Emotional positioning
    competitive_context: Dict[str, Any]  # Market positioning
    authenticity_markers: List[str]  # What makes this brand genuine
    creative_opportunities: List[str]  # Unique storytelling angles
    visual_identity_cues: Dict[str, Any]  # Brand-aligned visual direction

class UniversalBrandIntelligenceGenerator:
    """Generates deep brand intelligence for any business using GPT-4o creativity."""
    
    def __init__(self):
        self.openai_provider = OpenAIProvider()
    
    def analyze_brand_deeply(self, 
                           brand_name: str,
                           company_description: str,
                           additional_context: Optional[Dict[str, Any]] = None) -> UniversalBrandIntelligence:
        """
        Use GPT-4o to deeply analyze and understand any brand/business.
        
        Args:
            brand_name: The brand or company name
            company_description: Description of the business
            additional_context: Any additional context available
            
        Returns:
            Deep brand intelligence across all dimensions
        """
        try:
            logger.info(f"Analyzing brand intelligence for {brand_name}",
                       action="brand.intelligence.analysis.start")
            
            # Generate comprehensive brand analysis using GPT-4o
            brand_analysis = self._generate_comprehensive_brand_analysis(
                brand_name, company_description, additional_context
            )
            
            # Extract customer insights
            customer_insights = self._extract_customer_intelligence(
                brand_name, company_description, brand_analysis
            )
            
            # Generate creative storytelling opportunities
            creative_opportunities = self._discover_creative_opportunities(
                brand_name, brand_analysis, customer_insights
            )
            
            # Synthesize into universal brand intelligence
            brand_intelligence = self._synthesize_brand_intelligence(
                brand_name, brand_analysis, customer_insights, creative_opportunities
            )
            
            logger.info("Universal brand intelligence generated successfully",
                       action="brand.intelligence.analysis.complete")
            
            return brand_intelligence
            
        except Exception as e:
            logger.error(f"Brand intelligence analysis failed: {e}", exc_info=True)
            return self._generate_fallback_intelligence(brand_name, company_description)
    
    def _generate_comprehensive_brand_analysis(self, 
                                             brand_name: str, 
                                             company_description: str,
                                             additional_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive brand analysis using GPT-4o."""
        
        context_info = ""
        if additional_context:
            context_info = f"\n**ADDITIONAL CONTEXT:**\n{json.dumps(additional_context, indent=2)}"
        
        prompt = f"""
You are the world's foremost brand strategist and business intelligence analyst, with deep expertise across every industry and business model. You've worked with startups to Fortune 500 companies across technology, retail, services, manufacturing, healthcare, finance, and beyond.

**BRAND ANALYSIS ASSIGNMENT:**
Brand: {brand_name}
Business Description: {company_description}{context_info}

**YOUR MISSION:**
Conduct a comprehensive, unbiased analysis of this brand to understand its true essence, market position, and customer reality. Use your deep business intelligence to uncover insights that go beyond surface descriptions.

**ANALYSIS FRAMEWORK:**

**1. BUSINESS REALITY ASSESSMENT**
• What type of business is this actually? (Don't assume - analyze)
• What industry/category does it truly operate in?
• Is this B2B, B2C, B2B2C, marketplace, platform, service, product, hybrid?
• What's the actual business model and revenue structure?
• What stage is this business in? (startup, growth, mature, transformation)

**2. VALUE ARCHITECTURE ANALYSIS**
• What core problem does this business solve?
• What's the real value proposition (not marketing speak)?
• How does value flow between this business and its customers?
• What's the actual competitive differentiation?
• What makes this business essential vs. nice-to-have?

**3. CUSTOMER REALITY MAPPING**
• Who are the actual customers (not target demographics)?
• What's their real situation, needs, and motivations?
• What's their journey and decision-making process?
• What emotional and functional jobs are they hiring this business for?
• What are their genuine pain points and desired outcomes?

**4. BRAND ESSENCE DISCOVERY**
• What personality traits does this brand naturally embody?
• What values are truly demonstrated (not just claimed)?
• What's the authentic brand voice based on business reality?
• What cultural and emotional territory does this brand own?
• What promises can this brand authentically make and keep?

**5. MARKET CONTEXT ANALYSIS**
• Who are the real competitors and alternatives?
• What's this brand's genuine position in the market?
• What trends and forces shape this business environment?
• What opportunities and threats exist?
• How is this industry/category evolving?

**CRITICAL REQUIREMENTS:**
• Base analysis on business reality, not marketing claims
• Identify genuine differentiators, not generic advantages
• Understand real customer motivations, not assumed demographics
• Discover authentic brand traits, not aspirational positioning
• Consider industry dynamics and competitive forces

**DELIVERABLE FORMAT:**
Provide analysis in structured JSON format with:

```json
{{
    "business_classification": {{
        "business_type": "Specific business model/type",
        "industry_category": "Primary industry classification",
        "business_model": "How the business operates and generates value",
        "market_stage": "Current business lifecycle stage"
    }},
    "value_proposition": {{
        "core_problem_solved": "Primary customer problem addressed",
        "unique_value_delivery": "How value is uniquely delivered", 
        "competitive_differentiation": "Genuine competitive advantages",
        "value_chain_position": "Role in broader value ecosystem"
    }},
    "customer_reality": {{
        "primary_customer_profile": "Actual customer characteristics",
        "customer_motivations": "Real drivers of customer behavior",
        "customer_journey_insights": "Key journey moments and decisions",
        "emotional_drivers": "Underlying emotional motivations",
        "functional_needs": "Practical needs being fulfilled"
    }},
    "brand_essence": {{
        "authentic_personality": "Genuine brand personality traits",
        "core_values_demonstrated": "Values actually demonstrated",
        "brand_voice_characteristics": "Natural communication style",
        "emotional_territory": "Emotional space the brand occupies",
        "brand_promises": "Promises the brand can authentically make"
    }},
    "market_dynamics": {{
        "competitive_landscape": "Real competitive context",
        "market_position": "Current market standing",
        "industry_trends": "Relevant industry developments",
        "strategic_opportunities": "Key growth/positioning opportunities",
        "market_challenges": "Primary market obstacles"
    }}
}}
```

Analyze deeply. Think strategically. Uncover truth.
        """
        
        try:
            response = self.openai_provider.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world-class brand strategist and business intelligence analyst with expertise across all industries and business models. Your analysis is based on deep business understanding and market reality."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            response_content = response.choices[0].message.content
            
            # Parse JSON response
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
            else:
                json_content = response_content
            
            analysis = json.loads(json_content)
            return analysis
            
        except Exception as e:
            logger.error(f"GPT-4o brand analysis failed: {e}")
            raise
    
    def _extract_customer_intelligence(self, 
                                     brand_name: str,
                                     company_description: str, 
                                     brand_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract deep customer intelligence using GPT-4o."""
        
        customer_context = brand_analysis.get('customer_reality', {})
        value_prop = brand_analysis.get('value_proposition', {})
        
        prompt = f"""
You are a world-renowned customer intelligence expert and behavioral psychologist. You understand human behavior, decision-making, and the complex relationship between customers and businesses.

**CUSTOMER INTELLIGENCE ASSIGNMENT:**
Brand: {brand_name}
Business: {company_description}

**EXISTING ANALYSIS:**
Customer Reality: {json.dumps(customer_context, indent=2)}
Value Proposition: {json.dumps(value_prop, indent=2)}

**YOUR MISSION:**
Go deeper into customer psychology, behavior, and relationship with this brand. Understand the human truth behind the business relationship.

**ANALYSIS DIMENSIONS:**

**1. CUSTOMER PSYCHOLOGY**
• What are customers really thinking and feeling?
• What internal narratives drive their decisions?
• What fears, hopes, and aspirations are at play?
• How do they perceive themselves in relation to this business?

**2. BEHAVIORAL REALITY** 
• How do customers actually discover and engage with this business?
• What's the real decision-making process (not the rational one)?
• What influences and triggers move them to action?
• What stops them or creates friction?

**3. RELATIONSHIP DYNAMICS**
• What kind of relationship do customers have with this brand?
• What role does this business play in their life/work?
• How do they talk about this brand to others?
• What would they miss if this business disappeared?

**4. DEEPER MOTIVATIONS**
• What job is this customer really hiring this business to do?
• What transformation are they seeking?
• What status, identity, or belonging needs are involved?
• What success means to them through this relationship?

**DELIVERABLE:**
```json
{{
    "customer_psychology": {{
        "internal_narrative": "What customers tell themselves",
        "emotional_drivers": "Deep emotional motivations",
        "identity_connection": "How brand relates to customer identity",
        "transformation_seeking": "Change/outcome they desire"
    }},
    "behavioral_patterns": {{
        "discovery_behavior": "How they find and evaluate",
        "decision_triggers": "What moves them to action", 
        "usage_patterns": "How they actually engage",
        "advocacy_behavior": "How they share and recommend"
    }},
    "relationship_dynamics": {{
        "brand_relationship_type": "Nature of customer-brand relationship",
        "emotional_bond_level": "Depth of emotional connection",
        "trust_factors": "What builds and maintains trust",
        "loyalty_drivers": "What keeps them coming back"
    }},
    "success_metrics": {{
        "customer_success_definition": "How customers define success",
        "value_realization_moments": "When value becomes clear",
        "satisfaction_indicators": "Signs of customer satisfaction",
        "growth_opportunities": "How relationship can deepen"
    }}
}}
```

Understand the human truth. Reveal genuine insights.
        """
        
        try:
            response = self.openai_provider.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world-class customer intelligence expert and behavioral psychologist who understands the deep psychology behind customer-business relationships."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,
                max_tokens=2500
            )
            
            response_content = response.choices[0].message.content
            
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
            else:
                json_content = response_content
            
            return json.loads(json_content)
            
        except Exception as e:
            logger.error(f"Customer intelligence extraction failed: {e}")
            return {}
    
    def _discover_creative_opportunities(self, 
                                       brand_name: str,
                                       brand_analysis: Dict[str, Any],
                                       customer_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Discover creative storytelling opportunities using GPT-4o."""
        
        prompt = f"""
You are a legendary Creative Director who has created iconic campaigns across every industry. You see storytelling opportunities that others miss and understand how to turn business truths into compelling narratives.

**CREATIVE DISCOVERY ASSIGNMENT:**
Brand: {brand_name}

**BRAND INTELLIGENCE:**
{json.dumps(brand_analysis, indent=2)}

**CUSTOMER INSIGHTS:**
{json.dumps(customer_insights, indent=2)}

**YOUR MISSION:**
Discover unique, authentic storytelling opportunities that bring this brand to life through genuine human stories. Find the creative angles that competitors can't replicate because they're rooted in this brand's unique truth.

**CREATIVE EXPLORATION:**

**1. AUTHENTIC STORY TERRITORIES**
• What human stories are uniquely this brand's to tell?
• What moments of transformation, discovery, or connection are authentic to this business?
• What customer journey moments contain inherent drama or emotion?
• What behind-the-scenes realities could become compelling narratives?

**2. VISUAL STORYTELLING OPPORTUNITIES**
• What environments, settings, and contexts authentically represent this brand?
• What visual metaphors or symbols naturally align with the brand essence?
• What colors, textures, and aesthetics emerge from the business reality?
• What camera angles and perspectives serve the authentic story?

**3. CHARACTER ARCHETYPES**
• Who are the authentic heroes in this brand's ecosystem?
• What customer personas become compelling story characters?
• What employee/founder stories embody the brand essence?
• What community figures represent the brand's impact?

**4. EMOTIONAL LANDSCAPES** 
• What emotional territories does this brand naturally own?
• What feelings and moods authentically surround this business?
• What emotional transformations happen through customer interaction?
• What aspirational yet achievable emotional states does this brand enable?

**5. UNIQUE CREATIVE ANGLES**
• What aspects of this business have never been showcased?
• What unexpected perspectives could reveal brand truth?
• What industry assumptions could this brand challenge or reframe?
• What creative approaches would competitors find impossible to copy?

**DELIVERABLE:**
```json
{{
    "story_territories": {{
        "primary_narratives": ["Unique stories this brand can authentically tell"],
        "customer_journey_moments": ["Compelling moments in customer experience"],
        "transformation_stories": ["Change/growth narratives"],
        "community_impact_stories": ["How brand affects broader community"]
    }},
    "visual_opportunities": {{
        "authentic_environments": ["Settings that genuinely represent brand"],
        "visual_metaphors": ["Symbols and imagery that embody brand essence"],
        "aesthetic_direction": ["Colors, textures, styles that feel authentic"],
        "cinematographic_approach": ["Camera and lighting that serves story"]
    }},
    "character_archetypes": {{
        "hero_profiles": ["Protagonist types for brand stories"],
        "supporting_characters": ["Secondary characters in brand ecosystem"],
        "authentic_casting": ["Real person types vs. actors"],
        "diversity_opportunities": ["Authentic representation opportunities"]
    }},
    "emotional_opportunities": {{
        "primary_emotional_territories": ["Core emotions brand owns"],
        "emotional_journey_mapping": ["Emotional arc opportunities"],
        "mood_and_tone": ["Authentic emotional atmosphere"],
        "aspirational_emotions": ["Achievable aspirational feelings"]
    }},
    "creative_differentiation": {{
        "unique_angles": ["Creative approaches only this brand can take"],
        "untold_stories": ["Aspects of business never showcased"],
        "industry_disruption": ["Ways to challenge category norms"],
        "competitive_moats": ["Creative territories competitors can't claim"]
    }}
}}
```

Find the stories only this brand can tell. Discover authentic creative gold.
        """
        
        try:
            response = self.openai_provider.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a legendary Creative Director with the ability to see unique storytelling opportunities in any business and turn authentic truths into compelling narratives."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.9,  # Higher creativity for creative opportunities
                max_tokens=3000
            )
            
            response_content = response.choices[0].message.content
            
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
            else:
                json_content = response_content
            
            return json.loads(json_content)
            
        except Exception as e:
            logger.error(f"Creative opportunities discovery failed: {e}")
            return {}
    
    def _synthesize_brand_intelligence(self, 
                                     brand_name: str,
                                     brand_analysis: Dict[str, Any],
                                     customer_insights: Dict[str, Any], 
                                     creative_opportunities: Dict[str, Any]) -> UniversalBrandIntelligence:
        """Synthesize all analyses into universal brand intelligence."""
        
        try:
            # Extract business classification
            business_class = brand_analysis.get('business_classification', {})
            business_type = business_class.get('business_type', 'Business')
            industry_category = business_class.get('industry_category', 'General')
            
            # Extract brand essence
            brand_essence = brand_analysis.get('brand_essence', {})
            
            # Extract customer profile
            customer_reality = brand_analysis.get('customer_reality', {})
            customer_psychology = customer_insights.get('customer_psychology', {})
            
            # Combine customer understanding
            customer_profile = {
                **customer_reality,
                **customer_psychology,
                'behavioral_patterns': customer_insights.get('behavioral_patterns', {}),
                'relationship_dynamics': customer_insights.get('relationship_dynamics', {})
            }
            
            # Extract value architecture
            value_architecture = brand_analysis.get('value_proposition', {})
            
            # Build emotional landscape
            emotional_landscape = {
                'brand_emotional_territory': brand_essence.get('emotional_territory', ''),
                'customer_emotional_drivers': customer_psychology.get('emotional_drivers', []),
                'emotional_opportunities': creative_opportunities.get('emotional_opportunities', {}),
                'transformation_emotions': customer_psychology.get('transformation_seeking', '')
            }
            
            # Extract competitive context
            competitive_context = brand_analysis.get('market_dynamics', {})
            
            # Generate authenticity markers
            authenticity_markers = self._extract_authenticity_markers(
                brand_analysis, customer_insights, creative_opportunities
            )
            
            # Extract creative opportunities list
            creative_opps_list = []
            story_territories = creative_opportunities.get('story_territories', {})
            if story_territories.get('primary_narratives'):
                creative_opps_list.extend(story_territories['primary_narratives'])
            if creative_opportunities.get('creative_differentiation', {}).get('unique_angles'):
                creative_opps_list.extend(creative_opportunities['creative_differentiation']['unique_angles'])
            
            # Build visual identity cues
            visual_identity_cues = creative_opportunities.get('visual_opportunities', {})
            
            return UniversalBrandIntelligence(
                brand_name=brand_name,
                business_type=business_type,
                industry_category=industry_category,
                brand_essence=brand_essence,
                customer_profile=customer_profile,
                value_architecture=value_architecture,
                emotional_landscape=emotional_landscape,
                competitive_context=competitive_context,
                authenticity_markers=authenticity_markers,
                creative_opportunities=creative_opps_list,
                visual_identity_cues=visual_identity_cues
            )
            
        except Exception as e:
            logger.error(f"Brand intelligence synthesis failed: {e}")
            return self._generate_fallback_intelligence(brand_name, "Business description not available")
    
    def _extract_authenticity_markers(self, 
                                    brand_analysis: Dict[str, Any],
                                    customer_insights: Dict[str, Any],
                                    creative_opportunities: Dict[str, Any]) -> List[str]:
        """Extract authenticity markers from all analyses."""
        
        markers = []
        
        # From brand essence
        brand_essence = brand_analysis.get('brand_essence', {})
        if brand_essence.get('core_values_demonstrated'):
            markers.append(f"Demonstrates {', '.join(brand_essence['core_values_demonstrated'])} through actions")
        
        if brand_essence.get('brand_promises'):
            markers.append(f"Makes authentic promises: {', '.join(brand_essence['brand_promises'])}")
        
        # From customer relationship
        relationship_dynamics = customer_insights.get('relationship_dynamics', {})
        if relationship_dynamics.get('trust_factors'):
            markers.append(f"Builds trust through {', '.join(relationship_dynamics['trust_factors'])}")
        
        # From creative differentiation  
        creative_diff = creative_opportunities.get('creative_differentiation', {})
        if creative_diff.get('unique_angles'):
            markers.extend(creative_diff['unique_angles'][:2])  # Top 2 unique angles
        
        return markers
    
    def _generate_fallback_intelligence(self, 
                                      brand_name: str, 
                                      company_description: str) -> UniversalBrandIntelligence:
        """Generate fallback intelligence when GPT-4o analysis fails."""
        
        logger.warning(f"Using fallback intelligence for {brand_name}")
        
        return UniversalBrandIntelligence(
            brand_name=brand_name,
            business_type="Business",
            industry_category="General",
            brand_essence={
                'authentic_personality': ['professional', 'reliable'],
                'core_values_demonstrated': ['quality', 'service'],
                'brand_voice_characteristics': 'clear and trustworthy',
                'emotional_territory': 'trust and reliability'
            },
            customer_profile={
                'primary_customer_profile': 'Value-seeking customers',
                'customer_motivations': 'Quality solutions',
                'emotional_drivers': ['trust', 'satisfaction']
            },
            value_architecture={
                'core_problem_solved': 'Customer needs fulfillment',
                'unique_value_delivery': 'Reliable service/product delivery',
                'competitive_differentiation': 'Customer-focused approach'
            },
            emotional_landscape={
                'brand_emotional_territory': 'trust and reliability',
                'customer_emotional_drivers': ['confidence', 'satisfaction'],
                'transformation_emotions': 'peace of mind'
            },
            competitive_context={
                'market_position': 'Established player',
                'competitive_landscape': 'Competitive market'
            },
            authenticity_markers=[
                'Demonstrates quality through consistent delivery',
                'Builds trust through reliable service',
                'Maintains customer-first approach'
            ],
            creative_opportunities=[
                'Customer success stories',
                'Behind-the-scenes quality processes',
                'Community impact narratives'
            ],
            visual_identity_cues={
                'authentic_environments': ['professional settings', 'customer environments'],
                'aesthetic_direction': ['clean', 'professional', 'trustworthy']
            }
        )

# Convenience function for universal brand analysis
def analyze_any_brand(brand_name: str, 
                     company_description: str,
                     additional_context: Optional[Dict[str, Any]] = None) -> UniversalBrandIntelligence:
    """
    Analyze any brand using GPT-4o deep intelligence.
    
    Args:
        brand_name: Name of the brand/company
        company_description: Description of what the business does
        additional_context: Any additional context (optional)
        
    Returns:
        Comprehensive brand intelligence
    """
    generator = UniversalBrandIntelligenceGenerator()
    return generator.analyze_brand_deeply(brand_name, company_description, additional_context)