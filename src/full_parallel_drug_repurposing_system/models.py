from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class AgentInsight(BaseModel):
    """Represents a specific insight or finding from an agent."""
    summary: str = Field(..., description="Concise summary of the finding")
    details: str = Field(..., description="Detailed explanation of the finding")
    source: str = Field(..., description="Source of the information (e.g., 'PubMed', 'DrugBank', 'Analysis')")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0) for this finding")
    url: Optional[str] = Field(None, description="URL to the source if available")

class AgentResponse(BaseModel):
    """Structured response from a specialist agent."""
    agent_name: str = Field(..., description="Name of the agent")
    key_findings: List[AgentInsight] = Field(..., description="List of key findings")
    summary: str = Field(..., description="High-level summary of the agent's work")
    next_steps: Optional[List[str]] = Field(None, description="Suggested next steps or areas for further research")

class Recommendation(BaseModel):
    """Specific strategic recommendation."""
    action: str = Field(..., description="Recommended action")
    rationale: str = Field(..., description="Reasoning behind the recommendation")
    priority: str = Field(..., description="Priority level (High, Medium, Low)")

class MasterResponse(BaseModel):
    """Final synthesized response from the Master Coordinator."""
    executive_summary: str = Field(..., description="High-level executive summary of the repurposing opportunity")
    drug_name: str = Field(..., description="Name of the drug analyzed")
    agent_summaries: Dict[str, str] = Field(..., description="Brief summary of what each agent found, keyed by agent name")
    key_insights: List[AgentInsight] = Field(..., description="Top consolidated insights from all agents")
    recommendations: List[Recommendation] = Field(..., description="Strategic recommendations")
    overall_confidence: float = Field(..., description="Overall confidence score (0.0 to 1.0) in the repurposing potential")
    success_probability: float = Field(..., description="Estimated probability of success (0.0 to 1.0)")
