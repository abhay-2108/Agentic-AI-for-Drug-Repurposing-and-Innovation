import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
	ArxivPaperTool,
	ScrapeWebsiteTool
)

from full_parallel_drug_repurposing_system.tools.duckduckgo_search_tool import DuckDuckGoSearchTool


from full_parallel_drug_repurposing_system.models import AgentResponse, MasterResponse

llm = LLM(
    model="ollama/qwen3:8b",
    base_url="http://localhost:11434"
)

@CrewBase
class FullParallelDrugRepurposingSystemCrew:
    """FullParallelDrugRepurposingSystem crew"""

    
    @agent
    def drug_repurposing_master_coordinator(self) -> Agent:
        
        return Agent(
            config=self.agents_config["drug_repurposing_master_coordinator"],
            
            
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=llm,
            
        )
    
    @agent
    def scientific_literature_research_specialist(self) -> Agent:
        
        return Agent(
            config=self.agents_config["scientific_literature_research_specialist"],
            
            
            tools=[				ArxivPaperTool(),
				ScrapeWebsiteTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=llm,
            
        )
    
    @agent
    def molecular_database_search_agent(self) -> Agent:
        
        return Agent(
            config=self.agents_config["molecular_database_search_agent"],
            
            
            tools=[DuckDuckGoSearchTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=llm,
            
        )
    
    @agent
    def safety_profile_analyzer(self) -> Agent:
        
        return Agent(
            config=self.agents_config["safety_profile_analyzer"],
            
            
            tools=[				DuckDuckGoSearchTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=llm,
            
        )
    
    @agent
    def competitive_intelligence_agent(self) -> Agent:
        
        return Agent(
            config=self.agents_config["competitive_intelligence_agent"],
            
            
            tools=[				DuckDuckGoSearchTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=llm,
            
        )
    
    @agent
    def regulatory_pathway_analyst(self) -> Agent:
        
        return Agent(
            config=self.agents_config["regulatory_pathway_analyst"],
            
            
            tools=[				DuckDuckGoSearchTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=llm,
            
        )
    
    @agent
    def disease_area_expert(self) -> Agent:
        
        return Agent(
            config=self.agents_config["disease_area_expert"],
            
            
            tools=[				DuckDuckGoSearchTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=llm,
            
        )
    

    
    @task
    def scientific_literature_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["scientific_literature_analysis"],
            markdown=False,
            output_pydantic=AgentResponse
            
            
        )
    
    @task
    def molecular_database_research(self) -> Task:
        return Task(
            config=self.tasks_config["molecular_database_research"],
            markdown=False,
            output_pydantic=AgentResponse
            
            
        )
    
    @task
    def safety_profile_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["safety_profile_analysis"],
            markdown=False,
            output_pydantic=AgentResponse
            
            
        )
    
    @task
    def competitive_intelligence_research(self) -> Task:
        return Task(
            config=self.tasks_config["competitive_intelligence_research"],
            markdown=False,
            output_pydantic=AgentResponse
            
            
        )
    
    @task
    def regulatory_pathway_assessment(self) -> Task:
        return Task(
            config=self.tasks_config["regulatory_pathway_assessment"],
            markdown=False,
            output_pydantic=AgentResponse
            
            
        )
    
    @task
    def disease_area_opportunity_mapping(self) -> Task:
        return Task(
            config=self.tasks_config["disease_area_opportunity_mapping"],
            markdown=False,
            output_pydantic=AgentResponse
            
            
        )
    
    @task
    def independent_strategy_synthesis(self) -> Task:
        return Task(
            config=self.tasks_config["independent_strategy_synthesis"],
            markdown=False,
            output_pydantic=MasterResponse
            
            
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the FullParallelDrugRepurposingSystem crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

    def _load_response_format(self, name):
        with open(os.path.join(self.base_directory, "config", f"{name}.json")) as f:
            json_schema = json.loads(f.read())

        return SchemaConverter.build(json_schema)
