from typing import Literal

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    goal: str = Field(description="What we are trying to answer")
    search_queries: list[str] = Field(description="Specific queries to execute")
    sources_to_check: list[str] = Field(
        description="Use 'knowledge_base', 'web', or both"
    )
    output_format: str = Field(
        description="What the final report should look like"
    )


class CritiqueResult(BaseModel):
    verdict: Literal["APPROVE", "REVISE"] = Field(
        default="REVISE",
        description="Final reviewer decision"
    )
    is_fresh: bool = Field(
        default=False,
        description="Is the data up-to-date and based on recent sources?"
    )
    is_complete: bool = Field(
        default=False,
        description="Does the research fully cover the user's original request?"
    )
    is_well_structured: bool = Field(
        default=False,
        description="Are findings logically organized and ready for a report?"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="What is good about the research"
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="What is missing, outdated, or poorly structured"
    )
    revision_requests: list[str] = Field(
        default_factory=list,
        description="Specific things to fix if verdict is REVISE"
    )
    is_error: bool = Field(
        default=False,
        description="True when the critique step failed due to an exception (not a quality verdict). Supervisor must not count this as a revise cycle."
    )
