"""LangGraph workflow for orchestrating the resume screening agents."""

import asyncio
from typing import Any, Annotated, TypedDict

from langgraph.graph import StateGraph, END

from .agents import (
    ResumeParserAgent,
    SkillExtractorAgent,
    JobAnalyzerAgent,
    SkillsMatcherAgent,
    ExperienceEvaluatorAgent,
    DecisionSynthesizerAgent,
)
from .models import ScreeningState, ScreeningOutput
from .document_parser import parse_document


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    """Reducer function to merge dictionaries from parallel branches."""
    result = dict(a) if a else {}
    if b:
        result.update(b)
    return result


def merge_lists(a: list | None, b: list | None) -> list:
    """Reducer function to merge lists from parallel branches."""
    result = list(a) if a else []
    if b:
        for item in b:
            if item not in result:
                result.append(item)
    return result


class WorkflowState(TypedDict, total=False):
    """State dictionary for the LangGraph workflow with reducers for parallel updates."""
    # Inputs
    resume_path: str
    resume_raw_text: str
    job_description: str
    
    # Agent outputs
    resume_data: dict | None
    # None = SkillExtractorAgent has not actually run yet (distinct from
    # [], which means it ran and genuinely found no recognizable skills).
    # match_skills relies on this distinction -- see _match_skills_node.
    extracted_skills: list | None
    job_requirements: dict | None
    skills_match: dict | None
    experience_eval: dict | None
    
    # Final output
    final_output: dict | None
    
    # Metadata - these use Annotated with reducers for parallel updates
    errors: Annotated[list, merge_lists]
    agent_confidences: Annotated[dict, merge_dicts]
    workflow_complete: bool


class ResumeScreeningWorkflow:
    """
    LangGraph workflow for agentic resume screening.
    
    This workflow orchestrates multiple agents to:
    1. Parse the resume document
    2. Extract and structure resume information
    3. Analyze job requirements
    4. Match skills and evaluate experience
    5. Synthesize a final recommendation
    
    The workflow uses parallel branches where possible for efficiency,
    and converges results for final decision making.
    """
    
    def __init__(self):
        """Initialize the workflow with all agents."""
        self.resume_parser = ResumeParserAgent()
        self.skill_extractor = SkillExtractorAgent()
        self.job_analyzer = JobAnalyzerAgent()
        self.skills_matcher = SkillsMatcherAgent()
        self.experience_evaluator = ExperienceEvaluatorAgent()
        self.decision_synthesizer = DecisionSynthesizerAgent()
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph."""
        
        # Create the graph with our state schema
        workflow = StateGraph(WorkflowState)
        
        # Add nodes (each node is an agent's process function)
        workflow.add_node("parse_document", self._parse_document_node)
        workflow.add_node("parse_resume", self._parse_resume_node)
        workflow.add_node("analyze_job", self._analyze_job_node)
        workflow.add_node("extract_skills", self._extract_skills_node)
        workflow.add_node("match_skills", self._match_skills_node)
        workflow.add_node("evaluate_experience", self._evaluate_experience_node)
        workflow.add_node("synthesize_decision", self._synthesize_decision_node)
        
        # Define the workflow edges
        # Entry point: parse the document
        workflow.set_entry_point("parse_document")
        
        # After document parsing, parse resume and analyze job in parallel
        workflow.add_edge("parse_document", "parse_resume")
        workflow.add_edge("parse_document", "analyze_job")
        
        # After resume parsing, extract skills
        workflow.add_edge("parse_resume", "extract_skills")
        
        # Skills matching depends on both skill extraction and job analysis
        workflow.add_edge("extract_skills", "match_skills")
        workflow.add_edge("analyze_job", "match_skills")
        
        # Experience evaluation depends on resume parsing and job analysis
        workflow.add_edge("parse_resume", "evaluate_experience")
        workflow.add_edge("analyze_job", "evaluate_experience")
        
        # Decision synthesis depends on skills matching and experience evaluation
        workflow.add_edge("match_skills", "synthesize_decision")
        workflow.add_edge("evaluate_experience", "synthesize_decision")
        
        # End after decision synthesis
        workflow.add_edge("synthesize_decision", END)
        
        return workflow.compile()
    
    def _parse_document_node(self, state: WorkflowState) -> dict:
        """Node: Parse the resume document to extract raw text."""
        # If a caller already supplied fully-hydrated resume_data (e.g.
        # the service layer reusing a previously-parsed Resume row by
        # content hash), there's nothing for this node OR the parser node
        # after it to do -- skip straight through.
        if state.get("resume_data"):
            return {}

        resume_path = state.get("resume_path", "")
        
        if not resume_path:
            # If raw text is already provided, skip parsing
            if state.get("resume_raw_text"):
                return {}
            return {
                "errors": ["No resume path or text provided"]
            }
        
        # Parse the document
        result = parse_document(resume_path)
        
        if not result.success:
            return {
                "errors": [f"Document parsing failed: {result.error_message}"],
                "resume_raw_text": ""
            }
        
        return {
            "resume_raw_text": result.text,
            "agent_confidences": {"DocumentParser": result.confidence}
        }
    
    async def _parse_resume_node(self, state: WorkflowState) -> dict:
        """Node: Parse resume text into structured data (skipped if the
        caller already supplied resume_data, e.g. reused from a resume
        with matching content_hash -- avoids an LLM call for content
        we've already parsed once)."""
        if state.get("resume_data"):
            return {}
        return await self.resume_parser.process(dict(state))
    
    async def _analyze_job_node(self, state: WorkflowState) -> dict:
        """Node: Analyze job description (skipped if already provided)."""
        if state.get("job_requirements"):
            # Already parsed once at the position level and passed in —
            # don't burn an LLM call re-analyzing the same JD per candidate.
            return {}
        return await self.job_analyzer.process(dict(state))
    
    async def _extract_skills_node(self, state: WorkflowState) -> dict:
        """Node: Extract skills from parsed resume (skipped if the caller
        already supplied extracted_skills, e.g. reused alongside a
        resume with matching content_hash)."""
        if state.get("extracted_skills"):
            return {}
        return await self.skill_extractor.process(dict(state))
    
    async def _match_skills_node(self, state: WorkflowState) -> dict:
        """Node: Match skills against requirements.

        Guarded against premature fan-in. match_skills has two incoming
        edges -- extract_skills (parse_resume -> extract_skills, depth 3
        from parse_document) and analyze_job (depth 2) -- of different
        lengths, so LangGraph schedules this node once per incoming edge
        that resolves rather than once after BOTH are ready. Whichever
        edge resolves first (in practice: analyze_job, since it depends
        on an LLM call whose latency varies, could resolve before or
        after extract_skills, which is fully deterministic) triggers an
        invocation where the OTHER input is still missing.

        Before this guard, that premature invocation still ran
        SkillsMatcherAgent, which happily scored the requirement list
        against an empty (not-yet-populated) extracted_skills list --
        producing a fully-formed but WRONG SkillsMatchResult (e.g.
        "0 of 7 required skills met") that got written into shared
        state. Because extracted_skills defaults to [] rather than a
        distinguishable "not computed yet" sentinel, that wrong result
        was indistinguishable from a resume that genuinely has zero
        matching skills, so nothing downstream (including
        DecisionSynthesizerAgent's own guard, which locks in whatever
        skills_match is present the first time it runs) could tell it
        apart from the real answer that a later, correctly-timed
        invocation would have computed once extract_skills actually
        finished. This is the exact same "idempotency guard locks in a
        premature result" failure mode Batch 4 diagnosed for this node
        -- reintroduced one hop upstream once DecisionSynthesizerAgent
        gained its own idempotency guard.

        Fix: extracted_skills is now None (not []) until
        SkillExtractorAgent has genuinely run once (see WorkflowState),
        so a premature firing -- either input still missing -- is simply
        skipped (no-op) instead of computing a wrong result. The later
        invocation, triggered once the slower of the two real
        predecessors finishes, does the real (and only) computation.
        """
        if state.get("extracted_skills") is None or not state.get("job_requirements"):
            return {}
        return await self.skills_matcher.process(dict(state))
    
    async def _evaluate_experience_node(self, state: WorkflowState) -> dict:
        """Node: Evaluate work experience."""
        return await self.experience_evaluator.process(dict(state))
    
    async def _synthesize_decision_node(self, state: WorkflowState) -> dict:
        """Node: Synthesize final decision.

        Guarded on two independent conditions:

        1. Already complete -- once this node has produced final_output
           in this graph run, later invocations are a no-op. This is
           what prevents DecisionSynthesizerAgent's LLM call from firing
           twice (this node also has two incoming edges of different
           depths -- match_skills, evaluate_experience -- for the same
           reason described on _match_skills_node).
        2. Not yet ready -- skills_match/experience_eval can each still
           be None the first time this node is invoked (same fan-in
           race as above, one hop downstream). Previously this node
           would fall through to DecisionSynthesizerAgent, which treats
           a missing skills_match/experience_eval as a hard error and
           returns an error ScreeningOutput with workflow_complete=True
           -- and guard #1 would then permanently lock in that error
           result once the *real* skills_match/experience_eval arrived
           moments later. Skipping (without setting workflow_complete)
           when either input is still missing lets the later,
           genuinely-ready invocation run for real -- exactly once.
        """
        if state.get("workflow_complete") and state.get("final_output"):
            return {}
        if state.get("skills_match") is None or state.get("experience_eval") is None:
            return {}
        return await self.decision_synthesizer.process(dict(state))
    
    async def run(
        self,
        resume_path: str = "",
        resume_text: str = "",
        job_description: str = ""
    ) -> ScreeningOutput:
        """
        Run the complete screening workflow.
        
        Args:
            resume_path: Path to resume file (PDF, DOCX, or TXT)
            resume_text: Raw resume text (alternative to file path)
            job_description: The job description text
            
        Returns:
            ScreeningOutput with match score, recommendation, and reasoning
        """
        # Initialize state
        initial_state: WorkflowState = {
            "resume_path": resume_path,
            "resume_raw_text": resume_text,
            "job_description": job_description,
            "resume_data": None,
            "extracted_skills": None,
            "job_requirements": None,
            "skills_match": None,
            "experience_eval": None,
            "final_output": None,
            "errors": [],
            "agent_confidences": {},
            "workflow_complete": False,
        }
        
        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        # Extract and return the final output
        output_data = final_state.get("final_output")
        
        if output_data:
            if isinstance(output_data, dict):
                return ScreeningOutput.model_validate(output_data)
            return output_data
        
        # Fallback if no output
        return ScreeningOutput(
            match_score=0.0,
            recommendation="Error - workflow did not complete",
            requires_human=True,
            confidence=0.0,
            reasoning_summary="The workflow failed to produce a result. Please review manually.",
            flags=["Workflow error"]
        )
    
    async def run_full(
        self,
        resume_path: str = "",
        resume_text: str = "",
        job_description: str = "",
        job_requirements: dict | None = None,
        resume_data: dict | None = None,
        extracted_skills: list | None = None,
    ) -> WorkflowState:
        """
        Run the complete screening workflow and return the FULL final
        state (not just the synthesized output).

        This exists because a caller that wants to persist results
        (SkillMatchResult, ExperienceEvaluation, an Explanation built from
        the match/eval detail) needs more than ScreeningOutput alone --
        run() below only ever returned final_output and threw the rest
        away. Any code doing DB persistence should call this instead of
        run().

        Args:
            resume_path: Path to resume file (PDF, DOCX, or TXT)
            resume_text: Raw resume text (alternative to file path)
            job_description: The job description text
            job_requirements: Optional pre-parsed requirements dict. Pass
                this when screening many candidates against the same
                position so JobAnalyzerAgent only runs once per JD
                version instead of once per resume.
            resume_data: Optional pre-parsed resume dict (shape of
                src.models.ResumeData). Pass this when the caller has
                already parsed this exact resume content before (see
                services.screening_service's content-hash resume reuse)
                so document parsing + ResumeParserAgent are both skipped.
            extracted_skills: Optional pre-extracted skills list (shape of
                list[src.models.Skill] as dicts). Pass alongside
                resume_data to also skip SkillExtractorAgent.

        Returns:
            The full WorkflowState dict after the graph finishes --
            includes resume_data, extracted_skills, job_requirements,
            skills_match, experience_eval, final_output, errors, etc.
        """
        
        initial_state: WorkflowState = {
            "resume_path": resume_path,
            "resume_raw_text": resume_text,
            "job_description": job_description,
            "resume_data": resume_data,
            # Preserve None (not yet extracted) vs. [] (caller reused a
            # resume that was genuinely found to have zero skills) --
            # collapsing both to [] is what let match_skills mistake "not
            # computed yet" for "computed, found nothing". See
            # WorkflowState.extracted_skills and _match_skills_node.
            "extracted_skills": extracted_skills,
            "job_requirements": job_requirements,
            "skills_match": None,
            "experience_eval": None,
            "final_output": None,
            "errors": [],
            "agent_confidences": {},
            "workflow_complete": False,
        }

        return await self.graph.ainvoke(initial_state)

    async def run(
            self,
            resume_path: str = "",
            resume_text: str = "",
            job_description: str = "",
            job_requirements: dict | None = None,
            resume_data: dict | None = None,
            extracted_skills: list | None = None,
        ) -> ScreeningOutput:
            """
            Run the complete screening workflow.

            Args:
                resume_path: Path to resume file (PDF, DOCX, or TXT)
                resume_text: Raw resume text (alternative to file path)
                job_description: The job description text
                job_requirements: Optional pre-parsed requirements dict (see
                    run_full for why you'd pass this)
                resume_data: Optional pre-parsed resume dict (see run_full)
                extracted_skills: Optional pre-extracted skills list (see run_full)

            Returns:
                ScreeningOutput with match score, recommendation, and reasoning
            """
            final_state = await self.run_full(
                resume_path=resume_path,
                resume_text=resume_text,
                job_description=job_description,
                job_requirements=job_requirements,
                resume_data=resume_data,
                extracted_skills=extracted_skills,
            )

            output_data = final_state.get("final_output")

            if output_data:
                if isinstance(output_data, dict):
                    return ScreeningOutput.model_validate(output_data)
                return output_data

            return ScreeningOutput(
                match_score=0.0,
                recommendation="Error - workflow did not complete",
                requires_human=True,
                confidence=0.0,
                reasoning_summary="The workflow failed to produce a result. Please review manually.",
                flags=["Workflow error"]
        )
        
    def run_sync(
        self,
        resume_path: str = "",
        resume_text: str = "",
        job_description: str = "",
        job_requirements: dict | None = None,
    ) -> ScreeningOutput:
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(resume_path, resume_text, job_description, job_requirements))


def create_screening_workflow() -> ResumeScreeningWorkflow:
    """Factory function to create a screening workflow."""
    return ResumeScreeningWorkflow()


async def screen_resume(
    resume_path: str = "",
    resume_text: str = "",
    job_description: str = ""
) -> ScreeningOutput:
    """
    Convenience function to screen a resume.
    
    Args:
        resume_path: Path to resume file
        resume_text: Raw resume text (alternative to path)
        job_description: Job description text
        
    Returns:
        ScreeningOutput with recommendation and reasoning
    """
    workflow = create_screening_workflow()
    return await workflow.run(resume_path, resume_text, job_description)
