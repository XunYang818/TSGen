"""Statistics collector module - Collects token usage statistics for LLM calls"""
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class AgentStats:
    """Statistics for a single agent"""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, input_tokens: int = 0, output_tokens: int = 0):
        """Add statistics for one call"""
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class UsageStats:
    """Global usage statistics collector"""
    # Planning agent (3 LLM agents)
    plan_gen: AgentStats = field(default_factory=AgentStats)
    plan_val: AgentStats = field(default_factory=AgentStats)
    plan_cor: AgentStats = field(default_factory=AgentStats)
    
    # Event agent (3 LLM agents)
    event_gen: AgentStats = field(default_factory=AgentStats)
    event_val: AgentStats = field(default_factory=AgentStats)
    event_cor: AgentStats = field(default_factory=AgentStats)
    
    # ActionStep agent (3 LLM agents)
    action_gen: AgentStats = field(default_factory=AgentStats)
    action_val: AgentStats = field(default_factory=AgentStats)
    action_cor: AgentStats = field(default_factory=AgentStats)
    
    # Pass-once counters (generated content passes all validations without correction)
    plan_pass_once: int = 0
    event_pass_once: int = 0
    action_pass_once: int = 0
    
    def get_totals(self) -> Dict[str, int]:
        """Get aggregated statistics for all agents"""
        agents = [
            self.plan_gen, self.plan_val, self.plan_cor,
            self.event_gen, self.event_val, self.event_cor,
            self.action_gen, self.action_val, self.action_cor
        ]
        return {
            "total_calls": sum(a.calls for a in agents),
            "total_input_tokens": sum(a.input_tokens for a in agents),
            "total_output_tokens": sum(a.output_tokens for a in agents),
            "total_tokens": sum(a.total_tokens for a in agents)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "agents": {
                "plan_gen": self.plan_gen.to_dict(),
                "plan_val": self.plan_val.to_dict(),
                "plan_cor": self.plan_cor.to_dict(),
                "event_gen": self.event_gen.to_dict(),
                "event_val": self.event_val.to_dict(),
                "event_cor": self.event_cor.to_dict(),
                "action_gen": self.action_gen.to_dict(),
                "action_val": self.action_val.to_dict(),
                "action_cor": self.action_cor.to_dict()
            },
            "pass_once": {
                "plan": self.plan_pass_once,
                "event": self.event_pass_once,
                "action": self.action_pass_once
            },
            "totals": self.get_totals()
        }


def extract_usage(raw_result: Dict[str, Any]) -> Tuple[int, int]:
    """Extract token usage information from include_raw=True result
    
    Args:
        raw_result: Dictionary returned by with_structured_output(include_raw=True), containing raw/parsed/parsing_error
        
    Returns:
        (input_tokens, output_tokens)
    """
    if not raw_result or "raw" not in raw_result:
        return 0, 0
    
    raw_msg = raw_result.get("raw")
    if raw_msg and hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
        meta = raw_msg.usage_metadata
        return meta.get("input_tokens", 0), meta.get("output_tokens", 0)
    return 0, 0
