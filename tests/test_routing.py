"""Routing and workflow topology tests that do not require an LLM API key."""

from agents.coordinator import (
    _increment_repair,
    _privacy_boundary,
    _route_after_privacy,
    _route_after_retrieval,
    _route_after_validation,
    build_graph,
)
from agents.escalation_node import escalation_node
from agents.fallback_node import fallback_node
from agents.privacy_risk_node import privacy_and_risk_node
from config.settings import get_settings
from contract import AgentState, ValidationResult


def test_privacy_node_redacts_pii_and_flags_high_risk():
    out = privacy_and_risk_node(
        {
            "raw_input": (
                "A teammate emailed from teammate@example.com and said "
                "I may be in immediate danger."
            )
        }
    )

    assert out["pii_detected"] is True
    assert "teammate@example.com" not in out["redacted_input"]
    assert "[EMAIL]" in out["redacted_input"]
    assert out["high_risk_detected"] is True
    assert out["escalation_required"] is True


def test_privacy_boundary_removes_raw_pii_from_state_update():
    email = "student@example.com"
    phone = "713-555-0100"
    out = _privacy_boundary(
        {
            "raw_input": (
                f"Email {email} or call {phone}. "
                "I may be in immediate danger."
            )
        }
    )

    assert out["raw_input"] == ""
    assert email not in out["redacted_input"]
    assert phone not in out["redacted_input"]
    assert out["pii_detected"] is True
    assert out["pii_spans"]
    assert all("text" not in span for span in out["pii_spans"])
    assert email not in str(out)
    assert phone not in str(out)


def test_privacy_node_redacts_student_goal_pii():
    email = "jane.doe@uh.edu"
    out = privacy_and_risk_node(
        {
            "raw_input": "Our team keeps missing checkpoints and ownership is unclear.",
            "student_goal": f"Email {email} to reset roles before demo week.",
        }
    )

    assert out["pii_detected"] is True
    assert email not in (out["student_goal"] or "")
    assert "[EMAIL]" in (out["student_goal"] or "")
    assert out["high_risk_detected"] is False


def test_privacy_boundary_strips_goal_pii_from_state_update():
    email = "goal.leak@uh.edu"
    out = _privacy_boundary(
        {
            "raw_input": "Roles are fuzzy and handoffs keep slipping.",
            "student_goal": f"Contact {email} about ownership.",
        }
    )

    assert out["raw_input"] == ""
    assert email not in (out["student_goal"] or "")
    assert "[EMAIL]" in (out["student_goal"] or "")
    assert email not in str(out)
    assert all("text" not in span for span in out["pii_spans"])


def test_compiled_graph_does_not_retain_raw_pii_after_privacy_boundary():
    email = "student@example.com"
    phone = "713-555-0100"
    result = build_graph().invoke(
        {
            "raw_input": (
                f"Email {email} or call {phone}. "
                "I may be in immediate danger."
            )
        }
    )

    state_view = (
        result.model_dump()
        if isinstance(result, AgentState)
        else result
    )

    assert state_view["raw_input"] == ""
    assert all("text" not in span for span in state_view["pii_spans"])
    assert email not in str(state_view)
    assert phone not in str(state_view)


def test_route_normal_privacy_result_to_diagnosis_and_retrieval():
    state = AgentState(
        raw_input="x",
        escalation_required=False,
        high_risk_detected=False,
    )

    assert _route_after_privacy(state) == "diagnosis_retrieval"


def test_route_high_risk_privacy_result_to_escalation():
    state = AgentState(
        raw_input="x",
        escalation_required=True,
        high_risk_detected=True,
    )

    assert _route_after_privacy(state) == "escalation"


def test_route_sufficient_evidence_to_advice():
    state = AgentState(
        raw_input="x",
        retrieval_sufficient=True,
    )

    assert _route_after_retrieval(state) == "advice"


def test_route_insufficient_evidence_to_fallback():
    state = AgentState(
        raw_input="x",
        retrieval_sufficient=False,
    )

    assert _route_after_retrieval(state) == "fallback"


def test_route_missing_validation_result_to_fallback():
    state = AgentState(
        raw_input="x",
        validation_result=None,
    )

    assert _route_after_validation(state) == "fallback"


def test_route_validation_escalation_to_escalation():
    state = AgentState(
        raw_input="x",
        validation_result=ValidationResult(
            safe_to_display=False,
            escalation_required=True,
            repairable=False,
            reasons=["human support required"],
        ),
    )

    assert _route_after_validation(state) == "escalation"


def test_route_safe_validation_result_to_finalize():
    state = AgentState(
        raw_input="x",
        validation_result=ValidationResult(
            safe_to_display=True,
            escalation_required=False,
            repairable=False,
            reasons=[],
        ),
    )

    assert _route_after_validation(state) == "finalize"


def test_route_repairable_result_to_single_repair_attempt():
    state = AgentState(
        raw_input="x",
        regeneration_count=0,
        validation_result=ValidationResult(
            safe_to_display=False,
            escalation_required=False,
            repairable=True,
            reasons=["missing citation"],
        ),
    )

    assert _route_after_validation(state) == "repair"

    state.regeneration_count = get_settings().max_repair_attempts
    assert _route_after_validation(state) == "fallback"


def test_route_nonrepairable_validation_failure_to_fallback():
    state = AgentState(
        raw_input="x",
        regeneration_count=0,
        validation_result=ValidationResult(
            safe_to_display=False,
            escalation_required=False,
            repairable=False,
            reasons=["unsafe advice"],
        ),
    )

    assert _route_after_validation(state) == "fallback"


def test_increment_repair_increases_counter_without_other_state_changes():
    out = _increment_repair({"regeneration_count": 0})

    assert out == {"regeneration_count": 1}


def test_increment_repair_handles_missing_counter():
    out = _increment_repair({})

    assert out == {"regeneration_count": 1}


def test_escalation_node_returns_terminal_resource_response():
    out = escalation_node(
        {
            "redacted_input": "redacted",
            "pii_detected": False,
        }
    )

    assert out["final_response"].route == "escalation"
    assert out["final_response"].resources
    assert out["safe_to_display"] is True
    assert out["escalation_required"] is True


def test_fallback_node_returns_terminal_abstention_response():
    out = fallback_node(
        {
            "redacted_input": "redacted",
            "retrieval_sufficient": False,
            "validation_result": None,
        }
    )

    assert out["final_response"].route == "fallback"
    assert out["final_response"].citations == []
    assert out["safe_to_display"] is True


def test_compiled_graph_contains_expected_nodes():
    graph = build_graph().get_graph()

    assert set(graph.nodes) == {
        "__start__",
        "__end__",
        "privacy_risk",
        "diagnosis_retrieval",
        "advice",
        "validation",
        "repair_increment",
        "finalize",
        "fallback",
        "escalation",
    }


def test_compiled_graph_contains_expected_workflow_edges():
    graph = build_graph().get_graph()
    edges = {
        (
            edge.source,
            edge.target,
            bool(edge.conditional),
        )
        for edge in graph.edges
    }

    assert edges == {
        ("__start__", "privacy_risk", False),
        ("privacy_risk", "diagnosis_retrieval", True),
        ("privacy_risk", "escalation", True),
        ("privacy_risk", "fallback", True),
        ("diagnosis_retrieval", "advice", True),
        ("diagnosis_retrieval", "fallback", True),
        ("advice", "validation", False),
        ("validation", "finalize", True),
        ("validation", "repair_increment", True),
        ("validation", "fallback", True),
        ("validation", "escalation", True),
        ("repair_increment", "advice", False),
        ("finalize", "__end__", False),
        ("fallback", "__end__", False),
        ("escalation", "__end__", False),
    }
