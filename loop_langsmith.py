"""Operations Agent: LangGraph loop with HITL, retry, and LangSmith traces.

LangSmith traces are emitted when LANGSMITH_TRACING=true and the LangSmith
SDK is configured. LangGraph records graph/node and LangChain LLM runs;
custom tool attempts and approval decisions are instrumented below.
"""
import json
import logging
import os

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import task
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt
from pydantic import ValidationError

from harness.agent.prompt import SYSTEM_PROMPT
from harness.agent.state import AgentState
from harness.core.executor import ToolExecutor
from harness.core.registry import get_tool, get_tool_definitions
from harness.observability.langsmith_tracing import (
    trace_approval_decision,
    trace_tool_execution,
)

logger = logging.getLogger(__name__)

READ_TOOLS = {
    "get_service_status",
    "get_error_logs",
    "search_knowledge_base",
}

RETRYABLE_ERRORS = {
    "NETWORK_ERROR",
    "NETWORK_TIMEOUT",
    "TOOL_TIMEOUT",
    "RATE_LIMITED",
    "UPSTREAM_UNAVAILABLE",
}


class RetryableToolError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@task(
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.5,
        retry_on=RetryableToolError,
    )
)
async def execute_read_task(
    tool_name: str,
    arguments: dict,
    tool_call_id: str | None = None,
) -> dict:
    """Run a single read tool, with one LangSmith span per retry attempt."""
    if tool_name not in READ_TOOLS:
        raise ValueError("Only read tools are allowed")

    executor = ToolExecutor()

    # Wrapper is INSIDE @task: every retry attempt produces its own span.
    result = await trace_tool_execution(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        operation=lambda: executor.execute(
            tool_name=tool_name,
            arguments=arguments,
        ),
    )

    error = result.get("error") or {}
    error_code = error.get("code")
    if error_code in RETRYABLE_ERRORS:
        # LangGraph RetryPolicy only retries exceptions, not error dicts.
        raise RetryableToolError(error_code)

    return result


class OperationsAgent:
    def __init__(
        self,
        model: str | None = None,
        max_steps: int = 8,
        llm=None,
        checkpointer=None,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        self.max_steps = max_steps
        self.llm = llm if llm is not None else ChatOpenAI(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0,
            timeout=30,
            max_retries=1,
        )
        self.llm_with_tools = self.llm.bind_tools(
            get_tool_definitions(),
            parallel_tool_calls=False,
        )
        self.executor = ToolExecutor()
        self.checkpointer = (
            checkpointer if checkpointer is not None else InMemorySaver()
        )
        self.graph = self.build_graph()

    async def call_llm(self, state: AgentState) -> dict:
        if state["status"] == "waiting_approval":
            return {}

        if state["step_count"] >= self.max_steps:
            return {
                "status": "failed",
                "messages": [AIMessage(content="Maximum tool steps reached.")],
            }

        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        try:
            # ChatOpenAI + LangGraph are automatically traced by LangSmith
            # when LANGSMITH_TRACING=true. No extra @traceable needed here.
            response = await self.llm_with_tools.ainvoke(messages)
        except Exception:
            logger.exception("LLM invocation failed")
            return {
                "status": "failed",
                "messages": [AIMessage(content="LLM request failed.")],
            }

        if response.invalid_tool_calls:
            return {"messages": [response], "status": "failed"}

        if not response.tool_calls:
            return {"messages": [response], "status": "completed"}

        if len(response.tool_calls) > 1:
            return {
                "status": "failed",
                "messages": [AIMessage(content="Multiple tool calls are not supported.")],
            }

        return {"messages": [response], "status": "running"}

    def route_after_llm(self, state: AgentState):
        if state["status"] != "running":
            return END

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "execute_tools"
        return END

    async def execute_tools(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        last_message = state["messages"][-1]

        # The current harness supports exactly one tool call per LLM turn.
        if not isinstance(last_message, AIMessage) or len(last_message.tool_calls) != 1:
            return {
                "status": "failed",
                "messages": [AIMessage(content="Expected exactly one tool call.")],
            }

        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]
        arguments = tool_call["args"]
        tool_call_id = tool_call["id"]
        step_count = state["step_count"]
        status = "running"

        def error_result(code: str, message: str) -> dict:
            return {
                "ok": False,
                "tool": tool_name,
                "data": None,
                "error": {"code": code, "message": message},
            }

        if not tool_call_id:
            return {
                "status": "failed",
                "messages": [AIMessage(content="Tool call ID is missing.")],
            }

        if step_count >= self.max_steps:
            result = error_result("STEP_LIMIT_REACHED", "Maximum tool steps reached.")
            status = "failed"

        elif tool_name == "create_incident":
            tool_spec = get_tool(tool_name)
            if tool_spec is None:
                result = error_result("UNKNOWN_TOOL", "Tool is not registered.")
            else:
                try:
                    validated_args = tool_spec.input_schema.model_validate(
                        arguments
                    ).model_dump(mode="json")
                except (ValidationError, TypeError, ValueError):
                    result = error_result("INVALID_ARGUMENTS", "Invalid incident arguments.")
                else:
                    # IMPORTANT: No database write or external side effect
                    # before interrupt(); LangGraph reruns this node on resume.
                    decision = interrupt(
                        {
                            "type": "incident_approval",
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "arguments": validated_args,
                            "message": "Approve creating this incident?",
                        }
                    )

                    valid_decision = (
                        isinstance(decision, dict)
                        and decision.get("tool_call_id") == tool_call_id
                        and decision.get("decision") in {"approve", "reject"}
                    )

                    decision_label = (
                        decision["decision"] if valid_decision else "invalid"
                    )
                    # This span belongs to the resumed run (not the paused run).
                    trace_approval_decision(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        decision=decision_label,
                    )

                    if not valid_decision:
                        result = error_result(
                            "INVALID_APPROVAL_RESPONSE", "Invalid approval response."
                        )
                    elif decision["decision"] == "reject":
                        result = error_result(
                            "USER_REJECTED", "User rejected incident creation."
                        )
                    else:
                        thread_id = config.get("configurable", {}).get("thread_id")
                        if not thread_id:
                            result = error_result(
                                "INVALID_THREAD_ID", "Thread ID is required."
                            )
                        else:
                            execution_id = f"{thread_id}:{tool_call_id}"
                            # Approval token is generated by trusted Harness code,
                            # not passed as an LLM tool argument.
                            approval_token = self.executor.issue_approval(
                                tool_name=tool_name,
                                arguments=validated_args,
                            )
                            result = await trace_tool_execution(
                                tool_name=tool_name,
                                tool_call_id=tool_call_id,
                                operation=lambda: self.executor.execute(
                                    tool_name=tool_name,
                                    arguments=validated_args,
                                    approval_token=approval_token,
                                    execution_id=execution_id,
                                ),
                            )

        elif tool_name in READ_TOOLS:
            try:
                # @task retry emits another tool span for each attempt.
                result = await execute_read_task(
                    tool_name,
                    arguments,
                    tool_call_id,
                )
            except RetryableToolError as exc:
                result = error_result(exc.code, "Retry attempts exhausted.")

        else:
            # Unknown/unsupported tool: let registry return a structured error.
            result = await trace_tool_execution(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                operation=lambda: self.executor.execute(
                    tool_name=tool_name,
                    arguments=arguments,
                ),
            )

        step_count += 1
        tool_message = ToolMessage(
            content=json.dumps(result, ensure_ascii=False, default=str),
            tool_call_id=tool_call_id,
            name=tool_name,
        )
        history = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
        }
        return {
            "messages": [tool_message],
            "tool_history": [history],
            "step_count": step_count,
            "status": status,
            "pending_approval": None,
        }

    def route_after_tools(self, state: AgentState):
        if state["status"] != "running":
            return END
        return "call_llm"

    def build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("call_llm", self.call_llm)
        builder.add_node("execute_tools", self.execute_tools)
        builder.add_edge(START, "call_llm")
        builder.add_conditional_edges("call_llm", self.route_after_llm)
        builder.add_conditional_edges("execute_tools", self.route_after_tools)
        return builder.compile(checkpointer=self.checkpointer)
