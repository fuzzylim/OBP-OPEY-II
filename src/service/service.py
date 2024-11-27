import json
import os
import warnings
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4
import asyncio
import logging

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client as LangsmithClient

from agent import opey_graph
from agent.components.chains import QueryFormulatorOutput
from fastapi import WebSocket
from schema import (
    ChatMessage,
    Feedback,
    FeedbackResponse,
    StreamInput,
    UserInput,
    convert_message_content_to_string,
    ToolCallApproval,
)

#logging.basicConfig(level=logging.DEBUG)
#logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Construct agent with Sqlite checkpointer
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        opey_graph.checkpointer = saver
        app.state.agent = opey_graph
        yield
    # context manager will clean up the AsyncSqliteSaver on exit


app = FastAPI(lifespan=lifespan)

# TODO: change to implement our own authentication checking (also decide what auth to use)
# NOTE: will be different when we use consents rather than a secret
@app.middleware("http")
async def check_auth_header(request: Request, call_next: Callable) -> Response:
    if auth_secret := os.getenv("AUTH_SECRET"):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response(status_code=401, content="Missing or invalid token")
        if auth_header[7:] != auth_secret:
            return Response(status_code=401, content="Invalid token")
    return await call_next(request)

def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    input_message = ChatMessage(type="human", content=user_input.message)
    kwargs = {
        "input": {"messages": [input_message.to_langchain()]},
        "config": RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model}, run_id=run_id
        ),
    }
    return kwargs, run_id, thread_id


def _remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]

# Global dictionary to track user approvals
_approval_condition = asyncio.Condition()  # Single instance
_pending_approvals = {}

async def wait_for_user_approval(thread_id: str) -> str:
    print(f"[Debug] Wait for approval event loop: {id(asyncio.get_running_loop())}")
    async with _approval_condition:
        print(f"[Waiter] Waiting for approval of {thread_id}")
        while thread_id not in _pending_approvals:
            await _approval_condition.wait()
            print(f"[Waiter] Woke up for {thread_id}, checking condition")
        return _pending_approvals.pop(thread_id)

@app.post("/approval/{thread_id}")
async def user_approval(user_response: ToolCallApproval) -> dict[str, str]:
    print(f"[Debug] Approval endpoint event loop: {id(asyncio.get_running_loop())}")
    async with _approval_condition:
        print(f"[Notifier] Setting approval for {user_response.thread_id}")
        _pending_approvals[user_response.thread_id] = user_response.approval
        _approval_condition.notify_all()
        print(f"[Notifier] Notified all for {user_response.thread_id}")
    return {"status": "received"}

async def _process_stream_event(event: StreamEvent, user_input: StreamInput, run_id: str) -> AsyncGenerator[str, None]:
    """Helper to process stream events consistently"""
    if not event:
        return
    
    print("DATA: ", event, "\n")
    
    # Handle messages after node execution
    if (
        event["event"] == "on_chain_end"
        and any(t.startswith("graph:step:") for t in event.get("tags", []))
        and event["data"].get("output") is not None
        and "messages" in event["data"]["output"]
        and event["metadata"].get("langgraph_node", "") != "human_review"
    ):
        new_messages = event["data"]["output"]["messages"]
        if not isinstance(new_messages, list):
            new_messages = [new_messages]
            
        for message in new_messages:
            print(f"{message.pretty_print()}")
            try:
                chat_message = ChatMessage.from_langchain(message)
                chat_message.run_id = str(run_id)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                continue

            if not (chat_message.type == "human" and chat_message.content == user_input.message):
                print(f"Sending message: {chat_message}")
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

    # Handle tokens streamed from LLMs
    if (
        event["event"] == "on_chat_model_stream"
        and user_input.stream_tokens
        and event['metadata'].get('langgraph_node', '') != "transform_query"
    ):
        content = _remove_tool_calls(event["data"]["chunk"].content)
        if content:
            yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"


@app.post("/invoke")
async def invoke(user_input: UserInput) -> ChatMessage:
    """
    Invoke the agent with user input to retrieve a final response.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    agent: CompiledStateGraph = app.state.agent
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        output = ChatMessage.from_langchain(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def message_generator(user_input: StreamInput) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: CompiledStateGraph = app.state.agent
    kwargs, run_id, thread_id = _parse_input(user_input)
    thread = {"configurable": {"thread_id": thread_id}}

    print(f"------------START STREAM-----------\n\n")
    # Process streamed events from the graph and yield messages over the SSE stream.
    async for event in agent.astream_events(**kwargs, version="v2"):
        async for msg in _process_stream_event(event, user_input, run_id):
            yield msg

    # Interruption for human in the loop
    # Wait for user approval via HTTP request
    agent_state = await agent.aget_state(thread)
    messages = agent_state.values.get("messages", [])
    tool_call_message = messages[-1] if messages else None
    
    if not tool_call_message:
        raise Exception("No tool call to approve found in the graph state.")
        
    
    tool_call = tool_call_message.tool_calls[0]
    print(f"Waiting for approval of tool call: {tool_call}")
    yield f"data: {json.dumps({'type': 'approval_request', 'for': tool_call})}\n\n"
    
    user_response = await wait_for_user_approval(thread_id)
    print(f"Received user response: {user_response}")
    # here we need to edit the current graph state with the user response I.e. add a tool message and route to Opey if disapproved

    # Restart graph streaming after user approval
    async for event in agent.astream_events(None, thread, version="v2"):
        async for msg in _process_stream_event(event, user_input, run_id):
            yield msg

    yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@app.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream_agent(user_input: StreamInput) -> StreamingResponse:
    """
    Stream the agent's response to a user input, including intermediate messages and tokens.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    """
    return StreamingResponse(message_generator(user_input), media_type="text/event-stream")


@app.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()