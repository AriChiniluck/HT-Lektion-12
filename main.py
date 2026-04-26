from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from langfuse import observe, propagate_attributes
from langgraph.types import Command

from config import settings
from observability import get_langfuse_client, get_langfuse_handler, infer_support_routing_metadata
from ingest import ingest
from user_memory import finish_session, get_or_create_active_user_id, save_message, start_new_session
from retriever import get_retriever
from supervisor import supervisor, get_last_critique_payload, reset_supervisor_limits, set_active_thread_id, clear_active_thread_id

try:
    from langchain_community.callbacks import get_openai_callback as _get_oai_cb
    _HAS_OAI_CB = True
except Exception:
    _HAS_OAI_CB = False

# One REPL session = one Langfuse session with many traces inside it.
# The user id is stored locally on the device in a pseudonymous form.
# We do NOT use IP or MAC addresses because they are unreliable and privacy-sensitive.
THREAD_ID = f"ht12-{uuid4().hex[:8]}"
ACTIVE_USER_ID = get_or_create_active_user_id()
SESSION_ID = start_new_session(ACTIVE_USER_ID)

TOOL_LABELS = {
    "plan": "[Supervisor -> Planner]",
    "research": "[Supervisor -> Researcher]",
    "critique": "[Supervisor -> Critic]",
    "save_report": "[Supervisor -> save_report]",
}

TOOL_RESULT_LABELS = {
    "plan": "[Planner -> Supervisor]",
    "research": "[Researcher -> Supervisor]",
    "critique": "[Critic -> Supervisor]",
    "save_report": "[save_report -> Supervisor]",
}


def debug_print(*args) -> None:
    if settings.debug:
        console_print(" ".join(str(arg) for arg in args))


def console_print(text: str, fallback: str | None = None) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        if fallback is not None:
            print(fallback)
            return
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe)


def _show_banner(title: str, width: int = 60, pad_top: bool = True) -> None:
    line = "=" * width
    prefix = "\n" if pad_top else ""
    console_print(f"{prefix}{line}")
    console_print(title)
    console_print(line)


@contextmanager
def _open_cb():
    """Context manager that tracks OpenAI token usage and cost if available."""
    if _HAS_OAI_CB:
        with _get_oai_cb() as cb:
            yield cb
    else:
        yield None


def _print_usage(t0: float, cb) -> None:
    """Print elapsed time and token cost after a pipeline run."""
    elapsed = time.monotonic() - t0
    parts = [f"\u23f1  {elapsed:.1f}s"]
    if cb is not None and cb.total_tokens:
        parts.append(f"tokens: {cb.total_tokens:,}")
        cost = cb.total_cost or 0.0
        if not cost:
            # Fallback: manual estimate from model name in settings
            try:
                from langchain_community.callbacks.openai_info import MODEL_COST_PER_1K_TOKENS
                m = settings.model_name.lower()
                input_rate = MODEL_COST_PER_1K_TOKENS.get(m, 0)
                output_rate = MODEL_COST_PER_1K_TOKENS.get(f"{m}-completion", 0)
                if input_rate:
                    prompt = getattr(cb, "prompt_tokens", 0) or 0
                    completion = getattr(cb, "completion_tokens", 0) or 0
                    cost = (prompt * input_rate + completion * output_rate) / 1000
            except Exception:
                pass
        if cost:
            parts.append(f"cost: ${cost:.5f} USD")
    console_print("  |  ".join(parts))


def ensure_knowledge_index() -> None:
    index_ok = Path(settings.index_dir).exists()
    chunks_ok = Path(settings.chunks_path).exists()

    if index_ok and chunks_ok:
        return

    print("Knowledge index not found. Running ingestion...")
    ingest()
    print("Knowledge index ready.")


def warmup_rag() -> None:
    try:
        print("Warming up RAG retriever...")
        _ = get_retriever().search("warmup query")
        print("RAG warm-up completed.")
    except Exception as exc:
        if settings.debug:
            print(f"[DEBUG] RAG warm-up skipped: {exc}")


def extract_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)

    return str(content) if content else ""


def _short_preview(value, limit: int = 1200) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = " ".join(str(value or "").split())

    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _render_debug_args(args: dict, limit: int = 4000) -> str:
    normalized = {
        key: _short_preview(extract_text(value), limit=limit)
        for key, value in (args or {}).items()
    }
    return json.dumps(normalized, ensure_ascii=False, indent=2)


def _parse_json_payload(content: str) -> dict | None:
    try:
        payload = json.loads(str(content or ""))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _show_debug_critic_handoff(critique_payload: dict, next_round: int) -> None:
    verdict = critique_payload.get("verdict", "?")
    revision_requests = critique_payload.get("revision_requests", []) or []
    gaps = critique_payload.get("gaps", []) or []

    _show_banner(
        f"[Supervisor note] Critic verdict: {verdict}; preparing Research round {next_round}",
        pad_top=False,
    )

    if revision_requests:
        console_print("  Based on critic revision_requests:")
        for item in revision_requests:
            console_print(f"    - {item}")
    elif gaps:
        console_print("  Based on critic gaps:")
        for item in gaps:
            console_print(f"    - {item}")


def _format_debug_payload(label: str, content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return f"{label}(<empty>)"

    try:
        payload = json.loads(text)
        pretty = json.dumps(payload, ensure_ascii=False, indent=2)
        return f"{label}(\n{pretty}\n)"
    except Exception:
        preview = text[:900] + ("..." if len(text) > 900 else "")
        return preview


def _show_debug_tool_call(tool_name: str, args: dict, research_round: int) -> None:
    label = TOOL_LABELS.get(tool_name, f"[Supervisor -> {tool_name}]")
    if tool_name == "research":
        label = f"{label}  (round {research_round})"

    _show_banner(label)

    if "request" in args:
        preview = _short_preview(args.get("request", ""), limit=240)
        console_print(
            f'🔧 {tool_name}(request="{preview}")',
            fallback=f'TOOL {tool_name}(request="{preview}")',
        )
    elif "plan" in args:
        preview = _short_preview(args.get("plan", ""), limit=240)
        console_print(
            f'🔧 {tool_name}(plan="{preview}")',
            fallback=f'TOOL {tool_name}(plan="{preview}")',
        )
    elif "findings" in args:
        preview_request = _short_preview(args.get("original_request", ""), limit=140)
        preview_findings = _short_preview(args.get("findings", ""), limit=240)
        console_print(
            f'🔧 {tool_name}(original_request="{preview_request}", findings="{preview_findings}")',
            fallback=f'TOOL {tool_name}(original_request="{preview_request}", findings="{preview_findings}")',
        )
    else:
        preview_args = {
            key: _short_preview(value, limit=240) for key, value in args.items()
        }
        rendered = json.dumps(preview_args, ensure_ascii=False)
        console_print(
            f"🔧 {tool_name}({rendered})",
            fallback=f"TOOL {tool_name}({rendered})",
        )

    if tool_name in {"research", "critique"} and args:
        console_print("  Full args:")
        console_print(_render_debug_args(args))


def _show_debug_tool_result(tool_name: str, content: str) -> None:
    result_label = TOOL_RESULT_LABELS.get(tool_name, f"[{tool_name} -> Supervisor]")
    _show_banner(result_label, pad_top=False)

    if tool_name == "plan":
        block = _format_debug_payload("ResearchPlan", content)
    elif tool_name == "critique":
        block = _format_debug_payload("CritiqueResult", content)
    else:
        limit = 3000 if tool_name == "research" else 1000
        block = content[:limit] + ("..." if len(content) > limit else "")

    indented = block.replace("\n", "\n  ")
    console_print(f"  📎 {indented}", fallback=f"  RESULT {indented}")


def stream_payload(payload, config, suppress_final_text: bool = False) -> list:
    interrupts = []
    final_messages: list[str] = []
    research_round = 0
    save_report_done = False
    thread_id = (config.get("configurable") or {}).get("thread_id", "")

    # Register thread_id (and session_id) for this OS thread BEFORE the stream so
    # wrap_model_call can resolve the correct thread_id on the very first model
    # invocation — before any tool call sets the ContextVar.
    # Safe for multi-user: threading.local() is isolated per OS thread.
    if thread_id:
        set_active_thread_id(thread_id, SESSION_ID)
    try:
        for chunk in supervisor.stream(
            payload,
            config=config,
            stream_mode=["updates"],
            version="v2",
        ):
            if chunk["type"] != "updates":
                continue

            data = chunk["data"]

            if "__interrupt__" in data:
                interrupts = list(data["__interrupt__"])
                continue

            model_payload = data.get("model") or {}
            for message in model_payload.get("messages", []):
                tool_calls = getattr(message, "tool_calls", []) or []

                if tool_calls:
                    if settings.debug:
                        for call in tool_calls:
                            if call.get("name") == "research":
                                research_round += 1
                                if research_round > 1:
                                    critique_payload = get_last_critique_payload(thread_id)
                                    if critique_payload:
                                        _show_debug_critic_handoff(critique_payload, research_round)
                            _show_debug_tool_call(
                                call.get("name", "tool"),
                                call.get("args", {}) or {},
                                research_round,
                            )
                    continue

                text = extract_text(getattr(message, "content", ""))
                if text.strip():
                    final_messages.append(text)

            tools_payload = data.get("tools") or {}
            for message in tools_payload.get("messages", []):
                tool_name = getattr(message, "name", "tool")
                content = extract_text(getattr(message, "content", ""))
                if not content.strip():
                    continue

                if tool_name == "save_report" and (
                    content.startswith("Report saved to")
                    or content == "Report saved successfully."
                ):
                    save_report_done = True

                if settings.debug:
                    if tool_name == "critique":
                        pass  # last_critique_payload now stored centrally in supervisor._RUN_LIMITS

                    _show_debug_tool_result(tool_name, content)
    finally:
        if thread_id:
            clear_active_thread_id()
    final_text = "\n".join(msg for msg in final_messages if msg.strip()).strip()
    if final_text and not suppress_final_text:
        # Save the assistant reply locally so a returning user can later continue the dialogue.
        save_message(SESSION_ID, "assistant", final_text)
    if final_text and not interrupts and not save_report_done and not suppress_final_text:
        console_print(final_text)

    return interrupts


def show_interrupt(interrupt) -> None:
    payload = getattr(interrupt, "value", {}) or {}
    action_requests = payload.get("action_requests", [])

    if not action_requests:
        console_print("\nWARNING: interrupt received, but no action details were found.")
        console_print(str(payload))
        return

    request = action_requests[0]
    name = request.get("name", "unknown")
    arguments = request.get("args") or request.get("arguments") or {}
    filename = arguments.get("filename", "report.md")
    content = extract_text(arguments.get("content", ""))
    # Show the full report content so the user can read it before deciding.
    # No length limit here — the user should see everything before approving.

    _show_banner("⏸️  ACTION REQUIRES APPROVAL")
    console_print(f"  Tool:  {name}")

    visible_args = {
        key: (extract_text(value)[:300] + ("..." if len(extract_text(value)) > 300 else ""))
        if key == "content"
        else value
        for key, value in arguments.items()
    }
    console_print(f"  Filename: {filename}")

    if settings.debug:
        rendered_args = json.dumps(visible_args, ensure_ascii=False)
        console_print(f"  Args:  {rendered_args}")
    else:
        console_print("\nReport content:\n")
        console_print(content or "(empty)")

    console_print("")


def ask_decision() -> str:
    while True:
        choice = input("approve / edit / reject > ").strip().lower()
        if choice in {"approve", "edit", "reject"}:
            return choice
        print("Будь ласка, введіть: approve, edit або reject.")


def resolve_interrupts(interrupts, config) -> None:
    pending = interrupts

    while pending:
        current = pending[0]
        show_interrupt(current)

        decision = ask_decision()

        if decision == "approve":
            payload = Command(resume={"decisions": [{"type": "approve"}]})

        elif decision == "edit":
            user_feedback = (
                input("Feedback > ").strip()
                or "Please revise the draft before saving and then call save_report again."
            )
            int_payload = getattr(current, "value", {}) or {}
            action_requests = int_payload.get("action_requests", [])
            original = action_requests[0] if action_requests else {}
            orig_name = original.get("name", "save_report")
            orig_args = dict(original.get("args") or original.get("arguments") or {})
            orig_args["feedback"] = (
                "Reviewer requested changes before saving. The report was NOT saved. "
                "Please apply the feedback below, pay attention to the critic's remarks, "
                "verify the current date if relevant, and then call save_report again.\n\n"
                f"Reviewer feedback: {user_feedback}"
            )
            payload = Command(
                resume={
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": orig_name,
                                "args": orig_args,
                            },
                        }
                    ]
                }
            )

        else:
            reason = (
                input("Reason (optional): ").strip()
                or "User rejected saving the report."
            )
            thread_id = (config.get("configurable") or {}).get("thread_id", "")
            reset_supervisor_limits(thread_id)
            payload = Command(
                resume={
                    "decisions": [
                        {
                            "type": "reject",
                            "message": reason,
                        }
                    ]
                }
            )

        print("\nAgent:")
        pending = stream_payload(payload, config, suppress_final_text=(decision == "reject"))
        if not pending and decision == "approve":
            print("Report saved.")
        if not pending and decision == "reject":
            console_print("Save canceled. The report was not saved.")


@observe(name="lecture12_user_turn")
def run_traced_turn(user_input: str, config: dict) -> str | None:
    """Run one full user turn under a single Langfuse trace.

    This includes the main agent execution and any save-report approval loop.
    The function returns the current trace id so it is easy to find in Langfuse UI.
    """
    langfuse_client = get_langfuse_client()

    save_message(SESSION_ID, "user", user_input)
    routing_meta = infer_support_routing_metadata(user_input)

    with propagate_attributes(
        session_id=SESSION_ID,
        user_id=ACTIVE_USER_ID,
        tags=["lecture-12", "ht10-base", "multi-agent", settings.langfuse_cloud_region.lower()],
        metadata={
            "homework": "lecture-12",
            "foundation": "HT10",
            "thread_id": (config.get("configurable") or {}).get("thread_id", ""),
            "agent_name": "supervisor",
            "category": str(routing_meta["category"]),
            "urgency": str(routing_meta["urgency"]),
            "confidence": f"{routing_meta['confidence']:.2f}",
            "langfuse_project_name": settings.langfuse_project_name,
            "langfuse_project_id": settings.langfuse_project_id,
            "langfuse_org_name": settings.langfuse_org_name,
            "langfuse_org_id": settings.langfuse_org_id,
            "langfuse_cloud_region": settings.langfuse_cloud_region,
            "user_identity_mode": "local_pseudonymous_id",
        },
    ):
        trace_id = langfuse_client.get_current_trace_id()
        # Build a fresh callback handler NOW — inside @observe — so it is linked
        # to the active trace. This covers supervisor LLM calls. Agent tools
        # (plan, research, critique) each create their own fresh handler too.
        run_config = {**config, "callbacks": [get_langfuse_handler()]}
        interrupts = stream_payload(
            {"messages": [{"role": "user", "content": user_input}]},
            run_config,
        )
        if interrupts:
            resolve_interrupts(interrupts, run_config)
        finish_session(SESSION_ID, topic_summary=user_input, resolution_status="processed", escalated=False)
        langfuse_client.flush()
        return trace_id


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("Multi-Agent Research System (Lecture 12 / HT10 base)")
    print("Debug mode:", "ON" if settings.debug else "OFF")
    print(f"Thread ID: {THREAD_ID}")
    print(f"Langfuse session: {SESSION_ID}")
    print(f"Langfuse user: {ACTIVE_USER_ID}")
    print(
        f"Limits: max_revise_cycles={settings.critique_max_rounds}, "
        f"recursion_limit={settings.graph_recursion_limit}, "
        f"llm_timeout={settings.llm_timeout_sec}s, "
        f"url_timeout={settings.url_fetch_timeout_sec}s"
    )
    print(f"FAISS index dir: {settings.index_dir}")
    print("Commands: 'debug on', 'debug off', '/ingest', 'exit'.")
    print("-" * 60)

    ensure_knowledge_index()
    warmup_rag()

    # callbacks are NOT set here — get_langfuse_handler() must be called inside
    # run_traced_turn() (which is @observe-decorated) so the handler is created
    # within an active Langfuse trace context and gets properly linked to the trace.
    config = {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": settings.graph_recursion_limit,
    }

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                print("Будь ласка, введіть запит.")
                continue

            lowered = user_input.lower()

            if lowered in {"exit", "quit"}:
                print("Goodbye!")
                break

            if lowered == "debug on":
                settings.debug = True
                print("Debug mode: ON")
                continue

            if lowered == "debug off":
                settings.debug = False
                print("Debug mode: OFF")
                continue

            if lowered == "/ingest":
                print("Rebuilding knowledge index...")
                ingest()
                print("Done.")
                continue

            print("\nAgent:")
            _t0 = time.monotonic()
            with _open_cb() as _cb:
                trace_id = run_traced_turn(user_input, config)
            _print_usage(_t0, _cb)
            if trace_id:
                console_print(f"Langfuse trace id: {trace_id}")

        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            break
        except Exception as exc:
            msg = str(exc)
            exc_type = type(exc).__name__
            is_dangling_tool_calls = (
                ("tool_calls" in msg and "tool messages" in msg)
                or "InvalidUpdateError" in exc_type
                or "GraphInterrupt" in exc_type
            )
            if is_dangling_tool_calls:
                new_thread = f"ht12-{uuid4().hex[:8]}"
                config["configurable"]["thread_id"] = new_thread
                reset_supervisor_limits(new_thread)
                console_print(
                    "\n[Recovery] Попередній запит завис і залишив незакриті tool-calls. "
                    f"Стартую новий thread: {new_thread}. Повторіть запит."
                )
            else:
                print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
