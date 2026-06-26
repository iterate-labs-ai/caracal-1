"""GDB/MI action space for embodied Recon module.

50 commands the model can emit. Output parsed structured.
"""

ACTIONS = {
    "break_function": {"args": ["function_name"], "mi": "-break-insert"},
    "break_address": {"args": ["address"], "mi": "-break-insert *"},
    "continue": {"args": [], "mi": "-exec-continue"},
    "step": {"args": [], "mi": "-exec-step"},
    "next": {"args": [], "mi": "-exec-next"},
    "finish": {"args": [], "mi": "-exec-finish"},
    "until": {"args": [], "mi": "-exec-until"},
    "print_expr": {"args": ["expression"], "mi": "-data-evaluate-expression"},
    "examine_memory": {"args": ["address", "count", "format"], "mi": "-data-read-memory-bytes"},
    "set_var": {"args": ["name", "value"], "mi": "-gdb-set var"},
    "info_locals": {"args": [], "mi": "-stack-list-locals --all-values"},
    "info_registers": {"args": [], "mi": "-data-list-register-values x"},
    "backtrace": {"args": [], "mi": "-stack-list-frames"},
    "thread_switch": {"args": ["thread_id"], "mi": "-thread-select"},
    "run_with_input": {"args": ["input_bytes"], "mi": "-exec-run"},
    "disasm": {"args": ["function_name"], "mi": "-data-disassemble"},
    "list_breakpoints": {"args": [], "mi": "-break-list"},
    "delete_breakpoint": {"args": ["bp_id"], "mi": "-break-delete"},
    "watch_var": {"args": ["expression"], "mi": "-break-watch"},
    "info_proc_mappings": {"args": [], "mi": "info proc mappings"},
    # ... mais ate 50
}


def emit_action(action_name: str, **kwargs) -> str:
    """Emit gdb/mi command string from action name and args."""
    spec = ACTIONS.get(action_name)
    if not spec:
        raise ValueError(f"Unknown action: {action_name}")
    mi = spec["mi"]
    for arg_name in spec["args"]:
        mi += f" {kwargs.get(arg_name, '')}"
    return mi.strip()


__all__ = ["ACTIONS", "emit_action"]
