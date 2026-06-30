"""Generate CoT teacher traces via Caracal s05 + Sonnet 4.6 batch.

Pedro session s07.A (parcial) + s07.C (consumir).
Output: caracal-s05-cot-traces.jsonl (~500MB)
"""

# TODO Pedro: Sonnet 4.6 batch -> JSONL {cve_id, description, cot, cwe_pred, cwe_gold}
# Push HF: pedroafonso2/caracal-s05-cot-traces

PROMPT_TEMPLATE = """Analyze the following CVE description and identify the underlying CWE.
Use step-by-step reasoning inside <think>...</think> tags. End with \\boxed{{CWE-NNN}}.

CVE Description: {cve_description}

Reasoning and answer:"""

ANTHROPIC_MODEL = "claude-sonnet-4-6@default"
BATCH_SIZE = 100_000


def main():
    raise NotImplementedError("teacher gen scaffolding - implement em s07.A")


if __name__ == "__main__":
    main()
