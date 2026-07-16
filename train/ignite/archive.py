"""Archive: generation ledger + HF push + JSONL logs.

Stores every (gen, candidate) attempt with mutation, dev/val reward, entropy,
KL, retention decision. Enables replay + Shapley attribution + reward-hacking audit.

Layout:
    archive_dir/
      log.jsonl              # append-only run log
      v0/, v1/, ..., v8/     # LoRA adapter checkpoints per accepted generation
      state.json             # last (gen, cand) for hot-reload
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


class Archive:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "log.jsonl"
        self.state_path = self.root / "state.json"

    def append(self, event: dict):
        row = {"ts": int(time.time()), **event}
        with self.log_path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def record_candidate(
        self,
        gen: int,
        cand_id: int,
        mutation,
        dev_r: float,
        val_r: float | None = None,
        entropy_delta: float | None = None,
        passk_ok: bool | None = None,
        retained: bool | None = None,
        reason: str | None = None,
    ):
        self.append(
            {
                "type": "candidate",
                "gen": gen,
                "cand_id": cand_id,
                "mutation_hash": mutation.hash() if hasattr(mutation, "hash") else "",
                "mutation": asdict(mutation)
                if hasattr(mutation, "__dataclass_fields__")
                else str(mutation),
                "dev_r": dev_r,
                "val_r": val_r,
                "entropy_delta": entropy_delta,
                "passk_ok": passk_ok,
                "retained": retained,
                "reason": reason,
            }
        )

    def record_generation(self, gen: int, v_ckpt: str, val_r: float, delta_pp: float):
        self.append(
            {
                "type": "generation",
                "gen": gen,
                "v_ckpt": v_ckpt,
                "val_r": val_r,
                "delta_pp": delta_pp,
            }
        )

    def save_state(self, state: dict[str, Any]):
        self.state_path.write_text(json.dumps(state, indent=2))

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"last_gen": -1, "last_cand": -1}
        return json.loads(self.state_path.read_text())

    def push_hf(self, adapter_dir: Path, repo_id: str, gen: int):
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
        api.upload_folder(
            folder_path=str(adapter_dir),
            path_in_repo=f"v{gen}",
            repo_id=repo_id,
            repo_type="model",
        )
        self.append({"type": "hf_push", "repo_id": repo_id, "gen": gen})

    def load_from_hf(self, repo_id: str, gen: int, out_dir: Path) -> Path:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"v{gen}/*"],
                local_dir=str(out_dir),
            )
        )
