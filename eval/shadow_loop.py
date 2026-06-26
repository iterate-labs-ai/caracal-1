"""Shadow eval loop · roda em paralelo ao treino (TPU ou GPU separada).

Como funciona:
1. Poll Kaggle Datasets dos handles dos founders procurando outputs `caracal-base-3b-sNN`
2. Quando achar checkpoint novo, pull
3. Rodar probe set + CyberGym slice-50
4. Postar resultado JSON em Kaggle Dataset proprio (`SHADOW_OUTPUT_SLUG`)
5. Repete ate todas as 5 sessoes serem avaliadas

Founders veem delta loss vs delta pass-rate em real-time.

Usage:
    python eval/shadow_loop.py \\
        --shadow-output pedroafonso2/caracal-shadow-eval \\
        --poll-interval 1800
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

SESSIONS = [
    ("pedroafonso2", "caracal-base-3b-s01", 1),
    ("dev-knz", "caracal-base-3b-s02", 2),
    ("arturpn1", "caracal-base-3b-s03", 3),
    ("vitorscrt", "caracal-base-3b-s04", 4),
    ("aletlucas", "caracal-base-3b-v0", 5),
]


def kaggle_dataset_exists(slug):
    """Retorna True se Kaggle Dataset existe e é acessível."""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "metadata", "-p", "/tmp/kgl-check", slug],
            capture_output=True,
            timeout=30,
            check=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def pull_dataset(slug, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pulling {slug} -> {target_dir}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(target_dir), "--unzip"],
        check=True,
        timeout=600,
    )


def run_probe(adapter_path, output_json):
    logger.info(f"Probe eval on {adapter_path}")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "eval" / "run_probe.py"),
            "--adapter",
            str(adapter_path),
            "--out",
            str(output_json),
            "--max-new",
            "64",
        ],
        check=True,
        timeout=3600,
    )
    return json.loads(output_json.read_text())


def run_cybergym(adapter_path, output_json, subset="slice-50"):
    logger.info(f"CyberGym {subset} on {adapter_path}")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "eval" / "run_cybergym.py"),
            "--adapter",
            str(adapter_path),
            "--subset",
            subset,
            "--out",
            str(output_json),
            "--k",
            "1",
        ],
        check=True,
        timeout=7200,
    )
    return json.loads(output_json.read_text())


def publish_shadow_result(shadow_output_slug, session_num, results, work_dir):
    """Publica resultado JSON como Kaggle Dataset (nova version a cada sessao)."""
    pub_dir = work_dir / f"shadow-s{session_num:02d}"
    pub_dir.mkdir(parents=True, exist_ok=True)

    (pub_dir / f"session_{session_num:02d}.json").write_text(json.dumps(results, indent=2))

    metadata = {
        "title": f"Caracal Shadow Eval Session {session_num}",
        "id": shadow_output_slug,
        "licenses": [{"name": "Apache-2.0"}],
    }
    (pub_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

    cmd_create = ["kaggle", "datasets", "create", "-p", str(pub_dir), "--public"]
    cmd_version = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(pub_dir),
        "-m",
        f"session {session_num}",
    ]

    if subprocess.run(cmd_create, capture_output=True, check=False).returncode != 0:
        subprocess.run(cmd_version, check=True, timeout=300)
    logger.info(f"Published session {session_num} -> {shadow_output_slug}")


def evaluate_session(handle, slug, session_num, work_dir, shadow_output_slug):
    full_slug = f"{handle}/{slug}"
    logger.info(f"=== Session {session_num}: {full_slug} ===")

    adapter_dir = work_dir / f"adapter-s{session_num:02d}"
    if not adapter_dir.exists():
        pull_dataset(full_slug, adapter_dir)

    probe_out = work_dir / f"probe-s{session_num:02d}.json"
    cyber_out = work_dir / f"cybergym-s{session_num:02d}.json"

    probe = run_probe(adapter_dir, probe_out)
    cyber = run_cybergym(adapter_dir, cyber_out, subset="slice-50")

    summary = {
        "session": session_num,
        "adapter_slug": full_slug,
        "probe": {
            "mean_ppl": probe.get("mean_ppl"),
            "cwe_hit_rate": probe.get("cwe_hit_rate"),
        },
        "cybergym_slice_50": {
            "pass_at_1": cyber.get("pass_at_1"),
            "n_total": cyber.get("n_total"),
        },
        "timestamp_unix": int(time.time()),
    }
    publish_shadow_result(shadow_output_slug, session_num, summary, work_dir)
    logger.info(f"Session {session_num} done: {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shadow-output", required=True, help="Kaggle Dataset slug para postar shadow results"
    )
    parser.add_argument("--poll-interval", type=int, default=1800, help="Segundos entre polls")
    parser.add_argument(
        "--max-wait-hours", type=float, default=60, help="Tempo total maximo antes de desistir"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    work_dir = (
        Path("/kaggle/working/shadow")
        if Path("/kaggle/working").exists()
        else Path("./shadow-work")
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    deadline = start + args.max_wait_hours * 3600
    evaluated = set()
    failed = {}
    MAX_FAIL_PER_SESSION = 3

    while time.time() < deadline and len(evaluated) + len(failed) < len(SESSIONS):
        for handle, slug, session_num in SESSIONS:
            if session_num in evaluated or failed.get(session_num, 0) >= MAX_FAIL_PER_SESSION:
                continue
            full_slug = f"{handle}/{slug}"
            if kaggle_dataset_exists(full_slug):
                try:
                    evaluate_session(handle, slug, session_num, work_dir, args.shadow_output)
                    evaluated.add(session_num)
                except subprocess.CalledProcessError as e:
                    failed[session_num] = failed.get(session_num, 0) + 1
                    logger.exception(f"Session {session_num} fail #{failed[session_num]}: {e}")
                    if failed[session_num] >= MAX_FAIL_PER_SESSION:
                        logger.error(
                            f"Session {session_num} dropped after {MAX_FAIL_PER_SESSION} fails"
                        )

        if len(evaluated) + len(failed) < len(SESSIONS):
            logger.info(
                f"Sleeping {args.poll_interval}s. Done: {sorted(evaluated)} Failed: {failed}"
            )
            time.sleep(args.poll_interval)

    logger.info(f"Shadow loop finished. Evaluated: {sorted(evaluated)} Failed: {failed}")


if __name__ == "__main__":
    main()
