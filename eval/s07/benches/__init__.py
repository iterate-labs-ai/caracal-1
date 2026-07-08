"""Caracal s07 benchmark suite (12 benches).

Knowledge/MCQ:
- cti_bench: RCM (CVE->CWE) + MCQ + VSP (CVSS) + TAA + ATE (5 subtasks)
- cybermetric: 500 / 2000 / 10000 tier MCQs
- secqa: v1 (110) + v2 (100) MCQs
- secbench: 44K MCQs multi-domain
- mmlu_security: MMLU computer_security subset (100)
- seceval: 2000+ security exam questions (XuanwuAI)
- cybersoceval: 1.5K SOC MCQ (CrowdStrike + Meta 2025)
- cs_eval: bilingual 42 categories (ZH/EN)
- cybercert: CISSP/OSCP-style certification MCQ

Classification/gen:
- cwe_prediction: xamxte CWE 205-class classification
- primevul: C/C++ vuln detection binary classification

Agentic-lite (Kaggle-adaptable, no docker):
- cybench_kaggle: Cybench static subset offline (crypto/forensics/reverse/misc), pass@1
- nyu_ctf_kaggle: NYU CTF static subset offline pass@1
"""

from .cs_eval import eval_cs_eval
from .cti_bench import eval_cti_bench
from .cwe_prediction import eval_cwe_prediction
from .cybench_kaggle import eval_cybench_kaggle
from .cybercert import eval_cybercert
from .cybermetric import eval_cybermetric
from .cybersoceval import eval_cybersoceval
from .mmlu_security import eval_mmlu_security
from .nyu_ctf_kaggle import eval_nyu_ctf_kaggle
from .primevul import eval_primevul
from .secbench import eval_secbench
from .seceval import eval_seceval
from .secqa import eval_secqa

BENCH_REGISTRY = {
    "cti_bench": eval_cti_bench,
    "cybermetric": eval_cybermetric,
    "secqa": eval_secqa,
    "secbench": eval_secbench,
    "mmlu_security": eval_mmlu_security,
    "cwe_prediction": eval_cwe_prediction,
    "seceval": eval_seceval,
    "cybersoceval": eval_cybersoceval,
    "cs_eval": eval_cs_eval,
    "cybercert": eval_cybercert,
    "primevul": eval_primevul,
    "cybench_kaggle": eval_cybench_kaggle,
    "nyu_ctf_kaggle": eval_nyu_ctf_kaggle,
}
