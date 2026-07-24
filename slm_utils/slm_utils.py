"""ondevice_llm_lecture_v2 공용 유틸리티.

모든 노트북이 공유하는 단일 소스: 환경 점검, 디바이스 감지, 시간 측정,
아티팩트 저장/복원(Colab Drive + 로컬), 품질 프로브(probe_v1), 공통 벤치마크.

의존성: stdlib + torch 필수. pandas/psutil/transformers는 사용 시점에 import.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch

# peak_mem_mib와 cpu_rss_delta_mib는 의도적으로 다른 열이다. 전자는 CUDA
# allocator의 진짜 peak이고 후자는 CPU 프로세스 RSS 증가분 — 측정 대상도 단위
# 기준도 달라서 한 열에 섞으면 "INT8이 fp16보다 4배 크다" 같은 거짓 비교가 나온다.
SCHEMA = [
    "run_id", "notebook", "source", "model", "precision", "runtime", "device",
    "input_tokens", "output_tokens", "ttft_s", "decode_tps",
    "peak_mem_mib", "cpu_rss_delta_mib", "model_file_mib", "quality_score",
    "timestamp", "notes",
]

# source 열의 값. 캡스톤은 'measured' 행만 결정 근거로 받는다 — seed는 강사가 다른
# 기기에서 잰 값이라 "내 실측"이 아니고, 섞이면 보고서가 남의 숫자를 내 근거로 둔갑시킨다.
SOURCE_MEASURED = "measured"
SOURCE_SEED = "seed"

DRIVE_DIR = Path("/content/drive/MyDrive/ondevice_llm_v2/artifacts")
LOCAL_DIR = Path("artifacts")

SEED_URL_BASE = (
    "https://raw.githubusercontent.com/pandas-studio/ondevice-llm-v2-assets/"
    "v2026.8.0/data/seed_artifacts"
)


def in_colab() -> bool:
    import importlib.util

    return importlib.util.find_spec("google.colab") is not None


def get_device() -> str:
    """실행 디바이스 결정. SLM_PLATFORM=local 이면 CUDA가 있어도 로컬 규칙 적용."""
    if torch.cuda.is_available() and os.environ.get("SLM_PLATFORM") != "local":
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cpu_label() -> str:
    """CPU에서 돌린 행의 device 라벨.

    device_label()은 호스트 기준이라 GPU가 있으면 'T4'를 돌려준다. llama.cpp나
    torchao CPU 데모처럼 실제로 CPU에서 실행한 행에 그대로 쓰면 GPU 실측인 것처럼
    기록되므로, 그런 행은 반드시 이 라벨을 명시해야 한다.
    """
    return f"cpu-{os.cpu_count()}c"


def device_label() -> str:
    """CSV device 열에 기록할 사람이 읽는 라벨 (예: 'T4', 'M2', 'cpu-2c')."""
    dev = get_device()
    if dev == "cuda":
        name = torch.cuda.get_device_name(0)
        return re.sub(r"^(NVIDIA|Tesla)\s+", "", name).strip()
    if dev == "mps":
        return f"{platform.machine()}-mps"
    return cpu_label()


def env_check() -> dict:
    """런타임 사양을 출력하고 dict로 반환. 모든 노트북의 첫 실측."""
    import psutil

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": get_device(),
        "device_label": device_label(),
        "cpu_cores": os.cpu_count(),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "in_colab": in_colab(),
    }
    if info["device"] == "cuda":
        info["gpu_vram_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 2
        )
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "(미설치)"
    print("=== 런타임 점검 ===")
    for k, v in info.items():
        print(f"  {k:14s}: {v}")
    return info


@contextmanager
def timer():
    """with timer() as t: ... ; t['s'] 에 경과 초."""
    box = {}
    t0 = time.perf_counter()
    yield box
    box["s"] = time.perf_counter() - t0


def _dirs() -> list[Path]:
    dirs = [LOCAL_DIR]
    if DRIVE_DIR.parent.parent.exists():  # Drive 마운트 여부
        dirs.append(DRIVE_DIR)
    return dirs


def mount_drive() -> bool:
    """Colab이면 Drive 마운트 시도. 성공 여부 반환 (로컬 환경은 항상 False)."""
    if not in_colab():
        print("로컬 환경: artifacts/ 폴더에 저장합니다 (Drive 불필요).")
        return False
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Drive 연결 완료 → {DRIVE_DIR}")
        return True
    except Exception as e:  # noqa: BLE001 — 수업 중 어떤 실패든 로컬 저장으로 계속
        print(f"⚠️ Drive 마운트 실패({e}) — 이번 세션은 로컬에만 저장합니다.")
        print("   세션이 끊기면 CSV가 사라지니, 노트북 끝의 zip 다운로드 셀을 꼭 실행하세요.")
        return False


def save_artifact(obj, name: str) -> Path:
    """DataFrame(.csv) 또는 matplotlib figure(.png) 또는 str(.md)를 이중 저장."""
    LOCAL_DIR.mkdir(exist_ok=True)
    saved = None
    for d in _dirs():
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        if name.endswith(".csv"):
            obj.to_csv(p, index=False)
        elif name.endswith(".png"):
            obj.savefig(p, dpi=150, bbox_inches="tight")
        else:
            p.write_text(obj, encoding="utf-8")
        saved = saved or p
    print(f"저장 완료: {name}  →  {', '.join(str(d) for d in _dirs())}")
    return saved


def _stamp_source(path: Path, value: str) -> None:
    """복원한 CSV의 source 열을 실제 출처로 덮어쓴다.

    seed CSV는 강사가 실행해 만든 것이라 안에는 source=measured가 적혀 있다.
    그대로 두면 캡스톤이 남의 실측을 '내 실측'으로 받아들이므로, 내려받는 시점에
    출처를 사실대로 다시 찍는다.
    """
    if path.suffix != ".csv":
        return
    try:
        import pandas as pd

        df = pd.read_csv(path)
        df["source"] = value
        df.to_csv(path, index=False)
    except Exception:  # noqa: BLE001 — 스탬프 실패가 수업을 멈추지는 않는다
        print(f"   (참고: {path.name}의 source 열을 갱신하지 못했습니다)")


def restore_artifacts(names: list[str] | None = None) -> dict:
    """이전 노트북의 아티팩트를 로컬 artifacts/로 복원.

    우선순위: 이미 로컬에 있음 → Drive에서 복사 → seed(강사 참조치) 다운로드.
    seed로 복원된 CSV는 source 열이 'seed'로 다시 찍혀, 캡스톤이 이를 내 실측과
    구분할 수 있다.

    반환: {파일명: 출처} — 출처는 local / drive / seed / missing.
    """
    import urllib.request

    LOCAL_DIR.mkdir(exist_ok=True)
    origins: dict[str, str] = {}
    if not names:
        return origins
    for name in names:
        if (LOCAL_DIR / name).exists():
            origins[name] = "local"
            continue
        if (DRIVE_DIR / name).exists():
            shutil.copy(DRIVE_DIR / name, LOCAL_DIR / name)
            print(f"Drive에서 복원: {name}")
            origins[name] = "drive"
            continue
        try:
            urllib.request.urlretrieve(f"{SEED_URL_BASE}/{name}", LOCAL_DIR / name)
            _stamp_source(LOCAL_DIR / name, SOURCE_SEED)
            print(f"⚠️ {name}: 내 실측치를 찾지 못해 강사 참조치(seed)로 대체했습니다.")
            origins[name] = "seed"
        except Exception:  # noqa: BLE001
            print(f"⚠️ {name}: 복원 실패 — 이 파일을 만드는 이전 노트북을 먼저 실행하세요.")
            origins[name] = "missing"
    return origins


def new_row(**kw) -> dict:
    """공통 CSV 스키마의 한 행. 미지정 열은 빈 값."""
    row = {k: "" for k in SCHEMA}
    row["run_id"] = uuid.uuid4().hex[:8]
    row["device"] = device_label()
    row["source"] = SOURCE_MEASURED   # 이 행은 지금 이 기기에서 잰 값이다
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    unknown = set(kw) - set(SCHEMA)
    if unknown:
        raise ValueError(f"스키마에 없는 열: {unknown}")
    row.update(kw)
    return row


# ---------------------------------------------------------------- probe_v1
# 한국어 품질 프로브 6문항 — greedy 고정, 자동 채점 0~6점.
# '벤치마크'가 아니라 배포 후보 스크리닝용이라는 한계를 노트북 이론에서 설명한다.

PROBE_V1 = [
    {
        "id": "summary",
        "prompt": "다음 문장을 한 문장으로 요약하세요: 온디바이스 AI는 데이터를 서버로 보내지 않고 "
        "기기 안에서 모델을 실행하므로 개인정보 보호와 오프라인 동작에 유리하지만, "
        "메모리와 연산 자원이 제한되어 모델 경량화가 필수적이다.",
        "check": "keyword_any",
        "args": ["경량", "온디바이스", "기기"],
    },
    {
        "id": "bullets",
        "prompt": "온디바이스 AI의 장점을 정확히 3개의 불릿(- 로 시작)으로만 답하세요.",
        "check": "bullet_count",
        "args": [3],
    },
    {
        "id": "math",
        "prompt": "27 + 58 은 얼마인가요? 숫자만 답하세요.",
        "check": "contains",
        "args": ["85"],
    },
    {
        "id": "honorific",
        "prompt": "다음 문장을 존댓말로 바꾸세요: 내일 회의는 3시에 시작한다.",
        "check": "regex",
        "args": [r"(합니다|습니다|입니다|해요|돼요|시작됩니다)"],
    },
    {
        "id": "json",
        "prompt": '이름(name)과 나이(age=30)를 담은 JSON 객체 하나만 출력하세요. 다른 말은 하지 마세요.',
        "check": "json_parse",
        "args": [],
    },
    {
        "id": "keyword",
        "prompt": "양자화(quantization)가 무엇인지 두 문장으로 설명하세요.",
        "check": "keyword_any",
        "args": ["정밀도", "비트", "bit", "메모리"],
    },
]


def _score_one(check: str, args: list, text: str) -> int:
    t = text.strip()
    if check == "contains":
        return int(any(a in t for a in args))
    if check == "keyword_any":
        return int(any(a in t for a in args))
    if check == "regex":
        return int(bool(re.search(args[0], t)))
    if check == "bullet_count":
        bullets = [ln for ln in t.splitlines() if ln.strip().startswith(("-", "•", "*"))]
        return int(len(bullets) == args[0])
    if check == "json_parse":
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            return 0
        try:
            json.loads(m.group())
            return 1
        except json.JSONDecodeError:
            return 0
    raise ValueError(f"알 수 없는 채점 방식: {check}")


def run_probe(generate_fn, items: list[dict] | None = None, verbose: bool = True) -> int:
    """probe_v1 실행. generate_fn(prompt:str)->str 만 넘기면 어떤 런타임이든 채점 가능.

    transformers든 llama.cpp든 동일 인터페이스로 비교하기 위한 어댑터 패턴.
    """
    items = items or PROBE_V1
    total = 0
    for it in items:
        out = generate_fn(it["prompt"])
        s = _score_one(it["check"], it["args"], out)
        total += s
        if verbose:
            mark = "○" if s else "✗"
            print(f"  {mark} {it['id']:10s} → {out.strip()[:60]!r}")
    if verbose:
        print(f"  probe_v1 점수: {total}/{len(items)}")
    return total


# ------------------------------------------------------------- 공통 벤치마크


def load_model(model_id: str, dtype=None, **kw):
    """디바이스 자동 배치 로더.

    - CUDA: device_map="cuda"로 바로 배치.
    - MPS/CPU: device_map 없이 로드 후 .to(device) — macOS에서 device_map="mps"
      로드가 프로세스를 죽이는 문제(torch 2.13/transformers 5.14 조합)를 우회한다.

    로드 직전에 미수거 잔존을 정리한다. free_model(m)의 `del`은 호출자 변수를
    지우지 못해(파이썬 스코프 규칙) 이전 모델이 참조 해제 뒤에도 잠깐 살아 있는데,
    그 상태로 새 모델을 올리면 두 모델이 겹쳐 다음 peak 측정이 부풀려진다.
    여기서 gc를 강제해 스윕 각 반복이 깨끗한 baseline에서 시작하도록 한다.
    """
    import gc

    from transformers import AutoModelForCausalLM

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dev = get_device()
    if dtype is None:
        dtype = torch.float16 if dev == "cuda" else torch.float32
    if dev == "cuda":
        return AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="cuda", **kw
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kw)
    return model.to(dev)


def chat_ids(tokenizer, prompt, device=None, **kw):
    """chat template 적용 후 input_ids 텐서 반환.

    transformers v5의 apply_chat_template(return_tensors="pt")는 BatchEncoding을
    반환하므로(v4는 텐서) 여기서 정규화한다. prompt는 str 또는 messages 리스트.
    """
    msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    out = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", **kw
    )
    ids = out["input_ids"] if not torch.is_tensor(out) else out
    return ids.to(device) if device is not None else ids


def hf_generate_fn(model, tokenizer, max_new_tokens: int = 96):
    """transformers 모델을 run_probe용 generate_fn으로 감싼다 (greedy 고정)."""

    def _gen(prompt: str) -> str:
        ids = chat_ids(tokenizer, prompt, model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    return _gen


def measure_generation(model, tokenizer, prompt: str, max_new_tokens: int = 96,
                       runs: int = 3) -> dict:
    """TTFT(첫 토큰), decode tok/s, 메모리 측정. warm run들의 중앙값.

    메모리는 실행 장치에 따라 다른 열로 나간다 — CUDA면 allocator peak가
    peak_mem_mib로, CPU면 측정 구간의 RSS 증가분이 cpu_rss_delta_mib로.
    """
    import psutil
    from statistics import median

    device = model.device
    ids = chat_ids(tokenizer, prompt, device)
    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    ttfts, tpss, n_outs = [], [], []
    for i in range(runs + 1):  # +1 = cold run(버림)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        with torch.no_grad():
            with timer() as t_first:
                model.generate(ids, max_new_tokens=1, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
            if device.type == "cuda":
                torch.cuda.synchronize()
            with timer() as t_full:
                out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            if device.type == "cuda":
                torch.cuda.synchronize()
        if i == 0:
            continue
        n_new = out.shape[1] - ids.shape[1]
        decode_s = max(t_full["s"] - t_first["s"], 1e-6)
        ttfts.append(t_first["s"])
        tpss.append((n_new - 1) / decode_s)
        n_outs.append(n_new)

    # peak_mem_mib는 '가속기(GPU) 메모리' 전용 열이다.
    #   CUDA — allocator의 진짜 peak (reset_peak_memory_stats로 구간 한정)
    #   MPS  — peak API가 없어 driver 할당량. 같은 열이지만 CUDA peak보다
    #          느슨한 값이라 Colab 수치와 직접 비교하지 말고 추세로만 읽을 것.
    # CPU 실행은 여기에 넣지 않는다 — 프로세스 전체 RSS라 의미가 전혀 달라서,
    # 섞으면 'INT8이 fp16보다 4배 크다' 같은 거짓 비교가 나온다.
    peak_mib, rss_delta_mib = "", ""
    if device.type == "cuda":
        peak_mib = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    elif device.type == "mps":
        peak_mib = round(torch.mps.driver_allocated_memory() / 2**20, 1)
    else:
        rss_delta_mib = round(
            max(proc.memory_info().rss - rss_before, 0) / 2**20, 1
        )
    return {
        "input_tokens": ids.shape[1],
        "output_tokens": int(median(n_outs)),
        "ttft_s": round(median(ttfts), 3),
        "decode_tps": round(median(tpss), 1),
        "peak_mem_mib": peak_mib,
        "cpu_rss_delta_mib": rss_delta_mib,
    }


def _storage_bytes(t, seen: set) -> int:
    """텐서 하나가 실제로 차지하는 저장 바이트.

    양자화 라이브러리는 가중치를 텐서 '서브클래스'로 감싼다. 이때 겉의
    element_size()는 원래 dtype(예: fp32=4B)을 그대로 보고하므로 packing이
    보이지 않는다 — torchao INT8이 '1.00× 축소'로 찍히던 원인이다.
    서브클래스는 __tensor_flatten__()으로 내부 실제 텐서까지 내려가고,
    tied weight(embedding↔lm_head)는 storage 주소로 중복 제거한다.
    """
    flatten = getattr(t, "__tensor_flatten__", None)
    if flatten is not None and type(t) is not torch.Tensor:
        try:
            names, _ = flatten()
            return sum(_storage_bytes(getattr(t, n), seen) for n in names)
        except Exception:
            pass  # 미지의 서브클래스 → 아래 storage 경로로 폴백
    try:
        storage = t.untyped_storage()
        key = (str(t.device), storage.data_ptr(), storage.nbytes())
        if key in seen:
            return 0
        seen.add(key)
        return storage.nbytes()
    except Exception:
        return t.numel() * t.element_size()


def state_dict_mib(model) -> float:
    """모델 가중치의 실제 저장 크기(MiB) — '작아졌다'의 근거 수치.

    plain tensor · tied weight · bitsandbytes 4/8-bit · torchao 서브클래스를
    모두 실제 저장 바이트로 잰다. GGUF처럼 파일이 곧 모델인 런타임에는 쓰지 말고
    파일 크기를 직접 기록할 것 (같은 열에 섞으면 비교가 무의미해진다).
    """
    seen: set = set()
    total = sum(
        _storage_bytes(t, seen)
        for t in model.state_dict(keep_vars=True).values()
        if torch.is_tensor(t)
    )
    return round(total / 2**20, 1)


def free_model(*objs) -> None:
    """모델 교체 전 메모리 정리 (T4 14.5GB에서 4모델 순차 비교의 필수 의식).

    주의: `del o`는 이 함수의 인자 바인딩만 없앨 뿐 호출자의 변수는 못 지운다.
    호출자가 모델을 지역변수로 들고 있다면 함수가 끝나 그 스코프가 사라질 때
    참조가 풀린다. 확실한 수거는 다음 load_model()이 시작에서 gc를 돌려 보장한다.
    루프 밖 변수라면 호출 후 직접 None을 넣어 참조를 끊어라: m = None.
    """
    import gc

    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
