from __future__ import annotations

from dataclasses import dataclass

from .i18n import Text, t
from .ollama import SUMMARY_MODELS

GIB = 1024 ** 3


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None = None
    # The git blob id (SHA-1 of "blob <size>\0" + content) for files the source keeps outside LFS.
    git_sha1: str | None = None


@dataclass(frozen=True)
class Term:
    """In Japanese only for now: legal wording is not translated yet."""

    text: str
    link: str = ""
    url: str = ""


@dataclass(frozen=True)
class ModelEntry:
    id: str
    component: str
    label: str
    folder: str
    source: str
    page: str
    license: str
    base_url: str
    files: tuple[RemoteFile, ...]
    summary: Text
    profile: str | None = None
    target: str = ""
    essential: tuple[str, ...] = ()
    license_page: str = ""
    terms: tuple[Term, ...] = ()

    @property
    def size(self) -> int:
        return sum(item.size for item in self.files)

    def url(self, item: RemoteFile) -> str:
        return f"{self.base_url}/{item.path}"


_HF = "https://huggingface.co"

WHISPER_TURBO = ModelEntry(
    id="whisper-turbo", component="asr", label="whisper-large-v3-turbo", folder="faster-whisper-large-v3-turbo",
    source="Hugging Face · dropbox-dash/faster-whisper-large-v3-turbo",
    page=f"{_HF}/dropbox-dash/faster-whisper-large-v3-turbo", license="MIT",
    base_url=f"{_HF}/dropbox-dash/faster-whisper-large-v3-turbo/resolve/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    files=(
        RemoteFile("config.json", 2263, git_sha1="0351d1d6870005e865747b781b5d7c23ea0459cd"),
        RemoteFile("model.bin", 1_617_884_929, sha256="e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"),
        RemoteFile("preprocessor_config.json", 340, git_sha1="931c77a740890c46365c7ae0c9d350ba3cca908f"),
        RemoteFile("tokenizer.json", 2_710_337, git_sha1="17456db595adc78a973f97d69d8cb50bc87c0b1c"),
        RemoteFile("vocabulary.json", 1_068_114, git_sha1="0adcd01e7c237205d593b707e66dd5d7bc785d2d"),
    ),
    summary=Text("高精度。GPU（CUDA）で 1 発話 0.1 秒ほどで認識し、話している途中の字幕も出ます。",
                 "High accuracy. On a GPU (CUDA) it recognizes an utterance in about 0.1 s and shows captions while people speak."),
    profile="full", target="gpu", essential=("config.json", "model.bin"),
)

WHISPER_SMALL = ModelEntry(
    id="whisper-small", component="asr", label="whisper-small", folder="faster-whisper-small",
    source="Hugging Face · Systran/faster-whisper-small", page=f"{_HF}/Systran/faster-whisper-small", license="MIT",
    base_url=f"{_HF}/Systran/faster-whisper-small/resolve/536b0662742c02347bc0e980a01041f333bce120",
    files=(
        RemoteFile("config.json", 2370, git_sha1="e5047537059bd8f182d9ca64c470201585015187"),
        RemoteFile("model.bin", 483_546_902, sha256="3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671"),
        RemoteFile("tokenizer.json", 2_203_239, git_sha1="7818adb6de9fa3064d3ff81226fdd675be1f6344"),
        RemoteFile("vocabulary.txt", 459_861, git_sha1="c9074644d9d1205686f16d411564729461324b75"),
    ),
    summary=Text("軽量。CPU（int8）で動きます。字幕は話し終わってから数秒で出ます（話している途中の字幕はなし）。",
                 "Light. Runs on the CPU (int8). Captions appear a few seconds after each utterance (none while people speak)."),
    profile="light", target="cpu", essential=("config.json", "model.bin"),
)

SPEAKER_CAMPPLUS = ModelEntry(
    id="speaker-campplus", component="speakers", label="CAM++（3D-Speaker）", folder="speaker",
    source="GitHub · k2-fsa/sherpa-onnx（speaker-recongition-models）",
    page="https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models", license="Apache-2.0",
    base_url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models",
    files=(
        RemoteFile("3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx", 28_281_164,
                   sha256="aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"),
    ),
    summary=Text("声の特徴から S1・S2 のような話者ラベルを付けます（自動判定のため誤ることがあります）。CPU で動きます。",
                 "Labels speakers as S1, S2 and so on from their voices (automatic, so it can be wrong). Runs on the CPU."),
    essential=("3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",),
)

# The zip is kept after unpacking: its presence and size say the runtime is here, its hash that it is intact.
LLAMA_CPP = ModelEntry(
    id="llama-cpp", component="llm-runtime", label="llama.cpp b11800（Vulkan）", folder="llama.cpp-b11800-vulkan",
    source="GitHub · ggml-org/llama.cpp", page="https://github.com/ggml-org/llama.cpp/releases/tag/b11800", license="MIT",
    base_url="https://github.com/ggml-org/llama.cpp/releases/download/b11800",
    files=(
        RemoteFile("llama-b11800-bin-win-vulkan-x64.zip", 36_859_398,
                   sha256="71b56d08a428ad7fbf1643a7a505e643d90bca7dc77138ebfae22adf0e845a01"),
    ),
    summary=Text("内蔵の要約と、字幕の翻訳（PLaMo）を動かす実行環境です。Vulkan で NVIDIA・AMD・Intel の GPU を使い、GPU がなければ CPU で動きます。",
                 "The runtime for built-in summaries and caption translation (PLaMo). Uses NVIDIA, AMD and Intel GPUs through "
                 "Vulkan, and the CPU when there is no GPU."),
    essential=("llama-b11800-bin-win-vulkan-x64.zip",),
)

BUILTIN_QWEN3_8B = ModelEntry(
    id="builtin-qwen3-8b", component="llm", label="Qwen3-8B（Q4_K_M）", folder="Qwen3-8B-GGUF",
    source="Hugging Face · Qwen/Qwen3-8B-GGUF", page=f"{_HF}/Qwen/Qwen3-8B-GGUF", license="Apache-2.0",
    base_url=f"{_HF}/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974",
    files=(
        RemoteFile("Qwen3-8B-Q4_K_M.gguf", 5_027_783_488, sha256="d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"),
    ),
    summary=Text("Ollama の qwen3:8b と同じモデルです。VRAM 8 GB 以上の GPU 向け。メモリ 16 GB 以上なら CPU でも動きますが、要約は遅れます。",
                 "The same model as Ollama's qwen3:8b. For GPUs with 8 GB of VRAM or more. With 16 GB of memory it also runs "
                 "on the CPU, but summaries lag."),
    essential=("Qwen3-8B-Q4_K_M.gguf",),
)

_PLAMO_TERMS = (
    Term("個人の利用（学習・趣味など）と、学術を含む非商用の利用は無償です。登録も要りません。"),
    Term("仕事など商用の利用は、利用者（または関連会社）の直近の年間売上が 10 億円以下なら、PFN の登録フォームで利用を登録すると無償で使えます。",
         "PLaMo 商用利用登録フォーム", "https://forms.gle/mTL8tBLrMYXKNZD56"),
    Term("年間売上が 10 億円を超える場合の商用利用には、PFN との別の契約が必要です。",
         "PLaMo 翻訳モデルのお問い合わせ", "https://translate.preferredai.jp/llm/contact/"),
    Term("翻訳した字幕を公開するときは、PLaMo の出力であることを明示してください。"),
)

PLAMO_TRANSLATE = ModelEntry(
    id="plamo-3-translate-2b", component="translator", label="PLaMo 3 Translate 2B（Q4_K_M）", folder="plamo-3-translate-2b-GGUF",
    source="Hugging Face · pfnet/plamo-3-translate-2b-GGUF", page=f"{_HF}/pfnet/plamo-3-translate-2b-GGUF",
    license="PLaMo Community License",
    base_url=f"{_HF}/pfnet/plamo-3-translate-2b-GGUF/resolve/f2b4599dff9b3e45c62567eac15e83186b4cb64e",
    files=(
        RemoteFile("plamo-3-translate-2b-Q4_K_M.gguf", 1_807_192_480, sha256="2c288ecc7ebfe60c3561726580da3583767f3c41ca3ef5c6825bf6c4cc060787"),
    ),
    summary=Text("字幕の翻訳だけを行う、Preferred Networks の翻訳専用モデルです。53 言語に対応（ポルトガル語は対象外）。"
                 "メモリ約 2 GB で、GPU なら 1 行 0.1 秒、CPU でも 1 行 1 秒ほどで訳します。",
                 "Preferred Networks' model made only for translation, used for captions. 53 languages (not Portuguese). "
                 "About 2 GB of memory; about 0.1 s per line on a GPU and about 1 s per line on a CPU."),
    essential=("plamo-3-translate-2b-Q4_K_M.gguf",),
    license_page="https://plamo.preferredai.jp/info/plamo-community-license-ja",
    terms=_PLAMO_TERMS,
)

# The smallest size of the same model: measured on the same lines as 2B (2026-11-06, RTX 5090 and 4 CPU threads),
# +0.75 GB of VRAM and 0.05 s per line on the GPU, 0.35 s per line on the CPU (2B: 1.1 s), but more lines
# lose a unit or shift in meaning ("140 ms" became "140 s").
PLAMO_TRANSLATE_SMALL = ModelEntry(
    id="plamo-3-translate-0.8b", component="translator", label="PLaMo 3 Translate 0.8B（Q4_K_M）",
    folder="plamo-3-translate-0.8b-GGUF", source="Hugging Face · pfnet/plamo-3-translate-0.8b-GGUF",
    page=f"{_HF}/pfnet/plamo-3-translate-0.8b-GGUF", license="PLaMo Community License",
    base_url=f"{_HF}/pfnet/plamo-3-translate-0.8b-GGUF/resolve/03891e907db8a43c97d3a7c75add125cc6d1f96a",
    files=(
        RemoteFile("plamo-3-translate-0.8b-Q4_K_M.gguf", 573_116_288, sha256="3dbb09295bfd3350231ba0dba92bc72cb14111ef1b034e37dad39d581bd86b9c"),
    ),
    summary=Text("同じ翻訳専用モデルの最小版です。メモリ約 0.8 GB と軽く、CPU でも 1 行 0.4 秒ほど（2B の約 3 倍速）で訳します。"
                 "そのぶん単位や意味を取り違える行が 2B より増えます。メモリの少ない PC や、GPU のない PC 向け。",
                 "The smallest size of the same translation model. Light, about 0.8 GB of memory, and about 0.4 s per line even "
                 "on a CPU (about 3 times as fast as 2B), but more lines get a unit or the meaning wrong than with 2B. For a "
                 "PC with little memory or without a GPU."),
    essential=("plamo-3-translate-0.8b-Q4_K_M.gguf",),
    license_page="https://plamo.preferredai.jp/info/plamo-community-license-ja",
    terms=_PLAMO_TERMS,
)

MODELS: tuple[ModelEntry, ...] = (WHISPER_TURBO, WHISPER_SMALL, SPEAKER_CAMPPLUS, LLAMA_CPP, BUILTIN_QWEN3_8B, PLAMO_TRANSLATE,
                                  PLAMO_TRANSLATE_SMALL)
ASR_MODELS = tuple(entry for entry in MODELS if entry.component == "asr")
BUILTIN_LLM_MODELS = tuple(entry for entry in MODELS if entry.component == "llm")
TRANSLATOR_MODELS = tuple(entry for entry in MODELS if entry.component == "translator")


@dataclass(frozen=True)
class LlmEntry:
    name: str
    size: int
    summary: Text

    @property
    def need(self) -> int:
        return memory_need(self.size)


LLM_MODELS: tuple[LlmEntry, ...] = (
    LlmEntry("qwen3.6:35b-a3b", 22_621_314_161, Text("最も高品質。VRAM 32 GB 級の GPU 向け。", "Highest quality. For GPUs with about 32 GB of VRAM.")),
    LlmEntry("qwen3.8:27b", 17_741_871_939, Text("高品質。VRAM 24 GB 以上の GPU 向け。", "High quality. For GPUs with 24 GB of VRAM or more.")),
    LlmEntry("qwen3:8b", 5_225_387_677, Text("許可しているうちで最小。VRAM 8 GB 以上の GPU 向け。メモリ 16 GB 以上なら CPU でも動きますが、要約は遅れます。",
                                             "The smallest allowed. For GPUs with 8 GB of VRAM or more. With 16 GB of memory it also "
                                             "runs on the CPU, but summaries lag.")),
    LlmEntry("mistral-nemo:12b", 7_071_712_741, Text("VRAM 12 GB 以上の GPU 向け。", "For GPUs with 12 GB of VRAM or more.")),
)
assert tuple(item.name for item in LLM_MODELS) == SUMMARY_MODELS, "the catalog follows the allowlist's order"


@dataclass(frozen=True)
class FoundryEntry:
    alias: str
    ollama: str
    summary: Text


FOUNDRY_MODELS: tuple[FoundryEntry, ...] = (
    FoundryEntry("qwen3-8b", "qwen3:8b", Text("Ollama の qwen3:8b と同じモデル（ONNX 版）です。VRAM 8 GB 以上の GPU 向け。",
                                             "The same model as Ollama's qwen3:8b (ONNX). For GPUs with 8 GB of VRAM or more.")),
    FoundryEntry("mistral-nemo-12b-instruct", "mistral-nemo:12b",
                 Text("Ollama の mistral-nemo:12b と同じモデル（ONNX 版）です。VRAM 12 GB 以上の GPU 向け。",
                      "The same model as Ollama's mistral-nemo:12b (ONNX). For GPUs with 12 GB of VRAM or more.")),
)


def memory_need(size: int) -> int:
    return int(size * 1.1 + 1.5 * GIB)

# Memory the ASR model takes on the GPU while a summary model shares it (turbo in float16, with headroom).
_ASR_VRAM = 3 * GIB


def translator_need(entry: ModelEntry = PLAMO_TRANSLATE) -> int:
    """Memory the caption translator needs loaded with its 4k context (measured: +1.8 GB VRAM for PLaMo Q4_K_M)."""
    return int(entry.size * 1.1 + 0.25 * GIB)


def translator_fits(hardware: dict, llm_need: int | None, entry: ModelEntry = PLAMO_TRANSLATE) -> bool:
    gpu = hardware.get("gpu") or {}
    vram = gpu.get("vram") or 0
    if not gpu.get("cuda") or not vram or llm_need is None:
        return False
    return _ASR_VRAM + llm_need + translator_need(entry) <= vram * 0.95


def recommend(hardware: dict) -> dict:
    gpu = hardware.get("gpu") or {}
    cuda = bool(gpu.get("cuda"))
    memory = (hardware.get("memory") or {}).get("total") or 0
    gpu_name = gpu.get("name") or "GPU"
    if cuda:
        asr, asr_reason = WHISPER_TURBO.id, t(f"{gpu_name} で動く高精度モデルです。", f"A high-accuracy model that runs on the {gpu_name}.")
    elif gpu.get("name"):
        asr, asr_reason = WHISPER_SMALL.id, t(f"{gpu_name} を CUDA で使えないため、CPU で動く軽量モデルを勧めます。",
                                              f"The {gpu_name} cannot be used with CUDA, so a light model that runs on the CPU is recommended.")
    else:
        asr, asr_reason = WHISPER_SMALL.id, t("CUDA で使える GPU がないため、CPU で動く軽量モデルを勧めます。",
                                              "There is no GPU that can use CUDA, so a light model that runs on the CPU is recommended.")
    vram = gpu.get("vram") or 0
    # The best allowlisted model that fits the GPU beside the ASR model (the allowlist is in preference order).
    fitting = next((item for item in LLM_MODELS if item.need <= vram * 0.95 - _ASR_VRAM), None) if cuda and vram else None
    if fitting is not None:
        llm, llm_reason = fitting.name, t(f"{gpu_name}（VRAM {vram / GIB:.0f} GB）に字幕のモデルと一緒に収まる、最も高品質なモデルです。",
                                          f"The highest-quality model that fits the {gpu_name} ({vram / GIB:.0f} GB of VRAM) together "
                                          "with the caption model.")
    elif memory >= 16 * GIB:
        llm = "qwen3:8b"
        llm_reason = t((f"VRAM {vram / GIB:.0f} GB には字幕のモデルと一緒に収まらないため、" if cuda else "GPU がないため、")
                       + "多くを CPU で動かします。要約は数十秒遅れ、字幕の認識とも CPU を取り合います。",
                       (f"It does not fit {vram / GIB:.0f} GB of VRAM together with the caption model, so " if cuda
                        else "There is no GPU, so ")
                       + "most of it runs on the CPU. Summaries lag by tens of seconds and share the CPU with the captions.")
    else:
        llm, llm_reason = None, t("メモリが 16 GB 未満のため、要約（LLM）は勧めません。字幕のみで使えます。",
                                  "With less than 16 GB of memory, summaries (an LLM) are not recommended. Captions only works.")
    # Summaries are recommended only where the LLM fits the GPU; on CPU it competes with the ASR.
    plan = "summaries" if fitting is not None else "captions"
    beside = next((entry for entry in TRANSLATOR_MODELS if translator_fits(hardware, fitting.need, entry)), None) if fitting else None
    with_summaries = beside is not None
    translator = beside.id if beside is not None else PLAMO_TRANSLATE.id if fitting is None and memory >= 8 * GIB else None
    if memory < 8 * GIB:
        translator_reason = t("メモリが 8 GB 未満のため、翻訳専用のモデルも勧めません。",
                              "With less than 8 GB of memory, the translation model is not recommended either.")
    elif cuda and vram and _ASR_VRAM + translator_need() <= vram * 0.95:
        translator_reason = t(f"{gpu_name} で 1 行 0.1 秒ほどで訳します（メモリ約 2 GB）。",
                              f"Translates a line in about 0.1 s on the {gpu_name} (about 2 GB of memory).")
        if fitting is None:
            translator_reason += t("要約の LLM は字幕のモデルと一緒に VRAM に収まりませんが、翻訳専用の PLaMo なら収まります。",
                                   " A summary LLM does not fit the VRAM together with the caption model, but PLaMo, made only "
                                   "for translation, does.")
    else:
        translator_reason = t("CUDA で使える GPU がないため、llama.cpp が使える GPU（Vulkan）か CPU で動かします。CPU なら 1 行 1 秒ほどです"
                              "（要約の LLM で翻訳すると 10 行で約 1 分）。",
                              "There is no GPU that can use CUDA, so it runs on a GPU llama.cpp can use (Vulkan) or on the CPU: "
                              "about 1 s per line on the CPU (a summary LLM takes about a minute for 10 lines).")
    # Summaries need no advice: the use is recommended and the LLM card says which model fits and why.
    plan_reason = ("" if plan == "summaries"
                   else t("この PC では字幕のみが快適です。要約も使えますが、遅くなります。",
                          "Captions only runs best on this PC. Summaries also work, but slowly.") if llm
                   else t("この PC では字幕のみを勧めます。", "Captions only is recommended on this PC."))
    if plan == "captions" and translator:
        plan_reason += t("字幕を翻訳するなら、翻訳専用の PLaMo を使う「字幕＋翻訳」が軽く動きます。",
                         " To translate captions, \"Captions + translation\" with PLaMo, made only for translation, runs light.")
    return {"plan": plan, "plan_reason": plan_reason, "asr": asr, "asr_reason": asr_reason, "llm": llm, "llm_reason": llm_reason,
            "translator": translator, "translator_reason": translator_reason, "translator_with_summaries": with_summaries}
