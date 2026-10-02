"""Jev-style beslismodel: Ollama, MLX (Apple Silicon) of PyTorch/CUDA.

Elke backend krijgt een state + vraag + opties en geeft per optie een logit/logprob
voor de optieletter (A, B, C, ...). Er wordt geen tekst gegenereerd; alleen de
verhoudingen tussen de opties tellen (zie softmax).

Het model wordt bij de eerste start automatisch gedownload van Hugging Face.
"""

import http.client
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import urllib.request

DEFAULT_MODEL = "jev-style-2b"
OLLAMA_HOST, OLLAMA_PORT = "localhost", 11434
HERE = os.path.dirname(os.path.abspath(__file__))

GGUF_REPO = "chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF"
GGUF_FILE = "Jev-Style-Qwen3.5-2B-Decision-Q8_0.gguf"
# Dezelfde v1-gewichten in bf16; MLX laadt ze direct, PyTorch na een kleine omzetting.
HF_REPO = "chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16"

BACKENDS = ("auto", "ollama", "mlx", "cuda")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_prompt(state, question, options):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return ("You are a decision function. Read the state, then answer the question "
            "by choosing exactly one option.\n\n"
            f"[State]\n{state}\n\n"
            f"[Question]\n{question}\n\n"
            f"[Options]\n{opts}\n\nAnswer:")


def softmax(scores):
    """Lijst logits -> genormaliseerde kansen (uniform als alles -inf is)."""
    mx = max(scores)
    if mx == -math.inf:
        return [1 / len(scores)] * len(scores)
    exp = [math.exp(s - mx) for s in scores]
    tot = sum(exp)
    return [e / tot for e in exp]


# --- Ollama --------------------------------------------------------------------

class JevClient:
    """Eén HTTP-verbinding met Ollama. Maak er per thread één (zie OllamaBackend)."""

    prompt = staticmethod(build_prompt)

    def __init__(self, model=DEFAULT_MODEL):
        self.model = model
        self.conn = None

    def _post(self, body):
        for attempt in (0, 1):
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=30)
                self.conn.request("POST", "/api/generate", body,
                                  {"Content-Type": "application/json"})
                return json.loads(self.conn.getresponse().read())
            except (http.client.HTTPException, OSError):
                if self.conn:
                    self.conn.close()
                self.conn = None
                if attempt:
                    raise

    def logprobs(self, state, question, options):
        """Logprob per optie (in volgorde van options); -inf als niet in de top."""
        body = json.dumps({
            "model": self.model, "raw": True, "stream": False,
            "prompt": build_prompt(state, question, options),
            "options": {"num_predict": 1, "temperature": 0},
            "logprobs": True, "top_logprobs": 20,
            "keep_alive": "10m",
        }).encode()
        data = self._post(body)
        letters = {LETTERS[i]: i for i in range(len(options))}
        scores = [-math.inf] * len(options)
        for t in data["logprobs"][0]["top_logprobs"]:
            i = letters.get(t["token"].strip())
            if i is not None:
                scores[i] = max(scores[i], t["logprob"])
        return scores


def ollama_models():
    """Namen van de modellen in Ollama, of None als Ollama niet bereikbaar is."""
    try:
        with urllib.request.urlopen(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags",
                                    timeout=3) as r:
            return [m["name"] for m in json.loads(r.read())["models"]]
    except Exception:
        return None


def _has_model(names, model):
    return any(n.split(":")[0] == model.split(":")[0] for n in names)


def download(url, dest):
    """Download met voortgang naar stderr; eerst naar .part, zodat een afgebroken download niet telt."""
    part = dest + ".part"
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB ({done / total:.0%})",
                      end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    os.replace(part, dest)


def ensure_ollama_model(model):
    """Zorgt dat het model in Ollama staat; het standaardmodel wordt zo nodig gedownload."""
    names = ollama_models()
    if names is None:
        sys.exit(f"Ollama niet bereikbaar op {OLLAMA_HOST}:{OLLAMA_PORT}. Start 'ollama serve', "
                 "of kies --backend mlx / cuda (zie README.md).")
    if _has_model(names, model):
        return
    if model != DEFAULT_MODEL:
        sys.exit(f"Model '{model}' niet gevonden in Ollama.")
    if not shutil.which("ollama"):
        sys.exit("Het 'ollama'-commando is nodig om het model aan te maken.")
    gguf = os.path.join(HERE, GGUF_FILE)
    if not os.path.exists(gguf):
        # 'ollama pull hf.co/...' werkt niet voor dit model, dus zelf downloaden.
        print(f"Model downloaden: {GGUF_REPO}/{GGUF_FILE} (±2 GB)...", file=sys.stderr)
        download(f"https://huggingface.co/{GGUF_REPO}/resolve/main/{GGUF_FILE}", gguf)
    print(f"Ollama-model '{model}' aanmaken...", file=sys.stderr)
    subprocess.run(["ollama", "create", model, "-f", "Modelfile"], cwd=HERE, check=True)


class OllamaBackend:
    name = "ollama"

    def __init__(self, model=DEFAULT_MODEL):
        ensure_ollama_model(model)
        self.model = model

    def client(self):
        # Eigen verbinding per brein, zodat pong de twee batjes parallel kan vragen.
        return JevClient(self.model)


# --- In-process (MLX / PyTorch) --------------------------------------------------

class LocalBackend:
    """Model in dit proces. De brein-threads delen het model, dus aanroepen gaan via een lock.

    De hidden state op de laatste positie wordt alleen vermenigvuldigd met de rijen van de
    (gedeelde) embedding-matrix voor ' A', ' B', ...: dat zijn precies de logits van de
    optieletters. De kalibratietemperatuur zit al in de gewichten.
    """

    prompt = staticmethod(build_prompt)

    def __init__(self):
        self.lock = threading.Lock()

    def client(self):
        return self

    def letter_ids(self, tok):
        return [tok.encode(" " + c, add_special_tokens=False)[0] for c in LETTERS]

    def logprobs(self, state, question, options):
        ids = self.tok.encode(build_prompt(state, question, options), add_special_tokens=False)
        with self.lock:
            return self._logits(ids, len(options))


class MLXBackend(LocalBackend):
    name = "mlx"

    def __init__(self, repo=HF_REPO):
        super().__init__()
        import mlx.core as mx
        from mlx_lm import load
        print(f"MLX-model laden: {repo} (eerste keer ±4 GB downloaden)...", file=sys.stderr)
        model, self.tok = load(repo)
        self.mx = mx
        self.inner = model.language_model.model
        self.rows = self.inner.embed_tokens.weight[mx.array(self.letter_ids(self.tok))]

    def _logits(self, ids, n):
        mx = self.mx
        last = self.inner(mx.array([ids]))[0, -1]  # inclusief laatste norm
        return (self.rows[:n] @ last).astype(mx.float32).tolist()


class TorchBackend(LocalBackend):
    """PyTorch (transformers). Bedoeld voor CUDA; werkt ook op 'mps' of 'cpu', maar trager."""

    name = "cuda"

    def __init__(self, repo=HF_REPO, device="cuda"):
        super().__init__()
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        if device == "cuda" and not torch.cuda.is_available():
            sys.exit("CUDA niet beschikbaar in deze PyTorch-installatie (zie README.md).")
        print(f"Model laden op {device}: {repo} (eerste keer ±4 GB downloaden)...",
              file=sys.stderr)
        path = snapshot_download(repo)
        self.tok = AutoTokenizer.from_pretrained(path)
        config = AutoConfig.from_pretrained(path).text_config
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        weights = self._from_mlx(load_file(os.path.join(path, "model.safetensors")), torch)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                model = Qwen3_5TextModel(config)
        finally:
            torch.set_default_dtype(prev)
        model.load_state_dict({k: v.to(dtype) for k, v in weights.items()}, strict=True)
        self.model = model.eval()
        self.torch, self.device = torch, device
        self.rows = model.embed_tokens.weight[self.letter_ids(self.tok)].float()

    @staticmethod
    def _from_mlx(weights, torch):
        """MLX-gewichten -> transformers: conv1d-assen terug en de '+1' van de RMSNorms eraf."""
        out = {}
        for k, v in weights.items():
            k = k.removeprefix("language_model.model.")
            if "conv1d.weight" in k:
                v = v.transpose(1, 2)
            elif k.endswith("norm.weight") and "linear_attn" not in k:
                v = v.float() - 1.0
            out[k] = v
        return out

    def _logits(self, ids, n):
        torch = self.torch
        with torch.inference_mode():
            x = torch.tensor([ids], device=self.device)
            last = self.model(input_ids=x, use_cache=False).last_hidden_state[0, -1].float()
            return (self.rows[:n] @ last).tolist()


# --- Kiezen -------------------------------------------------------------------------

def _importable(name):
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _cuda_available():
    if not _importable("torch") or not _importable("transformers"):
        return False
    import torch
    return torch.cuda.is_available()


def load_backend(name="auto", model=DEFAULT_MODEL):
    """Maakt de backend. 'auto': MLX op Apple Silicon, anders CUDA, anders Ollama."""
    if name == "auto":
        apple = sys.platform == "darwin" and platform.machine() == "arm64"
        if apple and _importable("mlx_lm"):
            name = "mlx"
        elif _cuda_available():
            name = "cuda"
        else:
            name = "ollama"
    if name == "ollama":
        return OllamaBackend(model)
    if name == "mlx":
        if not _importable("mlx_lm"):
            sys.exit("mlx-lm ontbreekt: pip install -r requirements.txt (Apple Silicon).")
        return MLXBackend()
    if name == "cuda":
        if not (_importable("torch") and _importable("transformers")):
            sys.exit("PyTorch/transformers ontbreken: pip install -r requirements-cuda.txt.")
        return TorchBackend()
    raise ValueError(f"onbekende backend: {name}")
