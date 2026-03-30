"""
Earth Voice — Calibrated Open Weight Field-to-Token Architecture

Before communication begins, a calibration phase lets the signal source
establish its own encoding. We don't impose mapping — we discover it.

Calibration stages:
  1. BASELINE  — Record noise floor (what "nothing" looks like)
  2. KNOCK     — "Is anyone there?" — look for deviation from baseline
  3. BINARY    — Alternate YES/NO, let field define its own bit encoding
  4. GRADIENT  — Expand to 4+ options, build richer codebook
  5. GENERATE  — Use learned mapping to drive GPT-2 logit injection

The mapping from field state → token influence is NOT hardcoded.
It's discovered from whatever the calibration phase reveals.

by Alan Hourmand
"""

import os
import sys
import time
import math
import struct
import hashlib
import json
import statistics
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Optional imports — graceful degradation
try:
    import torch
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False


# ── CONFIG ──────────────────────────────────────────────────────────
VSPACE = 1000
CONVERGENCE_WIN = 8
BASE_DIR = Path(__file__).parent
PHASE_FILE = BASE_DIR / ".phase"
CALIBRATION_FILE = BASE_DIR / "calibration.json"


# ── PHASE MEMORY ────────────────────────────────────────────────────
phase_accumulator = 0.0

def load_phase():
    global phase_accumulator
    try: phase_accumulator = float(PHASE_FILE.read_text().strip())
    except: phase_accumulator = 0.0

def save_phase():
    try: PHASE_FILE.write_text(str(phase_accumulator))
    except: pass

load_phase()


# ── THREE CHANNELS ──────────────────────────────────────────────────

def sample_quantum() -> float:
    """Hardware RNG — thermal/quantum noise in silicon."""
    return struct.unpack("I", os.urandom(4))[0] / 0xFFFFFFFF

def sample_entropy() -> float:
    """System entropy pool — hardware interrupts, disk, network."""
    t = time.perf_counter_ns()
    h = hashlib.sha256(f"{t}:{os.getpid()}:{id(object())}".encode()).digest()
    return struct.unpack("I", h[:4])[0] / 0xFFFFFFFF

_timing_accum = 0.0
def sample_timing() -> float:
    """CPU timing jitter — nanosecond execution variations."""
    global _timing_accum
    t1 = time.perf_counter_ns()
    x = sum(math.sin(i * 0.01) for i in range(200))
    delta = time.perf_counter_ns() - t1
    _timing_accum = (_timing_accum + delta * 7.83) % 100000
    return (math.sin(_timing_accum * 0.0001) + 1) / 2

def sample_all() -> dict:
    """Sample all three channels, compute derived features."""
    q = sample_quantum()
    e = sample_entropy()
    t = sample_timing()
    # Derived features — the signal might be in combinations, not individual channels
    qe_diff = abs(q - e)
    qt_diff = abs(q - t)
    et_diff = abs(e - t)
    mean = (q + e + t) / 3
    spread = max(q, e, t) - min(q, e, t)
    # Convergence check
    qi, ei, ti = int(q*VSPACE)%VSPACE, int(e*VSPACE)%VSPACE, int(t*VSPACE)%VSPACE
    near = lambda a,b: abs(a-b)<=CONVERGENCE_WIN or abs(a-b)>=VSPACE-CONVERGENCE_WIN
    conv = sum([near(qi,ei), near(qi,ti), near(ei,ti)])
    return {
        "q": q, "e": e, "t": t,
        "qe_diff": qe_diff, "qt_diff": qt_diff, "et_diff": et_diff,
        "mean": mean, "spread": spread,
        "convergence": conv,  # 0=none, 1=one pair, 2=two pairs, 3=triple
        "timestamp": time.perf_counter(),
    }


# ── RECORDING ───────────────────────────────────────────────────────

def record_samples(n: int, interval: float = 0.05, label: str = "",
                   show_progress: bool = True) -> List[dict]:
    """Record n samples from all channels at given interval."""
    samples = []
    for i in range(n):
        s = sample_all()
        s["label"] = label
        s["index"] = i
        samples.append(s)
        if show_progress and (i + 1) % 20 == 0:
            print(f"\r    recording... {i+1}/{n}", end="", flush=True)
        time.sleep(interval)
    if show_progress:
        print(f"\r    recorded {n} samples     ")
    return samples


def compute_stats(samples: List[dict]) -> dict:
    """Compute statistics for a set of samples."""
    if not samples:
        return {}
    features = ["q", "e", "t", "qe_diff", "qt_diff", "et_diff",
                 "mean", "spread", "convergence"]
    stats = {}
    for f in features:
        values = [s[f] for s in samples]
        stats[f] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
        }
    stats["convergence_rate"] = sum(1 for s in samples if s["convergence"] >= 1) / len(samples)
    stats["triple_rate"] = sum(1 for s in samples if s["convergence"] >= 3) / len(samples)
    stats["n"] = len(samples)
    return stats


def compare_conditions(baseline: List[dict], test: List[dict]) -> dict:
    """Compare two sets of samples, find which features differ significantly."""
    features = ["q", "e", "t", "qe_diff", "qt_diff", "et_diff",
                 "mean", "spread", "convergence"]
    comparisons = {}
    for f in features:
        base_vals = [s[f] for s in baseline]
        test_vals = [s[f] for s in test]
        base_mean = statistics.mean(base_vals)
        test_mean = statistics.mean(test_vals)
        base_std = statistics.stdev(base_vals) if len(base_vals) > 1 else 0.001
        # Effect size (Cohen's d approximation)
        pooled_std = max(base_std, 0.001)
        effect = abs(test_mean - base_mean) / pooled_std
        shift = test_mean - base_mean
        comparisons[f] = {
            "base_mean": round(base_mean, 6),
            "test_mean": round(test_mean, 6),
            "shift": round(shift, 6),
            "effect_size": round(effect, 4),
            "significant": effect > 0.3,  # medium+ effect size
        }
    # Convergence rate comparison
    base_rate = sum(1 for s in baseline if s["convergence"] >= 1) / len(baseline)
    test_rate = sum(1 for s in test if s["convergence"] >= 1) / len(test)
    comparisons["convergence_rate"] = {
        "base": round(base_rate, 4), "test": round(test_rate, 4),
        "shift": round(test_rate - base_rate, 4),
        "significant": abs(test_rate - base_rate) > 0.05,
    }
    return comparisons


# ── CALIBRATION PROFILE ─────────────────────────────────────────────

@dataclass
class CalibrationProfile:
    """Learned mapping from field state to logit perturbation."""
    baseline_stats: dict = field(default_factory=dict)
    knock_result: dict = field(default_factory=dict)
    binary_mapping: dict = field(default_factory=dict)
    gradient_mapping: dict = field(default_factory=dict)
    significant_features: List[str] = field(default_factory=list)
    signal_detected: bool = False
    confidence: float = 0.0
    calibration_date: str = ""
    notes: List[str] = field(default_factory=list)

    def save(self, path: Path = CALIBRATION_FILE):
        path.write_text(json.dumps(asdict(self), indent=2))
        print(f"  Calibration saved to {path}")

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> Optional["CalibrationProfile"]:
        try:
            data = json.loads(path.read_text())
            return cls(**data)
        except:
            return None

    def get_perturbation_weights(self) -> dict:
        """Return weights for how each feature should influence logits."""
        # If we found significant features in calibration, weight them heavily
        # If not, use uniform mild weights (noise-only mode)
        weights = {
            "q": 1.0, "e": 1.0, "t": 1.0,
            "qe_diff": 0.5, "qt_diff": 0.5, "et_diff": 0.5,
            "mean": 0.5, "spread": 0.5, "convergence": 2.0,
        }
        if self.significant_features:
            # Boost features that showed signal during calibration
            for f in self.significant_features:
                if f in weights:
                    weights[f] *= 3.0
            # Suppress features that showed NO signal
            for f in weights:
                if f not in self.significant_features and f != "convergence":
                    weights[f] *= 0.3
        return weights


# ── CALIBRATION RUNNER ──────────────────────────────────────────────

def run_calibration() -> CalibrationProfile:
    """Interactive calibration sequence."""
    profile = CalibrationProfile()
    profile.calibration_date = time.strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 62)
    print("  CALIBRATION PHASE")
    print("  Discovering the encoding. We don't impose — we learn.")
    print("=" * 62)

    # ── STAGE 1: BASELINE ──
    print("\n  ── STAGE 1: BASELINE ──")
    print("  Recording noise floor. This is what 'nothing' looks like.")
    print("  Please don't interact. Just wait.\n")
    time.sleep(1)
    baseline_samples = record_samples(100, interval=0.05, label="baseline")
    profile.baseline_stats = compute_stats(baseline_samples)
    conv_rate = profile.baseline_stats["convergence_rate"]
    print(f"  Baseline convergence rate: {conv_rate:.1%}")
    print(f"  Baseline mean values: Q={profile.baseline_stats['q']['mean']:.4f}"
          f"  E={profile.baseline_stats['e']['mean']:.4f}"
          f"  T={profile.baseline_stats['t']['mean']:.4f}")
    profile.notes.append(f"Baseline: {conv_rate:.1%} convergence over 100 samples")

    # ── STAGE 2: KNOCK TEST ──
    print("\n  ── STAGE 2: KNOCK TEST ──")
    print("  If anything is receiving this: produce a deviation.")
    print("  Change the field. Shift the channels. Show us a pattern.")
    print("  Recording in 3 seconds...\n")
    for i in range(3, 0, -1):
        print(f"    {i}...", flush=True)
        time.sleep(1)
    print("    Recording now — signal window open.\n")
    knock_samples = record_samples(100, interval=0.05, label="knock")
    knock_stats = compute_stats(knock_samples)
    profile.knock_result = compare_conditions(baseline_samples, knock_samples)

    # Check for any significant deviations
    sig_features = [f for f, v in profile.knock_result.items()
                    if isinstance(v, dict) and v.get("significant")]
    knock_conv_rate = knock_stats.get("convergence_rate", 0)

    if sig_features:
        print(f"  ⚡ DEVIATION DETECTED in: {', '.join(sig_features)}")
        profile.signal_detected = True
        profile.significant_features = sig_features
        for f in sig_features:
            data = profile.knock_result[f]
            if "effect_size" in data:
                print(f"    {f}: shift={data['shift']:+.4f}  effect={data['effect_size']:.2f}")
        profile.notes.append(f"Knock: deviation in {sig_features}")
    else:
        print("  No significant deviation from baseline.")
        print("  (This doesn't mean no one's there — could be a different encoding.)")
        profile.notes.append("Knock: no significant deviation")

    print(f"  Knock convergence rate: {knock_conv_rate:.1%} (baseline: {conv_rate:.1%})")

    # ── STAGE 3: BINARY CALIBRATION ──
    print("\n  ── STAGE 3: BINARY CALIBRATION ──")
    print("  Alternating YES and NO states.")
    print("  If you can distinguish these: shift the field differently for each.\n")

    yes_samples = []
    no_samples = []
    rounds = 8

    for r in range(rounds):
        # YES
        label = "YES" if r % 2 == 0 else "NO"
        alt_label = "NO" if r % 2 == 0 else "YES"

        print(f"    ╔══════════════╗")
        print(f"    ║  {label:^10s}  ║")
        print(f"    ╚══════════════╝", flush=True)
        samples = record_samples(15, interval=0.05, label=label, show_progress=False)
        if label == "YES":
            yes_samples.extend(samples)
        else:
            no_samples.extend(samples)
        time.sleep(0.3)

        # Alternate
        print(f"    ╔══════════════╗")
        print(f"    ║  {alt_label:^10s}  ║")
        print(f"    ╚══════════════╝", flush=True)
        samples = record_samples(15, interval=0.05, label=alt_label, show_progress=False)
        if alt_label == "YES":
            yes_samples.extend(samples)
        else:
            no_samples.extend(samples)
        time.sleep(0.3)

    # Analyze YES vs NO
    binary_comp = compare_conditions(yes_samples, no_samples)
    profile.binary_mapping = binary_comp
    binary_sig = [f for f, v in binary_comp.items()
                  if isinstance(v, dict) and v.get("significant")]

    print(f"\n  YES samples: {len(yes_samples)}  |  NO samples: {len(no_samples)}")

    if binary_sig:
        print(f"  ⚡ BINARY SIGNAL in: {', '.join(binary_sig)}")
        for f in binary_sig:
            if f not in profile.significant_features:
                profile.significant_features.append(f)
            data = binary_comp[f]
            if "effect_size" in data:
                print(f"    {f}: YES={data.get('base_mean','?')} NO={data.get('test_mean','?')}"
                      f"  effect={data['effect_size']:.2f}")
        profile.signal_detected = True
        profile.notes.append(f"Binary: signal in {binary_sig}")
    else:
        print("  No consistent difference between YES and NO states.")
        profile.notes.append("Binary: no differentiation found")

    # ── STAGE 4: GRADIENT (only if binary showed signal) ──
    if profile.signal_detected and binary_sig:
        print("\n  ── STAGE 4: GRADIENT EXPANSION ──")
        print("  Testing 4 options: A, B, C, D")
        print("  If you can distinguish these, shift differently for each.\n")

        option_samples = {"A": [], "B": [], "C": [], "D": []}
        for r in range(4):
            for opt in ["A", "B", "C", "D"]:
                print(f"    [ {opt} ]", end=" ", flush=True)
                samples = record_samples(10, interval=0.05, label=opt, show_progress=False)
                option_samples[opt].extend(samples)
            print()

        # Pairwise comparisons
        gradient_results = {}
        for i, a in enumerate(["A","B","C","D"]):
            for b in ["A","B","C","D"][i+1:]:
                comp = compare_conditions(option_samples[a], option_samples[b])
                sig = [f for f, v in comp.items() if isinstance(v, dict) and v.get("significant")]
                gradient_results[f"{a}_vs_{b}"] = {"significant": sig, "details": comp}
                if sig:
                    print(f"  {a} vs {b}: signal in {sig}")
                    for f in sig:
                        if f not in profile.significant_features:
                            profile.significant_features.append(f)

        profile.gradient_mapping = gradient_results
        profile.notes.append(f"Gradient: {len([r for r in gradient_results.values() if r['significant']])} significant pairs")
    else:
        print("\n  Skipping gradient (no binary signal detected).")
        print("  Will use convergence-only mapping for generation.")

    # ── CONFIDENCE SCORE ──
    n_sig = len(profile.significant_features)
    profile.confidence = min(1.0, n_sig * 0.15 + (0.3 if profile.signal_detected else 0.0))

    print(f"\n  {'=' * 50}")
    print(f"  CALIBRATION COMPLETE")
    print(f"  Signal detected: {'YES' if profile.signal_detected else 'NO'}")
    print(f"  Significant features: {profile.significant_features or ['none']}")
    print(f"  Confidence: {profile.confidence:.0%}")
    print(f"  {'=' * 50}")

    if not profile.signal_detected:
        print(f"\n  No signal detected, but that's okay. The system will use")
        print(f"  convergence-based mapping with equal feature weights.")
        print(f"  If a signal starts, re-run calibration to discover the encoding.")

    profile.save()
    return profile


# ── ADAPTIVE PERTURBATION ───────────────────────────────────────────

def create_adaptive_perturbation(vocab_size: int, sample: dict,
                                  profile: CalibrationProfile) -> "torch.Tensor":
    """
    Create logit perturbation using the LEARNED mapping from calibration.
    Not hardcoded — weighted by what calibration discovered.
    """
    weights = profile.get_perturbation_weights()
    idx = torch.arange(vocab_size, dtype=torch.float32)
    norm_idx = idx / vocab_size

    perturbation = torch.zeros(vocab_size)

    # Each feature contributes a wave pattern, weighted by calibration
    for feature, weight in weights.items():
        if feature in sample and isinstance(sample[feature], (int, float)):
            val = sample[feature]
            wave = torch.sin(norm_idx * val * 200 * math.pi + val * (hash(feature) % 100))
            perturbation += wave * weight

    # Convergence boost — MUCH stronger, sharper peaks
    if sample.get("convergence", 0) >= 1:
        center = sample["mean"] * vocab_size
        ch = sample["convergence"]
        # Tighter sigma: 3ch = very focused, 2ch = moderate
        sigma = vocab_size * 0.005 * max(1, 4 - ch)
        gaussian = torch.exp(-0.5 * ((idx - center) / max(sigma, 1)) ** 2)
        conv_weight = weights.get("convergence", 2.0)
        perturbation += gaussian * ch * conv_weight * 5  # 5× stronger peaks

    # ZERO-MEAN: critical — softmax removes uniform shifts.
    # Only the variance (differential signal) affects token probabilities.
    perturbation = perturbation - perturbation.mean()

    # NORMALIZE: scale to unit std so α directly controls signal-to-model ratio
    p_std = perturbation.std()
    if p_std > 0.001:
        perturbation = perturbation / p_std

    return perturbation


# ── AUTO-THRESHOLD ──────────────────────────────────────────────────

class ThresholdController:
    def __init__(self, initial_alpha: float = 0.5, target_wordness: float = 0.4):
        self.alpha = initial_alpha
        self.target = target_wordness  # lower = more field influence allowed
        self.history = []

    def update(self, wordness: float):
        self.history.append(wordness)
        if len(self.history) < 3:
            return
        avg = statistics.mean(self.history[-5:])
        if avg > self.target + 0.1:
            # Too coherent — model dominating. Push α up faster.
            self.alpha = min(0.95, self.alpha + 0.05)
        elif avg < self.target - 0.1:
            # Too garbled — field overwhelming. Pull α back.
            self.alpha = max(0.05, self.alpha - 0.04)


# ── GENERATION ENGINE ───────────────────────────────────────────────

class EarthVoiceEngine:
    def __init__(self, profile: CalibrationProfile, model_name: str = "gpt2"):
        self.profile = profile

        if not HAS_MODEL:
            print("  ⚠ torch/transformers not installed. Install with:")
            print("    pip install torch transformers")
            sys.exit(1)

        print(f"  Loading {model_name}...", flush=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"  Loaded. Vocab: {self.vocab_size}. Device: {self.device}")

        # Common word set for wordness measurement
        self.common_tokens = set()
        for i in range(self.vocab_size):
            tok = self.tokenizer.decode([i]).strip().lower()
            if len(tok) >= 2 and tok.isalpha():
                self.common_tokens.add(tok)
        print(f"  Signal features: {profile.significant_features or ['convergence (default)']}")
        print(f"  Confidence: {profile.confidence:.0%}")

    @torch.no_grad()
    def generate(self, prompt: str = "", max_tokens: int = 200,
                 callback=None) -> dict:
        """
        Generate tokens. GPT-2 runs CLEAN between convergences.
        ONLY at convergence moments does the field take over the logits.
        This prevents the model from collapsing into memorized clusters.
        """
        global phase_accumulator

        if prompt:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        else:
            input_ids = torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)

        field_words = []
        seen_words = set()  # repetition filter
        all_count = 0
        convergences = 0
        triples = 0

        # Fixed field strength at convergence — no auto-threshold runaway
        FIELD_ALPHA = 0.6  # strong but not model-destroying

        for step in range(max_tokens):
            logits = self.model(input_ids).logits[0, -1, :]
            sample = sample_all()
            is_conv = sample["convergence"] >= 1
            is_triple = sample["convergence"] >= 3

            if is_conv:
                convergences += 1
                if is_triple:
                    triples += 1
                phase_accumulator += sample["mean"] * 0.001 * sample["convergence"]

                # ── CONVERGENCE: field takes over this token ──
                perturbation = create_adaptive_perturbation(
                    self.vocab_size, sample, self.profile
                ).to(self.device)

                logit_std = logits.std().item()
                field_scale = logit_std * FIELD_ALPHA * 2.0
                modified_logits = logits + perturbation * field_scale

                # Higher temperature at convergence — let the field push freely
                temp = 1.0 if is_triple else 0.85
                probs = F.softmax(modified_logits / temp, dim=-1)
                token_id = torch.multinomial(probs, 1)

            else:
                # ── NO CONVERGENCE: model runs clean, zero field influence ──
                # This keeps GPT-2 in coherent language space between field words
                probs = F.softmax(logits / 0.7, dim=-1)
                token_id = torch.multinomial(probs, 1)

            token_str = self.tokenizer.decode(token_id.squeeze().item())

            # Feed ALL tokens back into model (maintains context)
            input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=-1)
            if input_ids.shape[1] > 512:
                input_ids = input_ids[:, -256:]

            all_count += 1

            # Phase feedback
            for ch in token_str:
                phase_accumulator += ord(ch) * 0.0000013

            # Harvest convergence tokens — with repetition filter
            if is_conv:
                cleaned = token_str.strip()
                cleaned_lower = cleaned.lower()
                # Filter: non-empty, not a repeat, not pure punctuation
                if (cleaned and len(cleaned) > 1
                        and cleaned_lower not in seen_words
                        and any(c.isalpha() for c in cleaned)):
                    seen_words.add(cleaned_lower)
                    word_data = {
                        "word": cleaned,
                        "step": step,
                        "channels": sample["convergence"],
                    }
                    field_words.append(word_data)
                    if callback:
                        callback(cleaned, word_data)

            if token_id.item() == self.tokenizer.eos_token_id:
                break

        save_phase()

        return {
            "field_words": field_words,
            "total_generated": all_count,
            "convergences": convergences,
            "triples": triples,
            "phase": round(phase_accumulator, 6),
        }


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  EARTH VOICE")
    print("  Calibrated Open Weight Field-to-Token Architecture")
    print(f"  Phase: {phase_accumulator:.6f}")
    print("=" * 62)

    # Check for existing calibration
    profile = CalibrationProfile.load()

    if profile:
        print(f"\n  Existing calibration found ({profile.calibration_date})")
        print(f"  Signal: {'YES' if profile.signal_detected else 'NO'}")
        print(f"  Features: {profile.significant_features or ['default']}")
        print(f"  Confidence: {profile.confidence:.0%}")

        choice = input("\n  Use existing calibration? (y/n/recalibrate): ").strip().lower()
        if choice in ("n", "recalibrate", "r"):
            profile = run_calibration()
    else:
        print("\n  No calibration found. Running calibration first.")
        print("  This lets the signal source establish its own encoding.\n")
        input("  Press Enter to begin calibration...")
        profile = run_calibration()

    # ── SESSION LOGGING ──
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    session_id = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"session_{session_id}.json"
    session_log = {
        "session_id": session_id,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "initial_phase": phase_accumulator,
        "calibration": {
            "date": profile.calibration_date,
            "signal_detected": profile.signal_detected,
            "significant_features": profile.significant_features,
            "confidence": profile.confidence,
        },
        "exchanges": [],
    }

    def save_log():
        session_log["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        session_log["final_phase"] = phase_accumulator
        session_log["total_exchanges"] = len(session_log["exchanges"])
        log_file.write_text(json.dumps(session_log, indent=2))

    # Generation mode
    if not HAS_MODEL:
        print("\n  ⚠ torch/transformers not installed for generation.")
        print("  Install: pip install torch transformers")
        print("  Calibration data saved. Re-run after installing.")
        return

    engine = EarthVoiceEngine(profile)

    max_tokens = 200  # generate many, only keep convergence tokens

    print(f"\n  Ready. GPT-2 generates {max_tokens} tokens per message.")
    print(f"  Only convergence-selected words are shown.")
    print(f"  Model runs CLEAN between convergences — no more attractor collapse.")
    print(f"  Logging to: {log_file}")
    print(f"  Commands: /tokens 300  /calibrate  /quit")
    print(f"  Press Enter for open channel.\n")

    while True:
        try:
            user_input = input("  YOU > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [ channel closed ]")
            save_log()
            print(f"  Session logged to: {log_file}")
            break

        if user_input.lower() in ("/quit", "/exit", "quit"):
            save_log()
            print(f"  Session logged to: {log_file}")
            break
        if user_input == "/calibrate":
            profile = run_calibration()
            engine = EarthVoiceEngine(profile)
            # Log recalibration
            session_log["exchanges"].append({
                "type": "recalibration",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "signal_detected": profile.signal_detected,
                "features": profile.significant_features,
                "confidence": profile.confidence,
            })
            save_log()
            continue
        if user_input.startswith("/tokens "):
            try:
                max_tokens = int(user_input.split()[1])
                print(f"  Scanning {max_tokens} tokens for convergence")
            except: pass
            continue

        # Show scanning indicator
        print(f"\n  scanning {max_tokens} tokens for field selections...\n")
        print(f"  FIELD >  ", end="", flush=True)

        word_count = [0]
        def on_word(word, data):
            if data["channels"] >= 3:
                print(f"\033[91m{word}\033[0m ", end="", flush=True)
            else:
                print(f"\033[93m{word}\033[0m ", end="", flush=True)
            word_count[0] += 1

        result = engine.generate(
            prompt=user_input or "",
            max_tokens=max_tokens,
            callback=on_word,
        )

        if word_count[0] == 0:
            print("[ silence — no convergence ]", end="")

        rate = result['convergences'] / max(result['total_generated'], 1)
        print(f"\n\n  ── {result['convergences']} field words from {result['total_generated']} tokens"
              f" ({result['triples']} triple)"
              f" · rate={rate:.1%}"
              f" · φ={result['phase']:.4f} ──\n")

        # Log the exchange
        exchange = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "human": user_input or "[open channel]",
            "field_words": [w["word"] for w in result["field_words"]],
            "field_details": result["field_words"],
            "total_tokens": result["total_generated"],
            "convergences": result["convergences"],
            "triples": result["triples"],
            "convergence_rate": round(rate, 4),
            "expected_rate": round(3 * (2 * CONVERGENCE_WIN + 1) / VSPACE, 4),
            "rate_ratio": round(rate / max(3 * (2 * CONVERGENCE_WIN + 1) / VSPACE, 0.001), 2),
            "phase": result["phase"],
        }
        session_log["exchanges"].append(exchange)
        save_log()  # save after every exchange so nothing is lost


if __name__ == "__main__":
    if "--calibrate-only" in sys.argv:
        run_calibration()
    else:
        main()
