"""Real-time chat moderation demo.

Each message is classified live (clean / offensive) by the selected model,
offensive words are masked, and a side panel logs per-message latency
against the CPU latency budget. Default model: INT8-quantized BERTurk (ONNX).

Run:
    python app/app.py
"""
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr

from moderation.models.lexicon import LexiconModerator
from moderation.preprocessing.normalize import join_spaced_letters
from moderation.registry import load_available

LATENCY_BUDGET_MS = 50.0
LOG_COLUMNS = ["time", "model", "message", "label", "P(off)", "latency ms", "budget"]

MODELS, MISSING = load_available()
if not MODELS:
    sys.exit("No trained models found in models/ - see README for the training commands.")
DEFAULT_MODEL = next(iter(MODELS))
# the lexicon is only used to decide *which* words to mask once a message is flagged
MASKER = next((m for m in MODELS.values() if isinstance(m, LexiconModerator)), None) or LexiconModerator()
WORD_RE = re.compile(r"\S+")

# first calls pay one-off initialization costs; keep them out of the latency log
for _model in MODELS.values():
    _model.predict_proba(["merhaba, iyi oyunlar"])


def mask_message(text: str) -> tuple[str, bool]:
    """Mask offensive words; returns (masked text, whether any word was masked)."""
    masked_any = False

    def repl(match: re.Match) -> str:
        nonlocal masked_any
        word = match.group(0)
        if MASKER.explain(word):
            masked_any = True
            return word[0] + "∗" * (len(word) - 1)
        return word

    # "S A L A K" is masked as one word
    return WORD_RE.sub(repl, join_spaced_letters(text)), masked_any


def classify(text: str, model_name: str) -> tuple[float, float]:
    model = MODELS[model_name]
    start = time.perf_counter()
    proba = float(model.predict_proba([text])[0])
    return proba, (time.perf_counter() - start) * 1000


def render(text: str, proba: float, offensive: bool) -> str:
    if not offensive:
        return f"🟢 **clean** · {1 - proba:.0%}\n\n{text}"
    masked, masked_any = mask_message(text)
    if not masked_any:
        # flagged by context, not by a known word: hide the whole message
        masked = "_[message hidden by moderation]_"
    return f"🔴 **offensive** · {proba:.0%}\n\n{masked}"


def latency_summary(log: list[list]) -> str:
    if not log:
        return f"Budget: **{LATENCY_BUDGET_MS:.0f} ms / message** on CPU. No messages yet."
    lines = [f"Budget: **{LATENCY_BUDGET_MS:.0f} ms / message** on CPU", "",
             "| model | msgs | mean ms | max ms | within budget |", "|---|---|---|---|---|"]
    for name in dict.fromkeys(row[1] for row in log):
        lat = [row[5] for row in log if row[1] == name]
        ok = sum(v <= LATENCY_BUDGET_MS for v in lat)
        lines.append(f"| {name} | {len(lat)} | {sum(lat) / len(lat):.1f} | {max(lat):.1f} | {ok}/{len(lat)} |")
    return "\n".join(lines)


def on_send(text: str, model_name: str, chat: list, log: list, last: str):
    text = (text or "").strip()
    if not text:
        return gr.update(), chat, log, log, latency_summary(log), last
    proba, ms = classify(text, model_name)
    offensive = proba >= MODELS[model_name].threshold
    chat = chat + [{"role": "user", "content": render(text, proba, offensive)}]
    shown = mask_message(text)[0] if offensive else text
    row = [datetime.now().strftime("%H:%M:%S"), model_name, shown[:60],
           "offensive" if offensive else "clean", round(proba, 3), round(ms, 2),
           "✅" if ms <= LATENCY_BUDGET_MS else "❌"]
    log = [row] + log
    return "", chat, log, log, latency_summary(log), text


def on_compare(text: str, last: str):
    # the textbox is cleared after sending, so fall back to the last sent message
    text = (text or "").strip() or last
    if not text:
        return []
    rows = []
    for name, model in MODELS.items():
        proba, ms = classify(text, name)
        rows.append([name, "offensive" if proba >= model.threshold else "clean",
                     round(proba, 3), round(model.threshold, 2), round(ms, 2)])
    return rows


def on_clear():
    return [], [], [], latency_summary([])


EXAMPLES = [
    "bu akşam maç kaç saatte başlıyor?",
    "s1kt1r git buradan gerizekalı",
    "takımın kadar ezik bir oyuncusun",
    "L A N  S A L A K  misin",
    "harika oynadın kanka, tebrikler!",
]

with gr.Blocks(title="Chat Moderation") as demo:
    gr.Markdown("## Real-time chat moderation · Turkish\n"
                "Type a chat message; it is labeled live, offensive words are masked, "
                "and the side panel logs response times.")
    log_state = gr.State([])
    last_msg = gr.State("")
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=460, label="Game chat")
            with gr.Row():
                msg = gr.Textbox(placeholder="Write a message and press Enter…", show_label=False, scale=5)
                send = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(EXAMPLES, inputs=msg)
            with gr.Accordion("Compare all models on the last message", open=False):
                compare_btn = gr.Button("Run all models")
                compare_out = gr.Dataframe(headers=["model", "label", "P(off)", "threshold", "latency ms"],
                                           interactive=False)
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Moderation model")
            if MISSING:
                gr.Markdown("Not loaded (not trained yet): " + ", ".join(MISSING))
            summary = gr.Markdown(latency_summary([]))
            log_df = gr.Dataframe(headers=LOG_COLUMNS, interactive=False, label="Response-time log",
                                  wrap=True)
            clear = gr.Button("Clear chat and log")

    inputs = [msg, model_dd, chat, log_state, last_msg]
    outputs = [msg, chat, log_state, log_df, summary, last_msg]
    msg.submit(on_send, inputs, outputs)
    send.click(on_send, inputs, outputs)
    compare_btn.click(on_compare, [msg, last_msg], compare_out)
    clear.click(on_clear, None, [chat, log_state, log_df, summary])


if __name__ == "__main__":
    demo.launch()
