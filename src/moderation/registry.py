"""Load every trained model of the ladder from models/ (skipping ones not trained yet)."""
from moderation.data import MODELS_DIR
from moderation.models.base import Moderator

# display name -> loader; the first entry is the application default (R5)
MODEL_LOADERS = {
    "BERTurk · INT8 · ONNX Runtime": lambda: _onnx("model_int8.onnx"),
    "BERTurk · FP32 · ONNX Runtime": lambda: _onnx("model_fp32.onnx"),
    "BERTurk · FP32 · PyTorch": lambda: _bert(),
    "word2vec + 1D-CNN": lambda: _cnn(),
    "tf-idf + LogReg": lambda: _tfidf(),
    "Lexicon + edit distance": lambda: _lexicon(),
}


def _onnx(file: str) -> Moderator:
    from moderation.models.onnx_model import OnnxBertModerator
    return OnnxBertModerator(MODELS_DIR / "berturk", file)


def _bert() -> Moderator:
    from moderation.models.bert import BertModerator
    return BertModerator(MODELS_DIR / "berturk")


def _cnn() -> Moderator:
    from moderation.models.w2v_cnn import W2VCNNModerator
    return W2VCNNModerator.load(MODELS_DIR / "w2v_cnn")


def _tfidf() -> Moderator:
    from moderation.models.tfidf import TfidfModerator
    return TfidfModerator.load(MODELS_DIR / "tfidf_lr.joblib")


def _lexicon() -> Moderator:
    from moderation.models.lexicon import LexiconModerator
    return LexiconModerator.load(MODELS_DIR / "lexicon.json")


def load_available() -> tuple[dict[str, Moderator], dict[str, str]]:
    models, missing = {}, {}
    for name, loader in MODEL_LOADERS.items():
        try:
            models[name] = loader()
        except (FileNotFoundError, OSError) as exc:
            missing[name] = str(exc).splitlines()[0]
    return models, missing
