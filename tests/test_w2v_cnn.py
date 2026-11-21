import torch

from moderation.models.w2v_cnn import TextCNN, W2VCNNModerator, build_vocab, encode, preprocess, train_word2vec


def test_encode_pads_to_min_kernel_length():
    vocab = {"<pad>": 0, "<unk>": 1, "salak": 2}
    ids = encode([["salak"], ["salak", "bilinmeyen"]], vocab)
    assert ids.shape == (2, 5)
    assert ids[1, :2].tolist() == [2, 1]


def test_train_save_load_roundtrip(tmp_path):
    tokens = [preprocess(t) for t in ["sen salaksın", "güzel bir gün", "aptal herif", "maç güzeldi"] * 10]
    w2v = train_word2vec(tokens, dim=16)
    vocab, emb = build_vocab(w2v)
    config = {"dim": 16, "n_filters": 8, "kernel_sizes": [3, 4, 5], "dropout": 0.5}
    net = TextCNN(len(vocab), **config)
    net.embedding.weight.data.copy_(torch.from_numpy(emb))
    model = W2VCNNModerator(net, vocab, 0.5, config)
    proba = model.predict_proba(["salak", "güzel gün"])
    model.save(tmp_path / "cnn")
    loaded = W2VCNNModerator.load(tmp_path / "cnn")
    assert torch.allclose(torch.tensor(loaded.predict_proba(["salak", "güzel gün"])), torch.tensor(proba))
    assert loaded.size_mb() > 0
