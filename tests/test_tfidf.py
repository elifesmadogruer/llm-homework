from moderation.models.tfidf import TfidfModerator, build_pipeline


def test_fit_predict_save_load(tmp_path):
    texts = ["sen salaksın", "çok güzel bir gün", "s1kt1r git", "maç harikaydı", "aptal herif", "kahve içelim"] * 5
    labels = [1, 0, 1, 0, 1, 0] * 5
    model = TfidfModerator(build_pipeline(C=1.0)).fit(texts, labels)
    proba = model.predict_proba(["salak", "güzel gün"])
    assert proba[0] > proba[1]
    model.save(tmp_path / "m.joblib")
    loaded = TfidfModerator.load(tmp_path / "m.joblib")
    assert abs(loaded.predict_proba(["salak"])[0] - proba[0]) < 1e-9
    assert loaded.size_mb() > 0
