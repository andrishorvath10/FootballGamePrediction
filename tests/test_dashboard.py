"""Smoke test: the Streamlit dashboard renders every tab on a synthetic database."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker

import football_predictor.db.session as db_session
from football_predictor.config import get_settings
from football_predictor.ingest.loader import LeagueLoader
from football_predictor.ingest.sample_data import generate_sample_league

APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"


@pytest.fixture()
def synthetic_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    url = f"sqlite:///{tmp_path / 'demo.db'}"
    engine = db_session.create_db_engine(url)
    db_session.init_db(engine)
    with sessionmaker(bind=engine)() as s:
        LeagueLoader(s).load(generate_sample_league(seed=5).records)
        s.commit()
    engine.dispose()

    monkeypatch.setenv("FP_DATABASE_URL", url)
    monkeypatch.setenv("FP_ANONYMIZE", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_session_factory", None)
    yield
    get_settings.cache_clear()


def test_dashboard_renders_all_tabs(synthetic_db: None) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.startswith("DEMO League")
    assert len(at.tabs) == 6
    assert at.toggle(key="anon").value is True  # FP_ANONYMIZE=true
    assert at.dataframe, "prediction table missing"

    assert at.toggle(key="uncertainty").value is True  # default model is mle
    assert at.toggle(key="drift").value is True
    # an expert assumption typed into the editor: "Csapat A" +20 % from now, wider drift
    at.session_state["assumptions|DEMO League (synthetic)|DEMO 2025/26 (synthetic)"] = {
        "edited_rows": {}, "deleted_rows": [],
        "added_rows": [{"Csapat": "Csapat A", "Változás (%)": 20, "Ettől a naptól": "2025-10-01",
                        "Bizonytalanság ×": 2.0}],
    }
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Saját feltevések: 1 csapat (2025-10-01-tól)" in c.value for c in at.caption)

    # save -> stored in the database -> checked against the matches since 1 October
    at.button(key="save_assumptions").click().run()
    assert not at.exception, [e.value for e in at.exception]
    from football_predictor.db.assumption_store import list_assumptions
    from football_predictor.db.session import session_scope

    with session_scope() as s:
        (stored,) = list_assumptions(s, "DEMO League (synthetic)", "DEMO 2025/26 (synthetic)")
    assert (stored.change, stored.drift_factor) == (0.2, 2.0)
    assert stored.start == pd.Timestamp("2025-10-01")
    assert any("Mentett feltevések ellenőrzése" in m.value for m in at.markdown)
    at.toggle(key="uncertainty").set_value(False)
    at.toggle(key="drift").set_value(False).run()
    assert not at.exception, [e.value for e in at.exception]

    at.toggle(key="anon").set_value(False)
    for model in ("simple", "dc"):
        at.selectbox(key="model").set_value(model).run()
        assert not at.exception, [e.value for e in at.exception]
