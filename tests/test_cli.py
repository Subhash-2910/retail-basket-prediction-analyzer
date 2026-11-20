"""The command line and the optional LLM settings."""
from app import llm
from app.cli import main


def test_info_says_what_to_run_next(tmp_path, capsys):
    main(["--data", str(tmp_path / "data"), "--model", str(tmp_path / "lm.pt"), "--results", str(tmp_path), "info"])
    out = capsys.readouterr().out
    assert "missing: run `python -m app.cli generate`" in out and "missing: run `python -m app.cli train`" in out


def test_llm_settings_default_to_a_local_ollama(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert llm.settings() == ("http://localhost:11434/v1", "", "")
