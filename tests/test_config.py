from pathlib import Path

from voice_rag import __version__
from voice_rag.config import Settings, load_settings
from voice_rag.types import Chunk, Retrieved


def test_package_imports() -> None:
    assert __version__


def test_defaults_are_sane() -> None:
    s = load_settings(env={})
    assert s == Settings()
    assert s.top_k >= 2  # multi-document questions need more than one passage
    assert s.temperature == 0.0


def test_env_overrides_are_typed() -> None:
    s = load_settings(
        env={
            "VOICE_RAG_LLM_MODEL": "qwen3:1.7b",
            "VOICE_RAG_TOP_K": "2",
            "VOICE_RAG_TEMPERATURE": "0.5",
            "VOICE_RAG_NUM_THREAD": "4",
            "VOICE_RAG_KB_DIR": "/tmp/kb",
            "VOICE_RAG_KEEP_ALIVE": "",  # empty means "keep default"
        }
    )
    assert s.llm_model == "qwen3:1.7b"
    assert s.top_k == 2
    assert s.temperature == 0.5
    assert s.num_thread == 4
    assert s.kb_dir == Path("/tmp/kb")
    assert s.keep_alive == Settings().keep_alive


def test_think_option_depends_on_model() -> None:
    assert Settings(llm_model="qwen3:1.7b").think_option() is False
    assert Settings(llm_model="qwen3:4b-instruct").think_option() is None


def test_types_are_hashable_values() -> None:
    chunk = Chunk(text="x", source="a.txt", index=0)
    assert Retrieved(chunk=chunk, score=0.5).chunk == chunk
    assert {chunk}  # frozen dataclass is hashable
