"""One question, end to end: guards, retrieval, facts, prompt, LLM, output guard, speech."""

from __future__ import annotations

import logging
import time

from voice_rag import calc, guards, prompt
from voice_rag.config import Settings
from voice_rag.embedder import Embedder
from voice_rag.index import Index, RefreshReport, load_or_build
from voice_rag.llm import LLM
from voice_rag.products import (
    Product,
    derive_products,
    mentioned_products,
    product_vocabulary,
    resolve_pronoun,
)
from voice_rag.retrieve import retrieve
from voice_rag.tts import Speaker
from voice_rag.types import Answer, Timings

log = logging.getLogger(__name__)

NOT_AVAILABLE_REPLY = "I'm sorry, that information is not available in the provided documents."


class Pipeline:
    """Holds the loaded index and model clients; ``answer`` runs one question through them."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        llm: LLM,
        speaker: Speaker | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.llm = llm
        self.speaker = speaker
        self.index: Index
        self.products: list[Product]
        self.product_vocab: frozenset[str]
        # Conversation memory: the single product the previous question named, if any.
        self.last_product: Product | None = None
        self.last_refresh: RefreshReport
        self.reload(force=False)

    def reload(self, force: bool = False) -> RefreshReport:
        """Re-check KB hashes and rebuild the index if anything changed; returns what happened."""
        self.index, report = load_or_build(
            self.settings.kb_dir, self.settings.index_dir, self.embedder, force=force
        )
        self.products = derive_products(self.index.chunks)
        self.product_vocab = product_vocabulary(self.index.chunks, self.products)
        self.last_product = None  # products may have changed
        self.last_refresh = report
        return report

    def answer(self, question: str, speak: bool = True) -> Answer:
        """Run the full pipeline for one question."""
        started = time.perf_counter()
        original = question
        timings = Timings()
        guard_messages: list[str] = []
        s = self.settings

        injection = guards.detect_injection(question)
        leak = guards.LEAK_LABEL in injection
        for label in injection:
            guard_messages.append(f"injection pattern: {label}; documents take precedence")

        # Follow-up handling: "How much does it cost?" after a question about the lamp.
        if self.last_product is not None and guards.needs_clarification(
            question, self.products, self.product_vocab
        ):
            question = resolve_pronoun(question, self.last_product)
            guard_messages.append(
                f"no product named; using {self.last_product.label} from the previous question: "
                f"{question!r}"
            )
        mentioned = mentioned_products(question, self.products)
        self.last_product = mentioned[0] if len(mentioned) == 1 else None
        anchors = [p.code for p in mentioned]
        result = retrieve(
            self.index, self.embedder, question, s.top_k, s.min_similarity, anchor_terms=anchors
        )
        timings.retrieval = result.seconds
        if result.low_confidence:
            guard_messages.append(
                f"low retrieval confidence ({result.best_cosine:.3f}), "
                f"using all {result.total_chunks} passages"
            )

        if guards.needs_clarification(question, self.products, self.product_vocab):
            guard_messages.append("question needs a product and names none: asking (no LLM call)")
            text = guards.clarification_reply(self.products)
            answer = Answer(original, text, result.hits, guard_messages, timings, llm_used=False)
            return self._finish(answer, speak, started)

        facts = calc.parse_facts(result.hits, self.products)
        computed = calc.compute_total(question, self.products, facts)
        if computed is not None:
            guard_messages.append(f"total computed in code: {computed.line}")
        facts_block = facts.block() if calc.is_numeric_question(question) else ""

        supporting = [hit.chunk.text for hit in result.hits]
        if computed is not None:
            supporting.append(computed.line)

        user = prompt.build_user_message(
            question,
            result.hits,
            facts_block=facts_block,
            computed_line=computed.line if computed else None,
            injection_flagged=bool(injection),
            leak_flagged=leak,
        )
        response = self.llm.generate(prompt.SYSTEM_PROMPT, user)
        timings.llm = response.seconds
        timings.ollama = response.metrics
        text = response.text

        check = guards.check_output(text, supporting, question)
        contradiction = guards.contradicts_itself(text)
        if not check.ok or contradiction:
            guard_messages.extend(f"output guard: {m}; regenerating once" for m in check.messages)
            if contradiction:
                guard_messages.append(f"output guard: {contradiction}; regenerating once")
            figures = guards.unsupported_figures(text, supporting, question)
            retry_user = prompt.build_user_message(
                question,
                result.hits,
                facts_block=facts_block,
                computed_line=computed.line if computed else None,
                injection_flagged=bool(injection),
                leak_flagged=leak,
                retry_figures=figures,
                retry_contradiction=contradiction,
            )
            retry = self.llm.generate(prompt.SYSTEM_PROMPT, retry_user)
            timings.llm += retry.seconds
            check = guards.check_output(retry.text, supporting, question)
            if check.ok:
                text = retry.text
                if guards.contradicts_itself(text):
                    text = guards.drop_contradictory_opening(text)
                    guard_messages.append(
                        "output guard: retry still contradicts itself; dropped its opening sentence"
                    )
            else:
                guard_messages.append("output guard: retry still unsupported; using safe reply")
                text = NOT_AVAILABLE_REPLY

        answer = Answer(original, text, result.hits, guard_messages, timings, llm_used=True)
        return self._finish(answer, speak, started)

    def _finish(self, answer: Answer, speak: bool, started: float) -> Answer:
        """Synthesize speech (not play it) when requested, then stamp the total time."""
        if speak and self.speaker is not None:
            speech = self.speaker.synthesize(answer.text, audio_name(answer.question))
            answer.timings.tts = speech.synthesis_seconds
            answer.audio_path = str(speech.path) if speech.path else None
            answer.spoken_text = speech.spoken_text
        answer.timings.total = time.perf_counter() - started
        return answer


def audio_name(question: str, limit: int = 40) -> str:
    """A file-system-safe stem derived from the question, e.g. ``does-the-l1-lamp-have-a-batt``."""
    safe = "".join(c.lower() if c.isalnum() else "-" for c in question).strip("-")
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe[:limit].rstrip("-") or "answer"
