"""Phase 5 — Label clusters using Ollama LLM with structured JSON output."""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import ollama

from src.models import Bookmark, ClusterResult, LabelResult

logger = logging.getLogger(__name__)

_TOPIC_PROMPT = """\
You are a professional librarian organizing a digital knowledge base.

The following {n} bookmarks belong to the same semantic topic cluster:

{bookmark_list}

Your task: Generate a concise, professional category name (1-3 words, no punctuation) \
for this cluster. The name must be broad enough to cover all items but specific enough \
to be meaningful. Avoid "Misc", "Stuff", "Links", "Resources", or brand names unless \
the cluster is entirely about that brand.

Respond ONLY with valid JSON (no markdown, no explanation):
{{"category_name": "...", "confidence": 0.0}}"""

_SUBDOMAIN_PROMPT = """\
You are a professional taxonomist.

The following topic categories belong to the same broader subdomain:

{topic_list}

Generate a 1-3 word subdomain name that accurately encompasses all of these topics.
Avoid "Misc", "Stuff", "General".

Respond ONLY with valid JSON:
{{"category_name": "...", "confidence": 0.0}}"""

_DOMAIN_PROMPT = """\
You are a professional taxonomist.

The following subdomain categories belong to the same high-level domain:

{subdomain_list}

Generate a 1-3 word domain name that accurately groups all of these subdomains.

Respond ONLY with valid JSON:
{{"category_name": "...", "confidence": 0.0}}"""


def _parse_llm_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response, handling common formatting issues."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object anywhere in the string
        match = re.search(r'\{[^{}]+\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse LLM JSON response: %r", text[:200])
    return None


class LLMLabeler:
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self._client = ollama.Client(host=base_url)

    def check_connection(self) -> None:
        try:
            models = self._client.list()
            available = [m.model.split(":")[0] for m in models.models]
        except Exception as exc:
            raise RuntimeError(f"Cannot reach Ollama: {exc}") from exc

        model_base = self.model.split(":")[0]
        if model_base not in available:
            raise RuntimeError(
                f"LLM model '{self.model}' not available. Run: ollama pull {self.model}"
            )

    def _call(self, prompt: str) -> Optional[LabelResult]:
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.1},
            )
            content = response["message"]["content"]
            data = _parse_llm_json(content)
            if data and "category_name" in data and "confidence" in data:
                name = str(data["category_name"]).strip()
                confidence = float(data.get("confidence", 0.5))
                # Enforce 1-3 word limit
                words = name.split()
                name = " ".join(words[:3]) if words else "Unknown"
                return LabelResult(category_name=name, confidence=confidence)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
        return None

    def label_topic(
        self,
        cluster: ClusterResult,
        bookmark_map: dict[str, Bookmark],
    ) -> LabelResult:
        """Label a Level-4 topic cluster from its representative bookmarks."""
        reps = [bookmark_map[rid] for rid in cluster.representative_ids if rid in bookmark_map]
        if not reps:
            return LabelResult(category_name="Unknown", confidence=0.0)

        lines = "\n".join(
            f"- {b.title} [{b.domain}]" for b in reps
        )
        prompt = _TOPIC_PROMPT.format(n=len(reps), bookmark_list=lines)
        result = self._call(prompt)
        if result is None:
            return LabelResult(category_name="Unknown", confidence=0.0)
        logger.debug("Topic label: %r (conf=%.2f)", result.category_name, result.confidence)
        return result

    def label_subdomain(self, topic_names: list[str]) -> LabelResult:
        """Label a Level-3 subdomain from a list of topic names."""
        lines = "\n".join(f"- {name}" for name in topic_names)
        prompt = _SUBDOMAIN_PROMPT.format(topic_list=lines)
        result = self._call(prompt)
        if result is None:
            return LabelResult(category_name="Unknown", confidence=0.0)
        logger.debug("Subdomain label: %r (conf=%.2f)", result.category_name, result.confidence)
        return result

    def label_domain(self, subdomain_names: list[str]) -> LabelResult:
        """Label a Level-2 domain from a list of subdomain names."""
        lines = "\n".join(f"- {name}" for name in subdomain_names)
        prompt = _DOMAIN_PROMPT.format(subdomain_list=lines)
        result = self._call(prompt)
        if result is None:
            return LabelResult(category_name="Unknown", confidence=0.0)
        logger.debug("Domain label: %r (conf=%.2f)", result.category_name, result.confidence)
        return result
