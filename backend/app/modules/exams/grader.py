import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import httpcore
import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from app.core.config import Settings, get_settings
from app.modules.exams.exceptions import ExamPermanentError, ExamTransientError

logger = logging.getLogger("app.modules.exams.grader")

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_NETWORK_ERRORS = (
    TimeoutError,
    ConnectionError,
    OSError,
    httpx.HTTPError,
    httpcore.NetworkError,
    httpcore.ProtocolError,
)
_OBJECTIVE_LETTER_PATTERN = re.compile(
    r"^(?:[\(\[]([A-Za-z])[\)\]][\s\.]*.*|([A-Za-z])[\.\:\)\-][\s]+.*|([A-Za-z])$)"
)


def normalize_objective_answer(answer: str) -> str:
    cleaned = answer.strip()
    if not cleaned:
        return ""
    lower = cleaned.lower()
    if lower in ("true", "t"):
        return "True"
    if lower in ("false", "f"):
        return "False"
    match = _OBJECTIVE_LETTER_PATTERN.match(cleaned)
    if match:
        letter = match.group(1) or match.group(2) or match.group(3)
        if letter:
            return letter.upper()
    return cleaned


def grade_objective_answer(
    *,
    question_type: str,
    student_answer: str,
    correct_answer: str,
) -> tuple[bool, float, str]:
    """
    Deterministically grades Multiple Choice and True/False questions.
    Returns (is_correct, score [0.0 or 1.0], feedback).
    """
    clean_student = normalize_objective_answer(student_answer)
    clean_correct = normalize_objective_answer(correct_answer)

    if not clean_student:
        return False, 0.0, "No answer was provided."

    if clean_student.upper() == clean_correct.upper():
        return True, 1.0, "Correct!"

    return False, 0.0, f"Incorrect. The correct answer is: {clean_correct}"


class SubjectiveGradeResult(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score between 0.0 (completely inaccurate) and 1.0 (complete mastery).",
    )
    is_correct: bool = Field(
        ...,
        description="Whether the response earns passing credit (score >= 0.6).",
    )
    feedback: str = Field(
        ...,
        description="Constructive, pedagogical feedback explaining strengths and missed points.",
    )


class SubjectiveGrader(Protocol):
    def grade_short_answer(
        self,
        *,
        prompt: str,
        correct_answer: str,
        grading_rubric: str | None,
        source_excerpt: str | None,
        student_answer: str,
    ) -> SubjectiveGradeResult: ...


SUBJECTIVE_EVALUATION_INSTRUCTION = (
    "You are an objective, fair academic evaluator assessing a student's short-answer "
    "exam response.\n"
    "1. Base your evaluation strictly on the question's grading rubric and the source "
    "reference excerpt.\n"
    "2. Be fair and award partial credit (0.0 to 1.0 scale):\n"
    "   - 1.0: Covers all required points with clear understanding.\n"
    "   - 0.5 - 0.8: Understands the core concept but missed secondary details.\n"
    "   - 0.1 - 0.4: Vague, tangential, or largely incomplete.\n"
    "   - 0.0: Irrelevant, contradictory, or completely incorrect.\n"
    "3. Provide constructive feedback that explains what was correct and what was missing."
)


class GeminiSubjectiveGrader:
    """
    Evaluates open-ended short-answer responses against ground-truth rubrics using Gemini.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str = "gemini-3.5-flash",
        fallback_models: Sequence[str] | None = None,
        max_retries: int = 3,
        initial_backoff_seconds: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        genai_client: object | None = None,
    ) -> None:
        self._api_key = api_key.strip() if api_key else ""
        self.model_name = model_name
        self.fallback_models = (
            list(fallback_models)
            if fallback_models is not None
            else []
        )
        self.max_retries = max(1, max_retries)
        self.initial_backoff_seconds = max(0.0, initial_backoff_seconds)
        self._sleep_fn = sleep_fn
        self._client = genai_client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ExamPermanentError(
                "GEMINI_API_KEY is not configured",
                code="EXAM_API_KEY_MISSING",
                user_message="AI answer grading is not configured on the server (missing API key).",
            )
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    def grade_short_answer(
        self,
        *,
        prompt: str,
        correct_answer: str,
        grading_rubric: str | None,
        source_excerpt: str | None,
        student_answer: str,
    ) -> SubjectiveGradeResult:
        clean_student = student_answer.strip()
        if not clean_student:
            return SubjectiveGradeResult(
                score=0.0,
                is_correct=False,
                feedback="No answer provided. Expected: " + correct_answer,
            )

        rubric_text = grading_rubric or "Check for accurate understanding of core concept."
        eval_prompt = (
            f"QUESTION PROMPT:\n{prompt}\n\n"
            f"IDEAL ANSWER:\n{correct_answer}\n\n"
            f"GRADING RUBRIC:\n{rubric_text}\n\n"
            f"SOURCE EXCERPT:\n{source_excerpt or 'N/A'}\n\n"
            f"STUDENT ANSWER TO EVALUATE:\n{clean_student}"
        )

        return self._evaluate_with_retry(eval_prompt)

    def _evaluate_with_retry(self, eval_prompt: str) -> SubjectiveGradeResult:
        candidate_models = [self.model_name] + [
            m for m in self.fallback_models if m != self.model_name
        ]
        last_transient: ExamTransientError | None = None

        for model in candidate_models:
            attempt = 0
            while attempt < self.max_retries:
                attempt += 1
                try:
                    return self._call_evaluator(eval_prompt, model=model)
                except ExamPermanentError:
                    raise
                except ExamTransientError as exc:
                    last_transient = exc
                    if attempt >= self.max_retries:
                        break
                    delay = self.initial_backoff_seconds * (2 ** (attempt - 1))
                    self._sleep_fn(delay)

        assert last_transient is not None
        raise last_transient

    def _call_evaluator(
        self,
        prompt: str,
        *,
        model: str | None = None,
    ) -> SubjectiveGradeResult:
        client = self._get_client()
        active_model = model or self.model_name
        config = types.GenerateContentConfig(
            system_instruction=SUBJECTIVE_EVALUATION_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=SubjectiveGradeResult,
            temperature=0.1,
        )

        try:
            response = client.models.generate_content(
                model=active_model,
                contents=prompt,
                config=config,
            )
        except (ExamPermanentError, ExamTransientError):
            raise
        except genai_errors.APIError as exc:
            status_code = getattr(exc, "code", None)
            if status_code == 429:
                raise ExamTransientError(
                    "Gemini rate limit exceeded during subjective evaluation",
                    code="GRADING_RATE_LIMITED",
                ) from exc
            if isinstance(status_code, int) and status_code in _RETRYABLE_STATUS_CODES:
                raise ExamTransientError(
                    f"Gemini API transient HTTP {status_code} error during evaluation",
                    code="GRADING_PROVIDER_UNAVAILABLE",
                ) from exc
            raise ExamPermanentError(
                f"Gemini rejected evaluation request: {status_code}",
                code="GRADING_PROVIDER_REJECTED",
            ) from exc
        except _NETWORK_ERRORS as exc:
            raise ExamTransientError(
                f"Network error during answer evaluation: {type(exc).__name__}",
                code="GRADING_NETWORK_ERROR",
            ) from exc
        except Exception as exc:
            raise ExamPermanentError(
                f"Unexpected error during answer evaluation: {type(exc).__name__}",
                code="GRADING_UNEXPECTED_ERROR",
            ) from exc

        parsed_obj = getattr(response, "parsed", None)
        if isinstance(parsed_obj, SubjectiveGradeResult):
            return parsed_obj

        raw_text = getattr(response, "text", None)
        try:
            if isinstance(parsed_obj, dict):
                return SubjectiveGradeResult.model_validate(parsed_obj)
            loaded = json.loads(raw_text or "{}")
            return SubjectiveGradeResult.model_validate(loaded)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ExamPermanentError(
                f"Subjective evaluation response failed schema validation: {type(exc).__name__}",
                code="INVALID_GRADING_OUTPUT",
            ) from exc


def create_subjective_grader(settings: Settings | None = None) -> SubjectiveGrader:
    cfg = settings or get_settings()
    return GeminiSubjectiveGrader(
        api_key=cfg.GEMINI_API_KEY,
        model_name=cfg.GEMINI_GENERATION_MODEL,
        fallback_models=["gemini-3.5-flash-lite", "gemini-flash-lite-latest"],
    )
