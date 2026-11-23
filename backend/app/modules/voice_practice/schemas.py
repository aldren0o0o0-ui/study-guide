import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Canonical Gemini Live input rate is 16000 Hz. Hardware fallback rates (24000, 44100, 48000)
# are deterministically resampled to 16000 Hz 16-bit mono PCM before forwarding to Gemini.
SUPPORTED_INPUT_SAMPLE_RATES = (16000, 24000, 44100, 48000)
GEMINI_CANONICAL_INPUT_SAMPLE_RATE_HZ: Literal[16000] = 16000
DEFAULT_INPUT_SAMPLE_RATE_HZ: Literal[16000] = 16000
GEMINI_OUTPUT_SAMPLE_RATE_HZ: Literal[24000] = 24000
PCM_ENCODING_NAME: Literal["pcm_s16le"] = "pcm_s16le"
PCM_CHANNELS: Literal[1] = 1
PCM_BYTES_PER_SAMPLE: Literal[2] = 2


class VoicePracticeMode(StrEnum):
    INTERVIEW = "interview"
    COMMUNICATION = "communication"
    DOCUMENT_ORAL = "document_oral"
    NEUTRAL = "neutral"


class VoiceGroundingStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    GROUNDED = "grounded"
    EMPTY_CONTEXT = "empty_context"
    DEGRADED_RETRIEVAL_ERROR = "degraded_retrieval_error"


class InterviewType(StrEnum):
    JOB = "job"
    INTERNSHIP = "internship"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"


class ExperienceLevel(StrEnum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"


class PracticeDifficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CommunicationScenario(StrEnum):
    GENERAL_CONVERSATION = "general_conversation"
    IMPROMPTU_SPEAKING = "impromptu_speaking"
    SITUATIONAL_COMMUNICATION = "situational_communication"
    EXPLANATION_PRESENTATION = "explanation_presentation"


class InterviewPracticeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interview_type: InterviewType = Field(
        default=InterviewType.BEHAVIORAL,
        description="Category of interview practice.",
    )
    target_role: str = Field(
        default="Software Engineer",
        min_length=2,
        max_length=120,
        description="Target job title or position.",
    )
    experience_level: ExperienceLevel = Field(
        default=ExperienceLevel.JUNIOR,
        description="Candidate experience level.",
    )
    difficulty: PracticeDifficulty = Field(
        default=PracticeDifficulty.MEDIUM,
        description="Difficulty level of interview questions.",
    )
    focus_area: str | None = Field(
        default=None,
        max_length=200,
        description="Specific competency or technical area to focus on.",
    )
    user_context: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional background or job description snippet provided by the user.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            exp = data.get("experience_level")
            if exp == "entry_level":
                data["experience_level"] = "junior"
            elif exp == "student":
                data["experience_level"] = "intern"
            itype = data.get("interview_type")
            if itype == "general":
                data["interview_type"] = "job"
        return data


class CommunicationPracticeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: CommunicationScenario = Field(
        default=CommunicationScenario.GENERAL_CONVERSATION,
        description="Communication or speaking practice scenario.",
    )
    speaking_topic: str | None = Field(
        default=None,
        max_length=200,
        description="Topic or situation to discuss or explain.",
    )
    difficulty: PracticeDifficulty = Field(
        default=PracticeDifficulty.MEDIUM,
        description="Complexity level of prompts and follow-ups.",
    )
    target_skill: str | None = Field(
        default=None,
        max_length=120,
        description="Target communication skill (e.g., structured explanation, conciseness).",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            sc = data.get("scenario")
            if sc in ("presentation_practice", "professional_explanation"):
                data["scenario"] = "explanation_presentation"
            elif sc == "academic_discussion":
                data["scenario"] = "situational_communication"
        return data


class DocumentOralPracticeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_ids: list[uuid.UUID] = Field(
        default_factory=list,
        description="Selected user-owned document UUIDs for grounded oral examination.",
    )
    focus_topic: str | None = Field(
        default=None,
        max_length=300,
        description="Optional topic or chapter focus within the selected documents.",
    )
    difficulty: PracticeDifficulty = Field(
        default=PracticeDifficulty.MEDIUM,
        description="Difficulty of oral examination questions.",
    )
    question_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Target number of oral examination questions for the session.",
    )


class ClientEventType(StrEnum):
    SESSION_START = "session_start"
    SESSION_STARTED = "session_started"
    AUDIO_INPUT_STARTED = "audio_input_started"
    AUDIO_INPUT_STOPPED = "audio_input_stopped"
    INTERRUPT = "interrupt"
    PING = "ping"
    SESSION_END = "session_end"


class ServerEventType(StrEnum):
    SESSION_READY = "session_ready"
    SESSION_STARTED = "session_started"
    AUDIO_INPUT_STARTED = "audio_input_started"
    AUDIO_INPUT_STOPPED = "audio_input_stopped"
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    AUDIO_OUTPUT_STARTED = "audio_output_started"
    AUDIO_OUTPUT_FINISHED = "audio_output_finished"
    INTERRUPTED = "interrupted"
    TURN_COMPLETED = "turn_completed"
    SESSION_WARNING = "session_warning"
    SESSION_ENDED = "session_ended"
    SESSION_ERROR = "session_error"
    PONG = "pong"


class VoiceAudioFormatSpec(BaseModel):
    encoding: Literal["pcm_s16le"] = PCM_ENCODING_NAME
    channels: Literal[1] = PCM_CHANNELS
    sample_width_bytes: Literal[2] = PCM_BYTES_PER_SAMPLE
    input_sample_rate_hz: int = Field(default=DEFAULT_INPUT_SAMPLE_RATE_HZ)
    gemini_input_sample_rate_hz: int = Field(default=GEMINI_CANONICAL_INPUT_SAMPLE_RATE_HZ)
    output_sample_rate_hz: int = Field(default=GEMINI_OUTPUT_SAMPLE_RATE_HZ)


class VoiceSessionLimitsSpec(BaseModel):
    max_duration_seconds: int
    warning_at_seconds: int
    max_concurrent_per_user: int
    daily_quota_limit: int
    daily_quota_remaining: int
    max_audio_frame_bytes: int


class VoiceSessionTicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: VoicePracticeMode = Field(
        default=VoicePracticeMode.NEUTRAL,
        description="Voice practice mode: interview, communication, document_oral, or neutral.",
    )
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Optional custom title for the practice session.",
    )
    input_sample_rate_hz: Literal[16000, 24000, 44100, 48000] = Field(
        default=DEFAULT_INPUT_SAMPLE_RATE_HZ,
        description="Client microphone PCM input sample rate in Hz (16kHz canonical).",
    )
    interview_config: InterviewPracticeConfig | None = Field(
        default=None,
        description="Configuration for Interview Practice mode.",
    )
    communication_config: CommunicationPracticeConfig | None = Field(
        default=None,
        description="Configuration for Communication & Speaking Practice mode.",
    )
    document_oral_config: DocumentOralPracticeConfig | None = Field(
        default=None,
        description="Configuration for Document-Based Oral Practice mode.",
    )
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Convenience list of document UUIDs for Document-Based Oral Practice.",
    )

    @model_validator(mode="after")
    def validate_mode_configuration(self) -> "VoiceSessionTicketRequest":
        if self.mode == VoicePracticeMode.INTERVIEW:
            if (
                self.communication_config is not None
                or self.document_oral_config is not None
                or self.document_ids
            ):
                raise ValueError("Only `interview_config` may be provided when mode is 'interview'")
            if self.interview_config is None:
                self.interview_config = InterviewPracticeConfig()
        elif self.mode in (VoicePracticeMode.COMMUNICATION, VoicePracticeMode.NEUTRAL):
            if (
                self.interview_config is not None
                or self.document_oral_config is not None
                or self.document_ids
            ):
                raise ValueError(
                    "Only `communication_config` may be provided when mode is 'communication'"
                )
            if self.communication_config is None:
                self.communication_config = CommunicationPracticeConfig()
        elif self.mode == VoicePracticeMode.DOCUMENT_ORAL:
            if self.interview_config is not None or self.communication_config is not None:
                raise ValueError(
                    "Only `document_oral_config` or `document_ids` may be provided "
                    "when mode is 'document_oral'"
                )
            if self.document_oral_config is None:
                self.document_oral_config = DocumentOralPracticeConfig(
                    document_ids=list(self.document_ids or [])
                )
            elif self.document_ids and not self.document_oral_config.document_ids:
                self.document_oral_config.document_ids = list(self.document_ids)

            deduped_docs = list(dict.fromkeys(self.document_oral_config.document_ids))
            if not deduped_docs:
                raise ValueError(
                    "At least one document_id is required for document_oral practice mode."
                )
            if len(deduped_docs) > 10:
                raise ValueError(
                    "A maximum of 10 documents can be selected for a voice practice session."
                )
            self.document_oral_config.document_ids = deduped_docs
            self.document_ids = deduped_docs
        return self


class VoiceSessionTicketResponse(BaseModel):
    ticket: str = Field(..., description="Short-lived, single-use WebSocket connection ticket.")
    session_id: uuid.UUID = Field(..., description="Unique identifier for the voice session.")
    websocket_path: str = Field(
        ...,
        description="Relative WebSocket endpoint path to connect with the issued ticket.",
    )
    expires_in_seconds: int = Field(..., description="Ticket validity window in seconds.")
    expires_at: datetime = Field(..., description="UTC timestamp when the ticket expires.")
    mode: VoicePracticeMode = Field(default=VoicePracticeMode.NEUTRAL)
    title: str = Field(default="Voice Practice Session")
    grounding_status: VoiceGroundingStatus = Field(
        default=VoiceGroundingStatus.NOT_APPLICABLE,
        description="Explicit RAG grounding status for document_oral sessions.",
    )
    audio_format: VoiceAudioFormatSpec
    limits: VoiceSessionLimitsSpec


class StoredSessionTicket(BaseModel):
    session_id: uuid.UUID
    user_id: uuid.UUID
    mode: VoicePracticeMode = VoicePracticeMode.NEUTRAL
    input_sample_rate_hz: int = DEFAULT_INPUT_SAMPLE_RATE_HZ
    system_instruction: str | None = None
    issued_at: datetime
    expires_at: datetime


class ActiveVoiceSession(BaseModel):
    session_id: uuid.UUID
    user_id: uuid.UUID
    mode: VoicePracticeMode = VoicePracticeMode.NEUTRAL
    input_sample_rate_hz: int = DEFAULT_INPUT_SAMPLE_RATE_HZ
    system_instruction: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ClientControlMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: ClientEventType
    payload: dict[str, Any] = Field(default_factory=dict)


class SessionReadyPayload(BaseModel):
    mode: VoicePracticeMode
    model: str
    audio_format: VoiceAudioFormatSpec
    max_duration_seconds: int
    warning_at_seconds: int
    max_audio_frame_bytes: int


class SessionStartedPayload(BaseModel):
    mode: VoicePracticeMode
    input_sample_rate_hz: int
    output_sample_rate_hz: int


class AudioInputStartedPayload(BaseModel):
    sample_rate_hz: int


class AudioInputStoppedPayload(BaseModel):
    turn_index: int


class TranscriptEventPayload(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    turn_index: int
    is_final: bool


class AudioOutputStartedPayload(BaseModel):
    turn_index: int
    sample_rate_hz: int = GEMINI_OUTPUT_SAMPLE_RATE_HZ
    encoding: Literal["pcm_s16le"] = PCM_ENCODING_NAME
    channels: Literal[1] = PCM_CHANNELS


class AudioOutputFinishedPayload(BaseModel):
    turn_index: int
    reason: Literal["completed", "interrupted"]


class InterruptedPayload(BaseModel):
    turn_index: int
    source: Literal["user_speech", "client_signal"]


class TurnCompletedPayload(BaseModel):
    turn_index: int
    interrupted: bool = False


class SessionWarningPayload(BaseModel):
    remaining_seconds: int
    max_duration_seconds: int
    message: str


class SessionEndedPayload(BaseModel):
    reason: Literal[
        "client_requested",
        "client_disconnected",
        "max_duration_reached",
        "provider_closed",
        "error",
    ]
    duration_seconds: float


class SessionErrorPayload(BaseModel):
    code: str
    message: str
    fatal: bool = False


class ServerEventEnvelope(BaseModel):
    event: ServerEventType
    session_id: uuid.UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# PERSISTENCE, EVALUATION & REST API SCHEMAS (PHASE 4)
# ============================================================================


class VoicePracticeDocumentRefRead(BaseModel):
    document_id: uuid.UUID
    original_filename: str
    file_extension: str
    processing_status: str


class VoicePracticeSourceReferenceRead(BaseModel):
    source_key: str
    document_id: uuid.UUID
    original_filename: str
    chunk_id: uuid.UUID
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    excerpt: str


class VoicePracticeTurnRead(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    turn_index: int
    speaker_role: Literal["user", "assistant"]
    transcript_text: str
    turn_status: Literal["completed", "interrupted"]
    started_at: datetime
    completed_at: datetime
    metadata: dict[str, Any] | None = None


class VoiceDimensionScore(BaseModel):
    dimension: str = Field(..., min_length=1, max_length=100)
    score: float = Field(..., ge=0.0, le=100.0)
    rationale: str = Field(..., min_length=1, max_length=1000)


class VoiceTranscriptEvidence(BaseModel):
    turn_index: int | None = Field(default=None, ge=1)
    quote: str = Field(..., min_length=1, max_length=600)
    observation: str = Field(..., min_length=1, max_length=600)


class VoicePracticeFeedbackRead(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    evaluation_status: str
    overall_score: float | None = Field(default=None, ge=0.0, le=100.0)
    summary: str | None = None
    dimension_scores: list[VoiceDimensionScore] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    areas_for_improvement: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    transcript_evidence: list[VoiceTranscriptEvidence] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    unsupported_statements: list[str] = Field(default_factory=list)
    source_references: list[VoicePracticeSourceReferenceRead] = Field(default_factory=list)
    model_name: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class VoicePracticeSessionSummaryRead(BaseModel):
    id: uuid.UUID
    mode: str
    title: str
    status: str
    evaluation_status: str
    grounding_status: VoiceGroundingStatus = VoiceGroundingStatus.NOT_APPLICABLE
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: float
    turns_count: int
    documents_count: int
    overall_score: float | None = None
    created_at: datetime
    updated_at: datetime


class VoicePracticeSessionDetailRead(BaseModel):
    id: uuid.UUID
    mode: str
    title: str
    status: str
    evaluation_status: str
    grounding_status: VoiceGroundingStatus = VoiceGroundingStatus.NOT_APPLICABLE
    configuration: dict[str, Any]
    termination_reason: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: float
    source_documents: list[VoicePracticeDocumentRefRead] = Field(default_factory=list)
    turns: list[VoicePracticeTurnRead] = Field(default_factory=list)
    feedback: VoicePracticeFeedbackRead | None = None
    created_at: datetime
    updated_at: datetime


class VoicePracticeSessionListResponse(BaseModel):
    items: list[VoicePracticeSessionSummaryRead]
    total: int
    limit: int
    offset: int


class VoicePracticeSessionDeleteResponse(BaseModel):
    deleted: bool
    session_id: uuid.UUID
