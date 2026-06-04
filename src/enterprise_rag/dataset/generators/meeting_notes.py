"""Meeting Notes document generator."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .._assembler import DocumentAssembler
from .._facts import (
    ACTION_STATUS_OPTIONS,
    DEPARTMENTS,
    MEETING_BLOCKER_POOL,
    MEETING_DECISION_POOL,
    MEETING_SEQUENCES,
    PROJECT_BASE_DATES,
    PROJECT_CODES,
    PROJECT_DESCRIPTIONS,
    ROLE_NAMES,
)
from .._rng import SeededRNG
from ..enums import (
    AuthorityLevel,
    ConfidentialityLevel,
    DocumentCategory,
    DocumentStatus,
    MeetingType,
    Region,
    ReviewCycle,
)
from ..schemas import DocumentMetadata, EnterpriseDocument, MeetingNotesMetadata

_MEETING_TYPE_MAP: dict[str, MeetingType] = {
    "sprint_planning": MeetingType.SPRINT_PLANNING,
    "architecture_review": MeetingType.ARCHITECTURE_REVIEW,
    "postmortem": MeetingType.POSTMORTEM,
    "launch_review": MeetingType.LAUNCH_REVIEW,
}

_AGENDA_BY_TYPE: dict[str, list[str]] = {
    "sprint_planning": [
        "Review velocity and capacity",
        "Prioritise backlog items for the sprint",
        "Assign stories and confirm acceptance criteria",
        "Identify blockers and dependencies",
        "Confirm sprint goal",
    ],
    "architecture_review": [
        "Present proposed architecture changes",
        "Review ADRs and design decisions",
        "Security and compliance review",
        "Performance and scalability discussion",
        "Approval or feedback collection",
    ],
    "postmortem": [
        "Timeline of the incident",
        "Root cause analysis",
        "Impact assessment",
        "Action items to prevent recurrence",
        "Communication review",
    ],
    "launch_review": [
        "Go/No-Go criteria review",
        "Monitoring and rollback plan",
        "Customer communication plan",
        "On-call rotation confirmation",
        "Launch date and time confirmation",
    ],
}

_DISCUSSION_NOTES_BY_TYPE: list[dict] = [
    {"topic": "Technical Discussion", "content": "The team reviewed the implementation approach and identified potential edge cases that need additional testing coverage."},
    {"topic": "Risk Assessment", "content": "Key risks were catalogued and mitigation strategies were discussed. Highest-priority risk assigned to the Security team for review."},
    {"topic": "Dependency Status", "content": "Cross-team dependencies were reviewed. Most are on track; one dependency on the HYDataLake team is delayed by approximately one week."},
]


class MeetingNotesGenerator:
    """Generates Meeting Notes — 5 projects × 6 meetings = 30 docs."""

    def __init__(self, rng: SeededRNG, assembler: DocumentAssembler) -> None:
        self._rng = rng
        self._assembler = assembler

    def generate_one(self, index: int, all_doc_ids: list[str]) -> EnterpriseDocument:
        project_idx = (index - 1) // len(MEETING_SEQUENCES)
        meeting_idx = (index - 1) % len(MEETING_SEQUENCES)

        project_code = PROJECT_CODES[project_idx % len(PROJECT_CODES)]
        meeting_seq = MEETING_SEQUENCES[meeting_idx]
        meeting_type_str = meeting_seq["type"]
        offset_weeks = meeting_seq["offset_weeks"]

        base_date = date.fromisoformat(PROJECT_BASE_DATES[project_code])
        meeting_date = base_date + timedelta(weeks=offset_weeks)

        doc_id = f"MTG-{index:03d}"
        meeting_type = _MEETING_TYPE_MAP[meeting_type_str]

        created_dt = datetime(meeting_date.year, meeting_date.month, meeting_date.day, 14, 0, 0)
        updated_dt = created_dt + timedelta(hours=2)

        version = "1.0"
        dept_list = DEPARTMENTS["Project Meeting Notes"]
        dept = self._rng.choice(dept_list)

        participants = self._rng.sample(ROLE_NAMES, self._rng.randint(4, 7))
        facilitator = participants[0]

        decisions_pool = MEETING_DECISION_POOL.get(meeting_type_str, [])
        num_decisions = self._rng.randint(2, min(4, len(decisions_pool)))
        chosen_decisions = self._rng.sample(decisions_pool, num_decisions)
        decisions = [
            {
                "topic": f"Decision {i + 1}",
                "text": text,
                "owner": self._rng.choice(participants),
            }
            for i, text in enumerate(chosen_decisions)
        ]

        num_action_items = self._rng.randint(3, 6)
        action_items = []
        for i in range(num_action_items):
            due_date = meeting_date + timedelta(days=self._rng.randint(5, 21))
            status = self._rng.choice(ACTION_STATUS_OPTIONS)
            action_items.append({
                "owner_role": self._rng.choice(participants),
                "action": f"Complete {meeting_type_str.replace('_', ' ')} item {i + 1} for {project_code}",
                "due_date": str(due_date),
                "status": status,
            })

        has_blockers = meeting_type_str in ("sprint_planning", "postmortem") and self._rng.random() < 0.6
        blockers = []
        if has_blockers:
            blocker_text = self._rng.choice(MEETING_BLOCKER_POOL)
            blockers.append({
                "area": "Engineering",
                "description": blocker_text,
                "owner": self._rng.choice(participants),
                "target_date": str(meeting_date + timedelta(days=14)),
            })

        related_documents: list[str] = []
        if all_doc_ids:
            # Always reference docs when available; vary the count for diversity
            sample_count = self._rng.choice([2, 2, 3, 3, 2, 3])
            sample_count = min(sample_count, len(all_doc_ids))
            related_documents = self._rng.sample(all_doc_ids, sample_count)

        participant_rows = [{"role": p, "status": "Present"} for p in participants]

        next_meeting_date_obj = meeting_date + timedelta(weeks=4)
        agenda_items = _AGENDA_BY_TYPE.get(meeting_type_str, ["General discussion"])

        title = (
            f"{meeting_type_str.replace('_', ' ').title()} – "
            f"{project_code} – {str(meeting_date)}"
        )

        context = {
            "title": title,
            "meeting_id": doc_id,
            "project_code": project_code,
            "meeting_type": meeting_type_str.replace("_", " ").title(),
            "meeting_date": str(meeting_date),
            "participants": participants,
            "department": dept,
            "facilitator": facilitator,
            "related_documents": related_documents,
            "project_description": PROJECT_DESCRIPTIONS[project_code],
            "duration_minutes": self._rng.choice([60, 90, 120]),
            "location": self._rng.choice(["Virtual – HYConf", "Seoul HQ – Room A4", "Singapore Office – Room 2B"]),
            "agenda_items": agenda_items,
            "participant_rows": participant_rows,
            "decisions": decisions,
            "action_items": action_items,
            "discussion_notes": self._rng.sample(_DISCUSSION_NOTES_BY_TYPE, min(2, len(_DISCUSSION_NOTES_BY_TYPE))),
            "blockers": blockers,
            "next_meeting_date": str(next_meeting_date_obj),
            "next_meeting_agenda": f"Follow-up on {meeting_type_str.replace('_', ' ')} action items",
            "next_meeting_scheduler": facilitator,
        }

        body = self._assembler.render("meeting_notes.j2", context)

        category_metadata = MeetingNotesMetadata(
            meeting_id=doc_id,
            project_code=project_code,
            meeting_type=meeting_type,
            meeting_date=meeting_date,
            participants=participants,
            decisions_count=len(decisions),
            action_items_count=len(action_items),
            related_documents=related_documents,
        )

        metadata = DocumentMetadata(
            document_id=doc_id,
            category=DocumentCategory.MEETING_NOTES,
            title=title,
            version=version,
            created_at=created_dt,
            updated_at=updated_dt,
            department=dept,
            document_owner=facilitator,
            confidentiality=ConfidentialityLevel.CONFIDENTIAL,
            region=Region.GLOBAL,
            language="en",
            effective_date=meeting_date,
            review_cycle=ReviewCycle.MONTHLY,
            status=DocumentStatus.ACTIVE,
            authority_level=AuthorityLevel.ADVISORY,
            tags=[project_code.lower(), meeting_type_str, "meeting"],
            source_path=f"/data/meetings/{doc_id}.md",
            category_metadata=category_metadata,
        )

        return self._assembler.build_document(metadata, body, references=related_documents)
