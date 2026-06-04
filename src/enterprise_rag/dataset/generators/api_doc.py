"""API Documentation document generator."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .._assembler import DocumentAssembler
from .._facts import API_SERVICES, API_VERSIONS, DEPARTMENTS
from .._rng import SeededRNG
from ..enums import (
    APIAuthMethod,
    APIServiceName,
    APIStability,
    AuthorityLevel,
    ConfidentialityLevel,
    DocumentCategory,
    DocumentStatus,
    Region,
    ReviewCycle,
)
from ..schemas import APIDocumentationMetadata, DocumentMetadata, EnterpriseDocument

_SERVICE_KEYS = list(API_SERVICES.keys())

_AUTH_MAP: dict[str, APIAuthMethod] = {
    "OAuth2": APIAuthMethod.OAUTH2,
    "API_key": APIAuthMethod.API_KEY,
    "mTLS": APIAuthMethod.MTLS,
    "service_token": APIAuthMethod.SERVICE_TOKEN,
}

_SERVICE_ENUM_MAP: dict[str, APIServiceName] = {
    "HYID": APIServiceName.HYID,
    "HYDeploy": APIServiceName.HYDEPLOY,
    "HYMonitor": APIServiceName.HYMONITOR,
    "HYDataLake": APIServiceName.HYDATALAKE,
}

_STABILITY_MAP: dict[str, APIStability] = {
    "deprecated": APIStability.DEPRECATED,
    "stable": APIStability.STABLE,
    "beta": APIStability.BETA,
    "experimental": APIStability.EXPERIMENTAL,
}

_BASE_DATE = date(2022, 1, 1)


def _build_endpoints(service_data: dict, version: str, rng: SeededRNG) -> list[dict]:
    all_eps = service_data["endpoints"]
    count = rng.randint(3, min(len(all_eps), 6))
    chosen = rng.sample(all_eps, count)
    result = []
    for ep in chosen:
        params = []
        for p in ep.get("required_params", []):
            params.append({"name": p, "location": "body/query", "type": "string", "required": True, "description": f"The {p.replace('_', ' ')}"})
        for p in ep.get("optional_params", []):
            params.append({"name": p, "location": "query", "type": "string", "required": False, "description": f"Optional: {p.replace('_', ' ')}"})
        is_v1_deprecated = version == "v1"
        result.append({
            "method": ep["method"],
            "path": ep["path"].replace("{ver}", version.lstrip("v")),
            "description": ep["purpose"],
            "deprecated": is_v1_deprecated and rng.random() < 0.3,
            "deprecated_replacement": ep["path"].replace("{ver}", "2").replace("v{ver}", "v2"),
            "parameters": params,
            "example_response": None,
        })
    return result


def _build_error_codes(service_data: dict) -> list[dict]:
    rows = []
    for code, (msg, action) in service_data["error_codes"].items():
        rows.append({"code": code, "message": msg, "description": msg, "action": action})
    return rows


class APIDocGenerator:
    """Generates API Documentation documents — 4 services × 5 version docs = 20 docs."""

    def __init__(self, rng: SeededRNG, assembler: DocumentAssembler) -> None:
        self._rng = rng
        self._assembler = assembler

    def generate_one(self, index: int, _all_doc_ids: list[str]) -> EnterpriseDocument:
        service_idx = (index - 1) // len(API_VERSIONS)
        version_idx = (index - 1) % len(API_VERSIONS)
        service_key = _SERVICE_KEYS[service_idx % len(_SERVICE_KEYS)]
        version_spec = API_VERSIONS[version_idx]

        service_data = API_SERVICES[service_key]
        version = version_spec["version"]
        stability_str = version_spec["stability"]

        doc_id = f"API-DOC-{index:03d}"
        is_deprecated = stability_str == "deprecated"
        status = DocumentStatus.DEPRECATED if is_deprecated else DocumentStatus.ACTIVE

        eff_date = _BASE_DATE + timedelta(days=(index - 1) * 45)
        created_dt = datetime(eff_date.year, eff_date.month, eff_date.day, 9, 0, 0)
        updated_dt = created_dt + timedelta(days=self._rng.randint(0, 60))

        doc_version = f"{version}.0"
        auth_method_str = service_data["auth_method"]
        owner_team = service_data["department"]
        adr_prefix = service_data["adr_prefix"]

        endpoints = _build_endpoints(service_data, version, self._rng)
        error_codes = _build_error_codes(service_data)

        deprecation_deadline = str(eff_date + timedelta(days=180))
        recommended_version = "v2" if version == "v1" else "v3"

        base_ver_num = version.lstrip("v")
        base_url = f"https://api.hytech-solutions.internal/{service_key.lower()}"

        auth_context: dict = {
            "token_endpoint": f"{base_url}/oauth/token",
            "required_scopes": [f"{service_key.lower()}:read", f"{service_key.lower()}:write"],
            "token_expiry_seconds": 3600,
            "key_rotation_days": 90,
            "jwks_endpoint": f"{base_url}/.well-known/jwks.json",
        }

        changelog = [
            {"version": version, "date": str(eff_date), "summary": "Initial release of this API version."},
        ]
        if version == "v2":
            changelog.append({"version": "v2.1", "date": str(eff_date + timedelta(days=60)), "summary": "Added pagination support and new error codes."})

        rev_history = [
            {"version": doc_version, "date": str(eff_date), "author_role": "Platform Architect", "change": "Initial documentation release"},
        ]

        title = f"{service_data['full_name']} API Reference – {version}"
        if is_deprecated:
            title += " (Deprecated)"

        context = {
            "title": title,
            "api_doc_id": doc_id,
            "document_owner": "Platform Architect",
            "owner_team": owner_team,
            "department": owner_team,
            "effective_date": str(eff_date),
            "version": doc_version,
            "status": status.value,
            "service_name": service_data["full_name"],
            "api_version": version,
            "stability": stability_str,
            "auth_method": auth_method_str.lower().replace("_", " "),
            "service_description": service_data["description"],
            "base_url": f"{base_url}/{version}",
            "token_endpoint": auth_context["token_endpoint"],
            "required_scopes": auth_context["required_scopes"],
            "token_expiry_seconds": auth_context["token_expiry_seconds"],
            "key_rotation_days": auth_context["key_rotation_days"],
            "jwks_endpoint": auth_context["jwks_endpoint"],
            "endpoints": endpoints,
            "error_codes": error_codes,
            "changelog": changelog,
            "support_sla": "2 business days",
            "deprecation_notice_days": 90,
            "deprecation_deadline": deprecation_deadline,
            "recommended_version": recommended_version,
            "revision_history": rev_history,
        }

        body = self._assembler.render("api_doc.j2", context)

        category_metadata = APIDocumentationMetadata(
            api_doc_id=doc_id,
            service_name=_SERVICE_ENUM_MAP[service_key],
            api_version=version,
            endpoint_count=len(endpoints),
            auth_method=_AUTH_MAP[auth_method_str],
            owner_team=owner_team,
            stability=_STABILITY_MAP[stability_str],
        )

        dept = DEPARTMENTS["API Documentation"][0]
        metadata = DocumentMetadata(
            document_id=doc_id,
            category=DocumentCategory.API_DOCUMENTATION,
            title=title,
            version=doc_version,
            created_at=created_dt,
            updated_at=updated_dt,
            department=dept,
            document_owner="Platform Architect",
            confidentiality=ConfidentialityLevel.CONFIDENTIAL,
            region=Region.GLOBAL,
            language="en",
            effective_date=eff_date,
            review_cycle=ReviewCycle.QUARTERLY,
            status=status,
            authority_level=AuthorityLevel.STANDARD,
            tags=[service_key.lower(), version, stability_str, "api"],
            source_path=f"/data/api/{doc_id}.md",
            category_metadata=category_metadata,
        )

        return self._assembler.build_document(metadata, body)
