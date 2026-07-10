"""Deterministic onboarding regressions for package-local nav and side-letter normalization."""

from __future__ import annotations

import json
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.onboard_contract_packages as onboard_module


@contextmanager
def _workspace_tempdir(prefix: str):
    root = Path("tmp_test_work")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{prefix}{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def _patched_onboarding_roots(data_root: Path):
    with ExitStack() as stack:
        stack.enter_context(patch.object(onboard_module, "DATA_DIR", data_root))
        stack.enter_context(patch.object(onboard_module, "CONTRACTS_ROOT", data_root / "contracts"))
        stack.enter_context(patch.object(onboard_module, "RUNTIME_MANIFESTS", data_root / "manifests"))
        stack.enter_context(patch.object(onboard_module, "RUNTIME_CHUNKS", data_root / "chunks"))
        stack.enter_context(patch.object(onboard_module, "RUNTIME_WAGES", data_root / "wages"))
        stack.enter_context(patch.object(onboard_module, "RUNTIME_TABLES", data_root / "tables"))
        stack.enter_context(patch.object(onboard_module, "RUNTIME_ONTOLOGIES", data_root / "ontologies"))
        stack.enter_context(patch.object(onboard_module, "RUNTIME_ENTITLEMENTS", data_root / "entitlements"))
        stack.enter_context(patch.object(onboard_module, "PACK_REGISTRY_FILE", data_root / "contracts" / "pack_registry.json"))
        yield


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _test_onboarding_generates_pdf_nav_and_side_letter_doc_types() -> None:
    contract_id = "onboard_nav_side_letter_contract"
    with _workspace_tempdir("onboard_pkg_") as tmp:
        data_root = tmp / "data"
        package_dir = data_root / "contracts" / contract_id
        source_dir = package_dir / "source"
        for rel in ("source", "chunks", "manifests", "tables", "wages", "entitlements", "ontology", "outline"):
            (package_dir / rel).mkdir(parents=True, exist_ok=True)

        md_text = "\n".join(
            [
                "# ARTICLE 1",
                "# GENERAL",
                "",
                "## Section 1",
                "Original document was signed dated 7/5/2025. Letter of Understanding regarding minimum wage shall remain in effect.",
                "",
                "## Section 2",
                "All regular full-time employees shall receive one (1) week's paid vacation after one (1) year",
                "of continuous service and two (2) weeks' paid vacation after two (2) years of continuous service.",
                "",
            ]
        )
        (source_dir / f"{contract_id}.md").write_text(md_text, encoding="utf-8")
        _write_json(
            source_dir / f"{contract_id}.json",
            {
                "pages": [
                    {
                        "page_number": 1,
                        "items": [
                            {"value": "ARTICLE 1"},
                            {"value": "SECTION 1"},
                            {"value": "SECTION 2"},
                        ],
                    }
                ]
            },
        )
        (source_dir / f"{contract_id}.pdf").write_bytes(b"%PDF-1.4\n%EOF\n")

        with _patched_onboarding_roots(data_root):
            result = onboard_module._process_package(
                package_dir=package_dir,
                build_tables=False,
                build_wages=False,
                sync_runtime=False,
                run_pack_gates=False,
                enforce_pack_gates=False,
                strict_pack_gates=False,
            )

        assert int(result.get("side_letter_doc_type_changes") or 0) >= 1
        pdf_nav_path = Path(result["pdf_nav_index_path"])
        outline_path = Path(result["contract_outline_path"])
        assert pdf_nav_path.exists()
        assert outline_path.exists()
        chunks = json.loads(Path(package_dir / "chunks" / f"contract_chunks_enriched_{contract_id}.json").read_text(encoding="utf-8"))
        assert any(str(row.get("doc_type") or "").lower() in {"loa", "lou"} for row in chunks if isinstance(row, dict))


def main() -> None:
    _test_onboarding_generates_pdf_nav_and_side_letter_doc_types()
    print("[OK] onboarding package checks passed")


if __name__ == "__main__":
    main()
