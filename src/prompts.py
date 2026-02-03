"""
Prompt generation system for P2Echo-new.

Ported from original P2Echo/voxtell/echo/prompts.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import torch


# =============================================================================
# Label mappings
# =============================================================================

LABEL_TO_ID: Dict[str, int] = {
    "BG": 0,
    "LV": 1,
    "MYO": 2,
    "LA": 3,
    "RV": 4,
    "RA": 5,
}

ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}

ID_TO_LONG: Dict[int, str] = {
    0: "background",
    1: "left ventricle cavity",
    2: "left ventricular myocardium",
    3: "left atrium",
    4: "right ventricle",
    5: "right atrium",
}

NUM_CLASSES: int = len(LABEL_TO_ID)


# =============================================================================
# View/Dataset structure mappings
# =============================================================================

VIEW_TO_STRUCTURES: Dict[str, Tuple[str, ...]] = {
    "4CH": ("LV", "LA", "RA", "RV", "MYO"),
    "2CH": ("LV", "LA", "MYO"),
    "3CH": ("LV", "LA", "MYO"),
    "PSAX": ("LV", "RV", "MYO"),
}

# Dataset -> label set (from datasets). Key must match DataFrame 'dataset'.
DATASET_TO_PRESENT: Dict[str, Set[str]] = {
    "Camus": {"LV", "MYO", "LA"},
    "CardiacNet": {"LV", "LA", "RV", "RA"},
    "CardiacUDA": {"LV", "LA", "RV", "RA"},
    "EchoCP": {"LV", "LA", "RV", "RA"},
    "EchoNet-Dynamic": {"LV"},
    "EchoNet-Pediatric": {"LV"},
    "HMCQU": {"LV", "MYO"},
    "SegRWMA": {"LV"},
    "Private": {"LV"},
}

VIEW_LABEL_DESC: Dict[str, Dict[str, str]] = {
    "4CH": {
        "LV": "The LV appears as a bullet-shaped chamber forming the apex.",
        "LA": "The LA is located posterior to the LV.",
        "RA": "The RA is located next to the LA and inferior to the RV.",
        "RV": "The RV appears triangular and usually smaller than the LV.",
        "MYO": "The LV myocardium includes inferoseptum, anterolateral, apical septal, and apical lateral segments.",
    },
    "2CH": {
        "LV": "The LV is fully visualized from base to apex.",
        "LA": "The LA is seen superior to the LV.",
        "MYO": "The LV myocardium includes anterior and inferior wall segments (basal, mid, apical).",
    },
    "3CH": {
        "LV": "The LV and LVOT are visible in long-axis.",
        "LA": "The LA is visible in long-axis.",
        "MYO": "The LV myocardium includes anteroseptum, inferolateral, apical septal, and apical lateral segments.",
    },
    "PSAX": {
        "LV": "The LV appears circular at mid-ventricle.",
        "RV": "The RV appears crescent-shaped, wrapping anteriorly around the LV.",
        "MYO": "The LV myocardium shows all 6 radial segments.",
    },
}


# =============================================================================
# Helper functions
# =============================================================================

def plane_to_text(plane: str) -> str:
    """Convert plane abbreviation to full text description."""
    p = str(plane).upper()
    if p == "2CH":
        return "apical 2 chamber echocardiography view"
    if p == "3CH":
        return "apical 3 chamber echocardiography view"
    if p == "4CH":
        return "apical 4 chamber echocardiography view"
    if p == "PSAX":
        return "parasternal short axis view"
    return "echocardiography view"


def plane_to_text_with_abbrev(plane: str) -> str:
    """Convert plane abbreviation to text with abbreviation in parentheses."""
    p = str(plane).upper()
    if p == "2CH":
        return "Apical 2-Chamber (2CH)"
    if p == "3CH":
        return "Apical 3-Chamber (3CH)"
    if p == "4CH":
        return "Apical 4-Chamber (4CH)"
    if p == "PSAX":
        return "Parasternal Short-Axis (PSAX)"
    return "Echocardiography View"


def present_label_set(dataset: str) -> Tuple[str, ...]:
    """Get the set of labels present in a dataset."""
    present = DATASET_TO_PRESENT.get(str(dataset))
    if present is None:
        present = {"LV", "MYO", "LA", "RV", "RA"}
    return tuple(sorted(present))


def view_label_set(plane: str) -> Tuple[str, ...]:
    """Get the set of labels visible in a view."""
    p = str(plane).upper()
    labels = VIEW_TO_STRUCTURES.get(p)
    if labels is None:
        labels = ("LV", "MYO", "LA", "RV", "RA")
    return tuple(labels)


# =============================================================================
# PromptSpec dataclass
# =============================================================================

@dataclass(frozen=True)
class PromptSpec:
    """
    Defines one prompt task.

    label_ids is a union (e.g., (1,2) for LV+MYO).
    """
    label_ids: Tuple[int, ...]
    user_prompt: str | None = None

    def key(self) -> Tuple[Tuple[int, ...], str | None]:
        return (tuple(self.label_ids), self.user_prompt)


def default_prompt_specs(include_lv_myo_combo: bool = False) -> List[PromptSpec]:
    """Get default prompt specs for all classes."""
    specs = [
        PromptSpec(label_ids=(LABEL_TO_ID["BG"],)),
        PromptSpec(label_ids=(LABEL_TO_ID["LV"],)),
        PromptSpec(label_ids=(LABEL_TO_ID["MYO"],)),
        PromptSpec(label_ids=(LABEL_TO_ID["LA"],)),
        PromptSpec(label_ids=(LABEL_TO_ID["RV"],)),
        PromptSpec(label_ids=(LABEL_TO_ID["RA"],)),
    ]
    if include_lv_myo_combo:
        specs.append(PromptSpec(label_ids=(LABEL_TO_ID["LV"], LABEL_TO_ID["MYO"])))
    return specs


# =============================================================================
# Prompt text generation
# =============================================================================

def build_background_prompt_text(*, plane: str) -> str:
    """Build prompt text for background-only segmentation."""
    plane_text = plane_to_text_with_abbrev(plane)
    present = set(view_label_set(plane))
    present_text = ", ".join(sorted(present)) if len(present) else "none"
    return f"{plane_text}: Structures visualized: {present_text}. Segment background only."


def build_prompt_text(
    *,
    plane: str,
    dataset: str,
    spec: PromptSpec,
    include_present_absent_context: bool = True,
) -> str:
    """
    Build prompt text for a given sample.
    
    Args:
        plane: View plane (2CH, 3CH, 4CH, PSAX)
        dataset: Dataset name (not used in prompt but kept for API compatibility)
        spec: PromptSpec defining the target labels
        include_present_absent_context: Whether to include view context
    
    Returns:
        Formatted prompt text string
    """
    plane_text = plane_to_text_with_abbrev(plane)
    present = set(view_label_set(plane))
    present_text = ", ".join(sorted(present)) if len(present) else "none"

    if spec.user_prompt is not None:
        if include_present_absent_context:
            return (
                f"{plane_text}. Structures present: {present_text}. "
                f"{spec.user_prompt} Only segment structures that are present."
            )
        return f"{plane_text}. {spec.user_prompt}"

    # Auto-generate from label ids
    label_ids = tuple(int(x) for x in spec.label_ids)
    if len(label_ids) == 0:
        raise ValueError("PromptSpec.label_ids must be non-empty unless user_prompt is provided.")

    if all(lid == 0 for lid in label_ids):
        instr = "Segment background only."
        if include_present_absent_context:
            return f"{plane_text}: Structures visualized: {present_text}. {instr}"
        return f"{plane_text}: {instr}"

    labels_short = []
    labels_long = []
    for lid in label_ids:
        if lid == 0:
            continue
        short = ID_TO_LABEL.get(lid, f"class{lid}")
        labels_short.append(short)
        labels_long.append(ID_TO_LONG.get(lid, short))

    requested = list(labels_short)
    requested_present = [s for s in requested if s in present]
    requested_absent = [s for s in requested if s not in present]

    desc_map = VIEW_LABEL_DESC.get(str(plane).upper(), {})
    
    if len(requested_present) == 0:
        # All requested structures are absent in this view
        if len(labels_short) == 1:
            instr = (
                f"The {labels_short[0]} is not visualized in this view; do not segment it. Background only."
            )
        else:
            joined_short = " and ".join(labels_short)
            instr = f"The {joined_short} are not visualized in this view; do not segment them. Background only."
    elif len(requested_absent) == 0:
        # All requested structures are present
        if len(labels_short) == 1:
            desc = desc_map.get(labels_short[0], f"The {labels_short[0]} is visible in this view.")
            instr = f"Segment the {labels_short[0]}: {desc}"
        else:
            joined_short = " and ".join(labels_short)
            descs = " ".join(desc_map.get(s, f"The {s} is visible in this view.") for s in labels_short)
            instr = f"Segment the {joined_short} together. {descs}"
    else:
        # Some present, some absent
        present_shorts = [s for s in labels_short if s in present]
        absent_shorts = [s for s in labels_short if s not in present]
        present_descs = " ".join(desc_map.get(s, f"The {s} is visible in this view.") for s in present_shorts)
        instr = (
            f"Segment the {' and '.join(present_shorts)}: {present_descs} "
            f"The {' and '.join(absent_shorts)} is not visualized in this view; do not segment it."
        )

    if include_present_absent_context:
        return f"{plane_text}: Structures visualized: {present_text}. {instr}"
    return f"{plane_text}: {instr}"


def build_prompt_text_from_view(
    *,
    plane: str,
    spec: PromptSpec,
    include_present_absent_context: bool = True,
) -> str:
    """Build prompt text using view-based visible structures."""
    return build_prompt_text(
        plane=str(plane),
        dataset="",
        spec=spec,
        include_present_absent_context=include_present_absent_context,
    )


def build_batch_prompt_texts(
    *,
    planes: Sequence[str],
    datasets: Sequence[str],
    prompt_specs: Sequence[PromptSpec],
    include_present_absent_context: bool = True,
) -> List[List[str]]:
    """
    Build prompt texts for a batch.
    
    Returns:
        Per-sample list of prompt strings (len(prompt_specs) each).
    """
    if len(planes) != len(datasets):
        raise ValueError("planes and datasets must have same length")
    out: List[List[str]] = []
    for p, ds in zip(planes, datasets):
        per = [
            build_prompt_text(
                plane=str(p),
                dataset=str(ds),
                spec=spec,
                include_present_absent_context=include_present_absent_context,
            )
            for spec in prompt_specs
        ]
        out.append(per)
    return out


# =============================================================================
# Prompt text parsing
# =============================================================================

_TOKEN_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bLV\b", re.IGNORECASE), "LV"),
    (re.compile(r"\bMYO\b", re.IGNORECASE), "MYO"),
    (re.compile(r"\bLA\b", re.IGNORECASE), "LA"),
    (re.compile(r"\bRV\b", re.IGNORECASE), "RV"),
    (re.compile(r"\bRA\b", re.IGNORECASE), "RA"),
    (re.compile(r"left\s+ventricle", re.IGNORECASE), "LV"),
    (re.compile(r"myocard", re.IGNORECASE), "MYO"),
    (re.compile(r"left\s+atri", re.IGNORECASE), "LA"),
    (re.compile(r"right\s+ventricle", re.IGNORECASE), "RV"),
    (re.compile(r"right\s+atri", re.IGNORECASE), "RA"),
]


def parse_prompt_to_label_ids(prompt: str) -> Tuple[int, ...]:
    """Extract known echo structures from a free-text prompt."""
    found: Set[str] = set()
    for pat, key in _TOKEN_PATTERNS:
        if pat.search(prompt) is not None:
            found.add(key)
    ids = sorted(LABEL_TO_ID[k] for k in found if k in LABEL_TO_ID and k != "BG")
    return tuple(ids)


def prompt_specs_from_free_text(prompts: Sequence[str]) -> List[PromptSpec]:
    """Convert free-text prompts to PromptSpec list."""
    specs: List[PromptSpec] = []
    for p in prompts:
        ids = parse_prompt_to_label_ids(p)
        if len(ids) == 0:
            raise ValueError(
                f"Could not map prompt to known labels: {p!r}. "
                f"Expected to find one of: LV, MYO, LA, RV, RA."
            )
        specs.append(PromptSpec(label_ids=ids, user_prompt=str(p)))
    specs.insert(0, PromptSpec(label_ids=(LABEL_TO_ID["BG"],), user_prompt="Segment background only."))
    return specs


# =============================================================================
# Target generation
# =============================================================================

def make_binary_targets(
    *,
    gt_mask: torch.Tensor,
    prompt_specs: Sequence[PromptSpec],
) -> torch.Tensor:
    """
    Convert a multi-class GT mask [B,H,W] with ids 0..5 to per-prompt binary targets [B,N,H,W].
    
    Args:
        gt_mask: [B, H, W] tensor with class IDs 0-5
        prompt_specs: List of PromptSpec objects
    
    Returns:
        [B, N, H, W] float tensor with binary targets
    """
    if gt_mask.ndim != 3:
        raise ValueError(f"gt_mask must be [B,H,W], got {tuple(gt_mask.shape)}")
    b, h, w = gt_mask.shape
    targets = torch.zeros((b, len(prompt_specs), h, w), device=gt_mask.device, dtype=torch.float32)
    for i, spec in enumerate(prompt_specs):
        ids = [int(x) for x in spec.label_ids if int(x) != 0]
        if len(ids) == 0:
            continue
        m = torch.zeros_like(gt_mask, dtype=torch.bool)
        for lid in ids:
            m |= (gt_mask == lid)
        targets[:, i] = m.float()
    return targets
