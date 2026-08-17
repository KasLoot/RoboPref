"""Build PrefMem thesis workflow diagrams as vector PDFs."""

from __future__ import annotations

from math import atan2, cos, pi, sin
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


OUT = Path(__file__).resolve().parent

PALETTE = {
    "agent": (HexColor("#EAF3FF"), HexColor("#2563EB")),
    "host": (HexColor("#F3EEFF"), HexColor("#7C3AED")),
    "data": (HexColor("#ECFDF5"), HexColor("#059669")),
    "human": (HexColor("#FFF7ED"), HexColor("#C2410C")),
    "decision": (HexColor("#FFFBEB"), HexColor("#D97706")),
    "safety": (HexColor("#FFF1F2"), HexColor("#DC2626")),
    "neutral": (HexColor("#F8FAFC"), HexColor("#475569")),
}
INK = HexColor("#172033")
MUTED = HexColor("#526070")
EDGE = HexColor("#334155")
CONTEXT = HexColor("#64748B")
RED = HexColor("#B91C1C")


def _wrap(text: str, width: float, font: str, size: float) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        words = raw.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if stringWidth(candidate, font, size) <= width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def box(c: canvas.Canvas, x: float, y: float, w: float, h: float,
        text: str, kind: str, size: float = 15.0) -> tuple[float, float, float, float]:
    fill, stroke = PALETTE[kind]
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1.5)
    c.roundRect(x, y, w, h, 7, stroke=1, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica", size)
    lines = _wrap(text, w - 14, "Helvetica", size)
    while size > 9 and len(lines) * (size + 2) > h - 8:
        size -= 0.5
        lines = _wrap(text, w - 14, "Helvetica", size)
        c.setFont("Helvetica", size)
    leading = size + 2
    total = leading * len(lines)
    baseline = y + (h + total) / 2 - leading
    for line in lines:
        c.drawCentredString(x + w / 2, baseline, line)
        baseline -= leading
    return (x, y, w, h)


def diamond(c: canvas.Canvas, x: float, y: float, w: float, h: float,
            text: str, size: float = 13.0) -> tuple[float, float, float, float]:
    fill, stroke = PALETTE["decision"]
    path = c.beginPath()
    path.moveTo(x + w / 2, y + h)
    path.lineTo(x + w, y + h / 2)
    path.lineTo(x + w / 2, y)
    path.lineTo(x, y + h / 2)
    path.close()
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1.5)
    c.drawPath(path, stroke=1, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica", size)
    lines = _wrap(text, w * 0.62, "Helvetica", size)
    leading = size + 1.5
    baseline = y + (h + leading * len(lines)) / 2 - leading
    for line in lines:
        c.drawCentredString(x + w / 2, baseline, line)
        baseline -= leading
    return (x, y, w, h)


def _head(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float,
          color=EDGE) -> None:
    angle = atan2(y2 - y1, x2 - x1)
    length = 9
    spread = pi / 7
    p1 = (x2 - length * cos(angle - spread), y2 - length * sin(angle - spread))
    p2 = (x2 - length * cos(angle + spread), y2 - length * sin(angle + spread))
    path = c.beginPath()
    path.moveTo(x2, y2)
    path.lineTo(*p1)
    path.lineTo(*p2)
    path.close()
    c.setFillColor(color)
    c.setStrokeColor(color)
    c.drawPath(path, stroke=1, fill=1)


def label(c: canvas.Canvas, x: float, y: float, text: str,
          size: float = 11.5) -> None:
    lines = text.split("\n")
    width = max(stringWidth(line, "Helvetica", size) for line in lines) + 6
    height = len(lines) * (size + 1) + 3
    c.setFillColor(white)
    c.rect(x - width / 2, y - height / 2, width, height, stroke=0, fill=1)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", size)
    baseline = y + (len(lines) - 1) * (size + 1) / 2 - size / 3
    for line in lines:
        c.drawCentredString(x, baseline, line)
        baseline -= size + 1


def path_arrow(c: canvas.Canvas, points: list[tuple[float, float]],
               text: str | None = None, label_xy: tuple[float, float] | None = None,
               dashed: bool = False, color=EDGE, both: bool = False) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(1.7)
    c.setDash(5, 4) if dashed else c.setDash()
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path, stroke=1, fill=0)
    c.setDash()
    _head(c, *points[-2], *points[-1], color=color)
    if both:
        _head(c, *points[1], *points[0], color=color)
    if text and label_xy:
        label(c, *label_xy, text)


def new_canvas(name: str, width: float, height: float) -> canvas.Canvas:
    c = canvas.Canvas(str(OUT / f"{name}.pdf"), pagesize=(width, height),
                      pageCompression=1)
    c.setTitle(name.replace("-", " ").title())
    return c


def finish(c: canvas.Canvas) -> None:
    c.showPage()
    c.save()


def overall_architecture() -> None:
    """High-level component graph; detailed loops are separate figures."""
    c = new_canvas("overall-architecture", 760, 500)
    box(c, 20, 380, 125, 58, "User / operator", "human")
    box(c, 180, 380, 150, 58, "HRI Agent\nuser-facing boundary", "agent")
    box(c, 180, 245, 150, 58, "Memory Agent\nscoped preferences", "agent")
    box(c, 20, 245, 125, 58, "Preference store\ntext + embeddings", "data")

    box(c, 385, 330, 180, 108,
        "Deterministic core\nPrefMemRuntime\n+ Controller", "host")
    box(c, 385, 205, 180, 64,
        "Observation and\npublication surfaces", "data")
    box(c, 385, 70, 180, 64, "Physical or\nMuJoCo scene", "data")

    box(c, 610, 350, 125, 88,
        "Semantic services\nPlanner\nMonitor\nValidator", "agent", 14.0)
    box(c, 610, 205, 125, 64, "Execution adapter\nhuman or MuJoCo", "host", 14.0)

    path_arrow(c, [(145, 409), (180, 409)], both=True)
    path_arrow(c, [(330, 409), (385, 409)], both=True)
    path_arrow(c, [(255, 380), (255, 303)], both=True)
    path_arrow(c, [(180, 274), (145, 274)], both=True)
    path_arrow(c, [(565, 394), (610, 394)], both=True)
    path_arrow(c, [(565, 350), (585, 350), (585, 237), (610, 237)], both=True)
    path_arrow(c, [(475, 330), (475, 269)], both=True)
    path_arrow(c, [(475, 205), (475, 134)], both=True)
    path_arrow(c, [(672, 205), (672, 102), (565, 102)], both=True)

    label(c, 475, 292, "typed state / fresh evidence", 10.5)
    label(c, 598, 289, "one current task", 10.5)
    finish(c)


def controller_state_machine() -> None:
    """Top-level controller states with the dense cycle factored out."""
    c = new_canvas("controller-state-machine", 760, 285)
    box(c, 15, 160, 90, 58, "IDLE", "host")
    box(c, 125, 150, 135, 78, "AWAITING\nCONFIRMATION", "host", 14.0)
    box(c, 285, 140, 175, 98,
        "ACTIVE LOOP\nplan - publish - observe\n(see cycle subgraph)", "host", 13.0)
    box(c, 485, 150, 135, 78, "FINAL\nVALIDATION", "host", 14.0)
    box(c, 645, 160, 100, 58, "COMPLETE", "host")
    box(c, 285, 25, 175, 64,
        "Guard and emergency paths\n(see exception subgraph)", "safety", 13.0)

    path_arrow(c, [(105, 189), (125, 189)])
    path_arrow(c, [(260, 189), (285, 189)])
    path_arrow(c, [(460, 189), (485, 189)])
    path_arrow(c, [(620, 189), (645, 189)])
    path_arrow(c, [(553, 150), (553, 115), (372, 115), (372, 140)],
               "incomplete", (500, 115))
    path_arrow(c, [(320, 140), (320, 89)], dashed=True, color=RED)
    finish(c)


def controller_cycle_workflow() -> None:
    c = new_canvas("controller-cycle-workflow", 760, 365)
    box(c, 35, 245, 150, 64, "Fresh Planner\ndecision", "agent")
    box(c, 305, 245, 150, 64, "Publish exactly\none STEP", "data")
    box(c, 575, 245, 150, 64, "Execute and\nsettle", "host")
    box(c, 575, 55, 150, 64, "Monitor fresh\npost-publication view", "agent", 13.5)
    box(c, 305, 55, 150, 64, "Controller applies\nstability guards", "host", 13.5)
    box(c, 35, 55, 150, 64, "Trusted history +\nnew scene state", "data")

    path_arrow(c, [(185, 277), (305, 277)])
    path_arrow(c, [(455, 277), (575, 277)])
    path_arrow(c, [(650, 245), (650, 119)])
    path_arrow(c, [(575, 87), (455, 87)])
    path_arrow(c, [(305, 87), (185, 87)])
    path_arrow(c, [(110, 119), (110, 245)], "replan", (82, 182))
    finish(c)


def controller_exception_workflow() -> None:
    c = new_canvas("controller-exception-workflow", 760, 330)
    box(c, 20, 205, 155, 70, "Model result or\nservice event", "data")
    box(c, 215, 195, 180, 90,
        "Deterministic guards\nidentity - phase - time\ncriteria - retry bounds",
        "host", 13.0)
    diamond(c, 435, 190, 145, 100, "Guarded\noutcome?")
    box(c, 610, 235, 130, 58, "Accept legal\ntransition", "host", 13.5)
    box(c, 610, 130, 130, 64, "NEEDS\nATTENTION", "safety", 13.5)
    box(c, 390, 25, 145, 64, "EMERGENCY\nSTOPPED", "safety", 13.5)
    box(c, 580, 25, 160, 64,
        "User-authorised\nresume or replan", "human", 13.5)

    path_arrow(c, [(175, 240), (215, 240)])
    path_arrow(c, [(395, 240), (435, 240)])
    path_arrow(c, [(580, 250), (610, 264)], "valid", (594, 282))
    path_arrow(c, [(580, 225), (595, 225), (595, 162), (610, 162)], "blocked", (597, 193))
    path_arrow(c, [(507, 190), (507, 110), (462, 89)], "danger", (530, 127), color=RED)
    path_arrow(c, [(675, 130), (660, 89)], "authorised", (700, 108), dashed=True)
    finish(c)


def hri_workflow() -> None:
    c = new_canvas("hri-workflow", 760, 455)
    box(c, 20, 350, 125, 58, "User request", "human")
    box(c, 180, 350, 150, 58, "HRI interprets\noutcome + constraints", "agent", 13.5)
    diamond(c, 380, 330, 150, 98, "Required field\nmissing?")
    box(c, 600, 350, 140, 58, "Clarified\ngoal specification", "data", 13.5)
    box(c, 380, 185, 150, 58, "Request-scoped\nMemory retrieval", "agent", 13.5)
    diamond(c, 380, 30, 150, 98, "One applicable,\nnon-conflicting record?")
    box(c, 590, 50, 150, 58, "Fill only the\nmissing field", "agent", 13.5)
    box(c, 20, 50, 160, 58, "Ask one focused\nclarification", "human", 13.5)

    path_arrow(c, [(145, 379), (180, 379)])
    path_arrow(c, [(330, 379), (380, 379)])
    path_arrow(c, [(530, 379), (600, 379)], "no", (565, 397))
    path_arrow(c, [(455, 330), (455, 243)], "yes", (477, 285))
    path_arrow(c, [(455, 185), (455, 128)])
    path_arrow(c, [(380, 79), (180, 79)], "no / conflict", (280, 97))
    path_arrow(c, [(530, 79), (590, 79)], "yes", (560, 97))
    path_arrow(c, [(665, 108), (665, 350)])
    path_arrow(c, [(100, 108), (100, 350)], "user answer", (62, 225), dashed=True)
    finish(c)


def hri_confirmation_workflow() -> None:
    c = new_canvas("hri-confirmation-workflow", 760, 330)
    box(c, 20, 220, 135, 58, "Clarified goal", "data")
    box(c, 190, 220, 145, 58, "Runtime requests\nscene-grounded preview", "host", 13.0)
    box(c, 370, 210, 160, 78,
        "HRI presents exact goal,\nconstraints, outcomes,\nID and revision", "agent", 12.5)
    diamond(c, 565, 200, 170, 98, "User confirms exact\nID and revision?")
    box(c, 565, 55, 170, 64,
        "Runtime confirms contract\nand authorises planning", "host", 13.0)
    box(c, 300, 55, 170, 64,
        "Retain proposal; revise\nor clarify without motion", "human", 13.0)

    path_arrow(c, [(155, 249), (190, 249)])
    path_arrow(c, [(335, 249), (370, 249)])
    path_arrow(c, [(530, 249), (565, 249)])
    path_arrow(c, [(650, 200), (650, 119)], "yes", (672, 158))
    path_arrow(c, [(565, 249), (550, 249), (550, 87), (470, 87)], "no", (550, 160))
    finish(c)


def memory_workflow() -> None:
    c = new_canvas("memory-workflow", 760, 350)
    box(c, 20, 235, 145, 64, "RETRIEVE request", "data")
    box(c, 220, 235, 145, 64, "Generate focused\nquery phrasing", "agent")
    box(c, 420, 235, 145, 64, "Embedding search\ntop-k per query", "host")
    box(c, 595, 130, 145, 64,
        "Rank, deduplicate,\nmerge and global cap", "host", 13.0)
    box(c, 360, 45, 165, 64,
        "Semantic applicability\nfilter over bounded set", "agent", 13.0)
    box(c, 90, 45, 165, 64,
        "Return unchanged\nvalidated records", "data", 13.5)

    path_arrow(c, [(165, 267), (220, 267)])
    path_arrow(c, [(365, 267), (420, 267)])
    path_arrow(c, [(565, 267), (667, 267), (667, 194)])
    path_arrow(c, [(595, 162), (525, 77)])
    path_arrow(c, [(360, 77), (255, 77)])
    finish(c)


def memory_mutation_workflow() -> None:
    c = new_canvas("memory-mutation-workflow", 760, 350)
    box(c, 20, 235, 145, 64, "MUTATE request\nwith explicit consent", "data", 13.0)
    box(c, 205, 235, 145, 64,
        "Retrieve within the\nsame request thread", "agent", 13.0)
    box(c, 390, 225, 165, 84,
        "Host validates request kind,\nretrieved target, scope,\nand write count", "host", 12.5)
    diamond(c, 590, 220, 150, 94, "Exactly one\njustified write?")
    box(c, 565, 55, 175, 64,
        "Persist text and regenerated\nembedding atomically", "data", 12.5)
    box(c, 280, 55, 175, 64,
        "Return no-op or\ntyped rejection", "safety", 13.5)

    path_arrow(c, [(165, 267), (205, 267)])
    path_arrow(c, [(350, 267), (390, 267)])
    path_arrow(c, [(555, 267), (590, 267)])
    path_arrow(c, [(665, 220), (665, 119)], "yes", (687, 169))
    path_arrow(c, [(590, 267), (570, 267), (570, 87), (455, 87)], "no", (570, 170))
    finish(c)


def planner_workflow() -> None:
    c = new_canvas("planner-workflow", 760, 245)
    box(c, 20, 100, 155, 70,
        "Clarified goal, constraints\nand fresh frame", "data", 13.0)
    box(c, 210, 100, 145, 70,
        "PREVIEW inference\nGoalProposal", "agent", 13.5)
    box(c, 390, 100, 145, 70,
        "Strict parse + bounded\nschema correction", "host", 13.0)
    box(c, 570, 90, 170, 90,
        "Runtime assigns goal ID\nand revision, then stages\nnon-executable proposal", "host", 12.5)

    path_arrow(c, [(175, 135), (210, 135)])
    path_arrow(c, [(355, 135), (390, 135)])
    path_arrow(c, [(535, 135), (570, 135)])
    finish(c)


def planner_cycle_workflow() -> None:
    c = new_canvas("planner-cycle-workflow", 760, 360)
    box(c, 20, 235, 165, 74,
        "Frozen goal + fresh frame\n+ trigger + trusted history", "data", 12.5)
    box(c, 220, 235, 145, 74,
        "PLAN_CYCLE\ninference", "agent", 13.5)
    box(c, 400, 235, 145, 74,
        "Strict parse + bounded\nschema correction", "host", 12.5)
    box(c, 580, 225, 160, 94,
        "PlannerDecision\nACT | FINAL VALIDATION\nBLOCKED | USER INPUT", "agent", 12.0)
    box(c, 580, 55, 160, 68,
        "Controller applies the\nlegal typed route", "host", 13.0)
    box(c, 310, 55, 180, 68,
        "If ACT: publish only\ncandidate_tasks[0]", "host", 13.0)
    box(c, 20, 55, 200, 68,
        "Next cycle receives new frame\nand host-owned history", "data", 12.5)

    path_arrow(c, [(185, 272), (220, 272)])
    path_arrow(c, [(365, 272), (400, 272)])
    path_arrow(c, [(545, 272), (580, 272)])
    path_arrow(c, [(660, 225), (660, 123)])
    path_arrow(c, [(580, 89), (490, 89)], "ACT", (535, 107))
    path_arrow(c, [(310, 89), (220, 89)])
    finish(c)


def monitor_workflow() -> None:
    c = new_canvas("monitor-workflow", 760, 390)
    box(c, 20, 275, 145, 64, "STEP publication", "data")
    box(c, 205, 275, 155, 64,
        "Bind publication ID\nand worker generation", "host", 13.0)
    diamond(c, 405, 260, 150, 94,
            "Frame strictly newer\nthan publication?")
    box(c, 595, 275, 145, 64, "Wait for a newer\nprefmem frame", "data", 13.0)
    box(c, 405, 105, 150, 64,
        "VLM assesses ordered\ncriteria + task status", "agent", 12.5)
    box(c, 205, 105, 155, 64,
        "Parse schema and verify\ncriterion + publication IDs", "host", 12.0)
    box(c, 20, 105, 145, 64,
        "Typed assessment,\nservice error, or emergency", "data", 12.0)

    path_arrow(c, [(165, 307), (205, 307)])
    path_arrow(c, [(360, 307), (405, 307)])
    path_arrow(c, [(555, 307), (595, 307)], "no", (575, 326))
    path_arrow(c, [(667, 275), (667, 235), (480, 235), (480, 260)],
               "poll again", (580, 235), dashed=True)
    path_arrow(c, [(480, 260), (480, 169)], "yes", (502, 214))
    path_arrow(c, [(405, 137), (360, 137)])
    path_arrow(c, [(205, 137), (165, 137)])
    finish(c)


def monitor_aggregation_workflow() -> None:
    c = new_canvas("monitor-aggregation-workflow", 760, 335)
    box(c, 20, 210, 145, 64, "Typed Monitor\nassessment", "data")
    diamond(c, 200, 195, 145, 94, "Still belongs to\nactive publication?")
    diamond(c, 390, 195, 145, 94, "Emergency\nenvelope valid?")
    box(c, 580, 210, 160, 64,
        "Controller updates streaks,\nprogress and timeout clock", "host", 12.0)
    box(c, 200, 45, 145, 64,
        "Discard stale\ncallback", "neutral", 13.5)
    box(c, 390, 45, 145, 64,
        "Latch process-wide\nemergency stop", "safety", 13.0)
    box(c, 580, 45, 160, 64,
        "Continue, append terminal\nhistory, or enter attention", "host", 12.0)

    path_arrow(c, [(165, 242), (200, 242)])
    path_arrow(c, [(272, 195), (272, 109)], "no", (294, 151))
    path_arrow(c, [(345, 242), (390, 242)], "yes", (367, 261))
    path_arrow(c, [(462, 195), (462, 109)], "yes", (484, 151), color=RED)
    path_arrow(c, [(535, 242), (580, 242)], "no", (557, 261))
    path_arrow(c, [(660, 210), (660, 109)])
    finish(c)


def validator_workflow() -> None:
    c = new_canvas("validator-workflow", 760, 245)
    box(c, 20, 100, 155, 70,
        "Confirmed GoalContract\n+ confirmation frame", "data", 13.0)
    box(c, 210, 100, 145, 70,
        "Compile detailed\nvisible criteria", "agent", 13.5)
    box(c, 390, 90, 155, 90,
        "Host verifies full coverage\nand one-to-one mapping to\nbroad outcomes", "host", 12.0)
    box(c, 580, 100, 160, 70,
        "Freeze immutable\nValidationContract", "data", 13.5)

    path_arrow(c, [(175, 135), (210, 135)])
    path_arrow(c, [(355, 135), (390, 135)])
    path_arrow(c, [(545, 135), (580, 135)])
    finish(c)


def validator_assessment_workflow() -> None:
    c = new_canvas("validator-assessment-workflow", 760, 370)
    box(c, 20, 245, 145, 64, "FINAL_VALIDATION\npublication", "data", 13.0)
    box(c, 205, 245, 145, 64,
        "Fresh post-publication\nprefmem view", "data", 13.0)
    box(c, 390, 245, 145, 64,
        "VLM assesses every\ndetailed criterion", "agent", 12.5)
    box(c, 575, 245, 165, 64,
        "Parse full ordered checklist\nand verify identity", "host", 12.0)
    box(c, 575, 60, 165, 64,
        "Accumulate evidence\nacross fresh views", "host", 13.0)
    box(c, 350, 60, 165, 64,
        "Aggregate detailed criteria\nto broad outcomes", "host", 12.0)
    box(c, 20, 50, 285, 84,
        "Derived route\nCOMPLETE | INCOMPLETE -> replan\nNEEDS_EVIDENCE -> new camera view", "data", 12.0)

    path_arrow(c, [(165, 277), (205, 277)])
    path_arrow(c, [(350, 277), (390, 277)])
    path_arrow(c, [(535, 277), (575, 277)])
    path_arrow(c, [(657, 245), (657, 124)])
    path_arrow(c, [(575, 92), (515, 92)])
    path_arrow(c, [(350, 92), (305, 92)])
    finish(c)


def executor_workflow() -> None:
    c = new_canvas("executor-workflow", 760, 365)
    box(c, 20, 245, 140, 64, "STEP\nPublishedTask", "data", 13.5)
    box(c, 200, 245, 140, 64,
        "Gemma compiler\nsymbolic pick/place", "agent", 13.0)
    box(c, 380, 245, 140, 64,
        "Host validates program\nand publication guards", "host", 12.0)
    box(c, 560, 245, 180, 64,
        "SAM source and target\nmask grounding", "agent", 13.0)
    box(c, 560, 55, 180, 64,
        "RGB-D filtering +\nworld-frame anchors", "host", 13.0)
    box(c, 380, 55, 140, 64,
        "IK + safe waypoint\nposition control", "host", 12.5)
    box(c, 200, 55, 140, 64,
        "Park, release\nand settle", "host", 13.0)
    box(c, 20, 55, 140, 64,
        "Typed execution\noutcome", "data", 13.5)

    path_arrow(c, [(160, 277), (200, 277)])
    path_arrow(c, [(340, 277), (380, 277)])
    path_arrow(c, [(520, 277), (560, 277)])
    path_arrow(c, [(650, 245), (650, 119)])
    path_arrow(c, [(560, 87), (520, 87)])
    path_arrow(c, [(380, 87), (340, 87)])
    path_arrow(c, [(200, 87), (160, 87)])
    finish(c)


def executor_assistance_workflow() -> None:
    c = new_canvas("executor-assistance-workflow", 760, 330)
    box(c, 20, 205, 160, 70,
        "Compiler or strict SAM\ngrounding failure", "safety", 13.0)
    box(c, 220, 205, 160, 70,
        "Record typed strict\nfailure unchanged", "data", 13.0)
    diamond(c, 420, 190, 155, 100,
            "Experiment-only\nassistance enabled?")
    box(c, 610, 225, 130, 58,
        "Stop before\nmotion", "safety", 13.5)
    box(c, 600, 100, 140, 70,
        "Oracle anchor for\ndownstream diagnostic", "neutral", 12.5)
    box(c, 300, 35, 220, 64,
        "Store strict and assisted results\nas separate experimental outcomes", "data", 12.0)

    path_arrow(c, [(180, 240), (220, 240)])
    path_arrow(c, [(380, 240), (420, 240)])
    path_arrow(c, [(575, 250), (610, 254)], "no", (592, 272))
    path_arrow(c, [(497, 190), (497, 135), (600, 135)], "yes", (548, 153), dashed=True)
    path_arrow(c, [(600, 115), (520, 67)], dashed=True)
    finish(c)


def interaction_protocol() -> None:
    c = new_canvas("interaction-protocol", 760, 435)
    box(c, 20, 205, 120, 58, "User", "human")
    box(c, 175, 205, 145, 58, "HRI Agent", "agent")
    box(c, 175, 60, 145, 64,
        "Memory Agent +\npreference store", "agent", 13.0)
    box(c, 360, 185, 180, 98,
        "Deterministic mediation\nPrefMemRuntime\n+ Controller", "host", 13.0)
    box(c, 600, 325, 140, 58, "Planner", "agent")
    box(c, 600, 235, 140, 58, "Monitor", "agent")
    box(c, 600, 145, 140, 58, "Validator", "agent")
    box(c, 600, 55, 140, 58, "Executor", "host")

    path_arrow(c, [(140, 234), (175, 234)], both=True)
    path_arrow(c, [(320, 234), (360, 234)], both=True)
    path_arrow(c, [(247, 205), (247, 124)], both=True)
    path_arrow(c, [(540, 258), (570, 258), (570, 354), (600, 354)], both=True)
    path_arrow(c, [(540, 242), (600, 264)], both=True)
    path_arrow(c, [(540, 218), (600, 174)], both=True)
    path_arrow(c, [(540, 198), (570, 198), (570, 84), (600, 84)], both=True)
    finish(c)


def interaction_sequence() -> None:
    c = new_canvas("interaction-sequence", 760, 465)
    rows = [
        (340, "PHASE 1", [
            ("Clarify", "human"), ("Retrieve if\nneeded", "agent"),
            ("Preview", "agent"), ("Exact\nconfirmation", "human"),
            ("Freeze\nchecklist", "host"),
        ]),
        (195, "PHASE 2", [
            ("Fresh plan", "agent"), ("Publish one\nSTEP", "data"),
            ("Execute", "host"), ("Monitor fresh\nevidence", "agent"),
            ("Record history\nand replan", "host"),
        ]),
        (50, "PHASE 3", [
            ("Request final", "agent"), ("Publish\nFINAL", "data"),
            ("Accumulate\nevidence", "agent"), ("Host derives\nstatus", "host"),
            ("Complete or\nreplan", "host"),
        ]),
    ]
    for y, phase, nodes in rows:
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(12, y + 29, phase)
        x_positions = [90, 225, 360, 495, 630]
        for x, (text, kind) in zip(x_positions, nodes):
            box(c, x, y, 115, 64, text, kind, 12.5)
        for left, right in zip(x_positions, x_positions[1:]):
            path_arrow(c, [(left + 115, y + 32), (right, y + 32)])
    finish(c)


def main() -> None:
    overall_architecture()
    controller_state_machine()
    controller_cycle_workflow()
    controller_exception_workflow()
    hri_workflow()
    hri_confirmation_workflow()
    memory_workflow()
    memory_mutation_workflow()
    planner_workflow()
    planner_cycle_workflow()
    monitor_workflow()
    monitor_aggregation_workflow()
    validator_workflow()
    validator_assessment_workflow()
    executor_workflow()
    executor_assistance_workflow()
    interaction_protocol()
    interaction_sequence()


if __name__ == "__main__":
    main()
