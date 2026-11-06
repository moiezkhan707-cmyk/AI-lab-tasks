"""Lab 1: Introduction to AI & Paradigms (single-file version).
Scholarship screening prototype. 'Review' = flag for a human reviewer, NOT an award decision.
All thresholds and probabilities are synthetic lab assumptions; no model is trained,
so Part A is a behaviour comparison, NOT an accuracy experiment.
Run: python lab1.py
"""

# ================= Part A: symbolic rule vs probability threshold =================


def symbolic(score, complete):
    return "Review" if complete and score >= 70 else "Hold"


def threshold(probability):
    return "Review" if probability >= 0.70 else "Hold"


# ================= Part B: extended transparent rule checker =================
# R1 score >= 70 | R2 documents complete | R3 attendance >= 75 (new) | R4 no disciplinary case (new)
SCORE_MIN = 70
ATTENDANCE_MIN = 75


def screen(score, documents_complete, attendance, disciplinary_cases):
    """Return (decision, list_of_failed_rules)."""
    failed = []
    if score < SCORE_MIN:
        failed.append(f"R1 score {score} < {SCORE_MIN}")
    if not documents_complete:
        failed.append("R2 documents incomplete")
    if attendance < ATTENDANCE_MIN:
        failed.append(f"R3 attendance {attendance}% < {ATTENDANCE_MIN}%")
    if disciplinary_cases != 0:
        failed.append(f"R4 disciplinary cases = {disciplinary_cases}")
    return ("Review" if not failed else "Hold"), failed


if __name__ == "__main__":
    print("=== Part A: symbolic vs threshold ===")
    # (name, score, documents_complete, invented_probability)
    cases = [
        ("complete_high", 82, True, 0.81),
        ("complete_low", 68, True, 0.74),
        ("incomplete_high", 91, False, 0.88),
        ("boundary", 70, True, 0.70),
    ]
    print("case | symbolic | threshold | disagreement")
    for name, score, complete, probability in cases:
        a, b = symbolic(score, complete), threshold(probability)
        print(name, a, b, a != b, sep=" | ")

    print("\nBoundary isolation (Section 1.6)")
    print("score=69, complete=True  ->", symbolic(69, True))
    print("probability=0.69         ->", threshold(0.69))

    print("\n=== Part B: extended rule checker (4 core tests) ===")
    tests = [
        ("T1_pass", dict(score=85, documents_complete=True, attendance=90, disciplinary_cases=0), "Review"),
        ("T2_fail_attendance", dict(score=85, documents_complete=True, attendance=60, disciplinary_cases=0), "Hold"),
        ("T3_fail_disciplinary", dict(score=85, documents_complete=True, attendance=90, disciplinary_cases=1), "Hold"),
        ("T4_boundary", dict(score=70, documents_complete=True, attendance=75, disciplinary_cases=0), "Review"),
    ]
    for name, inputs, expected in tests:
        actual, failed = screen(**inputs)
        reason = "; ".join(failed) if failed else "all rules satisfied"
        print(f"Case     : {name}")
        print(f"Input    : {inputs}")
        print(f"Expected : {expected}")
        print(f"Actual   : {actual}  ({'PASS' if actual == expected else 'FAIL'})")
        print(f"Reason   : {reason}")
        print("-" * 60)
