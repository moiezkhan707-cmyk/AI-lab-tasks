"""Lab 1 - Section 1.5: Scholarship screening, symbolic vs threshold comparison.
Note: probability values are invented demonstration inputs. No model is trained,
so this is a behaviour comparison, NOT an accuracy experiment.
Run: python lab1_compare.py
"""


def symbolic(score, complete):
    return "Review" if complete and score >= 70 else "Hold"


def threshold(probability):
    return "Review" if probability >= 0.70 else "Hold"


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

# Section 1.6: isolate the effect of the boundary values separately
print("\nBoundary isolation (Section 1.6)")
print("score=69, complete=True  ->", symbolic(69, True))
print("probability=0.69         ->", threshold(0.69))
