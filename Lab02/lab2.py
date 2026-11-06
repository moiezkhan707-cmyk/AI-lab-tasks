"""Lab 2: Intelligent Agents & the PEAS Framework (single-file version).
Smart classroom energy agent. ECO / COOL / WARM / IDLE are simulated mode labels
(no hardware). Thresholds are synthetic lab assumptions (degrees Celsius).
Run: python lab2.py
"""


# ---------- Reflex agent (no memory) ----------
class ClassroomAgent:
    def act(self, temp, occupied):
        if not occupied:          # empty room has priority over temperature
            return "ECO"
        if temp > 26:
            return "COOL"
        if temp < 20:
            return "WARM"
        return "IDLE"             # exactly 20 or 26 -> IDLE (strict < and >)


# ---------- Stateful agent (remembers previous mode) ----------
class StatefulClassroomAgent:
    def __init__(self):
        self.previous_mode = None

    def decide(self, temp, occupied):
        if not occupied:
            return "ECO"
        if temp > 26:
            return "COOL"
        if temp < 20:
            return "WARM"
        return "IDLE"

    def act(self, temp, occupied):
        """Return (target_mode, send_command). Send only if the mode changed."""
        target = self.decide(temp, occupied)
        send_command = target != self.previous_mode
        self.previous_mode = target
        return target, send_command


# ---------- Proposed improvement: hysteresis (dead band) ----------
class HysteresisAgent(StatefulClassroomAgent):
    COOL_OFF = 25                 # keep COOL until temp <= 25

    def decide(self, temp, occupied):
        if occupied and self.previous_mode == "COOL" and temp > self.COOL_OFF:
            return "COOL"
        return super().decide(temp, occupied)


if __name__ == "__main__":
    # Part 1: reflex agent
    print("=== Part 1: Reflex agent ===")
    reflex = ClassroomAgent()
    reflex_percepts = [(29, True), (29, True), (20, True),
                       (26, True), (19, True), (29, False)]
    for step, percept in enumerate(reflex_percepts, 1):
        print(step, percept, "target mode", reflex.act(*percept))

    # Part 2: stateful agent mode-transition trace
    print("\n=== Part 2: Stateful agent trace ===")
    agent = StatefulClassroomAgent()
    trace = [(29, True), (29, True), (26, True), (20, True),
             (19, True), (29, False), (29, False)]
    print("step | previous_mode | percept | target | command")
    for step, percept in enumerate(trace, 1):
        prev = agent.previous_mode
        target, send = agent.act(*percept)
        print(step, prev, percept, target, "SEND" if send else "no-command", sep=" | ")

    # Part 3: reflex vs stateful command count
    print("\n=== Part 3: Reflex vs stateful commands ===")
    reflex, stateful = ClassroomAgent(), StatefulClassroomAgent()
    stateful_cmds = 0
    for p in reflex_percepts:
        target, send = stateful.act(*p)
        assert reflex.act(*p) == target
        stateful_cmds += int(send)
    print("Reflex agent commands   :", len(reflex_percepts))
    print("Stateful agent commands :", stateful_cmds)

    # Part 4: edge case, noisy readings around 26 C
    print("\n=== Part 4: Noisy sequence around 26 C ===")
    noisy = [(25.9, True), (26.1, True), (25.9, True),
             (26.1, True), (25.9, True), (26.1, True)]
    for name, a in [("stateful", StatefulClassroomAgent()), ("hysteresis", HysteresisAgent())]:
        modes, cmds = [], 0
        for p in noisy:
            t, send = a.act(*p)
            modes.append(t)
            cmds += int(send)
        print(f"{name:10s} modes={modes} commands_sent={cmds}")
