"""Pong met twee AI-spelers, elk bestuurd door het Jev-style beslismodel.

Speelveld = hetzelfde grid als snake. De batjes staan in de linker- en
rechterkolom. De bal beweegt elke stap 1 cel horizontaal en 1 cel verticaal
en kaatst tegen boven- en onderrand. Elk batje kiest per stap: omhoog,
blijven of omlaag.
"""

import random
from concurrent.futures import ThreadPoolExecutor

import pygame

from jev import softmax

PADDLE = 4
MOVES = ["up", "stay", "down"]
DELTA = {"up": -1, "stay": 0, "down": 1}
LABELS = {"up": "move up", "stay": "stay", "down": "move down"}


class Pong:
    def __init__(self, w, h, rng=None):
        self.w, self.h = w, h
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        self.score = {"left": 0, "right": 0}
        self.top = {"left": (self.h - PADDLE) // 2, "right": (self.h - PADDLE) // 2}
        self.rally = 0
        self.steps = 0
        self.serve(self.rng.choice([-1, 1]))

    def serve(self, vx):
        self.ball = (self.w // 2, self.rng.randrange(self.h // 4, 3 * self.h // 4))
        self.vel = (vx, self.rng.choice([-1, 1]))
        self.rally = 0

    def covers(self, top, y):
        return top <= y < top + PADDLE

    def clamp(self, top):
        return max(0, min(self.h - PADDLE, top))

    def advance(self, ball, vel):
        """Eén balstap zonder batjes: alleen kaatsen tegen boven/onder."""
        (bx, by), (vx, vy) = ball, vel
        ny = by + vy
        if not 0 <= ny < self.h:
            vy = -vy
            ny = by + vy
        return (bx + vx, ny), (vx, vy)

    def column(self, side):
        return 0 if side == "left" else self.w - 1

    def intercept(self, side):
        """Waar en wanneer de bal de kolom van `side` bereikt. Gaat de bal van je
        af, dan wordt aangenomen dat de tegenstander hem terugslaat."""
        ball, vel = self.ball, self.vel
        towards = (vel[0] < 0) == (side == "left")
        other = self.column("right" if side == "left" else "left")
        for n in range(1, 4 * self.w):
            ball, vel = self.advance(ball, vel)
            if ball[0] == self.column(side):
                return ball[1], n, towards
            if ball[0] == other:
                ball = (ball[0] - vel[0], ball[1])
                vel = (-vel[0], vel[1])
        return self.h // 2, 4 * self.w, towards

    def step(self, moves):
        for side, m in moves.items():
            self.top[side] = self.clamp(self.top[side] + DELTA[m])
        self.steps += 1
        ball, vel = self.advance(self.ball, self.vel)
        for side, back in (("left", 1), ("right", -1)):
            if ball[0] != self.column(side):
                continue
            top = self.top[side]
            if self.covers(top, ball[1]):
                rel = ball[1] - top
                vy = -1 if rel == 0 else 1 if rel == PADDLE - 1 else vel[1]
                ball, vel = (ball[0] + back, ball[1]), (back, vy)
                self.rally += 1
            else:
                winner = "right" if side == "left" else "left"
                self.score[winner] += 1
                self.serve(-1 if side == "left" else 1)
                return
        self.ball, self.vel = ball, vel

    def move_facts(self, side):
        """Per zet: (afstand van batje tot landingsplek na de zet, op tijd haalbaar)."""
        y, n, towards = self.intercept(side)
        facts = {}
        for m in MOVES:
            top = self.clamp(self.top[side] + DELTA[m])
            need = 0 if self.covers(top, y) else (top - y if y < top else y - (top + PADDLE - 1))
            facts[m] = (need, need <= n - 1)
        return facts, (y, n, towards)

    def describe(self, side, facts, info):
        y, n, towards = info
        top = self.top[side]
        lines = [f"Pong. You control the {side} paddle ({PADDLE} cells tall). "
                 "Rows are counted from the top.",
                 f"Your paddle covers rows {top}-{top + PADDLE - 1}."]
        if towards:
            lines.append(f"The ball is coming towards you and lands at row {y} in {n} steps.")
        else:
            lines.append(f"The ball is moving away; it will probably come back at row {y} "
                         f"in {n} steps.")
        lines.append("Possible moves:")
        for m in MOVES:
            need, ok = facts[m]
            if need == 0:
                txt = "the ball lands on the paddle (catch)"
            elif ok:
                txt = f"paddle ends {need} cells from the landing spot, still reachable in time"
            else:
                txt = f"paddle ends {need} cells from the landing spot, TOO FAR, will miss"
            lines.append(f"'{LABELS[m]}': {txt}.")
        return "\n".join(lines)


def move_rank(fact):
    need, ok = fact
    return (ok, -need)


class PongBrain:
    QUESTION = ("Which move makes sure the paddle catches the ball? "
                "If moves are equally good, prefer to stay.")

    def __init__(self, backend, side, stay_bonus=3.0):
        self.client = backend.client()
        self.side = side
        self.stay_bonus = stay_bonus
        self.last_probs = {}

    def decide(self, game):
        facts, info = game.move_facts(self.side)
        scores = self.client.logprobs(game.describe(self.side, facts, info), self.QUESTION,
                                      [LABELS[m] for m in MOVES])
        scores = dict(zip(MOVES, scores))
        # Voorkeur voor blijven staan alleen als dat even goed is als de beste zet.
        if move_rank(facts["stay"]) == max(move_rank(f) for f in facts.values()):
            scores["stay"] += self.stay_bonus
        self.last_probs = dict(zip(MOVES, softmax(list(scores.values()))))
        return max(self.last_probs, key=self.last_probs.get)


class PongMode:
    """Pong als modus in de gedeelde pygame-loop (zie snake.py)."""

    title = "Pong"

    UP_KEYS = (pygame.K_w, pygame.K_UP)
    DOWN_KEYS = (pygame.K_s, pygame.K_DOWN)

    def __init__(self, backend, w, h, cell, stay_bonus=3.0, human=False):
        self.game = Pong(w, h)
        self.cell = cell
        self.brains = {s: PongBrain(backend, s, stay_bonus) for s in ("left", "right")}
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.human = human  # True = linkerbatje wordt door de speler bestuurd
        self.pending = None  # korte tik tussen twee stappen niet missen

    def active(self):
        return True

    def on_key(self, key):
        """Toetsen voor deze modus; True = afgehandeld."""
        if key == pygame.K_h:
            self.human = not self.human
            self.pending = None
            return True
        if self.human and key in self.UP_KEYS + self.DOWN_KEYS:
            self.pending = "up" if key in self.UP_KEYS else "down"
            return True
        return False

    def human_move(self):
        if self.pending:
            move, self.pending = self.pending, None
            return move
        held = pygame.key.get_pressed()
        if any(held[k] for k in self.UP_KEYS):
            return "up"
        if any(held[k] for k in self.DOWN_KEYS):
            return "down"
        return "stay"

    def update(self):
        ai = ["right"] if self.human else ["left", "right"]
        futures = {s: self.pool.submit(self.brains[s].decide, self.game) for s in ai}
        moves = {s: f.result() for s, f in futures.items()}
        if self.human:
            moves["left"] = self.human_move()
        self.game.step(moves)

    def on_space(self):
        self.game.reset()

    def draw(self, screen, font, small, status, panel_top):
        g, c = self.game, self.cell
        for x in range(1, g.w - 1):
            for y in range(g.h):
                pygame.draw.rect(screen, (28, 28, 36), (x * c, y * c, c - 1, c - 1))
        for y in range(0, g.h, 2):
            pygame.draw.rect(screen, (60, 60, 75), (g.w * c // 2 - 2, y * c + 4, 4, c - 8))
        colors = {"left": (90, 160, 255), "right": (255, 170, 70)}
        for side, col in colors.items():
            x = g.column(side)
            pygame.draw.rect(screen, col, (x * c + 4, g.top[side] * c + 2, c - 8,
                                           PADDLE * c - 4), border_radius=6)
        bx, by = g.ball
        pygame.draw.circle(screen, (240, 240, 240), (bx * c + c // 2, by * c + c // 2),
                           c // 2 - 4)
        left = "Links (jij)" if self.human else "Links"
        screen.blit(font.render(f"{left} {g.score['left']} - {g.score['right']} Rechts"
                                f"   Rally: {g.rally}", True, (230, 230, 230)),
                    (10, panel_top + 8))
        screen.blit(small.render(status, True, (150, 150, 170)), (10, panel_top + 38))
        arrows = {"up": "↑", "stay": "=", "down": "↓"}

        def probs(s):
            return " ".join(f"{arrows[m]}{p:.0%}" for m, p in self.brains[s].last_probs.items())

        if self.human:
            line = f"L: jij (W/S of ↑/↓)   R: {probs('right')}   H = AI"
        else:
            line = f"L: {probs('left')}   R: {probs('right')}   H = zelf"
        screen.blit(small.render(line, True, (150, 200, 255)), (10, panel_top + 58))
