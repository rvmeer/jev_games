"""Tetris, bestuurd door het Jev-style beslismodel.

Speelveld = hetzelfde grid als snake/pong. Stenen zijn de 7 standaard
tetromino's, gedefinieerd in een 5x5 vak en gedraaid rond het midden (2, 2).
Per stap kiest de AI: links, rechts, linksom draaien, rechtsom draaien of
niets doen; daarna valt de steen 1 rij.
"""

import random
from collections import deque

import pygame

from jev import softmax

# Cellen (x, y) binnen het 5x5 vak.
SHAPES = {
    "I": [(1, 2), (2, 2), (3, 2), (4, 2)],
    "O": [(2, 2), (3, 2), (2, 3), (3, 3)],
    "T": [(1, 2), (2, 2), (3, 2), (2, 1)],
    "S": [(2, 2), (3, 2), (1, 3), (2, 3)],
    "Z": [(1, 2), (2, 2), (2, 3), (3, 3)],
    "J": [(1, 1), (1, 2), (2, 2), (3, 2)],
    "L": [(3, 1), (1, 2), (2, 2), (3, 2)],
}
COLORS = {"I": (80, 210, 230), "O": (240, 210, 70), "T": (180, 100, 230),
          "S": (110, 210, 100), "Z": (230, 80, 80), "J": (80, 120, 230), "L": (240, 150, 60)}

ACTIONS = ["left", "right", "ccw", "cw", "none"]
LABELS = {"left": "move left", "right": "move right", "ccw": "rotate counterclockwise",
          "cw": "rotate clockwise", "none": "do nothing"}
SYMBOLS = {"left": "←", "right": "→", "ccw": "↺", "cw": "↻", "none": "·"}

# Gewichten van de bekende heuristiek van Yiyuan Lee.
W_HEIGHT, W_LINES, W_HOLES, W_BUMP = -0.510066, 0.760666, -0.35663, -0.184483


def rotate(cells, r):
    for _ in range(r % 4):
        cells = [(4 - y, x) for x, y in cells]  # rechtsom rond (2, 2)
    return cells


def shape_cells(kind, r):
    return SHAPES[kind] if kind == "O" else rotate(SHAPES[kind], r)


class Tetris:
    def __init__(self, w, h, rng=None):
        self.w, self.h = w, h
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        self.board = {}  # (x, y) -> kind
        self.lines = 0
        self.score = 0
        self.pieces = 0
        self.over = False
        self.next_kind = self.rng.choice(list(SHAPES))
        self.spawn()

    def spawn(self):
        self.kind, self.next_kind = self.next_kind, self.rng.choice(list(SHAPES))
        self.rot = 0
        self.px = self.w // 2 - 2
        self.py = -min(y for _, y in shape_cells(self.kind, 0))
        self.pieces += 1
        if not self.fits(self.rot, self.px, self.py):
            self.over = True

    def cells(self, r, x, y, kind=None):
        return [(x + cx, y + cy) for cx, cy in shape_cells(kind or self.kind, r)]

    def fits(self, r, x, y, board=None):
        board = self.board if board is None else board
        return all(0 <= cx < self.w and cy < self.h and (cx, cy) not in board
                   for cx, cy in self.cells(r, x, y))

    def apply(self, state, action):
        """Actie op (r, x, y); None als de actie niet kan."""
        if self.kind == "O" and action in ("cw", "ccw"):
            return None  # draaien van O verandert niets
        r, x, y = state
        new = {"left": (r, x - 1, y), "right": (r, x + 1, y), "ccw": ((r - 1) % 4, x, y),
               "cw": ((r + 1) % 4, x, y), "none": state}[action]
        return new if self.fits(*new) else None

    def step(self, action):
        if self.over:
            return
        state = self.apply((self.rot, self.px, self.py), action) or (self.rot, self.px, self.py)
        r, x, y = state
        if self.fits(r, x, y + 1):
            self.rot, self.px, self.py = r, x, y + 1
            return
        for c in self.cells(r, x, y):
            self.board[c] = self.kind
        if any(cy < 0 for _, cy in self.cells(r, x, y)):
            self.over = True
            return
        cleared = self.clear_lines()
        self.lines += cleared
        self.score += [0, 100, 300, 500, 800][cleared]
        self.spawn()

    def clear_lines(self):
        full = [y for y in range(self.h) if all((x, y) in self.board for x in range(self.w))]
        if not full:
            return 0
        new = {}
        for (x, y), k in self.board.items():
            if y in full:
                continue
            new[(x, y + sum(1 for f in full if f > y))] = k
        self.board = new
        return len(full)

    # --- lookahead ---------------------------------------------------------

    def evaluate(self, cells):
        """Bord na plaatsen van cells: (heuristische score, lijnen, gaten)."""
        board = set(self.board) | set(cells)
        full = [y for y in range(self.h) if all((x, y) in board for x in range(self.w))]
        if full:
            board = {(x, y + sum(1 for f in full if f > y)) for x, y in board if y not in full}
        heights, holes = [], 0
        for x in range(self.w):
            col = [y for (cx, y) in board if cx == x]
            top = min(col) if col else self.h
            heights.append(self.h - top)
            holes += sum(1 for y in range(top, self.h) if (x, y) not in board)
        bump = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
        score = W_HEIGHT * sum(heights) + W_LINES * len(full) + W_HOLES * holes + W_BUMP * bump
        return score, len(full), holes

    def holes(self):
        n = 0
        for x in range(self.w):
            col = [y for (cx, y) in self.board if cx == x]
            if col:
                n += sum(1 for y in range(min(col), self.h) if (x, y) not in self.board)
        return n

    def best_from(self, start, cache):
        """0-1 BFS over (r, x, y) met de echte stapregels (actie, dan vallen).
        Geeft (score, -zetten, lijnen, gaten, landing) van de beste bereikbare landing."""
        dist = {start: 0}
        dq = deque([start])
        best = None
        while dq:
            s = dq.popleft()
            d = dist[s]
            for a in ACTIONS:
                t = self.apply(s, a)
                if t is None:
                    continue
                cost = d + (a != "none")
                r, x, y = t
                if self.fits(r, x, y + 1):
                    nxt = (r, x, y + 1)
                    if cost < dist.get(nxt, 1 << 30):
                        dist[nxt] = cost
                        (dq.appendleft if a == "none" else dq.append)(nxt)
                else:
                    key = frozenset(self.cells(r, x, y))
                    if key not in cache:
                        cache[key] = self.evaluate(key)
                    score, lines, holes = cache[key]
                    cand = (score, -cost, lines, holes, t)
                    if best is None or cand[:2] > best[:2]:
                        best = cand
        return best

    def action_facts(self):
        """Per actie: None (kan niet) of (score, -extra zetten, lijnen, gaten, landing)."""
        cache, facts = {}, {}
        cur = (self.rot, self.px, self.py)
        for a in ACTIONS:
            t = self.apply(cur, a)
            if t is None:
                facts[a] = None
                continue
            r, x, y = t
            if not self.fits(r, x, y + 1):
                key = frozenset(self.cells(r, x, y))
                score, lines, holes = self.evaluate(key)
                facts[a] = (score, 0, lines, holes, t)
            else:
                facts[a] = self.best_from((r, x, y + 1), cache)
        return facts

    def describe(self, facts):
        best = max(f[0] for f in facts.values() if f)
        fastest = max(f[1] for f in facts.values() if f and f[0] >= best - 1e-9)
        base_holes = self.holes()
        top = min((y for _, y in self.board), default=self.h)
        lines = [f"Tetris on a {self.w}x{self.h} grid. The falling piece is a {self.kind}-piece.",
                 f"The stack is {self.h - top} rows high. Each step the piece falls one row "
                 "after your action.",
                 "Possible actions:"]
        for a in ACTIONS:
            f = facts[a]
            if f is None:
                lines.append(f"'{LABELS[a]}': not possible.")
                continue
            score, neg_moves, n_lines, holes, _ = f
            new_holes = max(0, holes - base_holes)
            moves = f"{-neg_moves} more move{'s' if neg_moves != -1 else ''}"
            if score < best - 1e-9:
                lines.append(f"'{LABELS[a]}': leads to a WORSE landing "
                             f"({n_lines} lines cleared, {new_holes} new holes).")
            elif neg_moves == fastest:
                lines.append(f"'{LABELS[a]}': keeps the BEST landing and is the FASTEST way "
                             f"there ({moves}).")
            else:
                lines.append(f"'{LABELS[a]}': keeps the BEST landing but is slower ({moves}).")
        return "\n".join(lines)


def action_rank(fact):
    return (-1e9, 0) if fact is None else (fact[0], fact[1])


class TetrisBrain:
    QUESTION = "Which action keeps the BEST landing and is the FASTEST way there?"

    def __init__(self, backend, none_bonus=0.0):
        self.client = backend.client()
        self.none_bonus = none_bonus
        self.last_probs = {}
        self.target = None

    def decide(self, game):
        facts = game.action_facts()
        best = max(facts.values(), key=action_rank)
        self.target = best[4] if best else None
        scores = dict(zip(ACTIONS, self.client.logprobs(
            game.describe(facts), self.QUESTION, [LABELS[a] for a in ACTIONS])))
        if action_rank(facts["none"]) == action_rank(best):
            scores["none"] += self.none_bonus
        self.last_probs = dict(zip(ACTIONS, softmax(list(scores.values()))))
        return max(self.last_probs, key=self.last_probs.get)


class TetrisMode:
    """Tetris als modus in de gedeelde pygame-loop (zie snake.py)."""

    title = "Tetris"

    def __init__(self, backend, w, h, cell):
        self.game = Tetris(w, h)
        self.cell = cell
        self.brain = TetrisBrain(backend)

    def active(self):
        return not self.game.over

    def update(self):
        self.game.step(self.brain.decide(self.game))

    def on_space(self):
        if self.game.over:
            self.game.reset()
            self.brain.target = None

    def on_key(self, key):
        return False

    def draw(self, screen, font, small, status, panel_top):
        g, c = self.game, self.cell
        for x in range(g.w):
            for y in range(g.h):
                pygame.draw.rect(screen, (28, 28, 36), (x * c, y * c, c - 1, c - 1))
        for (x, y), k in g.board.items():
            pygame.draw.rect(screen, COLORS[k], (x * c + 1, y * c + 1, c - 3, c - 3),
                             border_radius=4)
        if not g.over:
            if self.brain.target:
                for x, y in g.cells(*self.brain.target):
                    pygame.draw.rect(screen, (120, 120, 140), (x * c + 2, y * c + 2, c - 5, c - 5),
                                     width=2, border_radius=4)
            for x, y in g.cells(g.rot, g.px, g.py):
                if y >= 0:
                    pygame.draw.rect(screen, COLORS[g.kind], (x * c + 1, y * c + 1, c - 3, c - 3),
                                     border_radius=4)
        screen.blit(font.render(f"Lijnen: {g.lines}   Score: {g.score}   Volgende: {g.next_kind}",
                                True, (230, 230, 230)), (10, panel_top + 8))
        screen.blit(small.render(status, True, (150, 150, 170)), (10, panel_top + 38))
        probs = "  ".join(f"{SYMBOLS[a]} {p:.0%}" for a, p in self.brain.last_probs.items())
        screen.blit(small.render(probs, True, (150, 200, 255)), (10, panel_top + 58))
        if g.over:
            msg = font.render("GAME OVER - spatie = opnieuw", True, (255, 90, 90))
            screen.blit(msg, msg.get_rect(center=(g.w * c // 2, panel_top // 2)))
