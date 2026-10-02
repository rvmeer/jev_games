"""Snake, pong en tetris bestuurd door een lokaal Jev-style beslismodel (MLX, CUDA of Ollama).

Het model krijgt de spelsituatie (state) + een vraag + opties en geeft per
optie een kans terug. De snake draait links/rechts t.o.v. zijn kop
(of gaat rechtdoor). Met Tab wissel je tussen de spellen.

Start: python snake.py   (zie --help voor opties)
"""

import argparse
import math
import random
from collections import deque

import pygame

from jev import BACKENDS, DEFAULT_MODEL, load_backend, softmax
from pong import PongMode
from tetris import TetrisMode

GRID_W, GRID_H = 20, 20
CELL = 28
PANEL_H = 90

UP, RIGHT, DOWN, LEFT = (0, -1), (1, 0), (0, 1), (-1, 0)
DIRS = [UP, RIGHT, DOWN, LEFT]  # klokwijs


def turn(direction, move):
    i = DIRS.index(direction)
    if move == "left":
        return DIRS[(i - 1) % 4]
    if move == "right":
        return DIRS[(i + 1) % 4]
    return direction


class Game:
    def __init__(self, w=GRID_W, h=GRID_H, rng=None):
        self.w, self.h = w, h
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        cx, cy = self.w // 2, self.h // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = RIGHT
        self.alive = True
        self.score = 0
        self.steps = 0
        self.place_candy()

    def place_candy(self):
        free = [(x, y) for x in range(self.w) for y in range(self.h)
                if (x, y) not in self.snake]
        self.candy = self.rng.choice(free) if free else None

    def blocked(self, cell, ignore_tail=True):
        x, y = cell
        if not (0 <= x < self.w and 0 <= y < self.h):
            return True
        body = self.snake[:-1] if ignore_tail else self.snake
        return cell in body

    def inside(self, c):
        return 0 <= c[0] < self.w and 0 <= c[1] < self.h

    def path(self, snake, goal):
        """Kortste pad (BFS) van de kop naar goal; de staart telt als vrij
        omdat die meeschuift. Geeft lijst cellen (zonder kop) of None."""
        start = snake[0]
        obstacles = set(snake[1:-1])
        prev = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == goal:
                out = []
                while cur != start:
                    out.append(cur)
                    cur = prev[cur]
                return out[::-1]
            for dx, dy in DIRS:
                c = (cur[0] + dx, cur[1] + dy)
                if c not in prev and self.inside(c) and c not in obstacles:
                    prev[c] = cur
                    queue.append(c)
        return None

    def advance(self, snake, cell):
        new = [cell] + snake
        if cell != self.candy:
            new.pop()
        return new

    def can_reach_tail(self, snake):
        """Kan de kop zijn eigen staart bereiken? Dan zit hij nooit opgesloten."""
        if len(snake) < 3:
            return True
        return self.path(snake, snake[-1]) is not None

    def lookahead(self, nxt):
        """Analyse van één zet: (bij staart kunnen, stappen naar snoep, eten veilig)."""
        snake1 = self.advance(self.snake, nxt)
        tail_ok = self.can_reach_tail(snake1)
        if nxt == self.candy:
            return tail_ok, 1, tail_ok
        route = self.path(snake1, self.candy)
        if route is None:
            return tail_ok, None, False
        sim = snake1
        for cell in route:
            sim = self.advance(sim, cell)
        return tail_ok, len(route) + 1, self.can_reach_tail(sim)

    def step(self, move):
        if not self.alive:
            return
        self.direction = turn(self.direction, move)
        hx, hy = self.snake[0]
        new = (hx + self.direction[0], hy + self.direction[1])
        self.steps += 1
        if self.blocked(new):
            self.alive = False
            return
        self.snake.insert(0, new)
        if new == self.candy:
            self.score += 1  # staart blijft staan -> 1 unit langer
            self.place_candy()
        else:
            self.snake.pop()

    def describe(self, moves, facts=None):
        """Spelsituatie als tekst, relatief t.o.v. de kop van de snake."""
        d = self.direction
        right_d = turn(d, "right")
        hx, hy = self.snake[0]
        cx, cy = self.candy
        dx, dy = cx - hx, cy - hy
        # projectie van candy-offset op (vooruit, rechts) assen van de kop
        ahead = dx * d[0] + dy * d[1]
        right = dx * right_d[0] + dy * right_d[1]

        def side(n, pos, neg):
            if n == 0:
                return None
            return f"{abs(n)} cell{'s' if abs(n) != 1 else ''} {pos if n > 0 else neg}"

        parts = [p for p in (side(ahead, "ahead", "behind"),
                             side(right, "to the right", "to the left")) if p]
        candy_txt = " and ".join(parts)

        lines = ["Snake game on a grid. Positions are relative to the snake's head.",
                 f"The candy is {candy_txt}.", "Possible moves:"]
        facts = facts or self.move_facts(moves)
        for m in moves:
            label = JevBrain.LABELS[m]
            status, dist, eat_safe = facts[m]
            if status == "crash":
                lines.append(f"'{label}': CRASHES immediately (deadly).")
                continue
            if status == "trap":
                lines.append(f"'{label}': TRAP, the snake gets locked in and dies soon.")
                continue
            if dist is None:
                food = "candy not reachable from there"
            else:
                food = (f"candy in {dist} step{'s' if dist != 1 else ''}, eating it that way is "
                        + ("safe" if eat_safe else "DANGEROUS (gets locked in afterwards)"))
            lines.append(f"'{label}': safe, {food}.")
        return "\n".join(lines)

    def move_facts(self, moves):
        """Per zet: (status 'crash'/'trap'/'safe', stappen naar snoep of None, eten veilig)."""
        hx, hy = self.snake[0]
        facts = {}
        for m in moves:
            nd = turn(self.direction, m)
            nxt = (hx + nd[0], hy + nd[1])
            if self.blocked(nxt):
                facts[m] = ("crash", None, False)
                continue
            tail_ok, dist, eat_safe = self.lookahead(nxt)
            facts[m] = ("safe" if tail_ok else "trap", dist, eat_safe)
        return facts


def move_rank(fact):
    """Hoe goed is een zet volgens de feiten (hoger = beter)."""
    status, dist, eat_safe = fact
    return ({"crash": 0, "trap": 1, "safe": 2}[status], eat_safe,
            -dist if dist is not None else -math.inf)


class JevBrain:
    """Vraagt het Jev-style beslismodel om een snake-zet (Choice-primitive)."""

    LABELS = {"left": "turn left", "straight": "go straight", "right": "turn right"}

    def __init__(self, backend, moves, straight_bonus=0.0, prefer_hint=True):
        self.client = backend.client()
        self.moves = moves
        self.straight_bonus = straight_bonus  # extra logit voor 'straight' (beleid in code)
        self.prefer_hint = prefer_hint and "straight" in moves
        self.last_probs = {}

    def question(self):
        hint = " If moves are equally good, prefer going straight." if self.prefer_hint else ""
        return ("Which move avoids crashes and traps and gets the candy safely "
                f"in the fewest steps?{hint}")

    def prompt(self, state):
        return self.client.prompt(state, self.question(), [self.LABELS[m] for m in self.moves])

    def decide(self, game):
        facts = game.move_facts(self.moves)
        scores = dict(zip(self.moves, self.client.logprobs(
            game.describe(self.moves, facts), self.question(),
            [self.LABELS[m] for m in self.moves])))
        # Voorkeur voor rechtdoor alleen als dat volgens de feiten even goed is als de beste zet.
        if ("straight" in scores and scores["straight"] > -math.inf
                and move_rank(facts["straight"]) == max(move_rank(f) for f in facts.values())):
            scores["straight"] += self.straight_bonus
        self.last_probs = dict(zip(self.moves, softmax(list(scores.values()))))
        return max(self.last_probs, key=self.last_probs.get)


class SnakeMode:
    """Snake als modus in de gedeelde pygame-loop."""

    title = "Snake"

    def __init__(self, backend, moves, straight_bonus):
        self.game = Game()
        self.brain = JevBrain(backend, moves, straight_bonus=straight_bonus)

    def active(self):
        return self.game.alive

    def update(self):
        self.game.step(self.brain.decide(self.game))

    def on_space(self):
        if not self.game.alive:
            self.game.reset()

    def on_key(self, key):
        return False

    def draw(self, screen, font, small, status, panel_top):
        game = self.game
        for x in range(game.w):
            for y in range(game.h):
                pygame.draw.rect(screen, (28, 28, 36), (x * CELL, y * CELL, CELL - 1, CELL - 1))
        if game.candy:
            cx, cy = game.candy
            pygame.draw.circle(screen, (240, 70, 120),
                               (cx * CELL + CELL // 2, cy * CELL + CELL // 2), CELL // 2 - 4)
        for i, (x, y) in enumerate(game.snake):
            col = (120, 230, 120) if i == 0 else (60, 170, 80)
            pygame.draw.rect(screen, col, (x * CELL + 1, y * CELL + 1, CELL - 3, CELL - 3),
                             border_radius=6)
        screen.blit(font.render(f"Score: {game.score}   Lengte: {len(game.snake)}", True,
                                (230, 230, 230)), (10, panel_top + 8))
        screen.blit(small.render(status, True, (150, 150, 170)), (10, panel_top + 38))
        probs = "  ".join(f"{m}: {p:.0%}" for m, p in self.brain.last_probs.items())
        screen.blit(small.render(probs, True, (150, 200, 255)), (10, panel_top + 58))
        if not game.alive:
            msg = font.render("GAME OVER - spatie = opnieuw", True, (255, 90, 90))
            screen.blit(msg, msg.get_rect(center=(game.w * CELL // 2, panel_top // 2)))


def main():
    ap = argparse.ArgumentParser(description="Snake, pong en tetris bestuurd door een lokaal "
                                             "Jev-style model (Tab = wisselen)")
    ap.add_argument("--backend", choices=BACKENDS, default="auto",
                    help="waar het model draait (auto = mlx op Apple Silicon, anders cuda, "
                         "anders ollama)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama-modelnaam (alleen bij ollama)")
    ap.add_argument("--speed", type=float, default=0,
                    help="max stappen per seconde, 0 = zo snel als het model kan")
    ap.add_argument("--straight-bonus", type=float, default=3.0,
                    help="snake: voorkeur voor rechtdoor bij gelijkwaardige zetten (extra logit)")
    ap.add_argument("--stay-bonus", type=float, default=3.0,
                    help="pong: voorkeur voor stilstaan bij gelijkwaardige zetten (extra logit)")
    ap.add_argument("--no-straight", action="store_true",
                    help="snake: alleen links/rechts, nooit rechtdoor")
    ap.add_argument("--pong", action="store_true", help="start in pong")
    ap.add_argument("--tetris", action="store_true", help="start in tetris")
    ap.add_argument("--human", action="store_true",
                    help="start in pong en speel zelf het linkerbatje (H = wisselen)")
    args = ap.parse_args()

    backend = load_backend(args.backend, args.model)
    moves = ["left", "right"] if args.no_straight else ["left", "straight", "right"]
    modes = [SnakeMode(backend, moves, args.straight_bonus),
             PongMode(backend, GRID_W, GRID_H, CELL, args.stay_bonus, human=args.human),
             TetrisMode(backend, GRID_W, GRID_H, CELL)]
    current = 2 if args.tetris else 1 if args.pong or args.human else 0

    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL, GRID_H * CELL + PANEL_H))
    font = pygame.font.SysFont("menlo", 20, bold=True)
    small = pygame.font.SysFont("menlo", 15)
    clock = pygame.time.Clock()
    speed = args.speed
    steps = [0.0] * 10  # tijdstempels van de laatste stappen

    while True:
        mode = modes[current]
        pygame.display.set_caption(f"{mode.title} - Jev-style beslismodel via {backend.name} "
                                   "(Tab = wisselen)")
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN:
                if mode.on_key(ev.key):
                    pass
                elif ev.key == pygame.K_TAB:
                    current = (current + 1) % len(modes)
                    mode = modes[current]
                    steps = [0.0] * 10
                elif ev.key == pygame.K_SPACE:
                    mode.on_space()
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_UP):
                    speed = 0 if speed <= 0 or speed >= 40 else speed + 5
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_DOWN):
                    speed = 40 if speed <= 0 else max(1, speed - 5)
        if mode.active():
            mode.update()
            steps = steps[1:] + [pygame.time.get_ticks() / 1000]
        span = steps[-1] - steps[0]
        actual = (len(steps) - 1) / span if mode.active() and steps[0] and span > 0 else 0
        limit = "max" if speed <= 0 else f"{speed:g}"
        status = f"{actual:.1f} stappen/s (limiet {limit}, +/-)   Tab = wisselen"
        screen.fill((18, 18, 24))
        pygame.draw.rect(screen, (10, 10, 14), (0, GRID_H * CELL, GRID_W * CELL, PANEL_H))
        mode.draw(screen, font, small, status, GRID_H * CELL)
        pygame.display.flip()
        clock.tick(speed if speed > 0 else 0)


if __name__ == "__main__":
    main()
