# CLAUDE.md

Richtlijnen voor AI-assistenten die in deze repository werken.

## Project

Snake, Pong en Tetris (pygame-ce) waarin een lokaal **Jev-style beslismodel** elke zet kiest. Het model draait via MLX, PyTorch/CUDA of Ollama. Wisselen tussen de spellen gaat met Tab, in één venster en op hetzelfde 20×20-grid. Taal van code-commentaar, UI-teksten en docs: **Nederlands**. Prompts naar het model: **Engels**, want het model is Engelstalig getraind.

Bestanden:
- `jev.py`: backends en `load_backend(name, model)`. Elke backend heeft `name` en `client()`; een client heeft `logprobs(state, question, options)` en `prompt(...)`.
  - `OllamaBackend`: `client()` geeft per brein een nieuwe `JevClient` (eigen HTTP-verbinding). `ensure_ollama_model` downloadt de GGUF en doet `ollama create` als het standaardmodel ontbreekt.
  - `MLXBackend` / `TorchBackend` (`--backend cuda`): het model draait in het proces, `client()` geeft de backend zelf terug, en een lock zorgt dat threads één voor één gaan. Logits = embedding-rijen van ` A`, ` B`, ... × laatste hidden state. Beide laden `HF_REPO` (MLX-bf16). `TorchBackend._from_mlx` zet die om: conv1d-assen terug en `-1` op de RMSNorm-gewichten (niet `linear_attn.norm`).
  - `auto`: MLX (Apple Silicon + `mlx-lm`), anders CUDA, anders Ollama.
- `snake.py`: **startpunt**. Bevat de snake-logica (`Game`, `JevBrain`, `SnakeMode`) en `main()` met de gedeelde pygame-loop en het wisselen tussen modi.
- `pong.py`: `Pong` (fysica en lookahead), `PongBrain` (één per batje), `PongMode`.
- `tetris.py`: `Tetris` (bord, stenen in een 5×5 vak, BFS-lookahead), `TetrisBrain`, `TetrisMode`.
- `pong.py` en `tetris.py` importeren niets uit `snake.py`, zodat er geen circulaire import ontstaat.

Een **modus** (`SnakeMode`, `PongMode`) heeft: `title`, `active()`, `update()`, `on_space()`, `on_key(key)` en `draw(screen, font, small, status, panel_top)`. `on_key` krijgt elke toets als eerste; geeft hij `True` terug, dan doet de hoofdloop er niets meer mee. Zo neemt pong in mens-modus de pijltjes over. Een nieuw spel voeg je toe door zo'n klasse te maken en die in `main()` aan `modes` toe te voegen.

## Commando's

```bash
source .venv/bin/activate          # venv staat in .venv (niet committen)
pip install -r requirements.txt    # pygame-ce, plus mlx-lm op Apple Silicon
pip install -r requirements-cuda.txt  # NVIDIA: torch + transformers
python snake.py                    # starten (zie --help voor opties)
python -m py_compile snake.py      # snelle syntaxcheck
ollama ps                          # check of jev-style-2b geladen is
```

Er is geen testsuite. Valideer gedragswijzigingen **headless** door `Game` en `JevBrain` direct te importeren en een paar potjes te simuleren met vaste seeds (`Game(rng=random.Random(seed))`). Meet: score, aantal stappen, dood/levend, bochtpercentage. Een GUI-check doe je door `python snake.py` een paar seconden te draaien en stderr op tracebacks te controleren.

## Architectuur (`snake.py`)

- `turn(direction, move)`: relatieve zet → absolute richting. `DIRS` staat klokwijs.
- `Game`: spelregels en lookahead, zonder pygame-afhankelijkheid (dus headless te testen).
  - `step(move)`: beweegt de snake. Eet hij het snoepje, dan blijft de staart staan en wordt hij 1 langer.
  - `path()`: BFS waarbij de staart als vrij telt. `can_reach_tail()` is de anti-val-check. `lookahead()` simuleert de route naar het snoepje en checkt daarna opnieuw de staart.
  - `move_facts(moves)`: per zet `(status, dist, eat_safe)` met status `crash`/`trap`/`safe`.
  - `describe(moves, facts)`: de state-tekst voor het model.
- `move_rank(fact)`: ordening van zetten op basis van de feiten. Wordt gebruikt om te bepalen of rechtdoor "even goed" is.
- `JevBrain`: bouwt de snake-vraag en gebruikt `backend.client()`. Bij Ollama is dat `JevClient` (`/api/generate`, `raw: true`, `num_predict: 1`, `logprobs`, `top_logprobs: 20`, persistente `http.client`-verbinding). De `straight_bonus` wordt **alleen** opgeteld als `move_rank(straight)` gelijk is aan het maximum.
- `SnakeMode` / `main()`: de pygame-loop met een instelbare snelheid (`+`/`-`), Tab om van modus te wisselen en een paneel met score, werkelijke stappen/s en de kansen.

## Architectuur (`pong.py`)

- `Pong.step(moves)`: eerst bewegen de batjes, dan de bal. Een raak schot zet de bal terug in de kolom naast het batje. Een raakpunt op de rand verandert `vy`. Bij een misser krijgt de ander een punt en volgt een nieuwe opslag.
- `intercept(side)`: simuleert de bal tot hij de kolom van `side` bereikt. Gaat de bal van je af, dan wordt een perfecte terugslag van de tegenstander aangenomen.
- `move_facts(side)`: per zet `(need, reachable)`. `PongBrain` geeft `stay` een bonus als die even goed is als de beste zet.
- `PongMode.update()` roept de AI-batjes parallel aan (ThreadPoolExecutor; bij Ollama elk met een eigen `JevClient`/verbinding, bij MLX/CUDA via de lock na elkaar). Met `human=True` (toets H of `--human`) bestuurt de speler het linkerbatje: `pending` vangt een korte tik op, `pygame.key.get_pressed()` zorgt voor beweging bij ingedrukt houden. Alleen het rechterbatje vraagt dan het model.

## Architectuur (`tetris.py`)

- `SHAPES`: cellen in een 5×5 vak. `rotate()` draait rechtsom rond (2, 2), en O draait niet. `apply(state, action)` geeft `None` terug als de actie niet past. Draaien van O telt als niet mogelijk.
- `step(action)`: eerst de actie (bij een ongeldige actie gebeurt niets), dan zwaartekracht. Kan de steen niet verder vallen, dan wordt hij vastgezet, worden volle lijnen weggehaald en verschijnt de volgende steen. Game over als een nieuwe steen niet past of als een steen boven het veld vastloopt.
- `best_from()`: 0-1 BFS over `(r, x, y)` met de echte stapregels. Een zet kost 1, niets doen kost 0. Geeft de beste landing terug: (score, -zetten, ...). `evaluate()` gebruikt de heuristiek van Yiyuan Lee.
- `action_facts()` → `describe()`: labelt elke actie in code als BEST/FASTEST, BEST maar langzamer, WORSE of niet mogelijk. Met ruwe getallen koos het model maar in 45% van de stappen de beste actie, met labels in 100%. Laat de vergelijking dus in code.

## Model

- Ollama-naam: `jev-style-2b`, gemaakt met `Modelfile` uit `Jev-Style-Qwen3.5-2B-Decision-Q8_0.gguf`.
- Het `Modelfile` gebruikt een **pass-through template** (`{{ .Prompt }}`). Verander dat niet: het model verwacht exact het formaat `You are a decision function ... [State] ... [Question] ... [Options] A. ... Answer:`.
- Het is **geen chatmodel**. Gewone chatprompts leveren onzin op. Lees de beslissing altijd uit de logprobs van de optieletter (` A`, ` B`, ...), niet uit gegenereerde tekst.
- `ollama pull hf.co/...` werkt niet voor dit model ("blocked redirect"). `jev.py` downloadt de GGUF daarom zelf en roept `ollama create` aan.
- MLX/CUDA gebruiken `chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16` (dezelfde v1-gewichten). De kansen wijken < 2 procentpunt af van Ollama Q8_0. Controleer dat opnieuw als je de laadcode verandert: vergelijk dezelfde prompts op alle backends (`TorchBackend(device="cpu")` of `"mps"` werkt ook zonder GPU).

## Lessen uit eerdere experimenten

- Doodgaan komt bijna altijd door **insluiten**, niet door dom kiezen. Verbeter de state (lookahead), niet de modelgrootte. v2 van het model gaf geen winst.
- Een **onvoorwaardelijke** bonus voor rechtdoor maakt de snake blind voor het snoepje. Houd de bonus voorwaardelijk.
- Alleen links/rechts (`--no-straight`) presteert veel slechter. Rechtdoor is daarom standaard een optie.
- Volgens de Jev-filosofie: geef het model atomaire, feitelijke informatie en leg beleid (drempels, voorkeuren) vast **in code**, niet in steeds langere prompts.

## Conventies

- Houd `Game`, `Pong` en `Tetris` vrij van pygame en netwerk, zodat ze headless testbaar blijven.
- Wisselen tussen modi test je headless met `SDL_VIDEODRIVER=dummy`, door `pygame.event.get` te patchen zodat die Tab- en Esc-events teruggeeft.
- Draaien er tegelijk andere `snake.py`-instanties, dan delen ze Ollama en lijken metingen trager. Check dat met `ps` voordat je snelheid meet.
- Commentaar alleen waar nodig, in het Nederlands.
- Werk `README.md` bij als opties, de prompt of het gedrag veranderen.
- Commit geen `.venv/` of `.gguf`-bestanden.
