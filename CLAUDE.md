# CLAUDE.md

Richtlijnen voor AI-assistenten die in deze repository werken.

## Project

Snake-game (pygame-ce) waarin een lokaal **Jev-style beslismodel** via Ollama elke zet kiest: `left` / `straight` / `right`, gezien vanaf de kop. Alles staat in één bestand: `snake.py`. Taal van code-commentaar, UI-teksten en docs: **Nederlands**. Prompts naar het model: **Engels**, want het model is Engelstalig getraind.

## Commando's

```bash
source .venv/bin/activate          # venv staat in .venv (niet committen)
pip install -r requirements.txt    # alleen pygame-ce; HTTP gaat via de stdlib
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
- `JevBrain`: bouwt de prompt, roept Ollama aan (`/api/generate`, `raw: true`, `num_predict: 1`, `logprobs`, `top_logprobs: 20`) via een persistente `http.client`-verbinding, en normaliseert de logprobs van de optieletters naar kansen. De `straight_bonus` wordt **alleen** opgeteld als `move_rank(straight)` gelijk is aan het maximum.
- `draw()` / `main()`: pygame-loop met een instelbare snelheid (`+`/`-`) en een paneel met score, werkelijke stappen/s en de kansen.

## Model

- Ollama-naam: `jev-style-2b`, gemaakt met `Modelfile` uit `Jev-Style-Qwen3.5-2B-Decision-Q8_0.gguf`.
- Het `Modelfile` gebruikt een **pass-through template** (`{{ .Prompt }}`). Verander dat niet: het model verwacht exact het formaat `You are a decision function ... [State] ... [Question] ... [Options] A. ... Answer:`.
- Het is **geen chatmodel**. Gewone chatprompts leveren onzin op. Lees de beslissing altijd uit de logprobs van de optieletter (` A`, ` B`, ...), niet uit gegenereerde tekst.
- `ollama pull hf.co/...` werkt niet voor dit model ("blocked redirect"). Download de GGUF met `curl -L` en gebruik `ollama create`.

## Lessen uit eerdere experimenten

- Doodgaan komt bijna altijd door **insluiten**, niet door dom kiezen. Verbeter de state (lookahead), niet de modelgrootte. v2 van het model gaf geen winst.
- Een **onvoorwaardelijke** bonus voor rechtdoor maakt de snake blind voor het snoepje. Houd de bonus voorwaardelijk.
- Alleen links/rechts (`--no-straight`) presteert veel slechter. Rechtdoor is daarom standaard een optie.
- Volgens de Jev-filosofie: geef het model atomaire, feitelijke informatie en leg beleid (drempels, voorkeuren) vast **in code**, niet in steeds langere prompts.

## Conventies

- Houd `Game` vrij van pygame en netwerk, zodat het headless testbaar blijft.
- Commentaar alleen waar nodig, in het Nederlands.
- Werk `README.md` bij als opties, de prompt of het gedrag veranderen.
- Commit geen `.venv/` of `.gguf`-bestanden.
