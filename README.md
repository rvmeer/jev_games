# Snake met een lokaal Jev-model

Snake die bestuurd wordt door een lokaal **Jev-style beslismodel** via [Ollama](https://ollama.com). Er is geen menselijke speler: bij elke stap besluit het model of de snake linksaf, rechtdoor of rechtsaf gaat, gezien vanaf de kop.

## Wat is Jev?

[Jev](https://typesafe.ai) van TypeSafe AI is een *System One*-model. Het schrijft geen tekst. Je geeft het een **state**, een **vraag** en **opties**, en het geeft per optie een gekalibreerde **kans** terug. Jev zelf bestaat alleen als cloud-API. Dit project gebruikt daarom een open, lokale nabouw:

**[`chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF`](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF)** (Q8_0, ±2 GB, Apache-2.0). Dit model is niet verbonden aan TypeSafe AI.

## Hoe het werkt

1. **State**: de spelsituatie wordt in tekst beschreven, gezien vanaf de kop. Per mogelijke zet berekent een lookahead:
   - of de zet direct **dodelijk** is;
   - of het een **val** is: kan de kop daarna nog bij zijn eigen staart? Zo niet, dan raakt de snake ingesloten;
   - na hoeveel stappen het snoepje bereikt wordt en of het **veilig** is om het via die route te eten. Daarvoor wordt de route gesimuleerd en daarna opnieuw de staart-check gedaan.
2. **Vraag + opties**: `turn left` / `go straight` / `turn right`.
3. **Beslissing**: het model voorspelt één token, de optieletter. De logprobs van `A`/`B`/`C` worden omgerekend naar kansen. De kans van elke optie zie je onderin het scherm.
4. **Beleid in code**: als rechtdoor volgens de lookahead even goed is als de beste zet, krijgt het een bonus (`--straight-bonus`). Dat geeft rustige, rechte lijnen zonder dat de snake snoepjes laat liggen.

Er ligt steeds 1 snoepje op een willekeurige plek. Eet de snake het op, dan wordt hij 1 unit langer.

### Voorbeeldprompt

```
You are a decision function. Read the state, then answer the question by choosing exactly one option.

[State]
Snake game on a grid. Positions are relative to the snake's head.
The candy is 3 cells behind and 9 cells to the right.
Possible moves:
'turn left': safe, candy in 14 steps, eating it that way is safe.
'go straight': safe, candy in 14 steps, eating it that way is safe.
'turn right': safe, candy in 12 steps, eating it that way is safe.

[Question]
Which move avoids crashes and traps and gets the candy safely in the fewest steps? If moves are equally good, prefer going straight.

[Options]
A. turn left
B. go straight
C. turn right

Answer:
```

Een beslissing kost ongeveer 40–50 ms op een Apple Silicon Mac.

## Installatie

Vereisten: macOS/Linux, Python 3.10+, Ollama.

```bash
# 1. Model. Ollama kan de redirect van Hugging Face niet volgen, dus eerst handmatig downloaden.
curl -L -O https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF/resolve/main/Jev-Style-Qwen3.5-2B-Decision-Q8_0.gguf
ollama create jev-style-2b -f Modelfile

# 2. Python-omgeving
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> We gebruiken `pygame-ce` en niet `pygame`: de font-module van `pygame` 2.6 werkt niet onder Python 3.14.

## Starten

```bash
source .venv/bin/activate
python snake.py
```

| Optie | Betekenis |
|---|---|
| `--speed N` | max. stappen per seconde (standaard 15, `0` = zo snel als het model kan, ±20–25/s) |
| `--straight-bonus X` | voorkeur voor rechtdoor bij gelijkwaardige zetten (standaard 3, `0` = uit) |
| `--no-straight` | alleen links of rechts, nooit rechtdoor (werkt veel slechter) |
| `--model NAAM` | ander Ollama-model (standaard `jev-style-2b`) |

**Toetsen:** `+`/`-` of pijl omhoog/omlaag = snelheid aanpassen · spatie = opnieuw na game over · Esc = stoppen.

## Zelf het model aanroepen

In de Ollama-chat (`ollama run jev-style-2b`): plak de prompt tussen `"""` … `"""`. Het antwoord is de letter, bijvoorbeeld `B`.

Via de API, met kansen per optie:

```bash
jq -n --arg p "$PROMPT" '{model:"jev-style-2b", raw:true, stream:false, prompt:$p,
  options:{num_predict:1, temperature:0}, logprobs:true, top_logprobs:10}' \
| curl -s localhost:11434/api/generate -d @- \
| jq -r '.logprobs[0].top_logprobs[] | "\(.token)\t\(.logprob|exp*100|round)%"'
```

## Resultaten (eigen metingen)

| Variant | Resultaat |
|---|---|
| Alleen links/rechts, simpele state | 2–3 snoepjes, snel dood |
| Links/rechtdoor/rechts, simpele state | ±32 snoepjes, daarna ingesloten |
| + ruimte-check (flood fill) | ±45 snoepjes |
| + staart-check en eet-simulatie | 8/8 potjes leven nog na 3000 stappen, ±125 snoepjes |
| + bonus voor rechtdoor (altijd) | rechte lijnen, maar haalt geen snoepjes meer |
| + bonus alleen bij gelijkwaardige zet | rechte lijnen (±12% bochten) en snoepjes pakken (±16 stappen per snoepje) |

De nieuwere versie van het model (v2) scoorde hier niet beter dan v1.
