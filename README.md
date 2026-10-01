# Snake met een lokaal Jev-model

Snake die bestuurd wordt door een lokaal **Jev-style beslismodel** via Ollama.

[Jev](https://typesafe.ai) (TypeSafe AI) is een "System One"-model. Het schrijft geen tekst, maar krijgt een *state*, een *vraag* en *opties* en geeft per optie een kans terug. Jev zelf is alleen beschikbaar als cloud-API. Daarom gebruikt dit project de open, lokale nabouw
[`chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF`](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF) (Q8_0).

## Hoe het werkt
- Bij elke stap wordt de spelsituatie relatief t.o.v. de kop beschreven: waar het snoepje ligt, en per zet:
  - of de zet direct dodelijk is;
  - of het een **val** is: kan de kop daarna nog bij zijn eigen staart? Zo niet, dan raakt de snake ingesloten;
  - na hoeveel stappen het snoepje bereikt wordt en of het **veilig** is om het via die route te eten. Daarvoor wordt de route gesimuleerd en daarna opnieuw de staart-check gedaan.
- Het model kiest uit `turn left` / `go straight` / `turn right`. De kansen worden uitgelezen via de logprobs van de optieletter en onderin het scherm getoond.
- Voor rustige, rechte lijnen krijgt `go straight` een bonus (`--straight-bonus`, standaard 3). Die geldt alleen als rechtdoor volgens de lookahead **even goed** is als de beste zet: even veilig en even weinig stappen naar het snoepje.
- Er ligt steeds 1 snoepje op een willekeurige plek. Eet de snake het op, dan wordt hij 1 unit langer.

## Installatie
```bash
# 1. Model (Ollama kan de HF-redirect niet volgen, dus handmatig downloaden)
curl -L -O https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF/resolve/main/Jev-Style-Qwen3.5-2B-Decision-Q8_0.gguf
ollama create jev-style-2b -f Modelfile

# 2. Python-omgeving
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Starten
```bash
source .venv/bin/activate
python snake.py                # links / rechtdoor / rechts
python snake.py --no-straight  # alleen links of rechts
python snake.py --speed 0      # zo snel als het model kan (~25 stappen/s)
```
+/- (of pijl op/neer) = snelheid aanpassen, spatie = opnieuw na game over, Esc = stoppen.
