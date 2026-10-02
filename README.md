# Snake, Pong & Tetris met een lokaal Jev-model

Snake, Pong en Tetris, bestuurd door een lokaal **Jev-style beslismodel**. Het model draait in Python via **MLX** (Apple Silicon) of **PyTorch/CUDA** (NVIDIA), of via [Ollama](https://ollama.com). Bij de eerste start wordt het automatisch gedownload.
- **Snake:** het model kiest elke stap linksaf, rechtdoor of rechtsaf, gezien vanaf de kop.
- **Pong:** twee AI-spelers kiezen elk omhoog, blijven of omlaag. Je kunt ook zelf het linkerbatje spelen.
- **Tetris:** het model schuift en draait de vallende steen.

Met **Tab** wissel je tussen de spellen. Alle drie gebruiken hetzelfde 20×20-speelveld.

## Wat is Jev?

[Jev](https://typesafe.ai) van TypeSafe AI is een *System One*-model. Het schrijft geen tekst. Je geeft het een **state**, een **vraag** en **opties**, en het geeft per optie een gekalibreerde **kans** terug. Jev zelf bestaat alleen als cloud-API. Dit project gebruikt daarom een open, lokale nabouw:

**Jev-Style-Qwen3.5-2B-Decision** (v1, Apache-2.0) van chaoliangUNSW. Dit model is niet verbonden aan TypeSafe AI. Welke variant gedownload wordt, hangt af van de backend:

| Backend | Bestand | Grootte |
|---|---|---|
| `mlx`, `cuda` | [`Jev-Style-Qwen3.5-2B-Decision-MLX-bf16`](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16) (bf16 safetensors) | ±4 GB |
| `ollama` | [`Jev-Style-Qwen3.5-2B-Decision-Q8_0.gguf`](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF) | ±2 GB |

## Hoe het werkt

1. **State**: de spelsituatie wordt in tekst beschreven, gezien vanaf de kop. Per mogelijke zet berekent een lookahead:
   - of de zet direct **dodelijk** is;
   - of het een **val** is: kan de kop daarna nog bij zijn eigen staart? Zo niet, dan raakt de snake ingesloten;
   - na hoeveel stappen het snoepje bereikt wordt en of het **veilig** is om het via die route te eten. Daarvoor wordt de route gesimuleerd en daarna opnieuw de staart-check gedaan.
2. **Vraag + opties**: `turn left` / `go straight` / `turn right`.
3. **Beslissing**: het model voorspelt één token, de optieletter. De logprobs van `A`/`B`/`C` worden omgerekend naar kansen. De kans van elke optie zie je onderin het scherm.
4. **Beleid in code**: als rechtdoor volgens de lookahead even goed is als de beste zet, krijgt het een bonus (`--straight-bonus`). Dat geeft rustige, rechte lijnen zonder dat de snake snoepjes laat liggen.

Er ligt steeds 1 snoepje op een willekeurige plek. Eet de snake het op, dan wordt hij 1 unit langer.

### Pong

- De batjes (4 cellen hoog) staan in de linker- en rechterkolom. De bal beweegt elke stap 1 cel schuin en kaatst tegen boven- en onderrand. Raak je de bal met de rand van het batje, dan verandert zijn verticale richting.
- Elk batje is een eigen Jev-speler. Bij Ollama lopen de beide modelaanroepen parallel.
- Lookahead: het spel voorspelt waar en wanneer de bal bij jouw batje aankomt. Gaat de bal van je af, dan wordt aangenomen dat de tegenstander hem terugslaat. Per zet krijgt het model hoe ver het batje dan van die landingsplek staat en of dat nog op tijd te halen is.
- Beleid in code: als stilstaan even goed is als de beste zet, krijgt het een bonus (`--stay-bonus`). Dat voorkomt onrustig heen-en-weer schuiven.
- Omdat de feiten exact zijn, mist het model zelden. Rally's kunnen dus lang duren.

### Tetris

- De 7 standaard tetromino's (I, O, T, S, Z, J, L) zijn gedefinieerd in een **5×5 vak** en draaien rond het midden.
- Per stap kiest het model: `move left`, `move right`, `rotate counterclockwise`, `rotate clockwise` of `do nothing`. Daarna valt de steen 1 rij.
- Lookahead: per actie zoekt een BFS over alle bereikbare posities (draaiing, x, y), met dezelfde regels als het spel (actie, dan vallen), alle mogelijke landingen. Elke landing krijgt een score volgens de heuristiek van Yiyuan Lee: hoogte, volle lijnen, gaten en hobbeligheid.
- Het model krijgt per actie eenvoudige feiten: houdt die actie de **BEST** landing bereikbaar en is het de **FASTEST** weg daarheen? Of leidt hij tot een **WORSE** landing, of kan hij niet? De vergelijking gebeurt in code. Met ruwe getallen voor vijf opties koos het 2B-model maar in 45% van de stappen de beste actie, met deze labels in 100%.
- De beste landingsplek wordt als grijze omtrek (ghost) getekend.

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


## Installatie

Vereisten: Python 3.10+ en één van deze backends:

| Backend | Platform | Installatie | Beslissing (eigen meting, M-serie Mac) |
|---|---|---|---|
| `mlx` | Apple Silicon | `pip install -r requirements.txt` (installeert `mlx-lm` automatisch) | ±19 ms |
| `cuda` | NVIDIA-GPU (Linux/Windows) | `pip install -r requirements-cuda.txt` | niet gemeten |
| `ollama` | overal waar Ollama draait | [Ollama](https://ollama.com) installeren en `ollama serve` | ±37 ms |

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # Mac met Apple Silicon: klaar
pip install -r requirements-cuda.txt     # NVIDIA: installeer eerst PyTorch met CUDA (pytorch.org)
```

**Het model downloadt vanzelf** bij de eerste `python snake.py`:
- `mlx` en `cuda` halen de bf16-gewichten op via de Hugging Face-cache (`~/.cache/huggingface`).
- `ollama` downloadt de GGUF naar deze map en voert `ollama create jev-style-2b -f Modelfile` uit. Dat gaat buiten `ollama pull` om, omdat Ollama de redirect van Hugging Face niet kan volgen.

Alle drie de backends geven dezelfde kansen (verschil < 2 procentpunt) en dezelfde zetten.

**Hoe draait het model zonder Ollama?** De prompt gaat in één forward pass door het model. De hidden state op de laatste positie wordt vermenigvuldigd met de embedding-rijen van ` A`, ` B`, ... Zo krijg je direct de logits van de optieletters, zonder tekst te genereren. Voor PyTorch worden de MLX-gewichten bij het laden omgezet naar het transformers-formaat. Bij CUDA maakt `flash-linear-attention` (in `requirements-cuda.txt`) de Gated-DeltaNet-lagen veel sneller; zonder valt transformers terug op een tragere PyTorch-versie.

> We gebruiken `pygame-ce` en niet `pygame`: de font-module van `pygame` 2.6 werkt niet onder Python 3.14.

## Starten

```bash
source .venv/bin/activate
python snake.py
```

| Optie | Betekenis |
|---|---|
| `--speed N` | max. stappen per seconde (standaard `0` = zo snel als het model kan) |
| `--straight-bonus X` | snake: voorkeur voor rechtdoor bij gelijkwaardige zetten (standaard 3, `0` = uit) |
| `--stay-bonus X` | pong: voorkeur voor stilstaan bij gelijkwaardige zetten (standaard 3, `0` = uit) |
| `--no-straight` | snake: alleen links of rechts, nooit rechtdoor (werkt veel slechter) |
| `--pong` | start direct in pong |
| `--tetris` | start direct in tetris |
| `--human` | start in pong en speel zelf het linkerbatje tegen de AI rechts |
| `--backend B` | `auto` (standaard), `mlx`, `cuda` of `ollama`. `auto` kiest MLX op Apple Silicon als `mlx-lm` geïnstalleerd is, anders CUDA als PyTorch een GPU ziet, anders Ollama |
| `--model NAAM` | ander Ollama-model (standaard `jev-style-2b`, alleen bij `ollama`) |

**Toetsen:** Tab = volgende spel (snake → pong → tetris; de andere spellen pauzeren) · `+`/`-` of pijl omhoog/omlaag = snelheid aanpassen · spatie = opnieuw (snake/tetris: na game over, pong: score resetten) · Esc = stoppen.

**Zelf pong spelen:** druk in pong op **H** om het linkerbatje over te nemen; nog een keer H geeft het terug aan de AI. Je bestuurt het met **W/S** of **↑/↓**: ingedrukt houden blijft bewegen, een korte tik is één stap. Zolang je zelf speelt, zijn de pijltjes voor je batje en regel je de snelheid alleen met `+`/`-`.

Standaard draait alles op maximale snelheid: snake doet via Ollama ±25 en via MLX ±50 stappen per seconde. Pong is ongeveer half zo snel, omdat er per stap twee beslissingen zijn. Bij MLX en CUDA delen de twee batjes één model en lopen de aanroepen na elkaar.

De gebruikte backend staat in de titelbalk van het venster.

## Zelf het model aanroepen

Vanuit Python, met elke backend:

```python
import jev
b = jev.load_backend("auto")            # of "mlx", "cuda", "ollama"
scores = b.client().logprobs("x is 3", "Is x odd?", ["yes", "no"])
print(jev.softmax(scores))              # bijv. [0.87, 0.13]
```

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
