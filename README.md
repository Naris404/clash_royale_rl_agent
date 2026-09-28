# Trener do nauki gry w Clash Royale

Uproszczona symulacja [Clash Royale](https://supercell.com/en/games/clashroyale/) pod trening **reinforcement learning**. Arena, eliksir, kolejka kart i walka są zaimplementowane w Pythonie; agent RL uczy się grać przeciwko botowi regułowemu.

## Funkcje

- **Plansza 18×32** kafelków (zgodnie z oficjalną areną CR)
- **Dwa mosty** — jednostki nie przechodzą przez rzekę poza mostem (Hog skacze)
- **6 kart w talii**, **4 na ręce** — kolejka bez duplikatów (jak w grze)
- **Wieże** z zasięgiem ~7.5 kafelka; strzelają do wrogów zaraz po wejściu w range
- **Wizualizacja pygame** z HUD, ręką kart i historią meczu (pauza + strzałki)
- **`RLAgent`** — PPO (MaskablePPO) uczony przeciwko `LogicAgent`
- **`LogicAgent`** — bot oparty na regułach (obrona, Hog, Cannon itd.)

## Wymagania

- Python 3.10+
- Node.js 20+ (tylko interfejs webowy)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[test]"
```

## Szybki start

```bash
# Trening PPO vs LogicAgent (~1M kroków, kilka–kilkanaście min na CPU)
cr-rl-train

# Ewaluacja wytrenowanego modelu
cr-rl-evaluate --model models/ppo_cr_best.zip --episodes 100

# Raport statystyk (logi TB, win rate, wykresy)
cr-rl-stats
cr-rl-stats --plot --episodes 100

# Wizualizacja: P0 = agent RL, P1 = bot regułowy
cr-rl-viz
cr-rl-viz --model models/ppo_cr_best.zip

# Mniejsze okno
cr-rl-viz --scale 0.5

# Wolniejsza / szybsza symulacja (1 s gry = X s u Ciebie)
cr-rl-viz --tempo 1.0
```

### Sterowanie (okno pygame)

| Klawisz | Akcja |
|---------|--------|
| Spacja | Pauza / wznowienie |
| `+` / `-` | Przyspieszenie / zwolnienie |
| `[` / `]` | Zmiana tempa symulacji |
| `←` / `→` (pauza) | Cofnij / do przód po historii meczu |
| `↑` / `↓` (pauza) | Przesuń widok areny |
| `Q` / Esc | Wyjście |

## Struktura projektu

```
clash_royale_rl_agent/
├── src/cr_rl/
│   ├── game/          # plansza i karty
│   ├── agents/        # bot regułowy, losowy i PPO
│   ├── env/           # środowisko Gymnasium
│   ├── training/      # trening i ewaluacja
│   ├── coach/         # sugestie i ocena ruchów
│   ├── server/        # FastAPI i sesje WebSocket
│   ├── experiments/   # sweep, curriculum i self-play
│   ├── stats/         # raporty oraz wykresy
│   └── viz/           # wizualizacja pygame
├── web/               # React + PixiJS
├── tests/             # testy pytest
├── scripts/           # narzędzia deweloperskie
├── docs/              # dokumentacja projektu i materiały do pracy (docs/thesis)
├── models/            # modele i wyniki (gitignore)
└── pyproject.toml      # pakiet i komendy cr-rl-*
```

Cały kod źródłowy żyje w `src/cr_rl/`; punkty wejścia to komendy `cr-rl-*`
zdefiniowane w `pyproject.toml` (dostępne po `pip install -e .`).

## API środowiska (`Board`)

```python
from cr_rl.game.board import Board, NUM_ACTIONS, OBS_DIM

env = Board(seed=42)
obs = env.reset(seed=42)

# Akcja 0 = nic; 1–12 = slot ręki (0–3) × strefa rzutu (0–2)
result = env.step(action_p0, action_p1=None)

obs = result.observation   # wektor float32, OBS_DIM
reward = result.reward
done = result.terminated or result.truncated
```

### Karty w talii (domyślnie)

Knight, Giant, Cannon, Musketeer, Hog_Rider, Fireball

### Agent RL (PPO)

```python
from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.agents.rl import RLAgent

# Trening
# cr-rl-train --timesteps 1000000 --n-envs 8

# Gra z modelem
agent = RLAgent(model_path="models/ppo_cr_best.zip")
env = ClashRoyaleEnv()
obs, _ = env.reset()
action = agent.choose_action(env.board, player=0)
obs, reward, done, trunc, info = env.step(action)
```

**Obserwacja** (`OBS_DIM`): eliksir własny i wroga, czas, HP 6 wież, one-hot ręki (4×6), do 24 jednostek (posortowane).

**Nagroda**: wyłącznie zmiana łącznego HP wież `(utrata wroga − utrata własna) / max_HP_wież`.

`LogicAgent` używa `board.set_pending_play()` — w `gym_env` i `visualize.py` oba tryby są podłączone poprawnie.

## Konfiguracja

| Plik | Co zmienić |
|------|------------|
| `src/cr_rl/viz/pygame_viewer.py` → `UI_SCALE` | Rozmiar okna |
| `src/cr_rl/viz/pygame_viewer.py` → `REAL_SECONDS_PER_SIM_SECOND` | Tempo symulacji |
| `src/cr_rl/game/board.py` → `TOWER_LAYOUT`, `BRIDGE_LANE_X` | Pozycje wież / mostów |
| `src/cr_rl/game/cards.py` → `cards_dic` | Statystyki kart |
| `cr-rl-train --timesteps --n-envs` | Długość i równoległość treningu |

## Platforma webowa

Backend FastAPI jest źródłem prawdy dla symulacji, a interfejs React/PixiJS
renderuje arenę i komunikuje się przez WebSocket. Strona startowa jest
dwujęzycznym (PL/EN) portfolio projektu ze streszczeniem pracy.

### Uruchomienie bez Dockera

Po jednorazowym wykonaniu instalacji z sekcji „Wymagania”:

```powershell
cd web
npm ci
cd ..
.\scripts\dev.ps1
```

Otwórz **http://localhost:5173**. Vite przekazuje `/api` i `/ws` do backendu
FastAPI na porcie **8000**.

Możesz też uruchomić procesy ręcznie w dwóch terminalach:

```powershell
# terminal 1 — API
.\.venv\Scripts\Activate.ps1
uvicorn cr_rl.server.app:app --reload --port 8000

# terminal 2 — frontend
cd web
npm run dev
```

Po `npm run build` produkcyjny frontend jest serwowany przez API z `web/dist`
pod adresem http://localhost:8000.

## Eksperymenty do pracy

Każdy eksperyment zapisuje metadane i wyniki w JSON pod
`experiment_runs/`, dzięki czemu uruchomienia są powtarzalne i mogą być
porównane przez `cr-rl-stats`.

```bash
# sweep: learning rate, entropy, gamma i architektura sieci
cr-rl-sweep --timesteps 1000000 --eval-episodes 200

# curriculum: najpierw RandomAgent, potem LogicAgent
cr-rl-curriculum --phase1 300000 --phase2 700000

# raport zbiorczy i wykresy do rozdziału eksperymentalnego
cr-rl-stats --experiments --plot
```

Krótkie uruchomienia z mniejszą liczbą kroków służą wyłącznie jako smoke test;
wyniki do pracy powinny używać tych samych seedów, budżetu kroków i co najmniej
200 meczów ewaluacyjnych na wariant.

## Docker i hosting

```bash
docker build -t clash-royale-rl-coach .
docker run --rm -p 8000:8000 clash-royale-rl-coach
```

Obraz wieloetapowy buduje frontend i uruchamia API jako użytkownik bez
uprawnień root. Model `models/ppo_cr_best.zip` jest dołączany, jeśli znajduje
się w kontekście budowania. Alternatywnie można ustawić `MODEL_URL`; bez modelu
aplikacja uruchamia trenera heurystycznego. Instrukcje publikacji:
[DEPLOYMENT.md](DEPLOYMENT.md).

## Licencja

Projekt edukacyjny / własny — Clash Royale jest znakiem Supercell.
