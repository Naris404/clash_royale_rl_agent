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
- Zależności z `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Szybki start

```bash
# Trening PPO vs LogicAgent (~1M kroków, kilka–kilkanaście min na CPU)
pip install -r requirements.txt
python train.py

# Ewaluacja wytrenowanego modelu
python evaluate.py --model models/ppo_cr_best.zip --episodes 100

# Raport statystyk (logi TB, win rate, wykresy)
python statistics.py
python statistics.py --plot --episodes 100

# Wizualizacja: P0 = agent RL, P1 = bot regułowy
python visualize.py
python visualize.py --model models/ppo_cr_best.zip

# Mniejsze okno
python visualize.py --scale 0.5

# Wolniejsza / szybsza symulacja (1 s gry = X s u Ciebie)
python visualize.py --tempo 1.0
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
├── board.py        # Środowisko gry (step, reward, obs, ręka, mosty)
├── cards.py        # Statystyki kart i klasa Troop / Tower
├── gym_env.py      # Wrapper Gymnasium (RL vs LogicAgent)
├── logic_agent.py  # Bot regułowy (set_pending_play)
├── rl_agent.py     # Agent RL (wczytuje PPO z models/)
├── train.py        # Trening MaskablePPO
├── evaluate.py     # Win rate vs LogicAgent
├── statistics.py   # Raport uczenia + wykresy
├── visualize.py    # Podgląd pygame
├── requirements.txt
└── models/         # Zapisywane modele (gitignore)
```

## API środowiska (`Board`)

```python
from board import Board, NUM_ACTIONS, OBS_DIM

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
from gym_env import ClashRoyaleEnv
from rl_agent import RLAgent

# Trening
# python train.py --timesteps 1000000 --n-envs 8

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
| `visualize.py` → `UI_SCALE` | Rozmiar okna |
| `visualize.py` → `REAL_SECONDS_PER_SIM_SECOND` | Tempo symulacji |
| `board.py` → `TOWER_LAYOUT`, `BRIDGE_LANE_X` | Pozycje wież / mostów |
| `cards.py` → `cards_dic` | Statystyki kart |
| `train.py` → `--timesteps`, `--n-envs` | Długość i równoległość treningu |

## Licencja

Projekt edukacyjny / własny — Clash Royale jest znakiem Supercell.
