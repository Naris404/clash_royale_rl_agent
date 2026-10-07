export type Lang = "pl" | "en";

export const STRINGS = {
  pl: {
    title: "Clash Royale — Trener RL",
    subtitle: "Środowisko treningowe z agentem uczenia ze wzmacnianiem",
    mode_coach: "Trening z trenerem",
    mode_coach_desc: "Graj przeciwko botowi, a agent RL oceni Twoje ruchy na żywo i podpowie lepsze.",
    mode_vs_rl: "Mecz vs AI",
    mode_vs_rl_desc: "Zmierz się z wytrenowaną siecią neuronową (PPO).",
    mode_vs_bot: "Mecz vs bot",
    mode_vs_bot_desc: "Klasyczny mecz przeciwko botowi regułowemu — bez podpowiedzi.",
    play: "Graj",
    coach: "Trener",
    coach_waiting: "Analizuję planszę…",
    eval_bar: "Ocena pozycji",
    key_moments: "Kluczowe momenty",
    no_key_moments: "Brak — gra toczy się spokojnie.",
    pause: "Pauza",
    resume: "Wznów",
    speed: "Tempo",
    next_card: "Następna",
    grade_best: "Najlepszy ruch!",
    grade_good: "Dobry ruch",
    grade_inaccuracy: "Niedokładność",
    grade_blunder: "Błąd",
    grade_unknown: "Ruch zapisany",
    better_was: "Lepiej:",
    match_end: "Koniec meczu",
    you_won: "Wygrałeś!",
    you_lost: "Przegrałeś",
    draw: "Remis",
    moves: "Ruchy",
    mean_quality: "Średnia jakość zagrań",
    rematch: "Rewanż",
    back_to_menu: "Menu",
    move_history: "Historia Twoich ruchów",
    no_moves: "Nie zagrałeś żadnej karty.",
    how_to_play: "Kliknij kartę, potem pole na swojej połowie areny. Fireball możesz rzucić wszędzie.",
    footer_thesis: "Projekt dyplomowy — uczenie ze wzmacnianiem (PPO) i sieci neuronowe",
    connection_lost: "Połączenie przerwane — odśwież stronę.",
    elixir: "Eliksir",
    portfolio_eyebrow: "Interaktywny projekt dyplomowy",
    portfolio_intro: "Agent PPO nie tylko gra — analizuje pozycję, proponuje ruch i wyjaśnia błędy decyzyjne w czasie rzeczywistym.",
    try_demo: "Uruchom demo",
    about_title: "O projekcie",
    thesis_abstract_title: "Streszczenie pracy",
    thesis_abstract: "Celem pracy jest zbadanie, czy agent uczenia ze wzmacnianiem może pełnić rolę interaktywnego trenera w uproszczonym środowisku Clash Royale. Polityka MaskablePPO uczy się zarządzania eliksirem, doboru kart i rozmieszczenia jednostek przeciwko botom o różnym poziomie trudności. Moduł trenera wykorzystuje rozkład prawdopodobieństwa akcji oraz wartość stanu do podpowiadania ruchów, klasyfikowania błędów i wykrywania kluczowych momentów meczu. Jakość rozwiązania jest oceniana przez win rate, stabilność treningu i zgodność ocen trenera z późniejszym wynikiem na wieżach.",
    architecture_title: "Jak to działa",
    architecture_rl: "MaskablePPO",
    architecture_rl_desc: "Uczy politykę wyłącznie na legalnych akcjach.",
    architecture_coach: "Trener AI",
    architecture_coach_desc: "Łączy prawdopodobieństwa polityki, V(s) i reguły taktyczne.",
    architecture_live: "Gra na żywo",
    architecture_live_desc: "Python symuluje mecz, a WebSocket przesyła stan do areny WebGL.",
    research_title: "Zakres eksperymentów",
    research_items: "Sweep hiperparametrów|Curriculum: Random → Logic|Self-play ze starszym checkpointem|Walidacja ocen ruchów",
    stack_title: "Technologie",
    stack: "Python · Gymnasium · PyTorch · MaskablePPO · FastAPI · React · PixiJS",
    choose_mode: "Wybierz tryb gry",
    dashboard: "Dashboard treningu",
    dashboard_desc: "Metryki MaskablePPO odświeżane z logów TensorBoard",
    menu: "Menu",
    training_active: "Trening aktywny",
    training_idle: "Trening zatrzymany",
    latest_step: "Ostatni krok",
    latest_run: "Bieżący przebieg",
    metrics_count: "Metryki",
    model_files: "Modele",
    performance: "Wyniki agenta",
    optimization: "Optymalizacja",
    all_metrics: "Wszystkie metryki",
    no_metrics: "Brak logów treningowych. Uruchom python train.py.",
    updated: "Aktualizacja",
  },
  en: {
    title: "Clash Royale — RL Coach",
    subtitle: "Training environment with a reinforcement learning agent",
    mode_coach: "Training with coach",
    mode_coach_desc: "Play against the bot while the RL agent grades your moves live and suggests better ones.",
    mode_vs_rl: "Match vs AI",
    mode_vs_rl_desc: "Face the trained neural network (PPO).",
    mode_vs_bot: "Match vs bot",
    mode_vs_bot_desc: "Classic match against the rule-based bot — no hints.",
    play: "Play",
    coach: "Coach",
    coach_waiting: "Analyzing the board…",
    eval_bar: "Position eval",
    key_moments: "Key moments",
    no_key_moments: "None — the game is quiet.",
    pause: "Pause",
    resume: "Resume",
    speed: "Speed",
    next_card: "Next",
    grade_best: "Best move!",
    grade_good: "Good move",
    grade_inaccuracy: "Inaccuracy",
    grade_blunder: "Blunder",
    grade_unknown: "Move recorded",
    better_was: "Better:",
    match_end: "Match over",
    you_won: "You won!",
    you_lost: "You lost",
    draw: "Draw",
    moves: "Moves",
    mean_quality: "Mean move quality",
    rematch: "Rematch",
    back_to_menu: "Menu",
    move_history: "Your move history",
    no_moves: "You played no cards.",
    how_to_play: "Click a card, then a tile on your half of the arena. Fireball can be cast anywhere.",
    footer_thesis: "Thesis project — reinforcement learning (PPO) and neural networks",
    connection_lost: "Connection lost — refresh the page.",
    elixir: "Elixir",
    portfolio_eyebrow: "Interactive engineering thesis",
    portfolio_intro: "The PPO agent does more than play: it evaluates positions, recommends moves, and explains decision errors in real time.",
    try_demo: "Launch demo",
    about_title: "About the project",
    thesis_abstract_title: "Thesis abstract",
    thesis_abstract: "This thesis investigates whether a reinforcement-learning agent can act as an interactive coach in a simplified Clash Royale environment. A MaskablePPO policy learns elixir management, card selection, and unit placement against opponents of varying difficulty. The coaching module combines action probabilities and state value to recommend moves, classify mistakes, and identify key moments. The system is evaluated using win rate, training stability, and agreement between coach grades and subsequent tower-HP outcomes.",
    architecture_title: "How it works",
    architecture_rl: "MaskablePPO",
    architecture_rl_desc: "Learns a policy over legal actions only.",
    architecture_coach: "AI coach",
    architecture_coach_desc: "Combines policy probabilities, V(s), and tactical rules.",
    architecture_live: "Live match",
    architecture_live_desc: "Python simulates the game while WebSockets stream state to a WebGL arena.",
    research_title: "Experiments",
    research_items: "Hyperparameter sweep|Curriculum: Random → Logic|Self-play against an older checkpoint|Move-grading validation",
    stack_title: "Technology",
    stack: "Python · Gymnasium · PyTorch · MaskablePPO · FastAPI · React · PixiJS",
    choose_mode: "Choose a game mode",
    dashboard: "Training dashboard",
    dashboard_desc: "Live MaskablePPO metrics read from TensorBoard logs",
    menu: "Menu",
    training_active: "Training active",
    training_idle: "Training stopped",
    latest_step: "Latest step",
    latest_run: "Current run",
    metrics_count: "Metrics",
    model_files: "Models",
    performance: "Agent performance",
    optimization: "Optimization",
    all_metrics: "All metrics",
    no_metrics: "No training logs found. Run python train.py.",
    updated: "Updated",
  },
} as const;

export type StringKey = keyof (typeof STRINGS)["pl"];
export type Strings = Record<StringKey, string>;

export function gradeLabel(grade: string, lang: Lang): string {
  const s = STRINGS[lang];
  switch (grade) {
    case "best":
      return s.grade_best;
    case "good":
      return s.grade_good;
    case "inaccuracy":
      return s.grade_inaccuracy;
    case "blunder":
      return s.grade_blunder;
    default:
      return s.grade_unknown;
  }
}

export function zoneLabel(zone: number | null | undefined, lang: Lang): string {
  // Order mirrors the server's DEPLOY_ZONES (see cr_rl.game.board).
  const pl = [
    "tył, lewa aleja",
    "tył, prawa aleja",
    "środek, lewa aleja",
    "środek, prawa aleja",
    "lewy most",
    "prawy most",
    "lewy punkt przyciągania (pull)",
    "prawy punkt przyciągania (pull)",
    "lewa wieża wroga",
    "prawa wieża wroga",
    "król wroga",
  ];
  const en = [
    "back, left lane",
    "back, right lane",
    "mid, left lane",
    "mid, right lane",
    "left bridge",
    "right bridge",
    "left pull zone",
    "right pull zone",
    "enemy left tower",
    "enemy right tower",
    "enemy king",
  ];
  if (zone == null || zone < 0 || zone >= en.length) return "";
  return lang === "pl" ? pl[zone] : en[zone];
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
