import { useEffect, useSyncExternalStore } from "react";
import type { GameSocket } from "../net/client";
import type { GameStore } from "../game/store";
import type { Lang, Strings } from "../i18n";
import type { ArenaConfig, ModelInfo, PocketState } from "../net/types";
import { ArenaView } from "../scene/ArenaView";
import { CoachPanel } from "./CoachPanel";
import { HandView } from "./HandView";
import { Hud } from "./Hud";
import { SummaryModal } from "./SummaryModal";
import { Toasts } from "./Toasts";

interface Props {
  store: GameStore;
  socket: GameSocket;
  lang: Lang;
  strings: Strings;
  opponentModel: ModelInfo | null; // the model playing the RL Opponent
  onExit: () => void;
}

function isValidPlacement(config: ArenaConfig, pockets: PocketState[], card: string, x: number, y: number): boolean {
  const isSpell = config.cards[card]?.type === "spell";
  if (x < 1 || x > config.arena_width - 1) return false;
  const margin = config.back_edge_margin;
  if (isSpell) return y >= margin && y <= config.arena_length - margin;
  if (y >= margin && y <= config.river_y - config.river_half_width) return true; // P0's bank, never the water
  return pockets.some((p) => x >= p.x_min && x <= p.x_max && y >= p.y_min && y <= p.y_max);
}

export function GameScreen({ store, socket, lang, strings, opponentModel, onExit }: Props) {
  useSyncExternalStore(store.subscribe, store.getVersion);

  useEffect(() => {
    const interval = window.setInterval(() => store.pruneToasts(performance.now()), 1000);
    return () => window.clearInterval(interval);
  }, [store]);

  const snap = store.currSnap;
  const config = store.config;

  const handlePlace = (x: number, y: number) => {
    const card = store.selectedCard;
    const liveConfig = store.config;
    const liveSnap = store.currSnap;
    if (!card || !liveConfig || liveSnap?.done) return;
    if (!isValidPlacement(liveConfig, liveSnap?.pockets ?? [], card, x, y)) return;
    socket.play(card, x, y);
    store.selectCard(null);
  };

  const handlePause = (paused: boolean) => {
    store.setPaused(paused);
    socket.setPaused(paused);
  };

  const handleSpeed = (speed: number) => {
    store.setSpeed(speed);
    socket.setSpeed(speed);
  };

  const handleRematch = () => {
    store.resetMatchLocal();
    socket.rematch();
  };

  return (
    <div className="game-screen">
      <Hud
        snap={snap}
        paused={store.paused}
        speed={store.speed}
        opponent={store.mode === "vs_rl" ? [strings.hud_vs_rl, opponentModel?.label[lang]].filter(Boolean).join(" · ") : null}
        onPause={handlePause}
        onSpeed={handleSpeed}
        onMenu={onExit}
        strings={strings}
      />

      <div className="game-main">
        <div className="arena-wrap">
          <ArenaView store={store} onPlace={handlePlace} />
          <Toasts toasts={store.toasts} lang={lang} strings={strings} />
        </div>
        {store.coachEnabled && <CoachPanel store={store} strings={strings} />}
      </div>

      <HandView
        hand={snap?.hand ?? []}
        nextCard={snap?.next_card ?? null}
        elixir={snap?.elixir[0] ?? 0}
        selected={store.selectedCard}
        onSelect={(card) => store.selectCard(card)}
        config={config}
        strings={strings}
        suggestedSlot={store.coachEnabled ? store.hint?.slot ?? null : null}
      />

      <div className="how-to">{strings.how_to_play}</div>

      {store.matchEnd && (
        <SummaryModal detail={store.matchEnd} strings={strings} lang={lang} onRematch={handleRematch} onMenu={onExit} />
      )}
      {!store.connected && <div className="conn-lost">{strings.connection_lost}</div>}
    </div>
  );
}
