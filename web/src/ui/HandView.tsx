import { cardEmoji } from "../scene/emoji";
import type { Strings } from "../i18n";
import type { ArenaConfig } from "../net/types";

interface Props {
  hand: string[];
  nextCard: string | null;
  elixir: number;
  selected: string | null;
  onSelect: (card: string | null) => void;
  config: ArenaConfig | null;
  strings: Strings;
  suggestedSlot: number | null;
}

export function HandView({ hand, nextCard, elixir, selected, onSelect, config, strings, suggestedSlot }: Props) {
  return (
    <div className="hand-bar">
      <div className="elixir-column">
        <div className="elixir-bar" title={strings.elixir}>
          <div className="elixir-fill" style={{ width: `${Math.min(100, elixir * 10)}%` }} />
          <span className="elixir-text">{Math.floor(elixir)}</span>
        </div>
        {nextCard && (
          <div className="next-card" title={strings.next_card}>
            {strings.next_card}: {cardEmoji(nextCard)}
          </div>
        )}
      </div>
      <div className="hand-cards">
        {hand.map((card, slot) => {
          const cost = config?.cards[card]?.elisir ?? 0;
          const affordable = elixir >= cost;
          const isSelected = selected === card;
          const isSuggested = suggestedSlot === slot;
          return (
            <button
              key={`${card}-${slot}`}
              className={[
                "card",
                isSelected ? "card-selected" : "",
                !affordable ? "card-disabled" : "",
                isSuggested ? "card-suggested" : "",
              ].join(" ")}
              onClick={() => affordable && onSelect(isSelected ? null : card)}
              disabled={!affordable}
            >
              <span className="card-emoji">{cardEmoji(card)}</span>
              <span className="card-name">{card.replace("_", " ")}</span>
              <span className="card-cost">{cost}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
