import type { Toast } from "../game/store";
import { gradeLabel, zoneLabel, Strings } from "../i18n";

interface Props {
  toasts: Toast[];
  lang: "pl" | "en";
  strings: Strings;
}

const GRADE_CLASS: Record<string, string> = {
  best: "toast-best",
  good: "toast-good",
  inaccuracy: "toast-inaccuracy",
  blunder: "toast-blunder",
};

export function Toasts({ toasts, lang, strings }: Props) {
  return (
    <div className="toasts">
      {toasts.map((toast) => {
        const fb = toast.feedback;
        const better =
          fb.best_card && (fb.grade === "blunder" || fb.grade === "inaccuracy")
            ? ` ${strings.better_was} ${fb.best_card} (${zoneLabel(fb.best_zone, lang)})`
            : "";
        return (
          <div key={toast.id} className={`toast ${GRADE_CLASS[fb.grade] ?? ""}`}>
            <strong>{gradeLabel(fb.grade, lang)}</strong>
            {fb.card && <span> — {fb.card}</span>}
            {better && <span className="toast-better">{better}</span>}
          </div>
        );
      })}
    </div>
  );
}
