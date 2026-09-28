import { Application } from "pixi.js";
import { useEffect, useRef } from "react";
import type { GameStore } from "../game/store";
import { ArenaScene } from "./ArenaScene";

interface Props {
  store: GameStore;
  onPlace: (x: number, y: number) => void;
}

export function ArenaView({ store, onPlace }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const onPlaceRef = useRef(onPlace);
  onPlaceRef.current = onPlace;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const app = new Application();
    let scene: ArenaScene | null = null;
    let destroyed = false;

    app
      .init({ antialias: true, resizeTo: host, backgroundAlpha: 0 })
      .then(() => {
        if (destroyed) {
          app.destroy(true);
          return;
        }
        host.appendChild(app.canvas);
        scene = new ArenaScene(app, store, (x, y) => onPlaceRef.current(x, y));
        app.ticker.add(() => scene?.render(performance.now()));
      })
      .catch((error) => console.error("Pixi init failed", error));

    return () => {
      destroyed = true;
      scene = null;
      app.destroy(true, { children: true });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div ref={hostRef} className="arena-canvas" />;
}
