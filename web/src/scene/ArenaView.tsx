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
    let initialised = false;
    let destroyed = false;

    app
      .init({ antialias: true, resizeTo: host, backgroundAlpha: 0 })
      .then(() => {
        initialised = true;
        if (destroyed) {
          app.destroy(true, { children: true });
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
      // Pixi cannot destroy a half-initialised app (StrictMode remount); init's then() does it.
      if (initialised) app.destroy(true, { children: true });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div ref={hostRef} className="arena-canvas" />;
}
