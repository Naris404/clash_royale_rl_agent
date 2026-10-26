import { useEffect, useRef, useState } from "react";
import { GameSocket, createSession, fetchModels } from "./net/client";
import { GameStore } from "./game/store";
import { Lang, STRINGS } from "./i18n";
import { Dashboard } from "./ui/Dashboard";
import { GameScreen } from "./ui/GameScreen";
import { HomeScreen } from "./ui/HomeScreen";
import type { ModelCatalogue, ModelInfo } from "./net/types";

type Screen = "home" | "game" | "dashboard";

function screenFromPath(): Screen {
  return window.location.pathname === "/dashboard" ? "dashboard" : "home";
}

export default function App() {
  const [lang, setLang] = useState<Lang>("en");
  const [screen, setScreen] = useState<Screen>(screenFromPath);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<ModelCatalogue | null>(null);
  const [opponentModel, setOpponentModel] = useState<ModelInfo | null>(null);

  const storeRef = useRef<GameStore | null>(null);
  const socketRef = useRef<GameSocket | null>(null);
  if (!storeRef.current) storeRef.current = new GameStore();
  if (!socketRef.current) socketRef.current = new GameSocket();

  useEffect(() => {
    // without a catalogue the home screen simply has no model picker
    fetchModels().then(setModels).catch(() => undefined);
  }, []);

  useEffect(() => {
    const socket = socketRef.current!;
    const onPopState = () => {
      socket.close();
      setScreen(screenFromPath());
    };
    window.addEventListener("popstate", onPopState);
    return () => {
      window.removeEventListener("popstate", onPopState);
      socket.close();
    };
  }, []);

  const navigate = (next: Screen, path: string) => {
    window.history.pushState({}, "", path);
    setScreen(next);
  };

  const startGame = async (mode: string, model?: string) => {
    setBusy(true);
    setError(null);
    try {
      socketRef.current!.close();
      storeRef.current! = new GameStore();
      const session = await createSession(mode, lang, model);
      setOpponentModel(models?.models.find((m) => m.id === session.model) ?? null);
      // swap the subscription to the new store
      const socket = socketRef.current!;
      socket.connect(session.session_id);
      socket.onMessage((msg) => storeRef.current!.apply(msg));
      socket.onClose(() => {
        storeRef.current!.connected = false;
      });
      navigate("game", "/game");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const exitToMenu = () => {
    socketRef.current!.close();
    navigate("home", "/");
  };

  const strings = STRINGS[lang];

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  if (screen === "game") {
    return (
      <GameScreen
        store={storeRef.current}
        socket={socketRef.current}
        lang={lang}
        strings={strings}
        opponentModel={opponentModel}
        onExit={exitToMenu}
      />
    );
  }
  if (screen === "dashboard") {
    return (
      <Dashboard
        strings={strings}
        onMenu={exitToMenu}
        onPlay={() => navigate("home", "/")}
      />
    );
  }

  return (
    <>
      <HomeScreen
        lang={lang}
        strings={strings}
        models={models}
        onLang={setLang}
        onPlay={startGame}
        onDashboard={() => navigate("dashboard", "/dashboard")}
        busy={busy}
      />
      {error && <div className="conn-lost">{error}</div>}
    </>
  );
}
