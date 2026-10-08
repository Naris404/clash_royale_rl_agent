import { useEffect, useRef, useState } from "react";
import { GameSocket, createSession } from "./net/client";
import { GameStore } from "./game/store";
import { Lang, STRINGS } from "./i18n";
import { Dashboard } from "./ui/Dashboard";
import { GameScreen } from "./ui/GameScreen";
import { HomeScreen } from "./ui/HomeScreen";

type Screen = "home" | "game" | "dashboard";

function screenFromPath(): Screen {
  return window.location.pathname === "/dashboard" ? "dashboard" : "home";
}

export default function App() {
  const [lang, setLang] = useState<Lang>("en");
  const [screen, setScreen] = useState<Screen>(screenFromPath);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const storeRef = useRef<GameStore | null>(null);
  const socketRef = useRef<GameSocket | null>(null);
  if (!storeRef.current) storeRef.current = new GameStore();
  if (!socketRef.current) socketRef.current = new GameSocket();

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

  const startGame = async (mode: string) => {
    setBusy(true);
    setError(null);
    try {
      socketRef.current!.close();
      storeRef.current! = new GameStore();
      const session = await createSession(mode, lang);
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
    return <GameScreen store={storeRef.current} socket={socketRef.current} lang={lang} strings={strings} onExit={exitToMenu} />;
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
        onLang={setLang}
        onPlay={startGame}
        onDashboard={() => navigate("dashboard", "/dashboard")}
        busy={busy}
      />
      {error && <div className="conn-lost">{error}</div>}
    </>
  );
}
