import type { ClientMessage, ServerMessage } from "./types";

export interface SessionInfo {
  session_id: string;
  mode: string;
  coach_enabled: boolean;
}

export async function createSession(mode: string, lang: string): Promise<SessionInfo> {
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, lang }),
  });
  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.status}`);
  }
  return response.json();
}

export class GameSocket {
  private ws: WebSocket | null = null;
  private keepalive: ReturnType<typeof setInterval> | null = null;
  private listeners = new Set<(msg: ServerMessage) => void>();
  private closeListeners = new Set<() => void>();

  connect(sessionId: string): void {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${protocol}://${window.location.host}/ws/sessions/${sessionId}`);
    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data) as ServerMessage;
      this.listeners.forEach((listener) => listener(msg));
    };
    this.keepalive = setInterval(() => this.send({ type: "ping" }), 25_000);
    this.ws.onclose = () => {
      this.stopKeepalive();
      this.closeListeners.forEach((listener) => listener());
    };
  }

  onMessage(listener: (msg: ServerMessage) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  onClose(listener: () => void): () => void {
    this.closeListeners.add(listener);
    return () => this.closeListeners.delete(listener);
  }

  send(msg: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  play(card: string, x: number, y: number): void {
    this.send({ type: "play", card, x, y });
  }

  setPaused(paused: boolean): void {
    this.send({ type: "pause", paused });
  }

  setSpeed(value: number): void {
    this.send({ type: "speed", value });
  }

  rematch(): void {
    this.send({ type: "rematch" });
  }

  close(): void {
    this.stopKeepalive();
    this.ws?.close();
    this.ws = null;
    this.listeners.clear();
    this.closeListeners.clear();
  }

  private stopKeepalive(): void {
    if (this.keepalive !== null) {
      clearInterval(this.keepalive);
      this.keepalive = null;
    }
  }
}
