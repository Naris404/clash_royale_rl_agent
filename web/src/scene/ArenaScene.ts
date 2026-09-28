import { Application, Container, FederatedPointerEvent, Graphics, Text } from "pixi.js";
import type { GameStore } from "../game/store";
import type { Snapshot, SpellState, TowerState, TroopState } from "../net/types";
import { cardEmoji, OWNER_COLORS, TOWER_EMOJI } from "./emoji";

interface Layout {
  tile: number;
  ox: number;
  oy: number;
}

interface TroopView {
  root: Container;
  ring: Graphics;
  label: Text;
  hpBg: Graphics;
  hpFg: Graphics;
  flash: Graphics;
  flashUntil: number;
  lastHp: number;
  lastX: number;
  lastY: number;
}

interface TowerView {
  root: Container;
  label: Text;
  hpFg: Graphics;
  hpText: Text;
  rubble: Text;
}

interface GhostView {
  label: Text;
  bornAt: number;
}

interface SpellView {
  proj: Text;
  boom: Graphics;
  receivedAt: number;
  ageAtReceipt: number;
}

/** Scena Pixi: arena + wieże + jednostki + czary + podpowiedzi trenera. */
export class ArenaScene {
  private app: Application;
  private store: GameStore;
  private onPlace: (x: number, y: number) => void;

  private staticLayer = new Container();
  private overlayLayer = new Container();
  private towerLayer = new Container();
  private troopLayer = new Container();
  private fxLayer = new Container();

  private troops = new Map<number, TroopView>();
  private towers = new Map<string, TowerView>();
  private spells = new Map<number, SpellView>();
  private ghosts: GhostView[] = [];

  private hintRing = new Graphics();
  private deployOverlay = new Graphics();
  private ghostLabel = new Text({ text: "" });
  private pointerTile: { x: number; y: number } | null = null;
  private staticKey = "";

  constructor(app: Application, store: GameStore, onPlace: (x: number, y: number) => void) {
    this.app = app;
    this.store = store;
    this.onPlace = onPlace;

    app.stage.addChild(this.staticLayer, this.overlayLayer, this.towerLayer, this.troopLayer, this.fxLayer);
    app.stage.eventMode = "static";
    app.stage.hitArea = app.screen;
    app.stage.on("pointermove", (e: FederatedPointerEvent) => {
      this.pointerTile = this.toTile(e.global.x, e.global.y);
    });
    app.stage.on("pointerdown", (e: FederatedPointerEvent) => {
      const tile = this.toTile(e.global.x, e.global.y);
      if (tile) this.onPlace(tile.x, tile.y);
    });

    this.hintRing.visible = false;
    this.deployOverlay.visible = false;
    this.ghostLabel.alpha = 0.6;
    this.ghostLabel.visible = false;
    this.ghostLabel.anchor.set(0.5);
    this.fxLayer.addChild(this.hintRing, this.deployOverlay, this.ghostLabel);
  }

  // --- układ współrzędnych ---

  private layout(): Layout {
    const { width, height } = this.app.screen;
    const config = this.store.config;
    const aw = config?.arena_width ?? 18;
    const al = config?.arena_length ?? 32;
    const tile = Math.min(width / (aw + 1), height / (al + 1));
    return { tile, ox: (width - aw * tile) / 2, oy: (height - al * tile) / 2 };
  }

  private toScreen(x: number, y: number, l: Layout): [number, number] {
    const al = this.store.config?.arena_length ?? 32;
    return [l.ox + x * l.tile, l.oy + (al - y) * l.tile];
  }

  private toTile(px: number, py: number): { x: number; y: number } | null {
    const l = this.layout();
    const al = this.store.config?.arena_length ?? 32;
    const x = (px - l.ox) / l.tile;
    const y = al - (py - l.oy) / l.tile;
    if (x < 0 || y < 0 || x > (this.store.config?.arena_width ?? 18) || y > al) return null;
    return { x, y };
  }

  // --- statyczne tło ---

  private rebuildStatic(l: Layout): void {
    const config = this.store.config;
    if (!config) return;
    const g = this.staticLayer;
    g.removeChildren().forEach((c) => c.destroy());
    const bg = new Graphics();

    const aw = config.arena_width;
    const al = config.arena_length;
    for (let ix = 0; ix < aw; ix++) {
      for (let iy = 0; iy < al; iy++) {
        const dark = (ix + iy) % 2 === 0;
        bg.rect(l.ox + ix * l.tile, l.oy + iy * l.tile, l.tile, l.tile).fill(dark ? 0x2d6a30 : 0x2f7534);
      }
    }

    // rzeka
    const riverTop = this.toScreen(0, config.river_y + config.river_half_width, l)[1];
    const riverH = 2 * config.river_half_width * l.tile;
    bg.rect(l.ox, riverTop, aw * l.tile, riverH).fill(0x3787d2);

    // mosty
    for (const bx of config.bridge_lane_x) {
      const [cx] = this.toScreen(bx, 0, l);
      bg.roundRect(cx - 1.8 * l.tile, riverTop - 0.15 * l.tile, 3.6 * l.tile, riverH + 0.3 * l.tile, 4).fill(0x8b5a2b);
    }

    // strefy rzutu (dyskretne akcje RL) — delikatne punkty
    for (const zones of Object.values(config.deploy_zones)) {
      for (const [zx, zy] of zones) {
        const [sx, sy] = this.toScreen(zx, zy, l);
        bg.circle(sx, sy, 0.18 * l.tile).fill({ color: 0xffffff, alpha: 0.25 });
      }
    }

    // ramka areny
    bg.rect(l.ox, l.oy, aw * l.tile, al * l.tile).stroke({ color: 0x1c4420, width: 3 });

    this.staticLayer.removeChildren();
    this.staticLayer.addChild(bg);
  }

  // --- klatka renderująca ---

  render(now: number): void {
    const config = this.store.config;
    const snap = this.store.currSnap;
    if (!config) return;

    const l = this.layout();
    const key = `${this.app.screen.width}x${this.app.screen.height}`;
    if (key !== this.staticKey) {
      this.staticKey = key;
      this.rebuildStatic(l);
    }
    if (!snap) return;

    const alpha = this.store.interpolationAlpha(now);
    this.renderTowers(snap, l);
    this.renderTroops(snap, l, alpha, now);
    this.renderSpells(snap, l, now);
    this.renderGhosts(now);
    this.renderHint(l, now);
    this.renderTargeting(l);
  }

  private renderTowers(snap: Snapshot, l: Layout): void {
    for (const tower of snap.towers) {
      let view = this.towers.get(tower.id);
      if (!view) {
        view = this.createTower(tower, l);
        this.towers.set(tower.id, view);
        this.towerLayer.addChild(view.root);
      }
      this.updateTower(view, tower, l);
    }
  }

  private createTower(tower: TowerState, l: Layout): TowerView {
    const root = new Container();
    const isKing = tower.name === "King_Tower";
    const base = new Graphics();
    const size = (isKing ? 2.6 : 2.1) * l.tile;
    base.roundRect(-size / 2, -size / 2, size, size, 6).fill(0x555c68);
    base.roundRect(-size / 2, -size / 2, size, size, 6).stroke({ color: OWNER_COLORS[tower.owner] ?? 0xffffff, width: 3 });
    const label = new Text({ text: TOWER_EMOJI[tower.name] ?? "🏰", style: { fontSize: size * 0.55 } });
    label.anchor.set(0.5);
    const hpBg = new Graphics();
    const barW = size;
    hpBg.rect(-barW / 2, -size / 2 - 10, barW, 6).fill(0x222222);
    const hpFg = new Graphics();
    const hpText = new Text({ text: "", style: { fill: 0xffffff, fontSize: Math.max(10, l.tile * 0.42), fontWeight: "bold" } });
    hpText.anchor.set(0.5, 1);
    hpText.position.set(0, -size / 2 - 12);
    const rubble = new Text({ text: "💥", style: { fontSize: size * 0.6 } });
    rubble.anchor.set(0.5);
    rubble.visible = false;
    root.addChild(base, label, hpBg, hpFg, hpText, rubble);
    return { root, label, hpFg, hpText, rubble };
  }

  private updateTower(view: TowerView, tower: TowerState, l: Layout): void {
    const [sx, sy] = this.toScreen(tower.x, tower.y, l);
    view.root.position.set(sx, sy);
    const frac = tower.max_hp > 0 ? Math.max(0, tower.hp / tower.max_hp) : 0;
    const isKing = tower.name === "King_Tower";
    const size = (isKing ? 2.6 : 2.1) * l.tile;
    view.hpFg.clear();
    if (tower.alive) {
      view.hpFg.rect(-size / 2, -size / 2 - 10, size * frac, 6).fill(frac > 0.35 ? 0x4ade80 : 0xef4444);
      view.hpText.text = String(Math.round(tower.hp));
    }
    view.rubble.visible = !tower.alive;
    view.label.visible = tower.alive;
    view.hpText.visible = tower.alive;
  }

  private renderTroops(snap: Snapshot, l: Layout, alpha: number, now: number): void {
    const prevById = new Map<number, TroopState>();
    this.store.prevSnap?.troops.forEach((t) => prevById.set(t.id, t));
    const seen = new Set<number>();

    for (const troop of snap.troops) {
      seen.add(troop.id);
      let view = this.troops.get(troop.id);
      if (!view) {
        view = this.createTroop(troop, l);
        this.troops.set(troop.id, view);
        this.troopLayer.addChild(view.root);
      }
      const prev = prevById.get(troop.id);
      const x = prev ? prev.x + (troop.x - prev.x) * alpha : troop.x;
      const y = prev ? prev.y + (troop.y - prev.y) * alpha : troop.y;
      view.lastX = x;
      view.lastY = y;
      const [sx, sy] = this.toScreen(x, y, l);
      view.root.position.set(sx, sy);
      view.root.scale.set(l.tile / 22);

      // pasek HP
      const frac = troop.max_hp > 0 ? Math.max(0, troop.hp / troop.max_hp) : 0;
      view.hpFg.clear();
      if (frac < 1) {
        view.hpBg.visible = true;
        view.hpFg.rect(-11, -17, 22 * frac, 3).fill(0x4ade80);
      } else {
        view.hpBg.visible = false;
      }

      // błysk obrażeń
      if (troop.hp < view.lastHp - 1e-6) {
        view.flashUntil = now + 180;
        view.flash.clear();
        view.flash.circle(0, 0, 13).fill({ color: 0xff3b30, alpha: 0.55 });
      }
      view.lastHp = troop.hp;
      const flashLeft = view.flashUntil - now;
      view.flash.alpha = flashLeft > 0 ? flashLeft / 180 : 0;
      view.flash.visible = flashLeft > 0;
    }

    // zniknięte jednostki → duch (animacja śmierci)
    for (const [id, view] of this.troops) {
      if (!seen.has(id)) {
        this.troops.delete(id);
        view.root.destroy({ children: true });
        const ghostText = new Text({ text: view.label.text, style: { fontSize: 20 } });
        ghostText.anchor.set(0.5);
        const [sx, sy] = this.toScreen(view.lastX, view.lastY, l);
        ghostText.position.set(sx, sy);
        this.fxLayer.addChild(ghostText);
        this.ghosts.push({ label: ghostText, bornAt: now });
      }
    }
  }

  private createTroop(troop: TroopState, _l: Layout): TroopView {
    const root = new Container();
    const ring = new Graphics();
    ring.circle(0, 0, 13).fill(OWNER_COLORS[troop.owner] ?? 0xffffff);
    ring.circle(0, 0, 13).stroke({ color: 0x0f172a, width: 2 });
    const label = new Text({ text: cardEmoji(troop.card), style: { fontSize: 18 } });
    label.anchor.set(0.5);
    const hpBg = new Graphics();
    hpBg.rect(-11, -17, 22, 3).fill(0x1f2937);
    hpBg.visible = false;
    const hpFg = new Graphics();
    const flash = new Graphics();
    flash.visible = false;
    root.addChild(ring, label, hpBg, hpFg, flash);
    return {
      root,
      ring,
      label,
      hpBg,
      hpFg,
      flash,
      flashUntil: 0,
      lastHp: troop.hp,
      lastX: troop.x,
      lastY: troop.y,
    };
  }

  private renderSpells(snap: Snapshot, l: Layout, now: number): void {
    const seen = new Set<number>();
    for (const spell of snap.spells) {
      seen.add(spell.id);
      let view = this.spells.get(spell.id);
      if (!view) {
        view = {
          proj: new Text({ text: cardEmoji(spell.card), style: { fontSize: 24 } }),
          boom: new Graphics(),
          receivedAt: now,
          ageAtReceipt: spell.age,
        };
        view.proj.anchor.set(0.5);
        view.proj.visible = false;
        this.fxLayer.addChild(view.proj, view.boom);
        this.spells.set(spell.id, view);
      }
      this.updateSpell(view, spell, l, now);
    }
    for (const [id, view] of this.spells) {
      if (!seen.has(id)) {
        view.proj.destroy();
        view.boom.destroy();
        this.spells.delete(id);
      }
    }
  }

  private updateSpell(view: SpellView, spell: SpellState, l: Layout, now: number): void {
    // wiek czaru interpolowany lokalnie między snapshotami
    const age = view.ageAtReceipt + ((now - view.receivedAt) / 1000) * this.store.speed;
    const travel = Math.max(spell.travel_time, 1e-3);
    const t = Math.min(1, age / travel);
    const [tx, ty] = this.toScreen(spell.x, spell.y, l);
    if (t < 1) {
      const [sx0, sy0] = this.toScreen(spell.start_x, spell.start_y, l);
      const px = sx0 + (tx - sx0) * t;
      const arc = Math.sin(t * Math.PI) * 2.5 * l.tile;
      view.proj.position.set(px, sy0 + (ty - sy0) * t - arc);
      view.proj.visible = true;
      view.boom.visible = false;
    } else {
      view.proj.visible = false;
      const fade = age - travel;
      const alpha = Math.max(0, 1 - fade / 0.5);
      view.boom.clear();
      view.boom.circle(tx, ty, spell.radius * l.tile * (0.6 + 0.4 * (1 - alpha))).fill({ color: 0xff7a18, alpha: 0.45 * alpha });
      view.boom.circle(tx, ty, spell.radius * l.tile).stroke({ color: 0xffd166, width: 2, alpha });
      view.boom.visible = alpha > 0;
    }
  }

  private renderGhosts(now: number): void {
    const lifeMs = 320;
    for (const ghost of this.ghosts) {
      const t = (now - ghost.bornAt) / lifeMs;
      ghost.label.alpha = Math.max(0, 1 - t);
      const s = 1 + t * 0.4;
      ghost.label.scale.set(s);
    }
    this.ghosts = this.ghosts.filter((ghost) => {
      if (now - ghost.bornAt >= lifeMs) {
        ghost.label.destroy();
        return false;
      }
      return true;
    });
  }

  private renderHint(l: Layout, now: number): void {
    const hint = this.store.hint;
    const config = this.store.config;
    if (!hint || hint.zone == null || hint.card == null || !config || this.store.selectedCard) {
      this.hintRing.visible = false;
      return;
    }
    const zone = config.deploy_zones["0"]?.[hint.zone];
    if (!zone) {
      this.hintRing.visible = false;
      return;
    }
    const [sx, sy] = this.toScreen(zone[0], zone[1], l);
    const pulse = 1 + 0.15 * Math.sin(now / 220);
    this.hintRing.clear();
    this.hintRing.circle(sx, sy, 1.1 * l.tile * pulse).stroke({ color: 0xfacc15, width: 3, alpha: 0.9 });
    this.hintRing.circle(sx, sy, 0.55 * l.tile * pulse).stroke({ color: 0xfacc15, width: 1.5, alpha: 0.6 });
    this.hintRing.visible = true;
  }

  private renderTargeting(l: Layout): void {
    const config = this.store.config;
    const card = this.store.selectedCard;
    this.deployOverlay.clear();
    if (!card || !config) {
      this.deployOverlay.visible = false;
      this.ghostLabel.visible = false;
      return;
    }
    const isSpell = config.cards[card]?.type === "spell";
    const aw = config.arena_width;
    const al = config.arena_length;

    if (isSpell) {
      this.deployOverlay
        .rect(l.ox + l.tile, l.oy + 2 * l.tile, (aw - 2) * l.tile, (al - 4) * l.tile)
        .fill({ color: 0xf97316, alpha: 0.12 });
    } else {
      // połowa gracza (P0): y od 2 do river_y - 0.5 → ekran od góry
      const topY = al - (config.river_y - 0.5);
      const bottomY = al - 2;
      this.deployOverlay
        .rect(l.ox + l.tile, l.oy + topY * l.tile, (aw - 2) * l.tile, (bottomY - topY) * l.tile)
        .fill({ color: 0x22c55e, alpha: 0.14 });
      for (const [zx, zy] of config.deploy_zones["0"] ?? []) {
        const [sx, sy] = this.toScreen(zx, zy, l);
        this.deployOverlay.circle(sx, sy, 0.9 * l.tile).stroke({ color: 0x22c55e, width: 2, alpha: 0.7 });
      }
    }
    this.deployOverlay.visible = true;

    if (this.pointerTile) {
      const [sx, sy] = this.toScreen(this.pointerTile.x, this.pointerTile.y, l);
      this.ghostLabel.text = cardEmoji(card);
      this.ghostLabel.style.fontSize = l.tile * 0.9;
      this.ghostLabel.position.set(sx, sy);
      this.ghostLabel.visible = true;
    } else {
      this.ghostLabel.visible = false;
    }
  }
}
