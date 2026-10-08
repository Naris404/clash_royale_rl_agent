import { Application, Assets, Container, FederatedPointerEvent, Graphics, Sprite, Text, Texture } from "pixi.js";
import type { GameStore } from "../game/store";
import type { HitState, Snapshot, SpellState, TowerState, TroopState } from "../net/types";
import { cardArtUrl } from "./cardArt";
import { cardEmoji, CARD_EMOJI, OWNER_COLORS, TOWER_EMOJI } from "./emoji";

interface Layout {
  tile: number;
  ox: number;
  oy: number;
}

// Troop tokens are drawn in local units and scaled by tile / TOKEN_UNITS_PER_TILE.
const TOKEN_UNITS_PER_TILE = 22;
const TOKEN_RADIUS = 13;
const FLASH_MS = 180;

interface TroopView {
  root: Container;
  icon: Container;
  hasArt: boolean;
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
  flash: Graphics;
  flashUntil: number;
  lastHp: number;
  wasAlive: boolean;
}

interface SpellView {
  proj: Text;
  boom: Graphics;
  receivedAt: number;
  ageAtReceipt: number;
}

/** Short-lived visual effect; `update` returns false once finished. */
interface Effect {
  node: Container;
  update: (now: number) => boolean;
}

/** Pixi scene: arena + towers + units + spells + combat effects + coach hints. */
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
  private effects: Effect[] = [];
  private art = new Map<string, Texture>();
  private lastSnap: Snapshot | null = null;

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

    void this.loadArt();
  }

  private async loadArt(): Promise<void> {
    await Promise.all(
      Object.keys(CARD_EMOJI).map(async (card) => {
        const url = cardArtUrl(card);
        if (!url) return;
        try {
          this.art.set(card, await Assets.load<Texture>(url));
        } catch {
          // no artwork → keep the emoji
        }
      }),
    );
  }

  // --- coordinate layout ---

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

  // --- static background ---

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

    // river
    const riverTop = this.toScreen(0, config.river_y + config.river_half_width, l)[1];
    const riverH = 2 * config.river_half_width * l.tile;
    bg.rect(l.ox, riverTop, aw * l.tile, riverH).fill(0x3787d2);

    // bridges
    const half = config.bridge_half_width;
    for (const bx of config.bridge_lane_x) {
      const [cx] = this.toScreen(bx, 0, l);
      bg.roundRect(cx - half * l.tile, riverTop - 0.15 * l.tile, 2 * half * l.tile, riverH + 0.3 * l.tile, 4).fill(0x8b5a2b);
    }

    // drop zones (discrete RL actions) — subtle markers
    for (const zones of Object.values(config.deploy_zones)) {
      for (const zone of zones) {
        const [sx, sy] = this.toScreen(zone.x, zone.y, l);
        bg.circle(sx, sy, 0.18 * l.tile).fill({ color: 0xffffff, alpha: 0.25 });
      }
    }

    // arena frame
    bg.rect(l.ox, l.oy, aw * l.tile, al * l.tile).stroke({ color: 0x1c4420, width: 3 });

    this.staticLayer.removeChildren();
    this.staticLayer.addChild(bg);
  }

  // --- render frame ---

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

    if (snap !== this.lastSnap) {
      this.lastSnap = snap;
      snap.hits?.forEach((hit, i) => this.spawnHit(hit, i, l, now));
    }

    const alpha = this.store.interpolationAlpha(now);
    this.renderTowers(snap, l, now);
    this.renderTroops(snap, l, alpha, now);
    this.renderSpells(snap, l, now);
    this.renderEffects(now);
    this.renderHint(l, now);
    this.renderTargeting(l);
  }

  // --- towers ---

  private renderTowers(snap: Snapshot, l: Layout, now: number): void {
    for (const tower of snap.towers) {
      let view = this.towers.get(tower.id);
      if (!view) {
        view = this.createTower(tower, l);
        this.towers.set(tower.id, view);
        this.towerLayer.addChild(view.root);
      }
      this.updateTower(view, tower, l, now);
    }
  }

  private towerSize(tower: TowerState, l: Layout): number {
    return (tower.name === "King_Tower" ? 2.6 : 2.1) * l.tile;
  }

  private createTower(tower: TowerState, l: Layout): TowerView {
    const root = new Container();
    const size = this.towerSize(tower, l);
    const base = new Graphics();
    base.roundRect(-size / 2, -size / 2, size, size, 6).fill(0x555c68);
    base.roundRect(-size / 2, -size / 2, size, size, 6).stroke({ color: OWNER_COLORS[tower.owner] ?? 0xffffff, width: 3 });
    const label = new Text({ text: TOWER_EMOJI[tower.name] ?? "🏰", style: { fontSize: size * 0.55 } });
    label.anchor.set(0.5);
    const hpBg = new Graphics();
    hpBg.rect(-size / 2, -size / 2 - 10, size, 6).fill(0x222222);
    const hpFg = new Graphics();
    const hpText = new Text({ text: "", style: { fill: 0xffffff, fontSize: Math.max(10, l.tile * 0.42), fontWeight: "bold" } });
    hpText.anchor.set(0.5, 1);
    hpText.position.set(0, -size / 2 - 12);
    const rubble = new Text({ text: "💥", style: { fontSize: size * 0.6 } });
    rubble.anchor.set(0.5);
    rubble.visible = false;
    const flash = new Graphics();
    flash.roundRect(-size / 2, -size / 2, size, size, 6).fill({ color: 0xffffff, alpha: 0.6 });
    flash.visible = false;
    root.addChild(base, label, flash, hpBg, hpFg, hpText, rubble);
    return { root, label, hpFg, hpText, rubble, flash, flashUntil: 0, lastHp: tower.hp, wasAlive: tower.alive };
  }

  private updateTower(view: TowerView, tower: TowerState, l: Layout, now: number): void {
    const [sx, sy] = this.toScreen(tower.x, tower.y, l);
    view.root.position.set(sx, sy);
    const frac = tower.max_hp > 0 ? Math.max(0, tower.hp / tower.max_hp) : 0;
    const size = this.towerSize(tower, l);
    view.hpFg.clear();
    if (tower.alive) {
      view.hpFg.rect(-size / 2, -size / 2 - 10, size * frac, 6).fill(frac > 0.35 ? 0x4ade80 : 0xef4444);
      view.hpText.text = String(Math.round(tower.hp));
    }
    if (tower.hp < view.lastHp - 1e-6) view.flashUntil = now + FLASH_MS;
    view.lastHp = tower.hp;
    this.updateFlash(view.flash, view.flashUntil, now);

    if (view.wasAlive && !tower.alive) this.spawnPoof(sx, sy, size * 0.7, now);
    view.wasAlive = tower.alive;
    view.rubble.visible = !tower.alive;
    view.label.visible = tower.alive;
    view.hpText.visible = tower.alive;
  }

  // --- units ---

  private renderTroops(snap: Snapshot, l: Layout, alpha: number, now: number): void {
    const prevById = new Map<number, TroopState>();
    this.store.prevSnap?.troops.forEach((t) => prevById.set(t.id, t));
    const seen = new Set<number>();
    const scale = l.tile / TOKEN_UNITS_PER_TILE;

    for (const troop of snap.troops) {
      seen.add(troop.id);
      let view = this.troops.get(troop.id);
      if (!view) {
        view = this.createTroop(troop);
        this.troops.set(troop.id, view);
        this.troopLayer.addChild(view.root);
      }
      if (!view.hasArt && this.art.has(troop.card)) this.replaceIcon(view, troop.card);

      const prev = prevById.get(troop.id);
      const x = prev ? prev.x + (troop.x - prev.x) * alpha : troop.x;
      const y = prev ? prev.y + (troop.y - prev.y) * alpha : troop.y;
      view.lastX = x;
      view.lastY = y;
      const [sx, sy] = this.toScreen(x, y, l);
      view.root.position.set(sx, sy);
      view.root.scale.set(scale);

      // pasek HP
      const frac = troop.max_hp > 0 ? Math.max(0, troop.hp / troop.max_hp) : 0;
      view.hpFg.clear();
      view.hpBg.visible = frac < 1;
      if (frac < 1) view.hpFg.rect(-11, -18, 22 * frac, 3).fill(0x4ade80);

      if (troop.hp < view.lastHp - 1e-6) view.flashUntil = now + FLASH_MS;
      view.lastHp = troop.hp;
      this.updateFlash(view.flash, view.flashUntil, now);
    }

    // vanished units → death poof
    for (const [id, view] of this.troops) {
      if (seen.has(id)) continue;
      this.troops.delete(id);
      view.root.destroy({ children: true });
      const [sx, sy] = this.toScreen(view.lastX, view.lastY, l);
      this.spawnPoof(sx, sy, TOKEN_RADIUS * scale, now);
    }
  }

  private makeIcon(card: string): { icon: Container; hasArt: boolean } {
    const icon = new Container();
    const texture = this.art.get(card);
    if (!texture) {
      const label = new Text({ text: cardEmoji(card), style: { fontSize: 18 } });
      label.anchor.set(0.5);
      icon.addChild(label);
      return { icon, hasArt: false };
    }
    // the card has a frame and transparent margin — crop the circle onto the character's face
    const r = TOKEN_RADIUS - 2;
    const sprite = new Sprite(texture);
    sprite.anchor.set(0.5, 0.56);
    sprite.scale.set((2.6 * r) / texture.width);
    const mask = new Graphics().circle(0, 0, r).fill(0xffffff);
    sprite.mask = mask;
    icon.addChild(sprite, mask);
    return { icon, hasArt: true };
  }

  private replaceIcon(view: TroopView, card: string): void {
    const { icon, hasArt } = this.makeIcon(card);
    const index = view.root.getChildIndex(view.icon);
    view.root.removeChild(view.icon);
    view.icon.destroy({ children: true });
    view.root.addChildAt(icon, index);
    view.icon = icon;
    view.hasArt = hasArt;
  }

  private createTroop(troop: TroopState): TroopView {
    const root = new Container();
    const ring = new Graphics();
    ring.circle(0, 0, TOKEN_RADIUS).fill(0x0f172a);
    ring.circle(0, 0, TOKEN_RADIUS - 1).stroke({ color: OWNER_COLORS[troop.owner] ?? 0xffffff, width: 3 });
    const { icon, hasArt } = this.makeIcon(troop.card);
    const hpBg = new Graphics();
    hpBg.rect(-11, -18, 22, 3).fill(0x1f2937);
    hpBg.visible = false;
    const hpFg = new Graphics();
    const flash = new Graphics();
    flash.circle(0, 0, TOKEN_RADIUS).fill({ color: 0xffffff, alpha: 0.7 });
    flash.visible = false;
    root.addChild(ring, icon, flash, hpBg, hpFg);
    return {
      root,
      icon,
      hasArt,
      hpBg,
      hpFg,
      flash,
      flashUntil: 0,
      lastHp: troop.hp,
      lastX: troop.x,
      lastY: troop.y,
    };
  }

  private updateFlash(flash: Graphics, flashUntil: number, now: number): void {
    const left = flashUntil - now;
    flash.visible = left > 0;
    flash.alpha = left > 0 ? left / FLASH_MS : 0;
  }

  // --- czary ---

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
    // spell age interpolated locally between snapshots
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

  // --- combat effects ---

  private addEffect(node: Container, update: (now: number) => boolean): void {
    this.fxLayer.addChild(node);
    this.effects.push({ node, update });
  }

  private renderEffects(now: number): void {
    // update() may spawn follow-up effects, which land in the fresh list
    const active = this.effects;
    this.effects = [];
    for (const effect of active) {
      if (effect.update(now)) this.effects.push(effect);
      else effect.node.destroy({ children: true });
    }
  }

  /** Hit: projectile arc (ranged), then the damage number above the target. */
  private spawnHit(hit: HitState, index: number, l: Layout, now: number): void {
    const [tx, ty] = this.toScreen(hit.x, hit.y, l);
    if (!hit.ranged) {
      this.spawnDamageNumber(hit.damage, tx, ty, index, l, now);
      return;
    }
    const [fx, fy] = this.toScreen(hit.from_x, hit.from_y, l);
    const dist = Math.hypot(hit.x - hit.from_x, hit.y - hit.from_y);
    const flightMs = Math.max(90, 40 * dist) / this.store.speed;
    const height = (0.2 + 0.08 * dist) * l.tile;
    const proj = new Graphics().circle(0, 0, Math.max(2, 0.16 * l.tile)).fill(OWNER_COLORS[hit.owner] ?? 0xffffff);
    proj.circle(0, 0, Math.max(2, 0.16 * l.tile)).stroke({ color: 0xffffff, width: 1 });
    this.addEffect(proj, (t) => {
      const p = (t - now) / flightMs;
      if (p >= 1) {
        this.spawnDamageNumber(hit.damage, tx, ty, index, l, t);
        return false;
      }
      proj.position.set(fx + (tx - fx) * p, fy + (ty - fy) * p - Math.sin(p * Math.PI) * height);
      return true;
    });
  }

  private spawnDamageNumber(damage: number, x: number, y: number, index: number, l: Layout, now: number): void {
    const lifeMs = 700;
    const label = new Text({
      text: String(Math.round(damage)),
      style: {
        fill: 0xffffff,
        fontSize: Math.max(11, 0.5 * l.tile),
        fontWeight: "bold",
        stroke: { color: 0x7f1d1d, width: 3 },
      },
    });
    label.anchor.set(0.5);
    const dx = ((index % 3) - 1) * 0.35 * l.tile;
    this.addEffect(label, (t) => {
      const p = (t - now) / lifeMs;
      if (p >= 1) return false;
      label.position.set(x + dx, y - 0.6 * l.tile - p * 0.9 * l.tile);
      label.alpha = p < 0.6 ? 1 : 1 - (p - 0.6) / 0.4;
      return true;
    });
  }

  /** Unit death / tower destruction: an expanding smoke puff. */
  private spawnPoof(x: number, y: number, radius: number, now: number): void {
    const lifeMs = 420;
    const puffs = 6;
    const poof = new Container();
    poof.position.set(x, y);
    const g = new Graphics();
    poof.addChild(g);
    this.addEffect(poof, (t) => {
      const p = (t - now) / lifeMs;
      if (p >= 1) return false;
      const alpha = 1 - p;
      g.clear();
      g.circle(0, 0, radius * (0.6 + 0.8 * p)).fill({ color: 0xe5e7eb, alpha: 0.35 * alpha });
      for (let i = 0; i < puffs; i++) {
        const angle = (i / puffs) * Math.PI * 2;
        const d = radius * (0.4 + 1.1 * p);
        g.circle(Math.cos(angle) * d, Math.sin(angle) * d, radius * 0.35 * (1 - 0.5 * p)).fill({ color: 0xf8fafc, alpha: 0.7 * alpha });
      }
      return true;
    });
  }

  // --- hints and targeting ---

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
    const [sx, sy] = this.toScreen(zone.x, zone.y, l);
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

    const cardType = config.cards[card]?.type ?? "";
    const zones = (config.deploy_zones["0"] ?? []).filter((z) => z.card_types.includes(cardType));
    const zoneColor = isSpell ? 0xf97316 : 0x22c55e;
    if (isSpell) {
      this.deployOverlay
        .rect(l.ox + l.tile, l.oy + 2 * l.tile, (aw - 2) * l.tile, (al - 4) * l.tile)
        .fill({ color: zoneColor, alpha: 0.12 });
    } else {
      // player's half (P0): y from 2 to river_y - 0.5 → screen from the top
      const topY = al - (config.river_y - 0.5);
      const bottomY = al - 2;
      this.deployOverlay
        .rect(l.ox + l.tile, l.oy + topY * l.tile, (aw - 2) * l.tile, (bottomY - topY) * l.tile)
        .fill({ color: zoneColor, alpha: 0.14 });
    }
    for (const zone of zones) {
      const [sx, sy] = this.toScreen(zone.x, zone.y, l);
      this.deployOverlay.circle(sx, sy, 0.9 * l.tile).stroke({ color: zoneColor, width: 2, alpha: 0.7 });
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
