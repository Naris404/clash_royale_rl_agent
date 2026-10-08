# Fixed 8-card mirror deck, no deck selection

Status: accepted (2026-10-08)

Both players always use the same fixed 8-card deck (Knight, Giant, Hog Rider,
Musketeer, Cannon, Fireball, Arrows, Baby Dragon); there is no card pool and no deck
selection, unlike the real game. Training an agent across varied decks multiplies
the matchups it must learn and the training budget, which a laptop-scale thesis
cannot afford; one mirror deck keeps the observation layout fixed and the results
interpretable. The deck covers every env-v1 mechanic: air units and splash (Baby
Dragon), area spells (Fireball, Arrows), building-targeters (Giant, Hog Rider), and a
building (Cannon).
