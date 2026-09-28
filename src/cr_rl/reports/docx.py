"""Generuje raport Word (max ~4 strony): cel projektu + rozwiązanie + wyniki."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cr_rl.paths import MODELS_DIR, REPORT_OUTPUT

FIGURES = MODELS_DIR / "stats" / "figures"
OUTPUT = REPORT_OUTPUT
MLP_DIAGRAM = FIGURES / "architecture_mlp.png"


def _box(ax, x, y, w, h, text, color="#e8f0fe", ec="#1a73e8"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.2, edgecolor=ec, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
        linewidth=1.0, color="#444",
    ))


def generate_mlp_diagram(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Architektura MaskablePPO (MlpPolicy)", fontsize=11, fontweight="bold")

    _box(ax, 4.8, 6.0, 2.2, 0.7, "Wejście obs\n153 cech", "#fff8e1", "#f9a825")
    _arrow(ax, 5.9, 6.0, 3.5, 5.3)
    _arrow(ax, 5.9, 6.0, 8.5, 5.3)

    ax.text(2.5, 5.5, "π (policy)", fontsize=9, fontweight="bold", color="#1565c0")
    for i, t in enumerate(["153→256", "256→256", "256→13"]):
        _box(ax, 1.5, 4.2 - i * 1.1, 2.0, 0.65, f"Linear\n{t}")

    ax.text(8.0, 5.5, "V (value)", fontsize=9, fontweight="bold", color="#6a1b9a")
    for i, t in enumerate(["153→256", "256→256", "256→1"]):
        _box(ax, 7.5, 4.2 - i * 1.1, 2.0, 0.65, f"Linear\n{t}", "#f3e5f5", "#6a1b9a")

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build() -> None:
    generate_mlp_diagram(MLP_DIAGRAM)

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.paragraph_format.line_spacing = 1.1
    style.paragraph_format.space_after = Pt(4)

    # --- Strona tytułowa ---
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Agent RL do uproszczonego Clash Royale\n")
    r.bold, r.font.size = True, Pt(17)
    t.add_run("\nRaport z projektu\n\n").font.size = Pt(13)
    t.add_run("Autor: Piotr\nData: 16.06.2026\n").font.size = Pt(11)
    doc.add_page_break()

    # --- 1. Cel projektu ---
    doc.add_heading("1. Cel projektu", level=1)
    doc.add_paragraph(
        "Celem projektu było zaprojektowanie i implementacja uproszczonej symulacji gry "
        "inspirowanej Clash Royale oraz wytrenowanie agenta uczenia ze wzmocnieniem (RL), "
        "który podejmuje skuteczne decyzje taktyczne w czasie rzeczywistym — wybór karty, "
        "strefy zagrania i momentu ataku/obrony."
    )
    doc.add_paragraph(
        "Problem: jak nauczyć komputer grać w grę strategiczną bez podłączania się do "
        "prawdziwego Clash Royale. Wymaga to kontrolowanego środowiska, w którym da się "
        "zdefiniować stan gry, przestrzeń akcji, nagrodę i przeciwnika treningowego."
    )
    doc.add_paragraph("Cele szczegółowe:", style="List Bullet")
    for item in (
        "Zbudować symulację areny z eliksirem, kartami i walką jednostek.",
        "Zaimplementować bota regułowego jako benchmark.",
        "Wytrenować agenta MaskablePPO i porównać go z baseline.",
    ):
        p = doc.add_paragraph(item, style="List Bullet 2")

    # --- 2. Rozwiązanie ---
    doc.add_heading("2. Jak rozwiązano problem", level=1)

    doc.add_heading("2.1 Środowisko symulacji", level=2)
    doc.add_paragraph(
        "Rdzeń to moduł board.py: arena 18×32, dwa mosty, 6 kart w talii (Knight, Giant, "
        "Cannon, Musketeer, Hog_Rider, Fireball), eliksir, wieże i walka jednostek. "
        "Wrapper gym_env.py udostępnia API Gymnasium — gracz 0 (RL) vs gracz 1 (LogicAgent)."
    )

    doc.add_heading("2.2 Reprezentacja stanu i akcji", level=2)
    tbl = doc.add_table(rows=5, cols=3)
    tbl.style = "Table Grid"
    rows = [
        ("Sekcja", "Indeksy", "Opis"),
        ("Globalne", "0–2", "Eliksir ×2, czas meczu"),
        ("Wieże", "3–8", "HP 6 wież"),
        ("Ręka", "9–32", "One-hot 4 slotów × 6 kart"),
        ("Jednostki", "33–152", "24 jednostki × 5 cech"),
    ]
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            tbl.rows[i].cells[j].text = val
    doc.add_paragraph()
    doc.add_paragraph(
        "Przestrzeń akcji: 13 możliwości (0 = pass, 1–12 = slot karty × strefa deploy). "
        "Nagroda: zmiana łącznego HP wież wroga minus własnych. MaskablePPO odcina "
        "nielegalne akcje (brak eliksiru na kartę)."
    )

    doc.add_heading("2.3 Bot regułowy (LogicAgent)", level=2)
    doc.add_paragraph(
        "Przeciwnik treningowy oparty na regułach: snapshot planszy → generowanie kandydatów "
        "zagrania → scoring → wybór najlepszego. Obsługuje obronę (Hog, Giant), Fireball, "
        "push i cycle eliksiru. Stanowi trudny, powtarzalny benchmark."
    )

    doc.add_heading("2.4 Agent RL i trening", level=2)
    doc.add_paragraph(
        "Algorytm: MaskablePPO (Stable-Baselines3), sieć MLP z osobnymi gałęziami policy "
        "[256,256] i value [256,256]. Trening: 1 mln kroków, 8 środowisk równoległych, "
        "batch 256, entropy 0,02. Logi: TensorBoard, checkpointy co 100k."
    )
    if MLP_DIAGRAM.exists():
        doc.add_picture(str(MLP_DIAGRAM), width=Cm(13))
        cap = doc.add_paragraph("Rys. 1. Architektura sieci (wejście 153 → akcja 13 / wartość 1)")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)

    # --- 3. Wyniki ---
    doc.add_heading("3. Wyniki", level=1)
    doc.add_paragraph(
        "Ewaluacja w tym samym układzie co trening: agent na P0 vs LogicAgent na P1 (gym_env)."
    )
    res = doc.add_table(rows=4, cols=4)
    res.style = "Table Grid"
    for i, row in enumerate([
        ("Agent", "Epizody", "Wygrane", "Win rate"),
        ("Losowy (bez modelu)", "30", "14", "46,7%"),
        ("ppo_cr_final (po treningu)", "100", "69", "69,0%"),
        ("Logic vs Random (osobny test)", "50", "50 Logic", "100% Logic"),
    ]):
        for j, val in enumerate(row):
            res.rows[i].cells[j].text = val

    doc.add_paragraph()
    doc.add_paragraph(
        "Losowy baseline (46,7%) to RLAgent bez wczytanego modelu — losowy wybór legalnych akcji. "
        "Po treningu ppo_cr_final osiąga 69%, co potwierdza, że sieć nauczyła się strategii "
        "lepszej niż przypadkowe zagrania. Wzrost: ok. +22 p.p. win rate."
    )

    win_rate_fig = FIGURES / "win_rate_training.png"
    if win_rate_fig.exists():
        doc.add_picture(str(win_rate_fig), width=Cm(12))
        cap = doc.add_paragraph("Rys. 2. Win rate w trakcie treningu")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)

    # --- 4. Wnioski ---
    doc.add_heading("4. Wnioski", level=1)
    for item in (
        "Cel osiągnięty: agent RL wygrywa z botem regułowym znacznie częściej niż losowy gracz.",
        "Kluczowe elementy rozwiązania: jawna symulacja, maska akcji, nagroda z HP wież, LogicAgent jako przeciwnik.",
        "Prosta sieć MLP [256,256] wystarcza przy 153 cechach wejściowych.",
        "Ograniczenia: uproszczona gra (6 kart), brak self-play, trening vs jeden typ przeciwnika.",
    ):
        doc.add_paragraph(item, style="List Number")

    doc.add_heading("Bibliografia", level=2)
    for ref in (
        "Schulman J. et al., Proximal Policy Optimization Algorithms, arXiv:1707.06347.",
        "Sutton R. S., Barto A. G., Reinforcement Learning: An Introduction.",
        "Stable-Baselines3 / sb3-contrib (MaskablePPO).",
    ):
        doc.add_paragraph(ref, style="List Number")

    doc.save(OUTPUT)
    print(f"Zapisano: {OUTPUT}")


if __name__ == "__main__":
    build()
