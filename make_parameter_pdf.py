"""Erzeuge die Übersicht der Parameterbereiche als PDF.

Die PDF wird bewusst ohne GPU-Lauf und ohne externe Downloads erzeugt.
"""

from __future__ import annotations

from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


OUTPUT = "parameterbereiche_supernova_trainingsdaten.pdf"


def add_title(fig, title: str, subtitle: str | None = None) -> None:
    fig.text(0.06, 0.95, title, fontsize=18, weight="bold", va="top")
    if subtitle:
        fig.text(0.06, 0.915, subtitle, fontsize=10, va="top", color="#444444")


def add_table_page(pdf: PdfPages, title: str, subtitle: str, headers, rows) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    add_title(fig, title, subtitle)

    wrapped = []
    for row in rows:
        wrapped.append([fill(str(cell), width=34 if i else 24) for i, cell in enumerate(row)])

    table = ax.table(
        cellText=wrapped,
        colLabels=headers,
        colWidths=[0.20, 0.29, 0.51],
        cellLoc="left",
        colLoc="left",
        bbox=[0.04, 0.10, 0.92, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        cell.set_linewidth(0.5)
        cell.PAD = 0.012
        if row == 0:
            cell.set_facecolor("#D9EAF7")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        cell.get_text().set_va("top")
    fig.text(
        0.06,
        0.045,
        "Astro-Ai-Lab | Stand: 9. November 2026 | Bereiche sind vorläufig und müssen mit der Betreuung abgestimmt werden.",
        fontsize=7.5,
        color="#555555",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_text_page(pdf: PdfPages, title: str, sections) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    add_title(fig, title)
    y = 0.88
    for heading, paragraph in sections:
        fig.text(0.07, y, heading, fontsize=12, weight="bold", va="top")
        y -= 0.035
        for line in paragraph.split("\n"):
            wrapped = fill(line, width=105)
            fig.text(0.07, y, wrapped, fontsize=9.5, va="top", linespacing=1.35)
            y -= 0.055 + 0.018 * (wrapped.count("\n"))
        y -= 0.025
    fig.text(0.07, 0.045, "Astro-Ai-Lab | Stand: 9. November 2026", fontsize=7.5, color="#555555")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with PdfPages(OUTPUT) as pdf:
        add_table_page(
            pdf,
            "Parameterbereiche für Supernova-Trainingsdaten",
            "Aktuelle Referenzkonfiguration der bisher geplotteten Läufe",
            ["Parameter", "Aktueller Wert", "Bemerkung"],
            [
                ("Auflösung", "$128^3$ Zellen", "Gemeinsame räumliche Auflösung der aktuellen Läufe."),
                ("Boxgröße", "100 pc", "Physikalische Referenzskalierung."),
                ("Dichte", "$n_H=1\\,cm^{-3}$", "Bisher nur ein Referenzwert."),
                ("Temperatur", "$T_0=10^4$ K", "Warmes, ionisiertes Medium; Cooling-Boden ebenfalls $10^4$ K."),
                ("Supernovaenergie", "$10^{51}$ erg", "Kanonischer Einzel-Supernova-Referenzwert."),
                ("Explosionsradius", "0,03 Codeeinheiten ca. 3 pc", "Zusätzlich 2 Glättungszellen."),
                ("Turbulenz", "Amplitude 9, $k_f=25$, $t_{corr}=1$", "Amplitude ist nicht direkt eine Machzahl."),
                ("Seed", "42", "Reproduzierbare Medium-Erzeugung."),
                ("Cooling", "Schure2009-CIE", "Optisch dünne, tabellierte CIE-Kühlkurve."),
                ("Solver Phase 2", "PP-WENO + PP-Flux-Blending", "Zusätzlich REDISTRIBUTE und vacuum_rest."),
                ("CFL", "0,05 und 0,1 gültig; 0,5 in Prüfung", "Numerischer, kein physikalischer Parameter."),
            ],
        )

        add_table_page(
            pdf,
            "Vorläufige literaturbasierte Bereiche",
            "Vorschlag für die Staffelung zukünftiger Trainingsdaten",
            ["Parameter", "Literaturbezug", "Vorläufige Auswahl"],
            [
                ("Mittlere Dichte $n_H$", "Kim & Ostriker: 0,1 bis 100 cm$^{-3}$. Zhang & Chevalier: 1, 10, 100 cm$^{-3}$.", "0,1; 1; 10; 100 cm$^{-3}$"),
                ("Turbulente Machzahl", "Warmes ionisiertes Medium: $M_s\\lesssim2$. Molekülwolken: $M=3,10,30$.", "1; 2; 3; 10; 30; Regime getrennt kennzeichnen."),
                ("Supernovaenergie", "Einzel-Supernova typischerweise ungefähr $10^{51}$ erg.", "0,5; 1; 2 $\\times10^{51}$ erg als Explorationsspanne, nicht als vollständige Beobachtungsverteilung."),
                ("Temperatur", "Aktuelle Kühlkurve und Skalierung setzen ionisiertes Gas voraus.", "Zunächst $10^4$ K. Kalte Molekülwolken separat erweitern."),
                ("Cooling/Metallizität", "Schure et al.: optisch dünne CIE-Kurve auf SPEX-Basis.", "Zunächst Schure2009-CIE, solare Referenzzusammensetzung, Boden $10^4$ K."),
                ("Forcing-Amplitude", "Nicht direkt als Machzahl interpretierbar.", "Für jedes Ziel-$M_s$ kalibrieren; zunächst $k_f=25$ und $t_{corr}=1$ konstant halten."),
            ],
        )

        add_text_page(
            pdf,
            "Interpretation und offene Punkte",
            [
                (
                    "Machzahl und Forcing",
                    "Die Forcing-Amplitude 9 ist kein physikalisch universeller Machzahlwert. Für den Trainingsdatensatz sollte die tatsächlich erreichte RMS- oder lokale Machzahl bei t=5 gemessen werden. Anschließend wird die Amplitude auf Zielwerte kalibriert.",
                ),
                (
                    "ISM-Regime",
                    "Machzahlen 1 bis 2 passen zum warmen ionisierten Medium. Machzahlen 3, 10 und 30 stammen aus Modellierungen turbulenter Molekülwolken. Für diese kalten Medien wären eine andere mittlere Molekülmasse, ein tieferer Temperaturboden und eine passende Kühlfunktion nötig.",
                ),
                (
                    "Auflösung und Explosionsradius",
                    "Kim und Ostriker empfehlen näherungsweise Delta x und r_init kleiner als r_sf/3. Bei 128^3 Zellen in einer 100-pc-Box ist Delta x ungefähr 0,78 pc. Bei hoher Dichte kann ein Explosionsradius von ungefähr 3 pc deshalb zu groß sein und sollte angepasst oder als nicht konvergiert markiert werden.",
                ),
                (
                    "Empfohlene erste Staffelung",
                    "Zuerst das aktuelle warme Regime systematisch variieren: n_H in {0,1; 1; 10; 100} cm^-3, M_s in {1; 2} und E_SN in {0,5; 1; 2} x 10^51 erg. Danach separat die Molekülwolkenfälle mit M_s = 3, 10 und 30 untersuchen.",
                ),
                (
                    "Literatur",
                    "Kim & Ostriker (2015), Momentum Injection by Supernovae in the Interstellar Medium: https://arxiv.org/abs/1410.1537\nZhang & Chevalier (2018), Numerical simulations of supernova remnants in turbulent molecular clouds: https://arxiv.org/abs/1807.06603\nGaensler et al. (2011), Low-Mach-number turbulence in interstellar gas: https://arxiv.org/abs/1110.2896\nSchure et al. (2009), A new radiative cooling curve based on an up to date plasma emission code: https://arxiv.org/abs/0909.5204",
                ),
            ],
        )


if __name__ == "__main__":
    main()
