# Supernovae-Thesis

Dieses Repository enthält die Simulations- und Datenaufbereitungsarbeiten für
die Supernovae-Praktikums- und Bachelorarbeitsprojekte.

## Struktur

- `Praktikum_Datenzubereitung/`: Simulationsskripte, Auswertung und
  Datenaufbereitung
- `compare_sedov_profiles.py`, `plot_3d_slices.py`, `sedov_analytic.py`:
  Vergleich und Analyse
- `ARBEITSPROTOKOLL.md`: chronologische wissenschaftliche Dokumentation

Virtuelle Umgebungen, Python-Caches und große Simulationsergebnisse werden
nicht in Git gespeichert. Die erzeugten `.npz`-, `.gif`-, `.log`-, `.png`-,
`.csv`-, `.mp4`- und `.html`-Dateien bleiben lokal beziehungsweise in einem
separaten Datenbereich.

Das verwendete Python-Environment liegt außerhalb des Repositories. Der
aktuelle CPU-Interpreter ist in `AGENTS.md` dokumentiert.

## Serielle Trainingsdatenläufe

`run_training_dataset.py` liest standardmäßig die Experimentliste aus einer
JSON-Konfiguration. Energie- und Dichtesweeps können beim Start die dortige
`runs`-Liste ersetzen. Beispiel für 3 Supernovaenergien und eine
Wasserstoffdichte von 0.1 bis einschließlich 100 cm⁻³ in 5er-Schritten:

```bash
/export/home/bgschwen/micromamba/envs/pde/bin/python -u \
-m Praktikum_Datenzubereitung.run_training_dataset \
--config Praktikum_Datenzubereitung/training_dataset_config.example.json \
--sweep-energy-e51 0.5 1 2 \
--sweep-nh-range 0.1 100 5 \
--dry-run
```

Der Dichtesweep enthält `0.1, 5.1, 10.1, ..., 95.1, 100`; der letzte
Abstand ist daher 4.9 cm⁻³. Zusammen mit den drei Energien entstehen 63
serielle Experimente. `--dry-run` zeigt die Befehle, startet aber keine
Simulationen. Vor dem echten Start `--dry-run` entfernen und die
`CUDA_VISIBLE_DEVICES`-Variable auf eine freie GPU setzen. Der aktuelle
Runner erzeugt pro Energie-Dichte-Paar ein eigenes Medium und anschließend
die zugehörige Supernova.

### Machzahl-Kalibrierung

`calibrate_mach_forcing.py` führt ausschließlich Mediumläufe aus und misst
den finalen Zustand. Zusätzlich speichert es die vollständigen Zustände für
alle Snapshot-Zeitpunkte in der NPZ-Datei. Diese Datei reicht für spätere
Plots: `plot_anfangsmedium.py` kann daraus ein Medium-GIF erzeugen und
`plot_mean_velocity.py` bei manuellem Aufruf einen Velocity-Zeitplot samt
Zeitreihen-CSV. Der Kalibrierungsrunner selbst erzeugt keine PNG-Plots und
keine einzelnen Velocity-CSV-Dateien. Er speichert weiterhin eine gemeinsame
Machzahl-CSV mit den Endwerten und das JSON-Manifest. Der Pilot kann über
`--forcing-amplitudes` und
`--densities` angepasst werden. Der Runner schreibt außerdem die finalen
Machzahl-Messwerte in einen gemeinsamen CSV-Bericht und den Laufstatus in ein
JSON-Manifest. Vorhandene Ausgaben werden nicht überschrieben. Bei 128³
Zellen und 32 Snapshots kann eine einzelne NPZ-Datei ungefähr 1,2 GB groß
werden; zwölf Läufe benötigen daher grob 14 GB für die Snapshot-Dateien.

Für eine reine Befehlsvorschau kann das Skript mit `--dry-run` ohne GPU
ausgeführt werden. Bei echten Läufen ruft es `autocvd` auf, wählt genau eine
GPU ohne laufenden Compute-Prozess und wartet standardmäßig unbegrenzt, wenn
keine verfügbar ist. Die Prüfung erfolgt alle 30 Sekunden. `autocvd` setzt
`CUDA_VISIBLE_DEVICES` im Kalibrierungsprozess; die seriellen Mediumläufe
erben diese Auswahl. Die Standardkonfiguration stammt aus
`training_dataset_config.example.json`, damit die numerischen Einstellungen
mit dem geplanten Trainingsdatensatz übereinstimmen. `--continue-on-error`
führt die restlichen Kombinationen auch dann aus, wenn ein Mediumlauf
scheitert; ungültige Rohdaten werden separat markiert und nicht in die
Machzahl-CSV übernommen.

Startbefehl für den Pilotlauf:

```bash
cd /export/home/bgschwen/supernovae-thesis
JAX_PLATFORMS=cuda,cpu \
/export/home/bgschwen/micromamba/envs/pde/bin/python -u \
-m Praktikum_Datenzubereitung.calibrate_mach_forcing \
--config Praktikum_Datenzubereitung/training_dataset_config.example.json \
--forcing-amplitudes 3 6 9 \
--densities 0.1 1 100 \
--continue-on-error
```

Beispiel für den erweiterten Snapshot-Lauf mit vier Dichten und drei
Forcing-Amplituden (12 serielle Läufe):

```bash
cd /export/home/bgschwen/supernovae-thesis
JAX_PLATFORMS=cuda,cpu \
/export/home/bgschwen/micromamba/envs/pde/bin/python -u \
-m Praktikum_Datenzubereitung.calibrate_mach_forcing \
--config Praktikum_Datenzubereitung/training_dataset_config.example.json \
--forcing-amplitudes 3 9 15 \
--densities 0.1 1 10 100 \
--output-dir Praktikum_Datenzubereitung/mach_calibration_snapshots \
--continue-on-error
```

`autocvd` benötigt `nvidia-smi`. Die erkannte freie GPU wird ausgewählt, aber
`autocvd` reserviert sie nicht über einen Scheduler; gleichzeitig gestartete
Jobs könnten daher in einem sehr kurzen Zeitfenster dieselbe GPU auswählen.
