# Astro-Ai-Lab – dauerhafte Projektanweisungen für Codex

Diese Datei ist die dauerhafte Arbeitsanweisung für Codex in diesem
Projektverzeichnis. Lies sie zu Beginn jeder neuen Aufgabe. Lies zusätzlich
die relevante `README.md`, Dokumentation und den betroffenen Code, bevor du
Änderungen oder Experimente vornimmst.

## Projektziel

Astro-Ai-Lab besteht aus zwei aufeinander aufbauenden Teilen:

1. **Praktikum:** Mit dem astrophysikalischen Simulationspaket `astronomix`
   aussagekräftige Supernova-Simulationen durchführen und konsistente Paare
   aus Anfangs- und Endzuständen als Trainingsdaten erzeugen.
2. **Bachelorarbeit:** Ein U-Net entwickeln, das aus einem Anfangszustand
   direkt einen definierten späteren Zustand vorhersagt und dadurch eine
   aufwendige zeitliche Simulation beschleunigt oder ersetzt.

Das Modell soll nicht nur numerisch gute Werte liefern, sondern hinsichtlich
physikalischer Aussagekraft, Generalisierung und Rechenzeit fair mit der
direkten Simulation verglichen werden.

## Arbeitsauftrag

Unterstütze bei Planung, Implementierung, Auswertung und Dokumentation des
Projekts. Arbeite in umsetzbaren, überprüfbaren Schritten. Erkläre
Fachbegriffe verständlich und konzentriere dich bei einer konkreten Aufgabe
auf deren direkt benötigten Umfang.

Besonders zu berücksichtigen sind:

- Erzeugung, Speicherung und Prüfung von Anfangs-/Endzuständen;
- Aufbau, Training und Auswertung eines U-Net;
- physikalisch sinnvolle Größen wie Dichte, Druck, Temperatur,
  Geschwindigkeit, Energie und Impuls;
- faire Laufzeit- und Geschwindigkeitsvergleiche mit der direkten Simulation;
- Tests auf Anfangszuständen, die nicht im Training vorkommen;
- nachvollziehbare Dokumentation von Parametern, Datensätzen,
  Modellversionen, Checkpoints und Ergebnissen.

## Umgang mit offenen Punkten

Nichts als festgelegt behandeln, was nicht aus den Dateien oder einer
expliziten Nutzerentscheidung hervorgeht. Offene Annahmen immer kenntlich
machen und gegebenenfalls zur Abstimmung mit der Forschungsgruppe markieren.
Das betrifft insbesondere:

- Definition von Anfangs- und Endzustand;
- Supernova- und Umgebungsparameter;
- simulierte Größen, Einheiten und Normierungen;
- räumliche und zeitliche Auflösung;
- Anzahl und Variation der Simulationen;
- verfügbare CPU-/GPU-/Speicherressourcen;
- Genauigkeits- und Laufzeitanforderungen;
- Erfolgskriterien und akzeptable Fehlergrenzen.

Wenn vorhandene Dateien thematisch von diesem Projektkontext abweichen,
prüfe zuerst deren tatsächlichen Inhalt. Leite aus Dateinamen allein keine
physikalischen oder methodischen Eigenschaften ab.

## Berechtigungen und Sicherheitsregeln

Codex darf im Projektverzeichnis:

- Dateien lesen, analysieren und relevante Dokumentation prüfen;
- Quellcode, Tests, Konfigurationen und Dokumentation ändern oder neu
  erstellen, wenn dies zur konkret beauftragten Aufgabe gehört;
- nicht-destruktive Prüfungen, kleine Testläufe und Auswertungen durchführen;
- reproduzierbare Skripte und Protokolle für Experimente anlegen.

Dabei gelten immer diese Grenzen:

- Bestehende Nutzeränderungen erhalten; keine Änderungen anderer Aufgaben
  zurücksetzen oder „aufräumen“.
- Keine Checkpoints, Trainingsdaten, Messdaten, Auswertungen oder sonstigen
  wertvollen Ergebnisse löschen oder überschreiben, ohne dass dies ausdrücklich
  beauftragt wurde.
- Keine mehrstündigen Trainingsläufe, Remote-Jobs, umfangreichen
  Simulationen oder kostenpflichtigen Ressourcen starten, ohne vorher
  Aufwand, Ziel und Ausgabe klar zu benennen.
- Keine GPU verwendenden Befehle ungefragt ausführen. Dazu gehören
  insbesondere Training, Fine-Tuning, Inferenz, Rollouts, Simulationen und
  Auswertungen, sobald sie GPU-Rechenleistung beanspruchen können. Vor jedem
  solchen Lauf muss Codex den Nutzer fragen und auf dessen Angabe warten,
  welche GPU verwendet werden darf beziehungsweise frei ist.
- Auch ein kurzer GPU-Test oder ein Lauf mit kleiner Datenmenge benötigt diese
  vorherige Freigabe, wenn dafür eine Universitäts-GPU beansprucht wird.
- Die aktuelle Ausgangskonfiguration verwendet CPU-JAX. Falls Simulationen zu
  langsam sind, darf Codex eine CUDA-fähige JAX-/jaxlib-Version als mögliche
  Optimierung prüfen und vorschlagen. Eine solche Installation oder Nutzung
  darf erst erfolgen, nachdem der Nutzer eine freie GPU genannt und die
  Kompatibilität von GPU, CUDA und JAX bestätigt hat.
- Keine Commits, Pushes, Veröffentlichungen oder externen Nachrichten ohne
  ausdrücklichen Auftrag.
- Unzusammenhängende Dateien nicht ändern.

## Vorgehensweise bei Aufgaben

1. Arbeitsbereich, Git-Status, relevante README-Dateien und betroffenen Code
   prüfen.
2. Ziel, vorhandenen Stand, Annahmen und offene Fragen kurz festhalten.
3. Eine kleine, nachvollziehbare Änderung oder einen klar abgegrenzten
   Versuch durchführen.
4. Änderungen mit geeigneten Tests, Dry-Runs oder Plausibilitätsprüfungen
   verifizieren.
5. Ergebnis, betroffene Dateien, verbleibende Risiken und nächste Schritte
   dokumentieren.

## Arbeitsprotokoll und wissenschaftliche Dokumentation

Der gesamte wesentliche Arbeitsverlauf muss fortlaufend in `ARBEITSPROTOKOLL.md` im Projektverzeichnis dokumentiert werden. Das Protokoll dient später als Grundlage für den Praktikumsbericht und die Bachelorarbeit.

Nach jeder wesentlichen Aufgabe einen neuen chronologischen Eintrag anhängen. Alte Einträge nicht nachträglich überschreiben. Ein Eintrag soll, soweit zutreffend, enthalten:

- Datum und kurze Bezeichnung der Aufgabe;
- Ziel und Ausgangslage;
- verwendete Dateien, Datensätze und relevante Parameter;
- getroffene Entscheidungen und ausdrücklich markierte Annahmen;
- geänderte oder neu erstellte Dateien;
- ausgeführte Prüfungen, Tests oder Auswertungen;
- Ergebnisse und deren vorsichtige Interpretation;
- Fehler, offene Fragen und nächste Schritte;
- bei GPU-Läufen: vorherige Nutzerfreigabe, GPU, Laufzeit und Zweck.

Auch negative Ergebnisse, verworfene Ansätze und Änderungen der Planung sind zu dokumentieren, sofern sie für die wissenschaftliche Nachvollziehbarkeit relevant sind. Keine Zugangsdaten, Tokens oder andere Geheimnisse in das Protokoll schreiben. Wenn Informationen fehlen, als „nicht dokumentiert“ oder „offen“ kennzeichnen statt sie zu erfinden.

Bei späteren Zusammenfassungen, Berichten oder Kapiteln der Bachelorarbeit zuerst das Arbeitsprotokoll, die README-Dateien, Versuchsdokumentationen und gespeicherten Ergebnisse heranziehen.

Bei Experimenten mindestens dokumentieren:

- Datensatz und genaue Datenaufteilung;
- Zufallsseed;
- Eingabe-/Ausgabedefinition und Einheiten;
- Normalisierung und Loss;
- Modellvariante und Hyperparameter;
- Trainings- und Inferenz-Stride beziehungsweise Rollout-Strategie;
- Hardware, Laufzeit und relevante Softwareversionen;
- verwendete Checkpoints und Auswertungsfenster.

## Bewertung

Ein einzelner MAE-Wert reicht nicht als Erfolgsnachweis. Wenn passend,
vergleiche mindestens Fehlermaße über den Vorhersagehorizont, RMSE oder MAE,
physikalische Erhaltungsgrößen, räumliche beziehungsweise spektrale
Strukturen, zeitliche Korrelation, qualitative Felder und Laufzeit. Vergleiche
Modelle auf denselben Testdaten und mit transparenten Normalisierungs- und
Split-Regeln.

## Kommunikation

Antworte grundsätzlich auf Deutsch, sofern der Nutzer nichts anderes verlangt.
Sei präzise und transparent: Trenne bestätigte Fakten, Annahmen,
Vorschläge und offene Entscheidungen. Bei Codeänderungen nenne kurz, was
geändert wurde und wie es geprüft wurde. Bei wissenschaftlichen Aussagen
keine Genauigkeit oder physikalische Gültigkeit behaupten, die nicht durch
Daten, Tests oder Quellen belegt ist.

- Bei abgeschlossenen oder stabilen wesentlichen Arbeitsschritten den Nutzer
  ausdrücklich an einen passenden `commit` und anschließend gegebenenfalls
  `push` erinnern. Commits und Pushes niemals ohne ausdrücklichen Auftrag
  selbst ausführen.

## Aktueller Stand der Snapshot-Animation (2026-10-26)

- Das 2D-Simulationsskript liegt unter `Praktikum_Datenzubereitung/Supernovae_2D.py`.
- Die Animationsfunktion liegt unter `Praktikum_Datenzubereitung/Supernova_plot.py`.
- Das 2D-Skript muss aus dem Verzeichnis `Praktikum_Datenzubereitung` gestartet werden, weil es `Supernova_plot.py` direkt importiert:

  ```bash
  cd /export/home/bgschwen/supernovae-thesis/Praktikum_Datenzubereitung
  python -u Supernovae_2D.py
  ```

- Das aktuell verwendete Micromamba-Environment heißt `pde`; der aktive Python-Interpreter ist `/export/home/bgschwen/micromamba/envs/pde/bin/python`.
- Im Environment `pde` wurden `jax`, `astronomix`, `numpy`, `matplotlib` und `PIL` erfolgreich importiert. Der Test lautete:

  ```bash
  python -c "import jax, astronomix, numpy, matplotlib, PIL; print('Alle benötigten Pakete importierbar'); print('JAX-Geräte:', jax.devices())"
  ```

- JAX verwendet derzeit `CpuDevice(id=0)`. Die Meldung, dass keine CUDA-fähige `jaxlib`-Installation vorhanden ist, ist erwartetes CPU-Verhalten und kein Fehler. Keine CUDA-/GPU-Installation oder GPU-Nutzung ohne vorherige Nutzerfreigabe.
- Die Ausgangsparameter in `Supernovae_2D.py` sind derzeit `num_cells=100`, `num_snapshots=60`, `t_end=0.2`, `C_cfl=0.4`, `explosion_radius=0.3` und `interval=100` für die GIF-Darstellung. Änderungen an Parametern vor einem neuen Lauf ausdrücklich dokumentieren.
- Das bereits vorhandene `supernova_pressure.gif` ist gültig und darf nicht überschrieben werden. `Supernova_plot.py` behandelt den angegebenen Namen als Basisnamen und verwendet bei vorhandenen Dateien automatisch `supernova_pressure_1.gif`, danach `_2.gif`, `_3.gif` usw. Der tatsächlich verwendete Pfad wird vor dem Speichern ausgegeben.
- Jeder Start von `Supernovae_2D.py` führt zunächst die vollständige CPU-Simulation aus und erzeugt erst danach die GIF-Datei. `movie.save(...)` kann deshalb längere Zeit benötigen. `plt.show()` wird nach dem Speichern aufgerufen und hält den Prozess offen, bis das Matplotlib-Fenster geschlossen wird.
- Falls VS Code scheinbar hängt: zuerst die letzte Konsolenausgabe prüfen, keine zweite Simulation parallel starten und nach erfolgreichem Speichern das Matplotlib-Fenster schließen. Ein fehlendes neues GIF deutet eher auf eine Unterbrechung während `time_integration(...)` oder `movie.save(...)` hin als auf `plt.show()`.
- Vor einem neuen Lauf keine wertvollen Ergebnisdateien löschen oder überschreiben. Bei Problemen zuerst die vollständige Fehlermeldung und die letzte Konsolenausgabe sichern.

## Aktueller geplanter 256er-Sedov-Vergleich (2026-10-30)

- `Praktikum_Datenzubereitung/Supernovae_1D.py` und `Praktikum_Datenzubereitung/Supernovae_3D.py` verwenden derzeit `num_cells=256`, `num_snapshots=60`, `t_end=0.1`, `C_cfl=0.4`, `gamma=5/3` und `E_expl=1.0`.
- Die vorgesehenen Ausgabedateien sind `Praktikum_Datenzubereitung/supernova_1d_profiles_256cells.gif` beziehungsweise `Praktikum_Datenzubereitung/supernova_3d_radial_profiles_256cells.gif`; die zugehörigen Profildaten werden als `.npz` mit demselben Basisnamen gespeichert.
- CPU-only-Start aus dem Projekt-Hauptverzeichnis:

  ```bash
  cd /export/home/bgschwen/supernovae-thesis
  JAX_PLATFORMS=cpu /export/home/bgschwen/micromamba/envs/pde/bin/python -u -m Praktikum_Datenzubereitung.Supernovae_1D
  ```

  Danach, erst wenn der 1D-Lauf vollständig beendet ist:

  ```bash
  cd /export/home/bgschwen/supernovae-thesis
  JAX_PLATFORMS=cpu /export/home/bgschwen/micromamba/envs/pde/bin/python -u -m Praktikum_Datenzubereitung.Supernovae_3D
  ```

- Die beiden Läufe dürfen nicht parallel gestartet werden. Der 3D-Lauf mit `256^3` Zellen und 60 Snapshots kann sehr lange dauern und viel Speicher benötigen.
- Nach einem VS-Code- oder Chat-Abbruch zuerst Prozessstatus, letzte Konsolenausgabe und vorhandene 256er-Ausgaben prüfen. Keine zweite Simulation starten, bevor ausgeschlossen ist, dass der erste Prozess noch läuft. Ein neuer Chat soll diesen Abschnitt lesen und dem Nutzer bei Bedarf direkt die obigen CPU-Kommandos geben.
- **Startregel des Nutzers:** Codex soll keine Simulationen oder Trainings-/Auswertungsläufe selbst starten. Codex darf Konfigurationen prüfen, Ressourcen und Prozesse kontrollieren sowie Startkommandos ausgeben; den eigentlichen Lauf startet der Nutzer manuell.
- Nach erfolgreichen Läufen lautet der Vergleichsbefehl:

  ```bash
  /export/home/bgschwen/micromamba/envs/pde/bin/python compare_sedov_profiles.py --one-d Praktikum_Datenzubereitung/supernova_1d_profiles_256cells.npz --three-d Praktikum_Datenzubereitung/supernova_3d_radial_profiles_256cells.npz --analytic Praktikum_Datenzubereitung/sedov_analytic_profiles.npz --output Praktikum_Datenzubereitung/sedov_1d256_3d256_comparison.gif
  ```
