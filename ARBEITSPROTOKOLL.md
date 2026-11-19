# Arbeitsprotokoll – Astro-Ai-Lab

Dieses Protokoll dokumentiert den wesentlichen Arbeitsverlauf des Praktikums und der darauf aufbauenden Bachelorarbeit. Einträge werden chronologisch ergänzt und nicht überschrieben.

## 2026-10-23 – Protokoll eingerichtet

- **Ziel:** Dauerhafte Projektanweisungen und ein fortlaufendes Arbeitsprotokoll für zukünftige Codex-Chats einrichten.
- **Entscheidung:** `AGENTS.md` enthält die Arbeitsregeln; dieses Dokument enthält den chronologischen Verlauf, Entscheidungen, Ergebnisse und offene Punkte.
- **GPU-Regel:** GPU-verwendende Trainings-, Simulations-, Inferenz- und Auswertungsläufe dürfen nur nach vorheriger Nutzerfreigabe gestartet werden.
- **Status:** Die Projektanweisung wurde angelegt. Die fachlichen Details der Supernova-Simulationen, Datensätze und Erfolgskriterien sind noch nicht vollständig festgelegt.
- **Nächster Schritt:** Projektstruktur, vorhandene Daten und Anforderungen gemeinsam prüfen und die offenen Entscheidungen mit der Forschungsgruppe sammeln.

## 2026-10-23 – CPU-JAX als Ausgangspunkt

- **Entscheidung:** Die ersten Simulationen laufen zunächst mit CPU-JAX. Falls die Laufzeit zu hoch ist, wird eine CUDA-fähige JAX-/jaxlib-Version geprüft. GPU-Nutzung oder GPU-bezogene Installation erfolgt erst nach Angabe und Freigabe einer freien Universität-GPU durch den Nutzer.

## 2026-10-23 – Offener Vergleich 100 × 100 gegenüber 101 × 101

- **Forschungsfrage:** Im späteren Emulatortraining prüfen, ob eine gerade Gitterauflösung von `100 × 100` gegenüber `101 × 101` einen relevanten Unterschied macht.
- **Zu vergleichen:** Genauigkeit, Symmetrie der zentralen Explosion, Laufzeit und Speicherbedarf.
- **Kontrollbedingungen:** Gleiche physikalische Parameter, gleiche Gebietsgröße, gleiche Modell- und Trainingsbedingungen.
- **Wichtiger Hinweis:** Bei `101 × 101` liegt das Zentrum auf einer Zellmitte, bei `100 × 100` zwischen vier Zellen. Der Vergleich trennt daher Auflösungseffekt und Zentrierungseffekt nicht automatisch.
- **Status:** Noch nicht durchgeführt.

## 2026-10-23 – Vorbereitung der Snapshot-Animation

- **Ziel:** Eine separate Datei zur Darstellung der zeitlichen Entwicklung der 2D-Simulation als Animation bereitstellen.
- **Datei:** `Praktikum_Datenzubereitung/Supernova_plot.py`
- **Umsetzung:** Die Funktion `animate_snapshots(...)` verarbeitet astronomix-Snapshots und kann Druck oder Dichte als GIF beziehungsweise MP4 animieren.
- **Voraussetzung:** In der Simulation müssen `return_snapshots=True`, `num_snapshots` sowie `SnapshotSettings(return_states=True)` aktiviert werden.
- **Wichtig:** Die neue Datei startet selbst keine Simulation und wurde nicht ausgeführt.

## 2026-10-26 – Diagnose eines wiederholten Animationslaufs

- **Ausgangslage:** Beim erneuten Ausführen von `Praktikum_Datenzubereitung/Supernovae.py` reagierte VS Code nicht mehr sichtbar und es wurde kein weiteres GIF beobachtet.
- **Geprüfte Dateien:** `Praktikum_Datenzubereitung/Supernovae.py` und `Praktikum_Datenzubereitung/Supernova_plot.py`.
- **Relevante Parameter:** CPU-JAX-Ausgangskonfiguration mit `num_cells=100`, `num_snapshots=60`, `t_end=0.2`; Ausgabe `supernova_pressure.gif`.
- **Feststellung:** Das Skript startet bei jedem Aufruf zunächst die vollständige Simulation. Erst danach erzeugt `movie.save(...)` das GIF. `plt.show()` wird erst nach dem Speichern aufgerufen und kann anschließend blockierend auf das Schließen des Matplotlib-Fensters warten.
- **Vorhandenes Ergebnis:** `Praktikum_Datenzubereitung/supernova_pressure.gif` ist vorhanden und gültig; geprüft wurden 60 Frames bei 600×500 Pixeln und ungefähr 1,1 MB Dateigröße.
- **Prozessprüfung:** Zum Prüfzeitpunkt lief kein zugehöriger Python-/Simulationsprozess mehr.
- **Interpretation:** Der Hänger bzw. Abbruch beim zweiten Versuch ist nicht abschließend lokalisiert. Wahrscheinliche Stellen sind die erneute CPU-Simulation oder die synchrone GIF-Erzeugung; ein bloß offenes Plotfenster erklärt insbesondere nicht das Ausbleiben einer neuen Datei, da `plt.show()` erst nach `movie.save(...)` ausgeführt wird.
- **Offene Prüfung:** Beim nächsten Versuch sollte anhand der letzten Konsolenausgabe unterschieden werden, ob der Lauf noch in `time_integration(...)` oder bereits in `movie.save(...)` steckt. Ein optionales `show=False` beziehungsweise explizites Schließen der Figure kann danach als kleine Codeverbesserung geprüft werden.

## 2026-10-26 – Vorhandene GIF-Dateien vor Überschreiben schützen

- **Ziel:** Ein erneuter Animationslauf soll das bereits erzeugte GIF nicht überschreiben.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/Supernova_plot.py`.
- **Umsetzung:** `output_path` wird nun als Basisname behandelt. Falls der Pfad bereits existiert, wird automatisch der nächste freie Name mit fortlaufender Nummer gewählt, beispielsweise `supernova_pressure_1.gif`, danach `supernova_pressure_2.gif`.
- **Zusatz:** Der tatsächlich verwendete Ausgabepfad wird vor dem Speichern in der Konsole ausgegeben.
- **Erwartetes Verhalten:** Bestehende GIF- beziehungsweise MP4-Dateien bleiben erhalten; jeder erfolgreiche neue Lauf erzeugt eine neue Datei.
- **Prüfung:** Der Code wurde statisch geprüft; ein vollständiger Simulations- oder Animationslauf wurde nicht gestartet.

## 2026-10-26 – Fehlendes JAX im aktiven Environment

- **Ausgangslage:** Der Start von `python -u Supernovae.py` im aktivierten Environment `pde` auf `compgpu3` schlug mit `ModuleNotFoundError: No module named 'jax'` fehl.
- **Prüfung:** Der verwendete Interpreter ist `/export/home/bgschwen/micromamba/envs/pde/bin/python`. In den vorhandenen Environments `pde` und `base` wurden keine Installationen von `jax`, `jaxlib` oder `astronomix` gefunden. Eine Projektdatei mit reproduzierbaren Abhängigkeiten ist nicht vorhanden.
- **Interpretation:** Der Fehler tritt vor dem eigentlichen Simulationslauf auf und hat nichts mit dem GIF-Dateinamen zu tun. Für die dokumentierte CPU-Ausgangskonfiguration müssen die fehlenden Pakete im Environment `pde` installiert werden.
- **GPU-Abgrenzung:** Es wurde keine GPU-Version installiert und kein Simulationslauf gestartet. Die offizielle JAX-Dokumentation nennt für CPU-Linux die normale JAX-Installation; eine CUDA-Variante bleibt ausdrücklich offen und benötigt vorherige Nutzerfreigabe.
- **Offen:** Die genaue Bezugsquelle beziehungsweise Version von `astronomix` ist im Projekt nicht dokumentiert und muss vor der Installation geklärt oder aus der bisherigen Projektumgebung rekonstruiert werden.

## 2026-10-26 – Abhängigkeiten erfolgreich geprüft

- **Prüfung:** Im Environment `pde` wurden `jax`, `astronomix`, `numpy`, `matplotlib` und `PIL` erfolgreich importiert.
- **JAX-Gerät:** `CpuDevice(id=0)`.
- **Hinweis:** JAX meldete, dass keine CUDA-fähige `jaxlib`-Installation vorhanden ist, und verwendet daher CPU. Es wurde kein GPU-Lauf gestartet.
- **Ergebnis:** Die Python-Umgebung ist grundsätzlich bereit, `Supernovae.py` auszuführen.

## 2026-10-26 – Dauerhafte Dokumentation des Animationsstands in AGENTS.md

- **Ziel:** Nach einem möglichen VS-Code- oder Chat-Abbruch soll der bisherige technische Stand ohne erneute Erklärung nachvollziehbar sein.
- **Geänderte Datei:** `AGENTS.md`.
- **Dokumentiert:** Startverzeichnis und Startbefehl, Environment `pde`, erfolgreich geprüfte Pakete, CPU-JAX-Verhalten, aktuelle Simulationsparameter, automatische GIF-Namensvergabe, bisherige Hänger-Diagnose und Verhalten von `plt.show()`.
- **Prüfung:** Es wurde keine Simulation gestartet und keine GPU verwendet.

## 2026-10-26 – Diagnose des erneuten Stillstands bei 0 %

- **Ausgangslage:** Beim Start von `Praktikum_Datenzubereitung/Supernovae.py` blieb die Fortschrittsanzeige in VS Code erneut bei `0 %`, und der Eindruck entstand, dass `jaxlib` für GPU-Ausführung fehle.
- **Geprüfte Dateien:** `Praktikum_Datenzubereitung/Supernovae.py`, `Praktikum_Datenzubereitung/Supernova_plot.py` sowie die lokal installierte `astronomix`-Implementierung.
- **Geprüfte Umgebung:** `/export/home/bgschwen/micromamba/envs/pde/bin/python`, Python `3.12.13`; `jax==0.12.0.1`, `jaxlib==0.12.0.1`, `astronomix==0.6.6`.
- **Feststellung:** `jaxlib` ist installiert, aber in der CPU-Variante aktiv. `jax.devices()` liefert ausschließlich `CpuDevice(id=0)`; es wurde kein GPU-Lauf gestartet.
- **Interpretation:** Die Fortschrittsanzeige von `astronomix` wird über `jax.debug.callback` aus der kompilierten Zeitschleife aktualisiert. Während der ersten JAX-Kompilierung kann sie daher unverändert bei `0 %` stehen. Zusätzlich wird der Fortschritt nur mit einer Nachkommastelle angezeigt, sodass kleine anfängliche Zeitschritte als `0.0 %` erscheinen können. Ein tatsächlicher Absturz ist damit noch nicht bewiesen.
- **Offen:** Für eine belastbare Ursachenanalyse fehlen die vollständige letzte Konsolenausgabe und die Laufzeit bis zum Abbruch beziehungsweise manuellen Beenden. Eine CUDA-fähige JAX-/jaxlib-Konfiguration darf erst nach Angabe und Freigabe einer geeigneten GPU geprüft oder installiert werden.

## 2026-10-26 – GPU-Lauf mit deaktiviertem CPU-Backend abgebrochen

- **Ausgangslage:** GPU 2 auf `compgpu3` wurde freigegeben. Die CUDA-12-Erweiterung für JAX wurde im Environment `pde` installiert und ein kurzer JAX-GPU-Test war erfolgreich.
- **Verwendeter Start:** `CUDA_VISIBLE_DEVICES=2 JAX_PLATFORMS=cuda /export/home/bgschwen/micromamba/envs/pde/bin/python -u Supernovae.py`.
- **Ergebnis:** Der Lauf brach in `jax.block_until_ready` mit `jax.errors.JaxRuntimeError` ab. `jax.debug.callback` konnte kein lokales CPU-Gerät für den Callback finden.
- **Ursache:** `JAX_PLATFORMS=cuda` schränkt JAX ausschließlich auf CUDA ein. Die `astronomix`-Fortschrittsanzeige verwendet jedoch einen CPU-seitigen `jax.debug.callback`.
- **Korrektur:** Für den nächsten manuellen Lauf muss CPU zusätzlich zugelassen werden, beispielsweise mit `JAX_PLATFORMS=cuda,cpu`, während `CUDA_VISIBLE_DEVICES=2` die physikalische GPU auswählt.
- **Hinweis:** Die Ausgabe `OPTIMAL_BACKEND: using the NATIVE_JAX backend (no compute capability >= 8.0 GPU found)` ist bei der TITAN Xp eine Backend-Auswahlmeldung und nicht die Ursache dieses Abbruchs. Es wurde kein GIF durch diesen fehlgeschlagenen Lauf erzeugt; vorhandene GIF-Dateien wurden nicht verändert.

## 2026-10-26 – Diagnose fehlender Zeitschritte bei Explosionradius 0.3

- **Ausgangslage:** Das neu erzeugte `supernova_pressure_1.gif` zeigt im Titel nur `t = 0` und enthält drei statt der konfigurierten 60 GIF-Frames.
- **Prüfung:** `supernova_pressure.gif` enthält 60 Frames mit unterschiedlichen Bildinhalten; `supernova_pressure_1.gif` enthält drei unterschiedliche Frames. Die aktuelle Konfiguration setzt `num_snapshots=60` und `explosion_radius=0.3`.
- **Codebefund 1:** In `Supernovae.py` wird außerhalb des Explosionsradius `rho=0.0`, aber gleichzeitig `p=0.1` gesetzt. Die Finite-Volume-CFL-Schätzung verwendet `sqrt(gamma*p/rho)` für die Schallgeschwindigkeit. Die Null-Dichte erzeugt damit eine singuläre beziehungsweise nicht endliche Schallgeschwindigkeit und kann einen ungültigen Zeitschritt (`0`/`NaN`) verursachen.
- **Codebefund 2:** In `Supernova_plot.py` läuft `FuncAnimation` mit `blit=True`; `update()` gibt nur das Bild, nicht aber den über `ax.set_title(...)` gesetzten Titel zurück. Der angezeigte Titel kann deshalb bei `t=0` stehen bleiben, selbst wenn sich das Feld ändert.
- **Interpretation:** Das beobachtete Verhalten ist wahrscheinlich eine Kombination aus numerisch problematischem Vakuumzustand und einer unabhängigen Einschränkung der Titelanimation. Eine positive Umgebungsdichte ist als physikalischer Parameter noch offen und muss vor einer endgültigen wissenschaftlichen Auswertung festgelegt werden.
- **Nächste Schritte:** Zuerst einen positiven, dokumentierten Umgebungsdichtewert testen und die Titelanimation mit `blit=False` oder einem zurückgegebenen Titel-Artist korrigieren. Danach Snapshot-Form, Zeitpunkte und `num_iterations` direkt vor der GIF-Erzeugung ausgeben und den GPU-Lauf erneut prüfen.

## 2026-10-26 – Prüfung des geänderten Anfangszustands

- **Aktueller Code:** `explosion_radius=0.3`, Dichte innen `1.0`, außen `0.1`; Druck innen `1.0`, außen `0.0`.
- **Bewertung:** Die positive Außendichte verhindert die vorherige Division durch null in der Schallgeschwindigkeit. Der Außendruck `0.0` ist jedoch eine neue Modellannahme und beschreibt ein druckloses Außenmedium; für einen numerisch weniger scharfen Test ist ein positiver, dokumentierter Umgebungsdruck wie `0.1` zu erwägen.
- **Offener Codepunkt:** `Supernova_plot.py` verwendet weiterhin `blit=True` und gibt im `update()`-Callback nur `image` zurück. Der Titel kann deshalb weiterhin bei `t=0` bleiben. Eine minimale Korrektur ist `blit=False` oder die Rückgabe eines animierten Titel-Artists.
- **Prüfung:** Nur statische Codeanalyse; kein neuer Simulations- oder GPU-Lauf.

## 2026-10-26 – Vorbereitung der 3D-Erweiterung

- **Ziel:** Die bisherige kartesische hydrodynamische 2D-Simulation schrittweise auf 3D erweitern.
- **Codeprüfung:** Die lokal installierte `astronomix`-Version unterstützt `dimensionality=3` für kartesische Finite-Volume-Hydrodynamik. `Supernovae.py` ist aktuell noch auf `dimensionality=2`, zwei Geschwindigkeitskomponenten und einen 2D-Zustand festgelegt. `Supernova_plot.py` prüft ausdrücklich auf 2D-Snapshots und kann 3D-Felder noch nicht direkt darstellen.
- **Vorgesehene technische Änderungen:** dreidimensionales Gitter, dreidimensionaler Mittelpunkt und Radius, zusätzliche `velocity_z`-Komponente sowie eine 3D-Snapshot-Auswertung. Für die erste Visualisierung soll ein zentraler 2D-Schnitt durch das 3D-Druckfeld verwendet werden.
- **Ressourcenannahme:** Ein erster Pilotlauf soll mit deutlich kleinerem Gitter, beispielsweise `32^3` oder `48^3`, und wenigen Snapshots beginnen. `100^3` Zellen mit 60 vollständigen Zustands-Snapshots wären bereits speicherintensiv; eine verbindliche Zielauflösung ist noch offen.
- **Wissenschaftliche Prüfungen:** Vor einer größeren 3D-Datenerzeugung sollen Kugelsymmetrie, radiale Profile, Energie/Masse und Auflösungsabhängigkeit geprüft werden. Es wurde kein 3D-Lauf gestartet.

## 2026-10-26 – Prüfung des aktuellen 3D-Codezwischenstands

- **Aktueller Stand:** `Supernovae.py` verwendet bereits `dimensionality=3`, definiert `uz` und übergibt `velocity_z`. Dichte und Druck besitzen positive Außenwerte.
- **Fehlende Korrektur:** Der Mittelpunkt ist noch zweidimensional (`[box_size/2, box_size/2]`). Für 3D muss eine dritte Koordinate ergänzt werden.
- **Fehlende Korrektur:** `Supernova_plot.py` schneidet weiterhin nur 2D-Snapshot-Arrays aus und weist 3D-Felder zurück. Für die erste Darstellung muss beispielsweise eine zentrale `z`-Ebene aus dem 3D-Druckfeld extrahiert werden.
- **Bestehender Darstellungsfehler:** `blit=True` und die Rückgabe ausschließlich von `image` können weiterhin verhindern, dass der Zeit-Titel aktualisiert wird. `blit=False` ist für die erste 3D-Schnittanimation die robuste Wahl.
- **Ressourcenhinweis:** `num_cells=100` bedeutet im 3D-Fall `100^3` Zellen und sollte zunächst durch einen kleineren Pilotwert wie `32` oder `48` ersetzt werden. Es wurde kein 3D-Lauf gestartet.

## 2026-10-26 – Neue getrennte 3D-Datei geprüft

- **Neue Datei:** `Supernovae_3D.py` im Projektwurzelverzeichnis; die 2D-Version liegt separat als `Praktikum_Datenzubereitung/Supernovae_2D.py` vor.
- **Bereits korrekt:** `dimensionality=3`, dreidimensionaler Mittelpunkt, `uz` und `velocity_z`, positive Außenwerte für Dichte und Druck.
- **Noch erforderlich:** Der Import `from Supernova_plot import ...` passt beim Start aus dem Projektwurzelverzeichnis nicht zum Speicherort der Plotdatei. Der Importpfad oder der Speicherort muss angepasst werden.
- **Noch erforderlich:** `Supernova_plot.py` erwartet weiterhin 2D-Snapshots. Für 3D muss vor `imshow` eine zentrale 2D-Schnittfläche extrahiert werden; zusätzlich sollte `blit=False` für den dynamischen Titel verwendet werden.
- **Ressourcenbefund:** `num_cells=100` bleibt für den ersten 3D-Lauf sehr groß. `supernovae_run.log` ist durch wiederholte `0.0%`-Fortschrittsausgaben bereits etwa 54 MB groß. Zum Prüfzeitpunkt lief kein zugehöriger Simulationsprozess.

## 2026-10-26 – Supernova_plot.py für 2D und 3D erweitert

- **Geänderte Datei:** `Praktikum_Datenzubereitung/Supernova_plot.py`.
- **Umsetzung:** Die Plotfunktion erkennt nun nach Auswahl des Feldes 2D- und 3D-Snapshots. Bei 3D wird automatisch die zentrale z-Schnittfläche ausgewählt und als 2D-Animation dargestellt.
- **Zusatz:** Die Anzahl der Zeitpunkte wird gegen die Anzahl der Feld-Snapshots geprüft. `blit=False` stellt sicher, dass der Zeit-Titel in jedem Frame aktualisiert wird.
- **Prüfung:** Die Datei wurde per `ast.parse` syntaktisch geprüft. Es wurde keine Simulation gestartet und keine GPU verwendet.

## 2026-10-26 – Startkontrolle des getrennten 3D-Skripts

- **Geprüfte Dateien:** `Supernovae_3D.py` und `Praktikum_Datenzubereitung/Supernova_plot.py`.
- **Startpfad:** Der Import in `Supernovae_3D.py` verweist jetzt auf `Praktikum_Datenzubereitung.Supernova_plot` und ist beim Start aus dem Projektwurzelverzeichnis auflösbar.
- **Statische Prüfung:** Beide Dateien sind syntaktisch gültig; der Plot-Import wurde mit `MPLBACKEND=Agg` erfolgreich getestet. Es wurde keine Simulation und keine GPU-Rechenoperation gestartet.
- **Verbleibender Ressourcenpunkt:** Das 3D-Skript verwendet weiterhin `num_cells=100` und `num_snapshots=60`. Der Codepfad ist damit vorbereitet, aber für einen ersten kontrollierten 3D-Lauf ist ein kleineres Gitter wie `32` oder `48` weiterhin empfohlen.
- **Prozessstatus:** Zum Prüfzeitpunkt lief kein Simulationsprozess. Die vorhandene große Logdatei wurde nicht gelöscht oder überschrieben.

## 2026-10-26 – Konzept für interaktive 3D-Visualisierung

- **Anforderung:** Im 3D-Feld sollen nur Zellen mit Druck oberhalb des anfänglichen Außendrucks sichtbar sein; zusätzlich soll eine zentrale z-Schnittfläche angezeigt und die Ansicht interaktiv gedreht werden können.
- **Umgebungsprüfung:** Matplotlib `3.11.1` und `mpl_toolkits.mplot3d` sind verfügbar. Plotly ist im Environment `pde` nicht installiert.
- **Vorgesehene Umsetzung:** Schwellenwertmaske `pressure > p_background` als transparente 3D-Punkt-/Voxel-Darstellung, zentrale z-Schnittfläche als farbige Fläche und interaktive Rotation über ein Matplotlib-3D-Fenster. Für `100^3` Zellen soll keine vollständige Voxelwolke verwendet werden; Schwellenwertfilter und gegebenenfalls Downsampling sind erforderlich.
- **Ausgabegrenze:** Ein GIF kann nicht per Maus gedreht werden. Die interaktive Drehung benötigt ein GUI-Fenster; `MPLBACKEND=Agg` ist dafür nicht geeignet.
- **Status:** Nur Konzeptprüfung; keine Datei geändert und kein Simulationslauf gestartet.

## 2026-10-26 – Interaktive 3D-Darstellung implementiert

- **Neue Datei:** `Praktikum_Datenzubereitung/Supernova_plot_3D.py`.
- **Umsetzung:** Druckzellen oberhalb von `background_pressure + pressure_margin` werden als transparente 3D-Punkte dargestellt. Zusätzlich wird die zentrale z-Schnittfläche als farbige Fläche angezeigt. Die Ansicht ist über ein interaktives Matplotlib-3D-Fenster drehbar; `cell_stride` reduziert die Darstellungskosten.
- **Anpassung:** `Supernovae_3D.py` verwendet nun die neue interaktive 3D-Funktion statt der 2D-GIF-Funktion.
- **Parameter im 3D-Skript:** `background_pressure=0.1`, `cell_stride=2`, `interval=100`.
- **Prüfungen:** Syntaxprüfung aller beteiligten Dateien und CPU-Test mit künstlichen 3D-Snapshot-Daten erfolgreich. Der erste Test fand die nicht mehr verfügbare API `cm.get_cmap` in Matplotlib 3.11; diese wurde auf `matplotlib.colormaps["viridis"]` korrigiert. Es wurde keine echte Simulation und keine GPU verwendet.
- **Bedienhinweis:** Für Mausrotation darf der Lauf nicht mit `MPLBACKEND=Agg` gestartet werden; das interaktive Matplotlib-Fenster muss einen GUI-Backend verwenden.

## 2026-10-26 – Boxgrößenfehler nach erfolgreicher 3D-Simulation korrigiert

- **Ausgangslage:** Die 3D-Simulation erreichte `100 %`, anschließend brach die Darstellung mit einem Broadcasting-Fehler bei `config.box_size` ab.
- **Ursache:** `finalize_config` liefert im 3D-Fall einen Vektor mit drei Kantenlängen; die Plotfunktion behandelte `box_size` noch als Skalar.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/Supernova_plot_3D.py` akzeptiert nun sowohl einen Skalar als auch einen Vektor `(box_x, box_y, box_z)` und setzt Achsengrenzen sowie Seitenverhältnis komponentenweise.
- **Prüfung:** Ein künstlicher CPU-Test mit 3D-Snapshots und einem dreielementigen Boxgrößenvektor war erfolgreich. Die echte Simulation wurde nach dem Fehler nicht erneut gestartet.

## 2026-10-26 – Keine interaktive 3D-Anzeige auf compgpu3

- **Lauf:** Die 3D-Simulation erreichte `100 %`; anschließend wurde nur eine Matplotlib-Warnung ausgegeben.
- **Diagnose:** Das Environment verwendet den nicht-interaktiven Backend `agg`; `DISPLAY` und `WAYLAND_DISPLAY` sind nicht gesetzt. Daher gibt es auf der aktuellen SSH-/Compute-Umgebung kein GUI-Fenster, in dem die Matplotlib-3D-Animation gerendert und gedreht werden kann.
- **Interpretation:** Die Warnung `Animation was deleted without rendering anything` betrifft die fehlende Darstellung, nicht die numerische Simulation. Für Mausrotation wird entweder eine Anzeigeverbindung mit GUI-Backend oder eine browserbasierte HTML-Ausgabe benötigt.
- **Offen:** Plotly ist im Environment nicht installiert. Eine HTML-Alternative oder die Einrichtung von X11/GUI-Weiterleitung ist noch zu entscheiden.

## 2026-10-26 – Browserbasierte interaktive 3D-Ausgabe umgesetzt

- **Entscheidung:** Wegen des fehlenden GUI-/X11-Displays auf `compgpu3` wird die interaktive 3D-Darstellung als HTML-Datei erzeugt.
- **Installation:** `plotly==7.1.0` und `narwhals==2.27.1` wurden im Environment `pde` installiert. Dabei wurde keine GPU verwendet.
- **Geänderte Dateien:** `Praktikum_Datenzubereitung/Supernova_plot_3D.py` um `save_interactive_3d_html(...)` erweitert; `Supernovae_3D.py` verwendet diese Funktion.
- **Funktion:** Die HTML-Datei enthält transparente Drucküberschreitungen, eine zentrale z-Schnittfläche, Mausrotation, Zoom, Zeitschieber sowie Start-/Pause-Steuerung.
- **Prüfung:** Syntax- und HTML-Erzeugungstest mit künstlichen 3D-Snapshots erfolgreich; Testdatei `/tmp/supernova_3d_plot_test.html` wurde erzeugt. Keine echte Simulation gestartet.
- **Zusatz:** Vorhandene HTML-Dateien werden nun wie die GIF-Dateien geschützt und bei Wiederholung automatisch nummeriert. Abschließende Syntax- und Plotly-Importprüfung erfolgreich.

## 2026-10-26 – Druckskalierung und Volumendarstellung angepasst

- **Anforderung:** Die 3D-Darstellung soll keine Mittelpunkt-Punkte zeigen. Hintergrundzellen mit dem Außendruck sollen transparent sein; mit wachsendem Druck sollen Farbe und Deckkraft zunehmen.
- **Geänderte Dateien:** `Praktikum_Datenzubereitung/Supernova_plot_3D.py` und `Supernovae_3D.py`.
- **Umsetzung:** Die HTML-Darstellung verwendet nun einen Plotly-`Volume`-Trace statt `Scatter3d`. Der Hintergrunddruck ist die transparente Untergrenze. Der anfängliche Druck innerhalb des Explosionsradius wird als Referenz übergeben; falls später höhere Druckwerte auftreten, wird der höchste tatsächlich vorkommende Wert zur neuen oberen Farbgrenze. Werte werden nicht am anfänglichen Innendruck abgeschnitten.
- **Zusatz:** Die zentrale z-Schnittfläche bleibt erhalten. Die Farbskalierung wird global über alle Snapshots bestimmt, damit die Farben während der Animation vergleichbar bleiben.
- **Prüfung:** HTML-Erzeugung mit künstlichen Snapshots mit Druckwerten `0.2`, `1.0` und `2.0` erfolgreich. Keine echte Simulation gestartet.

## 2026-10-26 – Logarithmische Druckskalierung ergänzt

- **Anforderung:** Druckwerte knapp oberhalb des Außendrucks `0.1` sollen trotz großer Dynamik sichtbar bleiben.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/Supernova_plot_3D.py`.
- **Umsetzung:** Die Drucknormierung für Volumen und z-Schnitt verwendet nun `log10(p)`. Der Außendruck bleibt die transparente Untergrenze; der höchste tatsächlich auftretende Druck bleibt die obere Farbgrenze. Die Farbskala zeigt logarithmische Druckstufen.
- **Prüfung:** Syntaxprüfung und HTML-Test mit künstlichen Drücken `0.11`, `1.0` und `2.0` erfolgreich. Keine echte Simulation gestartet.

## 2026-10-26 – Kontrastverstärkung der logarithmischen Farbskala getestet

- **Anforderung:** Druckwerte knapp oberhalb von `0.1` sollen noch deutlicher sichtbar sein.
- **Geänderte Dateien:** `Praktikum_Datenzubereitung/Supernova_plot_3D.py` und `Supernovae_3D.py`.
- **Umsetzung:** Nach der logarithmischen Normierung wird zusätzlich eine Potenztransformation mit `contrast_exponent=0.3` verwendet. Werte nahe der Hintergrundgrenze werden dadurch stärker auseinandergezogen; der exakte Hintergrunddruck bleibt bei null Sichtbarkeit.
- **Prüfung:** HTML-Erzeugung mit künstlichen Drücken `0.11`, `1.0` und `2.0` erfolgreich. Keine echte Simulation gestartet.

## 2026-10-26 – Kontrastparameter auf 0.5 geändert

- **Änderung:** In `Supernovae_3D.py` wurde `contrast_exponent` von `0.3` auf `0.5` gesetzt.
- **Bedeutung:** Die Kontrastverstärkung niedriger Drucküberschreitungen ist damit schwächer als bei `0.3`, aber weiterhin logarithmisch verstärkt.
- **Prüfung:** Parameteränderung statisch kontrolliert; keine Simulation gestartet.

## 2026-10-26 – Kontrastparameter auf 0.1 geändert

- **Änderung:** In `Supernovae_3D.py` wurde `contrast_exponent` von `0.5` auf `0.1` gesetzt.
- **Bedeutung:** Niedrige Drucküberschreitungen werden stärker hervorgehoben als bei den vorher getesteten Werten.
- **Prüfung:** Parameteränderung statisch kontrolliert; keine Simulation gestartet.

## 2026-10-26 – Prüfung der geplanten radialen 1D-Simulation

- **Datei:** `Supernovae_1D.py` wurde gefunden, ist derzeit jedoch leer.
- **Erforderlicher Modellwechsel:** Für eine radiale 1D-Simulation muss `geometry=SPHERICAL` zusammen mit `dimensionality=1` verwendet werden. `astronomix` setzt dafür unter anderem HLL, MUSCL sowie reflektierende innere und offene äußere Randbedingungen automatisch beziehungsweise gezielt.
- **Anfangszustand:** Der Radius kommt direkt aus `helper_data.geometric_centers`; ein 2D-/3D-Mittelpunkt entfällt. Es wird nur `velocity_x` übergeben. Die bisherige normierte Anfangsbedingung mit positiver Außen-Dichte und positivem Außen-Druck kann als vorläufige Annahme übernommen werden.
- **Darstellung:** Die vorhandenen Plotdateien erwarten 2D- beziehungsweise 3D-Felder. Für 1D wird eine eigene Linienanimation von Druck oder Dichte über dem Radius benötigt.
- **Status:** Noch kein 1D-Code erstellt und keine Simulation gestartet. Die äußere Radiusgröße, Auflösung und physikalischen Anfangswerte sind als Modellentscheidungen zu dokumentieren.

## 2026-10-26 – Prüfung des aktuellen Supernovae_1D.py

- **Aktueller Stand:** Das Skript enthält nun einen 1D-Entwurf, verwendet aber noch `geometry=CARTESIAN`, einen unnötigen Mittelpunkt-/Normierungsansatz, eine zusätzliche y-Geschwindigkeit und den 2D-Plotter.
- **Syntaxfehler:** `gas_pressure=p,p` erzeugt ein Tupel und muss `gas_pressure=p` lauten.
- **Erforderliche Modellkorrektur:** Für radiale 1D-Symmetrie muss `geometry=SPHERICAL` importiert und verwendet werden. Der Radius ist direkt `helper_data.geometric_centers`; ein `center` und `jnp.linalg.norm(..., axis=-1)` sind nicht erforderlich.
- **Erforderliche Zustandskorrektur:** In 1D nur `velocity_x=ux` an `construct_primitive_state` übergeben; `uy` entfällt.
- **Darstellung:** `Supernova_plot.py` erwartet 2D-Felder und ist für die 1D-Snapshots ungeeignet. Es wird ein eigener radialer Linienplot benötigt.
- **Prüfung:** Nur statische Codeanalyse; keine Simulation gestartet.

## 2026-10-26 – Aktuellen 1D-Code mit gewünschter HTML-Darstellung abgeglichen

- **Aktueller Befund:** `Supernovae_1D.py` verwendet `SPHEICAL` statt `SPHERICAL`, übergibt `velocity` statt `velocity_x` und importiert weiterhin den 2D-Plotter über `from Supernova_plot import ...`.
- **Darstellungsbefund:** Radiale 1D-Snapshots haben die Form `(Snapshots, Zellen)`. Die vorhandene interaktive 3D-HTML-Funktion erwartet dagegen `(Snapshots, Nx, Ny, Nz)` und kann daher nicht direkt mit der 1D-Simulation verwendet werden.
- **Erforderliche Entscheidung:** Für die physikalisch direkte 1D-Auswertung wird ein interaktiver radialer Plot benötigt. Eine 3D-Ansicht ist nur als abgeleitete Visualisierung möglich, indem das radiale Profil `p(r)` in ein kugelsymmetrisches 3D-Feld projiziert wird.
- **Prüfung:** Nur statische Codeanalyse; keine Simulation gestartet.

## 2026-10-26 – Pip-Installationen für Serverwechsel dokumentiert

- **Neue Datei:** `requirements_pde_gpu.txt` im Projektwurzelverzeichnis.
- **In diesem Chat ausgeführte Installationen:**
  - `python -m pip install --upgrade "jax[cuda12]"`; dadurch wurden die CUDA-12-JAX-Erweiterungen und NVIDIA-CUDA-Pythonpakete installiert, unter anderem `jax-cuda12-plugin==0.12.0.1`, `jax-cuda12-pjrt==0.12.0.1`, cuBLAS, cuDNN, cuFFT, cuSolver, cuSPARSE, NCCL und die CUDA-Runtime in den dort dokumentierten Versionen.
  - `python -m pip install --upgrade plotly`; installiert wurden `plotly==7.1.0` und `narwhals==2.27.1`.
- **Bereits vorhanden und nur verifiziert, nicht in diesem Installationsschritt installiert:** `jax==0.12.0.1`, `jaxlib==0.12.0.1`, `astronomix==0.6.6`, `numpy==2.5.2`, `matplotlib==3.11.1` und `Pillow==12.3.0`.
- **GPU-Hinweis:** Für die NVIDIA TITAN Xp muss die CUDA-12-Variante verwendet werden; die CUDA-13-JAX-Variante ist für diese GPU-Architektur nicht geeignet.

## 2026-10-26 – Interaktiven 1D-Plot erstellt

- **Ziel:** Die radiale 1D-Simulation soll nach ihrer manuellen Ausführung ähnlich bedienbar wie die interaktive 3D-Darstellung ausgewertet werden.
- **Neue Datei:** `Praktikum_Datenzubereitung/Supernova_plot_1D.py`.
- **Umsetzung:** `save_interactive_1d_html(...)` erzeugt eine eingebettete Plotly-HTML-Datei mit dem Feldverlauf über dem Radius, Zeit-Slider sowie Start-/Pause-Schaltflächen. Unterstützt werden Druck und Dichte.
- **Anpassungen:** `Supernovae_1D.py` verwendet `SPHERICAL`, übergibt die 1D-Geschwindigkeit als `velocity_x` und ruft den neuen 1D-Plotter mit `radius=r` auf.
- **Prüfungen:** Beide Python-Dateien wurden syntaktisch geprüft. Die HTML-Erzeugung wurde mit künstlichen 1D-Snapshots getestet und erfolgreich mit zwei Frames durchgeführt.
- **Zusatzprüfung:** Der Aufbau eines kleinen 1D-Primitive-Zustands wurde erfolgreich getestet (`shape=(3, 8)`, radiale Koordinaten `shape=(8,)`). Der Test wurde anschließend mit `JAX_PLATFORMS=cpu` wiederholt, um jede GPU-Nutzung auszuschließen; eine Simulation oder ein Integrationslauf fand nicht statt.
- **Nicht durchgeführt:** Die eigentliche 1D-Simulation wurde nicht gestartet und keine GPU verwendet.
- **Einordnung:** Der Plot ist eine direkte radiale 1D-Darstellung. Eine drehbare 3D-Kugel wäre nur eine nachträgliche Projektion des radialen Profils und ist nicht Bestandteil dieser Datei.

## 2026-10-26 – Startkontrolle der radialen 1D-Simulation

- **Geprüft:** `Supernovae_1D.py` sowie `Praktikum_Datenzubereitung/Supernova_plot_1D.py`.
- **Befund:** Die 1D-Konfiguration verwendet `SPHERICAL` und `dimensionality=1`; der Zustand wird mit `velocity_x` aufgebaut; der Plot-Import ist aus dem Projektwurzelverzeichnis auflösbar.
- **Ausgabe:** Nach der Simulation wird `supernova_1d.html` mit der radialen Druckanimation erzeugt.
- **Prüfung:** Beide Dateien wurden syntaktisch erfolgreich geprüft. Kein Simulationslauf wurde gestartet.

## 2026-10-26 – GIF-Ausgabe für die radiale 1D-Darstellung ergänzt

- **Anforderung:** Die 1D-Ergebnisse sollen direkt in VS Code geöffnet werden können, ohne eine interaktive HTML-Datei im Browser zu benötigen.
- **Geänderte Dateien:** `Praktikum_Datenzubereitung/Supernova_plot_1D.py` und `Supernovae_1D.py`.
- **Umsetzung:** Neue Funktion `save_1d_gif(...)` mit radialem Feldverlauf, Zeit-Titel und automatisch geschütztem Ausgabename. Der normale 1D-Lauf erzeugt nun `supernova_1d.gif`; die bestehende HTML-Funktion bleibt verfügbar.
- **Prüfung:** Syntaktische Prüfung und GIF-Test mit künstlichen Daten erfolgreich; der Test enthielt zwei Frames. Keine echte Simulation und keine GPU verwendet.

## 2026-10-26 – 3D-HTML-Ausgabe für Safari verkleinert

- **Ausgangslage:** Die 3D-HTML-Datei mit `100^3` Simulationszellen und 60 gespeicherten Zeitpunkten konnte in Safari möglicherweise wegen ihrer Größe nicht geladen werden.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/Supernovae_3D.py`.
- **Änderungen:** `num_snapshots` wurde von `60` auf `20` reduziert. `num_cells=100` bleibt unverändert. Für die Darstellung wurde `cell_stride` von `2` auf `4` gesetzt; dadurch wird nur die Visualisierung ausgedünnt, nicht die numerische Simulation.
- **Prüfung:** Syntaxprüfung erfolgreich. Kein 3D-Simulationslauf gestartet.

## 2026-10-29 – Paperanalyse zum 3D-Sedov-Test

- **Ziel:** Den von der Betreuung vorgeschlagenen 3D-Sedov-Test aus `Praktikum_Datenzubereitung/Astronomix_Paper.pdf` fachlich verstehen und mit dem aktuellen 3D-Code abgleichen.
- **Paperbefund:** Der Test beschreibt eine adiabatische, hydrodynamische Sedov-Taylor-Druckwelle. Eine thermische Energie `E_expl=1` wird in eine kleine kugelförmige Region eines ansonsten homogenen Mediums mit `rho=1` und `p_amb=10^-4` eingebracht; die Anfangsgeschwindigkeit ist null.
- **Anfangsdruck:** Das Paper verwendet die geglättete Kugelfunktion `p(r)=p_amb+Delta_p*w(r)` mit `w(r)=1/2*[1-tanh((r-r_expl)/delta)]`. `Delta_p` wird zellvolumenbezogen so bestimmt, dass die überschüssige thermische Energie exakt `E_expl` ergibt. Explosionsradius und Glättungsbreite sind im betrachteten Paperabschnitt nicht numerisch festgelegt und bleiben daher offen.
- **Auswertung:** Bei `t=0.1` werden radiale Profile von Dichte, Geschwindigkeitsbetrag und Druck mit der analytischen selbstähnlichen Sedov-Taylor-Lösung verglichen. Die Paperabbildung verwendet `256^3` Zellen, radiale Bins und zeigt zusätzlich die individuellen Zellwerte.
- **Abgrenzung zum aktuellen Code:** `Praktikum_Datenzubereitung/Supernovae_3D.py` verwendet derzeit `100^3` Zellen, `rho=1` innen und `0.1` außen sowie `p=1` innen und `0.1` außen. Die Energie wird nicht auf `E_expl=1` normiert; damit ist der aktuelle Lauf noch keine Paper-Reproduktion.
- **Prüfung:** PDF lokal ausgelesen und Sedov-Gleichungen sowie Abbildungsbeschriftung visuell kontrolliert. Aktueller 3D-Code statisch gelesen. Keine Simulation, keine Inferenz und keine GPU verwendet.
- **Offene Entscheidungen:** genaue Geometrie und Randbedingungen des Paper-Setups, `r_expl`, `delta`, Solvervariante für die erste Reproduktion sowie Umgang mit der hohen Kosten von `256^3` Zellen müssen vor der Implementierung festgelegt oder aus der Referenzimplementierung rekonstruiert werden.

## 2026-10-29 – Vorbereitung eines höher aufgelösten 1D-Laufs

- **Ziel:** Die Form der radialen Druckwelle bei höherer 1D-Auflösung qualitativ untersuchen.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/Supernovae_1D.py`.
- **Änderungen:** `num_cells` wurde für den geplanten Vergleich von `100` auf `400` erhöht; die GIF-Ausgabe wurde auf `Praktikum_Datenzubereitung/supernova_1d_400cells.gif` umgestellt, damit das bisherige Ergebnis nicht überschrieben wird.
- **Physikalische Parameter:** Die bisherigen Anfangsbedingungen bleiben für diesen ersten Auflösungstest unverändert. Damit ist der Lauf noch keine Reproduktion des Paper-Sedov-Tests.
- **Startversuch:** Ein direkter Dateistart scheiterte vor der Simulation an `ModuleNotFoundError` für den Paketimport. Der anschließend vorbereitete Modulstart wurde vom Nutzer abgebrochen. Es wurde kein erfolgreicher Simulationslauf und keine GPU-Nutzung durchgeführt; ein Prozess läuft nicht weiter.
- **Vorgesehener Nutzerstart:** Aus dem Projektwurzelverzeichnis mit `JAX_PLATFORMS=cpu` als Python-Modul starten. Für einen direkten Vergleich mit Figure 4 wäre zusätzlich `t_end=0.1` statt `0.2` zu setzen.

## 2026-10-29 – Logarithmische Druckachse für die 1D-Darstellung

- **Ziel:** Die Druckdarstellung des 1D-Ergebnisses an die logarithmische Druckachse der Sedov-Abbildung im Paper anpassen.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/Supernova_plot_1D.py`.
- **Umsetzung:** Für Druck wird sowohl im GIF-Plot als auch in der interaktiven HTML-Darstellung eine logarithmische y-Achse verwendet. Die Dichtedarstellung bleibt linear. Für den GIF-Plot werden nichtpositive Druckwerte mit einer Fehlermeldung abgewiesen.
- **Prüfung:** Nur Codeänderung; kein Simulationslauf und keine GPU verwendet.

## 2026-10-29 – Anpassungsplan für eine Reproduktion von Figure 4

- **Ziel:** Festlegen, welche Änderungen für den 3D-Sedov-Test aus Figure 4 des Astronomix-Papers erforderlich sind.
- **Ergebnis:** Die 1D-Simulation kann als Vorprüfung eines sphärischen Sedov-Profils dienen, reproduziert aber nicht die 3D-Gitterartefakte und die Darstellung von Figure 4. Für die Paper-Abbildung ist ein separater 3D-Cartesian-Lauf erforderlich.
- **Erforderliche Paperbedingungen:** `dimensionality=3`, `mhd=False`, uniformes `rho=1`, `p_amb=10^-4`, Nullgeschwindigkeit, `gamma=5/3`, `E_expl=1`, Druckprofil mit geglätteter Funktion `w(r)` und `Delta_p`-Normierung, Auswertung bei `t=0.1` und `256^3` Zellen.
- **Solver:** Für die erste Variante soll der Finite-Volume-Solver mit HLL und Minmod explizit gesetzt werden. HLLC und AM-HLLC wären getrennte Vergleichsläufe.
- **Auswertung:** Die bestehende 3D-HTML-Volumen-/Schnittdarstellung entspricht nicht Figure 4. Zusätzlich werden radiale Zellwerte, radiale Bins und Profile von Dichte, Geschwindigkeitsbetrag und Druck benötigt.
- **Offene Punkte:** `r_expl`, `delta`, genaue Box-/Randbedingungen sowie die analytische Sedov-Taylor-Referenz sind im betrachteten Paperabschnitt nicht numerisch beziehungsweise nicht als nutzbare Implementierung angegeben. Die lokal installierte `astronomix`-Datei `reference_solutions/sedov_solver.py` ist nur ein Platzhalter.
- **Prüfung:** Paper, aktueller 1D-/3D-Code und lokale astronomix-Testdateien gelesen. Kein Simulationslauf und keine GPU verwendet.

## 2026-10-29 – Abschlussprüfung des 1D-Sedov-Anfangszustands

- **Geprüfte Datei:** `Praktikum_Datenzubereitung/Supernovae_1D.py`.
- **Aktueller Stand:** `num_cells=512`, `t_end=0.1`, homogene Dichte `rho=1`, `p_amb=10^-4`, `E_expl=1`, geglättete Funktion `w(r)`, `delta=2 dr` und Zellvolumen-Normierung über `cell_volumes` sind im Code vorhanden.
- **Ausgabe:** Das Druck-GIF wird unter `Praktikum_Datenzubereitung/supernova_1d_512cells.gif` geschrieben.
- **Ergebnis:** Der Nutzer kann den Lauf nun CPU-only selbst starten. Es wurde durch Codex kein Simulationslauf gestartet und keine GPU verwendet.

## 2026-10-29 – Gemeinsame 1D-Profile für Dichte, Geschwindigkeit und Druck

- **Ziel:** Zusätzlich zum bisherigen Druck-GIF eine Darstellung der radialen Dichte und des Geschwindigkeitsbetrags analog zu Figure 4 vorbereiten.
- **Geänderte Datei:** Ausschließlich `Praktikum_Datenzubereitung/Supernova_plot_1D.py`.
- **Umsetzung:** Neue Funktion `save_1d_profiles_gif(...)` mit drei nebeneinanderliegenden Profilen für `rho`, `|v|` und `p` gegen den Radius. Die Druckachse ist logarithmisch, Dichte und Geschwindigkeit sind linear skaliert. Die Geschwindigkeit wird als Betrag der 1D-Komponente dargestellt.
- **Bewusste Abgrenzung:** `Supernovae_1D.py` wurde nicht verändert. Der Nutzer ergänzt Import und Funktionsaufruf selbst, damit der Simulationsablauf nachvollziehbar bleibt.
- **Prüfung:** Syntaxprüfung erfolgreich. Künstlicher Test mit zwei kleinen Snapshots erfolgreich; Testausgabe `/tmp/sedov_profiles_test.gif`. Keine Simulation und keine GPU verwendet.

## 2026-10-29 – Startprüfung vor dem 1D-Profillauf

- **Geprüfte Dateien:** `Praktikum_Datenzubereitung/Supernovae_1D.py` und `Praktikum_Datenzubereitung/Supernova_plot_1D.py`.
- **Befund:** Beide Dateien sind syntaktisch gültig. `Supernovae_1D.py` importiert `save_1d_profiles_gif`, ruft die Funktion aber derzeit noch nicht auf; ein Start würde daher nur das bisherige Druck-GIF erzeugen.
- **Parameterhinweis:** Aktuell sind `num_cells=64` eingestellt, während der Druck-Ausgabename noch `512cells` enthält. Das ist kein Laufzeitfehler, aber die Benennung ist irreführend und sollte vom Nutzer angepasst werden.
- **Ergebnis:** Nach Ergänzung des Profilaufrufs kann der Nutzer den CPU-only-Lauf selbst starten. Codex hat keinen Simulationslauf gestartet und keine GPU verwendet.

## 2026-10-29 – Dynamische y-Achsen der 1D-Profilanimation

- **Ziel:** Kleine momentane Geschwindigkeits-, Dichte- und Druckwerte sollen in der gemeinsamen Profilanimation weiterhin gut sichtbar bleiben.
- **Geänderte Datei:** Ausschließlich `Praktikum_Datenzubereitung/Supernova_plot_1D.py`.
- **Umsetzung:** Die drei y-Achsen werden in jedem Animationsframe neu gesetzt. Der aktuelle Maximalwert erhält standardmäßig fünf Prozent Abstand zum oberen Rand (`headroom=1.05`). Für Dichte und Geschwindigkeit beginnt die Achse bei null; für den logarithmischen Druck werden positive momentane Minimal- und Maximalwerte verwendet.
- **Prüfung:** Syntaxprüfung und künstlicher Test mit zwei Profilframes erfolgreich (`/tmp/sedov_profiles_dynamic_test.gif`). Keine Simulation und keine GPU verwendet.

## 2026-10-29 – Größerer oberer Rand für den Druckplot

- **Ziel:** Verhindern, dass das momentane Druckmaximum in der logarithmischen Achse optisch direkt am oberen Plotrand liegt.
- **Geänderte Datei:** Ausschließlich `Praktikum_Datenzubereitung/Supernova_plot_1D.py`.
- **Umsetzung:** Der dynamische obere Rand des Druckplots wurde unabhängig von Dichte und Geschwindigkeit auf `pressure_headroom=1.15` erhöht, also etwa 15 Prozent oberhalb des momentanen Druckmaximums gesetzt. Der bisherige Rand für Dichte und Geschwindigkeit bleibt bei `headroom=1.05`.
- **Prüfung:** Syntaxprüfung erfolgreich. Kein Simulationslauf und keine GPU verwendet.

## 2026-10-29 – Sedov-Anfangsbedingungen in Supernovae_3D.py übernommen

- **Ziel:** Die 3D-Simulation auf dieselben Sedov-Anfangsbedingungen wie den aktuellen 1D-Test umstellen.
- **Geänderte Datei:** Ausschließlich `Praktikum_Datenzubereitung/Supernovae_3D.py`.
- **Umsetzung:** Homogene Dichte `rho=1`, Umgebungsdruck `p_amb=10^-4`, Nullgeschwindigkeit, Explosionsenergie `E_expl=1`, geglättetes `w(r)` mit `r_expl=0.1` und `delta=2 dr`, sowie Energie-Normierung über das konstante kartesische Zellvolumen. Die Endzeit wurde auf `t_end=0.1` gesetzt. Die Ausgabe heißt nun `supernova_3d_sedov.html`.
- **Auflösung:** Die aktuell im Code vorhandene Pilotauflösung `64^3` wurde beibehalten. Die Paperauflösung `256^3` ist nicht gestartet.
- **Prüfung:** Syntaxprüfung erfolgreich. Kein 3D-Simulationslauf und keine GPU verwendet.

## 2026-10-29 – 3D-Ausgabe auf radiale Profile umgestellt

- **Ziel:** Für den 3D-Sedov-Lauf ausschließlich radiale Profile von Dichte, Geschwindigkeitsbetrag und Druck erzeugen; die räumliche HTML-Darstellung wird für diese Aufgabe nicht benötigt.
- **Geänderte Dateien:** `Praktikum_Datenzubereitung/Supernova_plot_3D.py` und `Praktikum_Datenzubereitung/Supernovae_3D.py`.
- **Umsetzung:** Neue Funktion `save_3d_radial_profiles_gif(...)`. Die kartesischen 3D-Zellen werden nach Abstand zum Boxzentrum radial gebinnt; pro Bin werden Mittelwerte von `rho`, `|v|` und `p` berechnet. Es werden drei nebeneinanderliegende Profile erzeugt, mit dynamischen y-Achsen sowie logarithmischer Druckachse. `Supernovae_3D.py` ruft nun diese GIF-Ausgabe statt der HTML-Volumendarstellung auf.
- **Ausgabe:** `Praktikum_Datenzubereitung/supernova_3d_radial_profiles_64cells.gif`.
- **Prüfung:** Syntaxprüfung und künstlicher 3D-Profiltest erfolgreich (`/tmp/sedov_3d_profiles_test.gif`). Keine Simulation und keine GPU verwendet.

## 2026-10-29 – Vergleich gemeinsamer 1D-/3D-Sedov-Profile vorbereitet

- **Ziel:** Dichte, Geschwindigkeitsbetrag und Druck der 1D- und 3D-Simulation bei demselben Zeitpunkt gemeinsam in drei Panels vergleichen.
- **Geänderte Dateien:** `Praktikum_Datenzubereitung/Supernova_plot_1D.py`, `Praktikum_Datenzubereitung/Supernova_plot_3D.py` und neu `compare_sedov_profiles.py`.
- **Umsetzung:** Die bestehenden 1D- und 3D-Profil-GIF-Funktionen speichern zusätzlich komprimierte numerische Profildaten als `.npz`. Das Vergleichsskript wählt den jeweils nächstliegenden Zeitpunkt, interpoliert die 1D-Werte auf die gültigen 3D-Radien und erzeugt drei gemeinsame Panels. 1D wird rot, 3D blau dargestellt; der Druck erhält eine logarithmische y-Achse.
- **Ausgabe:** Standardmäßig `Praktikum_Datenzubereitung/sedov_1d_3d_comparison_t010.png`. Bei bereits vorhandenen nummerierten Datendateien wird die zuletzt erzeugte passende Datei verwendet.
- **Prüfung:** Syntaxprüfung und künstlicher End-to-End-Vergleichstest erfolgreich (`/tmp/compare_result.png`). Keine Simulation und keine GPU verwendet.
- **Nächster Schritt:** Beide Simulationen müssen nach dieser Änderung erneut vom Nutzer ausgeführt werden, damit die `.npz`-Profildateien entstehen. Danach kann das Vergleichsskript gestartet werden.

## 2026-10-29 – Animierter gemeinsamer 1D-/3D-Profilvergleich

- **Anforderung:** Der 1D-/3D-Vergleich soll nicht nur den letzten Zustand als PNG, sondern die zeitliche Entwicklung als gemeinsames GIF zeigen.
- **Geänderte Datei:** `compare_sedov_profiles.py`.
- **Umsetzung:** Das Skript erzeugt nun ein animiertes GIF mit drei Panels. In jedem Frame werden die zeitlich nächstliegenden 1D- und 3D-Profile auf gemeinsame Radien gebracht, 1D rot und 3D blau gezeichnet. Die y-Achsen werden gemeinsam aus beiden Kurven skaliert; der Druck bleibt logarithmisch.
- **Standardausgabe:** `Praktikum_Datenzubereitung/sedov_1d_3d_comparison.gif`.
- **Prüfung:** Syntaxprüfung und künstlicher animierter Vergleichstest erfolgreich (`/tmp/compare_animation.gif`). Keine Simulation und keine GPU verwendet.

## 2026-10-29 – Radiusachse im animierten 1D-/3D-Vergleich korrigiert

- **Befund:** Die gespeicherten Radien waren korrekt (`1D` ungefähr `0…1`, `3D` ungefähr `0…0.85`). In der GIF-Animation blieb jedoch die automatische Matplotlib-Startgrenze der zunächst leeren Linien bei ungefähr `-0.05…0.05` bestehen.
- **Geänderte Datei:** `compare_sedov_profiles.py`.
- **Korrektur:** Die x-Achse wird nun in jedem Frame explizit von `0` bis zum maximalen gemeinsamen Radius gesetzt.
- **Prüfung:** Vergleichs-GIF aus den vorhandenen Profildaten neu erzeugt und visuell kontrolliert (`/tmp/actual_compare_fixed.gif`). Keine Simulation und keine GPU verwendet.

## 2026-10-29 – 3D-Scatterpunkte entlang der zentralen x-Achse

- **Anforderung:** Für den 1D-/3D-Vergleich sollen keine radialen 3D-Mittelwerte, sondern einzelne 3D-Zellen auf einer zentralen Achse verwendet werden.
- **Geänderte Dateien:** `Praktikum_Datenzubereitung/Supernova_plot_3D.py` und `compare_sedov_profiles.py`.
- **Umsetzung:** Die 3D-Profilfunktion speichert zusätzlich die Zellen der x-Achse durch das Explosionszentrum (`y` und `z` jeweils möglichst nahe `0.5`). Der Vergleich verwendet diese Werte als blaue Scatterpunkte gegen `r=|x-0.5|`; die 1D-Werte bleiben rote Linien. Die drei Panels für Dichte, Geschwindigkeit und Druck bleiben erhalten.
- **Wichtig:** Die vorhandenen 3D-`.npz`-Dateien enthalten diese Achsendaten noch nicht. Der 3D-Lauf muss nach dieser Änderung erneut durch den Nutzer ausgeführt werden; danach kann das neue Vergleichs-GIF erzeugt werden.
- **Prüfung:** Syntaxprüfung und künstlicher Achsen-Scattertest erfolgreich (`/tmp/axis_compare.gif`). Keine Simulation und keine GPU verwendet.

## 2026-10-29 – Vollständige 3D-Zellpunktwolke im Vergleich

- **Befund:** Eine einzelne zentrale x-Achse erzeugt pro Radius nur einzelne Werte und kann deshalb keine sichtbare Streuung wie in Figure 4 zeigen.
- **Geänderte Dateien:** `Praktikum_Datenzubereitung/Supernova_plot_3D.py` und `compare_sedov_profiles.py`.
- **Umsetzung:** Die 3D-Profilfunktion speichert nun zusätzlich Radius, Dichte, Geschwindigkeitsbetrag und Druck jeder einzelnen 3D-Zelle für alle Snapshots. Das Vergleichs-GIF verwendet diese vollständige Zellmenge als graue Scatterpunktwolke; die 1D-Profile bleiben rote Linien. Die drei Panels und die logarithmische Druckachse bleiben erhalten.
- **Ressourcenhinweis:** Wegen der Datenmenge ist die aktuelle `64^3`-Auflösung für diesen animierten Vergleich vorgesehen. Eine `256^3`-Punktwolke über viele Snapshots wäre deutlich speicher- und zeitaufwendiger.
- **Prüfung:** Syntaxprüfung und künstlicher vollständiger Zell-Scattertest erfolgreich (`/tmp/cells_compare.gif`). Keine Simulation und keine GPU verwendet.
- **Nächster Schritt:** Der 3D-Lauf muss nach dieser Änderung erneut durch den Nutzer ausgeführt werden, damit die vollständigen Zellwerte in der `.npz`-Datei gespeichert werden.

## 2026-10-29 – 3D-Mittelwertlinie im Zell-Scattervergleich beibehalten

- **Anforderung:** Zusätzlich zur grauen Punktwolke aller 3D-Zellen soll weiterhin die gemittelte blaue 3D-Kurve sichtbar bleiben.
- **Geänderte Datei:** `compare_sedov_profiles.py`.
- **Umsetzung:** Der animierte Vergleich zeichnet nun gleichzeitig die rote 1D-Kurve, den blauen radialen 3D-Mittelwert und die grauen Einzelzellwerte. Die y-Achsenskalierung berücksichtigt alle drei Darstellungen.
- **Prüfung:** Syntaxprüfung, synthetischer GIF-Test und visuelle Kontrolle erfolgreich (`/tmp/cells_compare_blue.gif`). Kein Simulationslauf und keine GPU verwendet.
- **Nächster Schritt:** Nach einem neuen 3D-Lauf kann der Nutzer das Vergleichs-GIF mit den erzeugten `.npz`-Dateien starten.

## 2026-10-29 – Importpfad beim Start von Supernovae_3D korrigiert

- **Befund:** Der Start aus `Praktikum_Datenzubereitung` mit `python -u Supernovae_3D.py` schlug beim Import `Praktikum_Datenzubereitung.Supernova_plot_3D` mit `ModuleNotFoundError` fehl.
- **Ursache:** Beim direkten Start aus dem Unterverzeichnis liegt der Projektordner nicht im Python-Suchpfad als Paket-Elternverzeichnis.
- **Lösung:** `Supernovae_3D.py` wird aus dem Projekt-Hauptverzeichnis mit `python -m Praktikum_Datenzubereitung.Supernovae_3D` gestartet. Der abgebrochene Versuch erreichte die Simulation nicht; keine GPU und kein Simulationslauf wurden ausgeführt.

## 2026-10-29 – CUDA-JAX für den 3D-Lauf geprüft

- **Freigabe:** Der Nutzer hat die aktuell unbenutzten GPUs auf `compgpu3` für den 3D-Lauf freigegeben.
- **Prüfung:** Im Environment `pde` wurden `jax==0.12.0.1` und `jaxlib==0.12.0.1` erkannt; JAX meldet neun `CudaDevice`-Geräte. Es wurde dabei keine Simulation gestartet.
- **Empfehlung:** Für den 64³-Pilotlauf wird nur `GPU 0` sichtbar gemacht. Das vermeidet eine unnötige Belegung aller GPUs; die übrigen Geräte bleiben frei.

## 2026-10-29 – Analytische Sedov-Taylor-Lösung vor dem Plotvergleich erklärt

- **Ziel:** Vor einer Implementierung klären, was die analytische Referenzlösung des Sedov-Problems beschreibt.
- **Bezug zum aktuellen Code:** `Supernovae_1D.py` verwendet `SPHERICAL` und ist damit ein eindimensionales Kugelsymmetrie-Modell; `Supernovae_3D.py` verwendet ein kartesisches 3D-Gitter, dessen Zellwerte radial ausgewertet werden.
- **Inhalt:** Die Sedov-Taylor-Lösung beschreibt eine starke, selbstähnliche Kugelwelle aus einer idealisierten Punktenergie in homogenem Medium. Die Schockfront folgt `R(t) ∝ (E t²/rho_0)^(1/5)` für drei Raumdimensionen. Die vollständigen Dichte-, Geschwindigkeits- und Druckprofile sind selbstähnliche Funktionen von `r/R(t)`.
- **Einschränkungen:** Die numerischen Anfangsbedingungen besitzen eine endliche geglättete Explosionskugel und einen kleinen Hintergrunddruck. Deshalb ist insbesondere der frühe numerische Zustand nicht exakt identisch mit der idealisierten analytischen Lösung.
- **Status:** Nur Erklärung und Dokumentation; keine Codeänderung und kein Simulationslauf.

## 2026-10-29 – Datei für analytische Sedov-Taylor-Referenzprofile erstellt

- **Ziel:** Eine separate analytische Referenz erzeugen, die später als grüne Kurve in den 1D-/3D-Vergleich aufgenommen werden kann.
- **Neue Datei:** `sedov_analytic.py`.
- **Umsetzung:** Für `gamma=5/3` werden die geschlossenen parametrischen Sedov-Profile für Dichte, radiale Geschwindigkeit und Druck ausgewertet. Die Energie-Normierung liefert `beta` numerisch aus dem Energieintegral. Die Datei erzeugt eine komprimierte `.npz`-Datei mit Zeitpunkten, Radien, Profilen, Schockradien und Schockgeschwindigkeiten.
- **Normierung:** Standardwerte sind `E_expl=1`, `rho_0=1`, `p_amb=1e-4`, `t_end=0.1`; wegen der Singularität der idealen Punkt-Explosion beginnt die analytische Zeitserie standardmäßig bei `t_start=1e-4` statt bei exakt `t=0`.
- **Prüfung:** Syntaxprüfung und lokale Profilprüfung erfolgreich. Die berechnete Sedov-Konstante beträgt `beta=1.151143`, der Schockradius bei `t=0.1` beträgt ungefähr `0.4583`. Kein hydrodynamischer Simulationslauf und keine GPU verwendet.
- **Nächster Schritt:** Die erzeugte analytische `.npz`-Datei in `compare_sedov_profiles.py` als grüne Linie einbinden.

## 2026-10-29 – Analytische Sedov-Kurve in das Vergleichs-GIF eingebunden

- **Ziel:** Die analytische Referenz als grüne Kurve gemeinsam mit den numerischen 1D-/3D-Ergebnissen darstellen.
- **Geänderte Datei:** `compare_sedov_profiles.py`.
- **Umsetzung:** Das Zell-Scatter-GIF lädt nun standardmäßig `Praktikum_Datenzubereitung/sedov_analytic_profiles.npz` und zeichnet in jedem Panel zusätzlich `Sedov analytisch` als grün gestrichelte Linie. Die Zeitpunkte werden jeweils dem nächstliegenden analytischen Zeitpunkt zugeordnet. Die Achsenskalierung berücksichtigt rote 1D-Linie, blaue 3D-Mittelwertlinie, grüne analytische Linie und graue 3D-Zellwolke.
- **Sonderfall:** Im numerischen Anfangsframe bei exakt `t=0` bleibt die analytische Linie leer, weil die ideale Punkt-Explosion dort singulär ist. Ab positiven Zeiten wird sie gezeichnet.
- **Prüfung:** Syntaxprüfung, synthetischer GIF-Test und visuelle Kontrolle eines positiven Zeitframes erfolgreich (`/tmp/cells_compare_analytic_fixed.gif`, `/tmp/cells_compare_analytic_frame1.png`). Keine Simulation und keine GPU verwendet.

## 2026-10-30 – Schreibweise Supernova vereinheitlicht

- **Ziel:** Die bisher uneinheitliche beziehungsweise fehlerhafte Schreibweise `Supanova`/`supanova` im Projekt auf `Supernova`/`supernova` korrigieren.
- **Umbenannte Quellcode-Dateien:** `Supernova_plot.py`, `Supernova_plot_1D.py`, `Supernova_plot_3D.py` und `Supernovae_2D.py`.
- **Umbenannte Ergebnisdateien:** Vorhandene GIF-, HTML-, NPZ- und Logdateien mit `supanova` im Dateinamen wurden entsprechend umbenannt. Inhalte der Ergebnisdateien, insbesondere der großen Logdatei, wurden nicht verändert.
- **Aktualisierte Referenzen:** Imports, Standardausgabepfade, `AGENTS.md` und die bisherigen Protokollverweise verwenden nun die Schreibweise `Supernova` beziehungsweise `Supernovae`.
- **Prüfungen:** Keine alte Schreibweise mehr in Python-, Markdown- oder Textdateien gefunden. Die acht Python-Dateien wurden mit `ast.parse` syntaktisch geprüft. Keine Simulation und keine GPU verwendet.

## 2026-10-30 – Laufstart und Wiederaufnahme nach VS-Code-Abbruch dokumentiert

- **Anlass:** Für den geplanten 256er-Sedov-Vergleich besteht die Sorge, dass VS Code während des langen 3D-Laufs scheinbar stehen bleibt und ein neuer Chat erforderlich wird.
- **Aktueller Codezustand:** 1D und 3D verwenden `num_cells=256`, beide verwenden `num_snapshots=60`, `t_end=0.1`, `C_cfl=0.4`, `gamma=5/3` und `E_expl=1.0`. Die 3D-Ausgabe verwendet `num_bins=128`.
- **Startstrategie:** Die Läufe werden aus dem Projekt-Hauptverzeichnis nacheinander mit `JAX_PLATFORMS=cpu` und dem Micromamba-Interpreter `/export/home/bgschwen/micromamba/envs/pde/bin/python` gestartet. Zuerst 1D, danach 3D; kein paralleler Start.
- **Ausgaben:** Erwartet werden `supernova_1d_profiles_256cells.gif/.npz` und `supernova_3d_radial_profiles_256cells.gif/.npz`. Danach kann `compare_sedov_profiles.py` mit diesen Dateien und `sedov_analytic_profiles.npz` ausgeführt werden.
- **Wiederaufnahme:** Nach einem Abbruch soll ein neuer Chat zuerst den Prozessstatus, die letzte Konsolenausgabe und die vorhandenen 256er-Dateien prüfen. Es darf keine zweite Simulation gestartet werden, solange ein vorheriger Prozess noch läuft. Die exakten CPU-Kommandos und der Vergleichsbefehl sind zusätzlich dauerhaft in `AGENTS.md` hinterlegt.
- **Durchführung:** In diesem Dokumentationsschritt wurde keine Simulation gestartet und keine GPU verwendet.

## 2026-10-30 – Statuskontrolle des laufenden 256³-3D-Laufs

- **Lauf:** `JAX_PLATFORMS=cpu ... -m Praktikum_Datenzubereitung.Supernovae_3D` mit `num_cells=256` und `num_snapshots=60`.
- **Beobachtung:** Die Fortschrittsanzeige stand nach den ersten Minuten weiterhin bei `0.00 %`, der Prozess lief jedoch aktiv mit ungefähr `986 %` CPU-Auslastung. Der beobachtete Prozess hatte eine Laufzeit von etwa 2:39 Minuten und belegte ungefähr 32 GB RAM.
- **Ressourcenprüfung:** Der Rechner verfügte zum Prüfzeitpunkt über etwa 503 GiB Arbeitsspeicher insgesamt und etwa 382 GiB verfügbaren Speicher. Es wurde keine GPU verwendet.
- **Interpretation:** Der Stand spricht für aktive Initialisierung beziehungsweise JAX-Kompilierung und nicht für einen nachgewiesenen Stillstand. Der Lauf soll nicht parallel ein zweites Mal gestartet werden; bei weiteren Kontrollen sind Prozessstatus, CPU-Auslastung, Speicher und die letzte Konsolenausgabe zu prüfen.

## 2026-10-30 – 256³-CPU-Lauf auf Nutzerwunsch abgebrochen

- **Abbruch:** Der laufende CPU-Prozess `Supernovae_3D` mit PID `273525` wurde auf ausdrücklichen Nutzerwunsch beendet.
- **Grund:** Der Nutzer möchte den 256³-Lauf anschließend auf einer GPU ausführen.
- **Kontrolle:** Eine anschließende Prozessprüfung ergab keinen laufenden Prozess mit dieser PID.
- **GPU-Status:** Noch kein GPU-Lauf gestartet. Die konkrete GPU muss vor dem Start vom Nutzer festgelegt und freigegeben werden.

## 2026-10-30 – Nutzer startet Läufe künftig selbst

- **Entscheidung:** Codex soll künftig keine Simulations-, Trainings- oder Auswertungsläufe selbst starten, sondern nur Konfigurationen prüfen und die passenden Kommandos bereitstellen.
- **GPU-Prüfung:** Auf `compgpu2` wurde nach Nutzerangabe GPU 2 ausschließlich durch einen kurzen JAX-Gerätetest geprüft. Mit `CUDA_VISIBLE_DEVICES=2` und `JAX_PLATFORMS=cuda,cpu` wurde `CudaDevice(id=0)` erkannt.
- **Wichtig:** Es wurde kein 1D- oder 3D-Simulationslauf gestartet. GPU 2 und GPU 3 waren zum Prüfzeitpunkt nahezu frei.

## 2026-10-30 – Herleitung der analytischen Sedov-Taylor-Referenz erklärt

- **Anlass:** Die Herkunft der in `sedov_analytic.py` verwendeten Formeln war unklar.
- **Erklärung:** Die Lösung basiert auf den sphärischen Euler-Gleichungen, der starken Rankine-Hugoniot-Sprungbedingung, der selbstähnlichen Variablen `xi=r/R(t)` und der parametrischen Sedov-Lösung für `gamma=5/3` aus den Gleichungen (7.28)–(7.30) der angegebenen Vorlesungsnotizen von Gordon I. Ogilvie.
- **Zuordnung:** Der Schockradius `R=beta*(E*t**2/rho_0)**(1/5)` folgt aus Dimensionsanalyse; `shock_speed=2*R/(5*t)` folgt durch Ableiten. Die Konstante `beta` wird durch das Gesamtenergieintegral numerisch normiert.
- **Wichtige Einschränkung:** Die Referenzlösung beschreibt eine ideale Punkt-Explosion mit vernachlässigbarem Umgebungsdruck. Die numerische Simulation besitzt dagegen eine endliche geglättete Explosionsregion, einen kleinen Hintergrunddruck und ein kartesisches Gitter.
- **Ergebnis:** Die Bedeutung der dimensionslosen Profile, des Parameters `v` und der Rückskalierung auf Dichte, Geschwindigkeit und Druck wurde fachlich erläutert. Keine Codeänderung und kein Simulationslauf.

## 2026-10-30 – Animierte 3D-Schnittflächen aus gespeicherten Sedov-Daten

- **Ziel:** Für die 3D-Sedov-Simulation drei zeitabhängige räumliche Schnittflächen von Dichte `rho`, Geschwindigkeitsbetrag `|v|` und Druck `p` in der zentralen x-y-Ebene darzustellen.
- **Neue Datei:** `plot_3d_slices.py`.
- **Umsetzung:** Der Plotter liest die bereits von `save_3d_radial_profiles_gif(...)` gespeicherten Felder `cell_density`, `cell_velocity`, `cell_pressure` und `time_points` aus einer `.npz`-Datei. Standardmäßig wird die zentrale Gitterebene `z_index=nz//2` verwendet. Bei der aktuellen Box von `0` bis `1` entspricht dies physikalisch `z≈0.5` beziehungsweise der zentrierten Koordinate `z'=0`.
- **Darstellung:** Drei nebeneinanderliegende x-y-Felder; lineare Farbskalen für Dichte und Geschwindigkeit, logarithmische Farbskala für den positiven Druck. Die Animation wird als GIF gespeichert, bestehende Dateien werden automatisch nummeriert.
- **Parameter:** Standardmäßig `box_size=1.0`, zentrale z-Ebene, `fps=10` und `interval=100` ms. Eine andere Ebene kann über `--z-index` gewählt werden.
- **Prüfung:** Statische Syntaxprüfung von `plot_3d_slices.py` erfolgreich. Kein neuer Simulationslauf und keine GPU-Nutzung durch Codex.
- **Nächster Schritt:** Nach Vorliegen der 128er-Profildatei den Plotter manuell auf dem Server ausführen und das erzeugte GIF visuell prüfen.

## 2026-11-02 – Fortsetzung der Projektarbeit in einer Zellij-Session

- **Dokumentation:** Der Nutzer führt diesen Chat mit Codex innerhalb einer Zellij-Session, um die Arbeit am Supernova-Projekt fortzusetzen.
- **Hinweis:** Der genaue Name beziehungsweise die technische Kennung der Zellij-Session ist nicht dokumentiert.

## 2026-11-02 – Neue Zielsetzung: Supernova in turbulentem Medium

- **Anforderung aus der Betreuung:** Vor der Supernova soll zunächst ein turbulentes Medium erzeugt werden. Die Supernova wird erst nach einer kurzen Vorlaufzeit beziehungsweise nach einer geeigneten turbulenten Crossing-/Wirbelzeit in diesen Zustand eingebracht.
- **Wichtige Abgrenzung:** Die bisherige 128er-Simulation mit homogenem Medium, Nullgeschwindigkeit und kugelsymmetrischer Druckerhöhung ist nur ein idealisierter Sedov-Test. Sie bildet noch nicht die eigentliche Zielsimulation mit eingebauten Strömungen ab.
- **Lokale astronomix-API geprüft:** Die installierte Version enthält `TurbulentForcingConfig` mit `turbulent_forcing`, `ou_forcing` und `vacuum_protection` sowie `TurbulentForcingParams` mit `energy_injection_rate`, `correlation_time`, `forcing_wavenumber` und `forcing_amplitude`.
- **Begriffsklärung:** Ein Parameter mit dem exakten Namen `turbulent_crossing_time` wurde in der installierten Version nicht gefunden. `correlation_time` der Ornstein-Uhlenbeck-Forcierung ist als ungefähr eine Eddy-Turnover-Zeit dokumentiert. Die tatsächliche Crossing-Zeit muss aus charakteristischer Länge und RMS-Geschwindigkeit des erzeugten Mediums bestimmt oder mit der Betreuung festgelegt werden.
- **Geplanter Ablauf:** (1) turbulente Forcierung einschalten und ein homogenes Medium bis zu einer festzulegenden Vorlaufzeit entwickeln, (2) den turbulenten Zustand einschließlich Geschwindigkeit und Druck speichern, (3) die Forcierung deaktivieren oder gemäß Versuchsdefinition fortführen, (4) die Supernova-Energie in diesem Zustand einbringen und (5) die räumlichen Geschwindigkeitsvektoren zusätzlich zu Dichte, Geschwindigkeitsbetrag und Druck auswerten.
- **Offene Entscheidungen:** genaue Definition der Crossing-Zeit, Vorlaufdauer, Forcierungsmethode (weiß oder OU), `forcing_wavenumber`, Forcierungsamplitude beziehungsweise Energieinjektionsrate, Randbedingungen, Speicherung des Zwischenzustands und ob die Turbulenz während der Supernova weiter angetrieben wird.
- **Prüfung:** Nur lokale Quelldateien und vorhandene Ergebnisdaten gelesen; kein Simulationslauf gestartet. Ein unbeschränkter Import von `astronomix` löste einmal JAX-CUDA-Geräteinitialisierung und Backend-/Speicherwarnungen aus, aber keine Simulation oder numerische Auswertung. Weitere Paketinspektionen erfolgen mit `JAX_PLATFORMS=cpu` beziehungsweise ohne Import.

## 2026-11-02 – Skript für das turbulente Anfangsmedium erstellt

- **Ziel:** Einen getrennten ersten Simulationsabschnitt für das turbulente Medium bereitstellen, bevor die Supernova-Energie eingebracht wird.
- **Neue Datei:** `Praktikum_Datenzubereitung/anfangsmedium.py`.
- **Umsetzung:** 3D-kartesisches homogenes Medium, zeitlich korrelierte Ornstein-Uhlenbeck-Forcierung, reproduzierbarer Zufallsseed, konfigurierbare Vorlaufzeit und Forcierungsparameter sowie Speicherung des finalen `primitive_state` als komprimierte `.npz`-Datei.
- **Vorläufige Defaultwerte:** `num_cells=128`, `box_size=1`, `t_end=1`, `rho=1`, `p=1`, `gamma=5/3`, `C_cfl=0.4`, `seed=42`, `correlation_time=1`, `forcing_wavenumber=4`, `forcing_amplitude=1`. Diese Werte sind Pilotparameter und noch nicht als Forschungsreferenz festgelegt.
- **Randbedingungen:** Periodische Randbedingungen sind der vorläufige Standard, da die Forcierung ein geschlossenes turbulentes Medium in einer Box erzeugen soll. Über `--boundary open` kann dies für einen Vergleich geändert werden.
- **Speicherung:** Standardausgabe ist `Praktikum_Datenzubereitung/turbulent_medium_128cells.npz`; gespeichert werden Endzustand, Zeitpunkte, Massen-/Energiediagnosen, Variablenindizes und JSON-Metadaten.
- **Bewusste Abgrenzung:** Das Skript führt keine Supernova aus und erzeugt noch keine zweite Phase mit Energieeintrag. Der gespeicherte Zustand ist für ein späteres Supernova-Skript vorgesehen.
- **Prüfung:** Syntaxprüfung, `git diff --check` und CPU-only-Konfigurations-Dry-Run mit `16^3` Zellen erfolgreich. Der Dry-Run bestätigte `state shape=(5,16,16,16)`, aktivierte OU-Forcierung und periodische Randbedingungen. Keine Zeitschrittintegration, keine Simulation und keine GPU-Nutzung.
- **Offen:** Die Defaultwerte, die genaue turbulente Crossing-/Eddy-Zeit, die Vorlaufdauer und die Frage, ob die Forcierung während der Supernova weiterläuft, müssen mit der Betreuung abgestimmt werden.

## 2026-11-02 – GIF-Darstellung des turbulenten Anfangsmediums ergänzt

- **Ziel:** Die Entstehung des turbulenten Mediums zeitlich als GIF darstellen und die Strömungsrichtungen sichtbar machen.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/anfangsmedium.py` unterstützt nun `--save-snapshots`. Bei Aktivierung werden die vollständigen Zwischenzustände zusätzlich zum finalen Zustand gespeichert.
- **Neue Datei:** `Praktikum_Datenzubereitung/plot_anfangsmedium.py` liest diese Snapshots und erzeugt einen zentralen z-Schnitt mit Dichte, Geschwindigkeitsbetrag, logarithmischem Druck sowie Geschwindigkeitsvektoren.
- **Ausgaben:** Das Plot-Skript erzeugt standardmäßig `Praktikum_Datenzubereitung/turbulent_medium_slices_128cells.gif`; vorhandene Dateien werden automatisch nummeriert.
- **Speicherhinweis:** Vollständige 3D-Zwischenzustände sind groß. Die Snapshot-Speicherung ist deshalb optional und nicht der Standard des Medium-Skripts.
- **Prüfung:** Beide Dateien wurden syntaktisch geprüft und mit `git diff --check` kontrolliert. Keine Simulation, keine GIF-Erzeugung aus echten Turbulenzdaten und keine GPU-Nutzung durch Codex.
- **Nächster manueller Ablauf:** Das Medium mit `--save-snapshots` erzeugen und anschließend `plot_anfangsmedium.py` mit dem tatsächlich ausgegebenen NPZ-Pfad starten.

## 2026-11-02 – Erster Medium-Lauf ohne Snapshots abgebrochen

- **Ausgangslage:** Der Nutzer startete `Praktikum_Datenzubereitung.anfangsmedium` auf `compgpu2` mit `JAX_PLATFORMS=cpu`, jedoch ohne `--save-snapshots`.
- **Beobachtung:** Der CPU-Lauf erreichte ungefähr `4.8 %` Fortschritt. Die Meldung zur Auswahl des NATIVE_JAX-Backends war erwartetes CPU-Verhalten.
- **Abbruch:** Auf ausdrücklichen Nutzerwunsch wurde der Prozess mit PID `1544500` zunächst per `SIGINT` und anschließend per `SIGTERM` beendet. Eine abschließende Prozessprüfung ergab keinen laufenden Prozess mehr.
- **Ergebnis:** Der Lauf wurde vor dem regulären Speichern beendet. Da keine Snapshots aktiviert waren, hätte der Lauf ohnehin keinen zeitlichen GIF-Datensatz erzeugt.
- **Nächster Schritt:** Für das GIF den Medium-Lauf erneut mit `--save-snapshots` und einem ausdrücklich benannten Ausgabepfad starten. Keine parallele zweite Simulation ausführen.

## 2026-11-02 – Zeitliche Statistik der mittleren Geschwindigkeit ergänzt

- **Ziel:** Die Geschwindigkeit des gesamten 3D-Mediums über die Zeit quantitativ auswerten.
- **Neue Datei:** `Praktikum_Datenzubereitung/plot_mean_velocity.py`.
- **Auswertung:** Für jeden gespeicherten Snapshot werden der mittlere Geschwindigkeitsbetrag `mean_speed = <|v|>`, die RMS-Geschwindigkeit, die gemittelten Komponenten `<v_x>`, `<v_y>`, `<v_z>` und der Betrag des Nettoflusses `|<v>|` berechnet.
- **Ausgaben:** Eine PNG-Grafik mit zwei Panels sowie eine CSV-Datei mit allen Zeitreihen. Das obere Panel zeigt typische Zellgeschwindigkeit und RMS-Geschwindigkeit; das untere Panel zeigt Nettofluss und gemittelte Komponenten.
- **Wissenschaftliche Begründung:** In einem turbulenten Medium kann `<|v|>` deutlich größer als null sein, während `|<v>|` wegen der gegenseitigen Aufhebung der Wirbel nahe null bleibt. Beide Größen werden deshalb getrennt dargestellt.
- **Prüfung:** Syntaxprüfung und `git diff --check` erfolgreich. Keine Simulation und keine GPU-Nutzung durch Codex.

## 2026-11-02 – Zweite Simulationsphase: Supernova im turbulenten Medium

- **Ziel:** Die Supernova nicht mehr in einem neu erzeugten homogenen Medium, sondern im gespeicherten turbulenten Endzustand bei `t=5` starten.
- **Neue Datei:** `Praktikum_Datenzubereitung/Supernovae_3D_turbulent.py`.
- **Eingabe:** Standardmäßig `turbulent_medium_128cells_t5_kf25_gpu2.npz`; geladen wird dessen finaler `primitive_state` einschließlich Dichte, Druck und aller drei Geschwindigkeitskomponenten.
- **Explosionsbehandlung:** Die vorhandene Turbulenz bleibt erhalten. Zum bestehenden Druck wird im Boxzentrum eine geglättete Druckerhöhung addiert. Die Normierung erfüllt numerisch `Delta E ≈ E_expl` über `Delta p/(gamma-1)` und das Zellvolumen.
- **Phase 2:** Die OU-Forcierung wird standardmäßig deaktiviert, damit die weitere Entwicklung eindeutig der Supernova in einem bereits turbulenten Medium zugeordnet werden kann. Periodische Randbedingungen werden aus den Medium-Metadaten übernommen.
- **Ausgabe:** Radiale Profile von Dichte, Geschwindigkeitsbetrag und Druck werden als GIF und NPZ über `save_3d_radial_profiles_gif(...)` gespeichert.
- **Vorläufige Parameter:** `E_expl=1`, `explosion_radius=0.01`, Glättung `2` Gitterzellen, Simulationsdauer `0.1`, `60` Snapshots. Der kleine Explosionsradius entspricht dem bisherigen idealisierten 128er-Sedov-Test, ist aber noch mit der Betreuung abzustimmen.
- **Prüfung:** Syntaxprüfung, `git diff --check` und CPU-Konfigurations-Dry-Run mit künstlichem `8^3`-Zustand erfolgreich. Die Energie-Normierung ergab `0.99999988`, der Zustand blieb `(5,8,8,8)`, und die Forcierung war in Phase 2 deaktiviert. Keine echte Simulation und keine GPU-Nutzung durch Codex.
- **Offene Entscheidung:** Ob die Turbulenz während der Supernova weiter forciert werden soll. Die aktuelle NPZ-Datei speichert den OU-Zustand nicht; für eine nahtlose Fortsetzung der Forcierung wäre Astronomix-Checkpointing beziehungsweise ein gemeinsamer Lauf erforderlich.

## 2026-11-02 – Gemeinsame Medium-/Supernova-Animation mit Zeitlupe

- **Ziel:** Das turbulente Medium soll bis kurz vor der Explosion wie im bisherigen Slice-GIF dargestellt werden. Ab absoluter Zeit `t=4.5` soll die Animation verlangsamt werden, damit der Übergang bei `t=5` und die Supernova-Ausbreitung besser sichtbar sind.
- **Geänderte beziehungsweise neue Dateien:** `Praktikum_Datenzubereitung/Supernovae_3D_turbulent.py` speichert zusätzlich zentrale z-Schnittdaten der Supernova-Phase. Neu ist `Praktikum_Datenzubereitung/plot_anfangsmedium_supernova.py`.
- **Darstellung:** Dichte, Geschwindigkeitsbetrag, logarithmischer Druck und Geschwindigkeitsvektoren entsprechen der bisherigen Medium-Darstellung. Der letzte Medium-Zustand bei `t=5` und der initiale Zustand der Supernova-Phase werden beide beibehalten, sodass der Drucksprung sichtbar bleibt.
- **Zeitlupe:** Standardmäßig werden alle Frames mit `t >= 4.5` fünfmal wiederholt. Die Zeitpunkte und der Verlangsamungsfaktor sind über `--slow-start` und `--slowdown-factor` änderbar.
- **Normierung:** Die Farbskalen werden ausschließlich aus dem Medium bestimmt, damit dessen Darstellung vor der Explosion unverändert bleibt. Sehr hohe Supernova-Druckwerte können dadurch im Druckpanel an der oberen Farbgrenze liegen; diese Konsequenz ist für die Vergleichbarkeit bewusst akzeptiert.
- **Prüfungen:** Beide Python-Dateien wurden per `ast.parse` syntaktisch geprüft. Es wurde kein Simulations-, GPU- oder GIF-Erzeugungslauf durch Codex gestartet.
- **Nächster Schritt:** `Supernovae_3D_turbulent.py` muss vom Nutzer erneut ausgeführt werden, damit die neue Schnitt-NPZ-Datei entsteht. Danach kann das kombinierte GIF separat erzeugt werden.

## 2026-11-02 – GPU-Speicherfehler beim Supernova-Lauf auf GPU 8

- **Lauf:** Der Nutzer startete `Supernovae_3D_turbulent.py` mit dem 128³-Medium, 60 Supernova-Snapshots und `CUDA_VISIBLE_DEVICES=8`.
- **Ergebnis:** Der Lauf brach bereits bei der GPU-Speicherreservierung mit `CUDA_ERROR_OUT_OF_MEMORY` ab. Angefordert wurden unter anderem etwa 8.92 GiB, 8.03 GiB und 7.23 GiB.
- **Interpretation:** Die Kombination aus 128³-Gitter, 3D-Zustandsfeldern, Zeitintegration und vollständiger Snapshot-Speicherung überschreitet den aktuell verfügbaren GPU-Speicher oder trifft auf eine belegte/fragmentierte GPU. Die Supernova-Simulation wurde nicht abgeschlossen; die neue Schnitt-NPZ-Datei wurde daher nicht erzeugt.
- **Nächster Versuch:** Zuerst GPU-Belegung mit `nvidia-smi -i 8` prüfen. Danach kann die Phase 2 mit deaktivierter JAX-Vorreservierung und weniger, beispielsweise 20, Snapshots erneut versucht werden. Das Eingangsmedium selbst muss dafür nicht neu erzeugt werden.

## 2026-11-02 – Zweiphasen-Skript mit absoluter Supernova-Zeit und Zeitlupenplot erstellt

- **Ziel:** Den vollständigen turbulenten Zustand bei `t=5` übernehmen, darin eine Supernova einsetzen und ihre Entwicklung bis `t=5.1` in der zentralen z-Ebene darstellen.
- **Neue Dateien:** `Praktikum_Datenzubereitung/supernovae_turbolenz.py` und `Praktikum_Datenzubereitung/plot_supernovae_turbolenz.py`.
- **Simulation:** `supernovae_turbolenz.py` lädt den vollständigen `primitive_state` mit Dichte, Druck und drei Geschwindigkeitskomponenten. Die OU-Forcierung bleibt in Phase 2 deaktiviert. Eine zusätzliche thermische Energie wird im Zentrum druckbasiert und zellvolumenbezogen normiert eingebracht. Die Integrationsdauer beträgt standardmäßig `Delta t=0.1`; die gespeicherten Zeiten werden absolut als `t=5...5.1` ausgegeben.
- **Darstellung:** `plot_supernovae_turbolenz.py` kombiniert die vollständigen Medium-Snapshots von `t=0...5` mit den Supernova-Schnittdaten von `t=5...5.1`. Dargestellt werden Dichte, Geschwindigkeitsbetrag und Druck einschließlich Geschwindigkeitsvektoren in der zentralen z-Ebene. Der Medium-Zustand vor der Explosion und der Supernova-Zustand direkt nach dem Einsetzen bleiben beide bei `t=5` erhalten.
- **Zeitlupe:** Bis `t=4.98` werden die Frames einmal dargestellt. Ab `t=4.98` werden sie standardmäßig achtmal wiederholt, damit der Übergang und die Supernova-Ausbreitung deutlich langsamer sichtbar sind.
- **Prüfung:** Beide neuen Dateien wurden mit `ast.parse` syntaktisch geprüft. Es wurde kein Simulations-, GPU- oder echter Plotlauf gestartet. Die bestehende `Supernovae_3D_turbulent.py` und vorhandene Ergebnisdateien wurden nicht verändert.
- **Ressourcenentscheidung:** Wegen des dokumentierten GPU-Speicherfehlers bei 128³ und 60 vollständigen Phase-2-Snapshots verwendet das neue Skript standardmäßig `20` Snapshots. Die Zeitlupe wird im Plot durch Frame-Wiederholungen erzeugt; die Snapshot-Anzahl bleibt über `--num-snapshots` veränderbar.
- **Offen:** Die tatsächliche Laufzeit und der Speicherbedarf hängen von Gittergröße, GPU, Snapshot-Anzahl und dem verwendeten turbulenten Medium ab. Vor einem Lauf muss die GPU-Auswahl über `CUDA_VISIBLE_DEVICES` gesetzt werden.

## 2026-11-02 – Zeitlupenfaktor und Übergang bei t=5 angepasst

- **Anlass:** Die erste kombinierte Animation war ab `t=4.98` zu langsam; außerdem wirkte der Übergang bei `t=5` durch zwei unterschiedlich dargestellte Zustände mit gleichem Zeitstempel zu stockend.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/plot_supernovae_turbolenz.py`.
- **Änderungen:** Der standardmäßige Zeitlupenfaktor wurde von `8` auf `4` reduziert. Der letzte Medium-Zustand vor der Explosion wird am doppelten Zeitstempel `t=5` nur einmal gehalten; der Supernova-Zustand nach dem Energieeintrag bleibt erhalten. Es wird keine künstliche Interpolation zwischen physikalischen Zuständen eingeführt.
- **Prüfung:** Die Änderung wurde statisch geprüft; die vorhandene Simulation wird nicht erneut ausgeführt. Für die neue GIF-Datei reicht ein erneuter Plotlauf mit `--slowdown-factor 4`.

## 2026-11-02 – Supernova-Snapshotanzahl an 128er-Slice-GIF angepasst

- **Anlass:** Die Supernova-Animation mit 20 Phase-2-Snapshots wirkte zeitlich zu grob. Die vorhandene Datei `supernova_3d_slices_128cells.gif` wurde lesend geprüft und enthält 60 Frames bei `100 ms` pro Frame.
- **Änderung:** `supernovae_turbolenz.py` verwendet nun standardmäßig `--num-snapshots 60`, damit die Supernova-Phase dieselbe zeitliche Auflösung wie das vorhandene 128er-Slice-GIF erhält.
- **Ressourcenhinweis:** Dadurch werden während der GPU-Integration deutlich mehr vollständige 3D-Zustände gehalten. Der Lauf benötigt deshalb mehr GPU-Speicher als die vorherige 20-Snapshot-Variante.
- **Prüfung:** Die GIF-Datei wurde mit Pillow ohne Änderung gelesen; es wurden 60 Frames festgestellt. Die Python-Datei wurde nach der Änderung syntaktisch geprüft. Kein neuer Simulationslauf wurde durch Codex gestartet.

## 2026-11-02 – Logarithmische Druckdarstellung für die Hintergrundturbulenz ergänzt

- **Anlass:** Nach dem Einsetzen der Supernova dominierte der hohe Explosionsdruck die gemeinsame Farbskala. Dadurch waren die Druckschwankungen des turbulenten Hintergrunds optisch kaum noch erkennbar.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/plot_supernovae_turbolenz.py`.
- **Umsetzung:** Der Druck wird nun logarithmisch mit `LogNorm` dargestellt. Standardmäßig werden `vmin` und `vmax` nur aus den positiven Druckwerten des turbulenten Mediums bestimmt; Supernova-Werte oberhalb dieser Referenz werden am oberen Farbrand gesättigt. Mit `--pressure-scale combined` kann alternativ eine gemeinsame logarithmische Skala über Medium und Supernova gewählt werden.
- **Prüfung:** Statische Syntaxprüfung erfolgreich. Die Supernova-Simulation muss für diese reine Plotänderung nicht erneut ausgeführt werden.

## 2026-11-02 – Logarithmische Dichtedarstellung ergänzt

- **Anlass:** Die gewünschte logarithmische Darstellung bezog sich auf die Dichte `rho` und nicht auf den Druck allein.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/plot_supernovae_turbolenz.py`.
- **Umsetzung:** Die Dichte kann nun mit `--density-scale log-medium` logarithmisch und mit Farbgrenzen aus dem turbulenten Medium dargestellt werden. Alternativ sind `linear` und `log-combined` verfügbar. Der Druck bleibt logarithmisch mit der separaten Option `--pressure-scale`.
- **Prüfung:** Statische Syntaxprüfung; kein Simulationslauf und keine GPU-Nutzung durch Codex. Für die neue Darstellung genügt ein erneuter Plotlauf aus den vorhandenen NPZ-Dateien.

## 2026-11-02 – Kontrastverstärkung der logarithmischen Dichte ergänzt

- **Anlass:** Eine einfache logarithmische Dichteskala machte die Hintergrundturbulenz weiterhin nicht ausreichend sichtbar. Eine doppelte Anwendung von `log` wäre für Dichtewerte nahe oder unter `1` nicht allgemein sinnvoll.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/plot_supernovae_turbolenz.py`.
- **Umsetzung:** Nach der logarithmischen Dichtenormierung wird standardmäßig eine Potenztransformation mit `density_contrast=0.3` angewendet. Kleinere Exponenten können über `--density-contrast` eine stärkere Hervorhebung kleiner Dichteunterschiede bewirken. Die Farblegende wird weiterhin in tatsächlichen Dichtewerten beschriftet.
- **Prüfung:** Statische Syntaxprüfung erfolgreich. Die vorhandenen NPZ-Daten reichen für einen neuen Plotlauf aus; die laufende Supernova-Simulation muss dafür nicht unterbrochen werden.

## 2026-11-02 – Robuste Druck-Farbgrenzen und Framezahl der vorhandenen Animation geprüft

- **Befund zur Flüssigkeit:** `supernovae_turbolenz_128cells_1.gif` enthält 52 unterschiedliche GIF-Frames; die Zeitlupenwiederholung verlängert nur deren Anzeige. Die laufende 60-Snapshot-Simulation muss abgeschlossen sein, bevor die höhere physikalische zeitliche Auflösung geplottet werden kann.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/plot_supernovae_turbolenz.py`.
- **Druckdarstellung:** Die logarithmische Druckskala verwendet bei `pressure_scale=medium` standardmäßig das 1. bis 95. Perzentil der positiven Medium-Drücke. Hohe Explosionsdrücke werden damit absichtlich am oberen Farbrand gesättigt, während niedrige Hintergrunddruckunterschiede mehr Farbkontrast erhalten.
- **Konfiguration:** Die Grenzen sind über `--pressure-low-percentile` und `--pressure-high-percentile` veränderbar.
- **Prüfung:** Vorhandene GIFs wurden mit Pillow nur lesend auf Framezahl und Dauer geprüft. Kein Simulationslauf wurde durch Codex gestartet.

## 2026-11-02 – Dichteskala anhand der zentralen Schnitte festgelegt

- **Datenbefund:** In den vorhandenen zentralen Medium-Schnitten liegen die Dichtewerte im 1.–99. Perzentil ungefähr zwischen `0.669` und `1.424`. In den vorhandenen Supernova-Schnitten reichen die Werte am Ende ungefähr von `0.012` bis `2.625`.
- **Entscheidung:** Für die Hintergrunddarstellung verwendet der Plotter nun standardmäßig eine logarithmische Dichteskala mit den zentralen Medium-Perzentilen `1...99` als Farbgrenzen. Die extremen Supernova-Dichten werden bewusst gesättigt, damit die Turbulenz sichtbar bleibt. Die zusätzliche Potenztransformation ist standardmäßig deaktiviert (`density_contrast=1`).
- **Druckkorrektur:** Die Druckskala wurde auf die vorherige vollständige Medium-Min/Max-Skala zurückgesetzt (`pressure`-Perzentile standardmäßig `0...100`).
- **Prüfung:** Die NPZ-Felder wurden lesend statistisch ausgewertet; anschließend wurde die Plotdatei syntaktisch geprüft. Kein Simulationslauf gestartet.

## 2026-11-02 – Ursprüngliche kombinierte Druckskala wiederhergestellt

- **Anlass:** Der erste Replot verwendete weiterhin `--pressure-scale medium` und entsprach deshalb nicht der ursprünglichen Druckdarstellung.
- **Änderung:** Die Standard-Druckskala in `plot_supernovae_turbolenz.py` ist wieder `combined`: logarithmische `LogNorm` über die vollständigen positiven Druckwerte von Medium und Supernova, ohne Perzentilbegrenzung (`0...100`).
- **Abgrenzung:** Die Dichte bleibt unabhängig davon logarithmisch mit den zentralen Medium-Perzentilen skaliert, damit die Hintergrundturbulenz sichtbar bleibt.
- **Prüfung:** Statische Codeänderung; kein Simulationslauf gestartet.

## 2026-11-02 – Mismatch des Medium-Datensatzes beim kombinierten Plot gefunden

- **Diagnose:** Die Supernova wurde aus `turbulent_medium_128cells_t5_kf25_gpu2.npz` gestartet, während der kombinierte Plot bisher `turbulent_medium_128cells_t5_gpu2_snapshots.npz` verwendete. Die Dateien beschreiben unterschiedliche turbulente Medien.
- **Nachweis:** Der erste Supernova-Dichte-Schnitt stimmt mit dem zentralen Schnitt des korrekten `kf25`-Finalzustands exakt überein (`max_abs_diff=0`), aber nicht mit dem bisher geplotteten Medium. Der künstliche Sprung bei `t=5` und das scheinbare Verschwinden der Hintergrundturbulenz wurden dadurch erklärt.
- **Befund:** `turbulent_medium_128cells_t5_kf25_gpu2.npz` enthält bereits die Felder `states` und `time_points`; eine erneute Medium-Simulation ist für die Animation nicht erforderlich.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/plot_supernovae_turbolenz.py` verwendet diese korrekte `kf25`-Datei nun als Standard-Mediuminput.
- **Nächster Schritt:** Die vorhandene Supernova-NPZ-Datei mit dem korrekten Mediuminput neu plotten. Die laufende 60-Snapshot-Simulation bleibt davon unabhängig.

## 2026-11-03 – Zweiseitig randverstärkte Dichteskala ergänzt

- **Ziel:** Niedrige und hohe Dichten sollen jeweils mehr Farbauflösung erhalten, während mittlere Dichten um die Hintergrunddichte weniger stark aufgeteilt werden.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/plot_supernovae_turbolenz.py`.
- **Umsetzung:** Neue Option `density_scale=log-two-sided`. Die Skala arbeitet mit `log10(rho)`, ist um die Mediane der Mediumdichte zentriert und verwendet standardmäßig den Randexponenten `0.35`. Kleinere Werte verstärken beide Randbereiche weiter; die mittleren Werte werden komprimiert.
- **Druck:** Die kombinierte logarithmische Druckskala bleibt unverändert.
- **Prüfung:** Statische Syntaxprüfung; kein Simulationslauf gestartet.

## 2026-11-03 – Logarithmische Geschwindigkeitsskala ergänzt

- **Anlass:** Die lineare Geschwindigkeitsskala wurde durch die hohen Explosionsgeschwindigkeiten dominiert. In den vorhandenen 120 Supernova-Schnitten reicht das Maximum bis ungefähr `22.9`, während 99 % der Werte nur bis ungefähr `3.1` und das turbulente Medium überwiegend bis ungefähr `1.4` reichen.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/plot_supernovae_turbolenz.py`.
- **Umsetzung:** Neue Standardoption `speed_scale=log-medium`. Die logarithmische Skala verwendet einen positiven Boden `speed_floor=0.01` und das 99. Perzentil des turbulenten Mediums als obere Referenz. Hohe Explosionsgeschwindigkeiten werden gesättigt; geringe Geschwindigkeiten erhalten deutlich mehr Farbauflösung.
- **Prüfung:** Die vorhandene 120-Snapshot-NPZ-Datei wurde lesend statistisch ausgewertet. Die Plotdatei wurde syntaktisch geprüft; kein Simulationslauf gestartet.

## 2026-11-03 – Geschwindigkeitsskala moderater eingestellt

- **Anlass:** Der Geschwindigkeitsboden `0.01` verstärkte die niedrigen Geschwindigkeitsunterschiede zu stark.
- **Änderung:** Der Standardwert `speed_floor` wurde auf `0.05` erhöht. Die logarithmische Skala bleibt erhalten, ist aber weniger aggressiv im unteren Bereich.
- **Prüfung:** Statische Codeänderung; für die neue Darstellung ist nur ein erneuter Plotlauf erforderlich.

## 2026-11-03 – Pfade und Imports an neues Repository angepasst

- **Ziel:** Nach dem Umzug in `/export/home/bgschwen/supernovae-thesis` sollen Imports und Standardausgaben unabhängig vom alten Repositorypfad funktionieren und im Ordner `Praktikum_Datenzubereitung` liegen.
- **Geänderte Dateien:** `Praktikum_Datenzubereitung/Supernovae_2D.py`, `Supernova_plot_1D.py` und `Supernova_plot_3D.py`.
- **Änderungen:** Der 2D-Plotimport unterstützt nun sowohl den Modulstart aus dem Repository-Hauptverzeichnis als auch den dokumentierten Direktstart aus `Praktikum_Datenzubereitung`. Der 2D-Standardausgabepfad sowie die Defaultausgaben der 1D-/3D-Plotmodule werden über den tatsächlichen Skriptordner aufgelöst.
- **Abgrenzung:** Bereits vorhandene explizite Ausgabeparameter und wissenschaftliche Ergebnisdateien wurden nicht überschrieben oder verschoben.
- **Prüfung:** Anschließend wird eine vollständige Python-Syntaxprüfung durchgeführt; keine Simulation und keine GPU-Nutzung durch Codex.

## 2026-11-03 – Geschwindigkeitsskala auf ursprüngliche lineare Darstellung zurückgesetzt

- **Anlass:** Die logarithmische Geschwindigkeitsskala hob die kleinen Geschwindigkeiten stärker hervor als gewünscht.
- **Änderung:** Die Standarddarstellung der Geschwindigkeit in `plot_supernovae_turbolenz.py` ist wieder linear über den globalen kombinierten Wertebereich. Die optionale logarithmische Variante bleibt im Code verfügbar, wird aber nicht mehr standardmäßig verwendet.
- **Abgrenzung:** Die angepassten Pfade sowie die zweiseitig randverstärkte Dichteskala und die bisherige Druckskala bleiben unverändert.
- **Prüfung:** Statische Syntaxprüfung; kein Simulationslauf gestartet.

## 2026-11-03 – Moderate Potenzskala für geringe Geschwindigkeiten ergänzt

- **Anlass:** Die lineare Geschwindigkeitsskala unterschied geringe Geschwindigkeiten zu wenig, während die logarithmische Skala zu stark verstärkte.
- **Änderung:** `plot_supernovae_turbolenz.py` unterstützt nun `speed_scale=power` mit standardmäßigem `speed_gamma=0.5`. Die PowerNorm hebt niedrige Geschwindigkeiten moderat hervor und verwendet weiterhin den globalen Geschwindigkeitsbereich.
- **Prüfung:** Statische Syntaxprüfung; keine Simulation und kein GPU-Lauf gestartet.

## 2026-11-03 – Moderate logarithmische Druck-Kontrastskala ergänzt

- **Anlass:** Die niedrigen Hintergrunddrücke waren trotz logarithmischer Skala gegenüber dem hohen Explosionsdruck zu schwer zu unterscheiden.
- **Änderung:** `plot_supernovae_turbolenz.py` verwendet standardmäßig `pressure_contrast=0.5`. Die logarithmisch normierten Druckwerte werden dadurch im unteren Bereich stärker aufgefächert; hohe Explosionsdrücke werden stärker komprimiert. Mit `pressure_contrast=1.0` wird die reine ursprüngliche LogNorm wiederhergestellt.
- **Prüfung:** Statische Syntaxprüfung; kein Simulationslauf gestartet.

## 2026-11-03 – Fokussierte Druckskala für den Bereich p=1...10 ergänzt

- **Datenbefund:** Der kombinierte Druckbereich der vorhandenen Daten reicht bis ungefähr `12300`; dadurch belegt der Bereich `p=1...10` nur einen kleinen Teil der logarithmischen Farbskala.
- **Änderung:** `plot_supernovae_turbolenz.py` akzeptiert nun `--pressure-display-min` und `--pressure-display-max`. Werte außerhalb dieses Bereichs werden für die Darstellung gesättigt, die Daten selbst bleiben unverändert.
- **Vorgesehene Darstellung:** Für den Hintergrundvergleich wird zunächst `p=0.3...10` empfohlen. Dadurch wird der Unterschied zwischen `p=1` und `p=10` auf einem großen Farbbereich sichtbar.
- **Prüfung:** Die Druckdaten wurden lesend statistisch ausgewertet; kein Simulationslauf gestartet.

## 2026-11-03 – Geschwindigkeitsstatistik auf Schallgeschwindigkeitsverhältnis umgestellt

- **Ziel:** Die Darstellung soll nur die typische Geschwindigkeit des 3D-Mediums relativ zur Schallgeschwindigkeit zeigen.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/plot_mean_velocity.py`.
- **Berechnung:** Für jeden Snapshot werden `mean_speed=<|v|>`, `mean_sound_speed=<sqrt(gamma*p/rho)>` und `speed_to_sound_ratio=mean_speed/mean_sound_speed` berechnet. Die Schallgeschwindigkeit wird lokal aus Druck, Dichte und `gamma` bestimmt und anschließend räumlich gemittelt.
- **Entfernt:** Gemittelter Geschwindigkeitsvektor, Nettofluss, einzelne Geschwindigkeitskomponenten und RMS-Geschwindigkeit werden nicht mehr geplottet oder in die Statistik-CSV geschrieben.
- **Darstellung:** Eine einzelne Kurve zeigt `mean_speed/mean_sound_speed`; eine gestrichelte Linie bei `1` markiert die Schallgeschwindigkeit.
- **Prüfung:** Statische Syntaxprüfung vorgesehen; keine Simulation und keine GPU-Nutzung durch Codex.

## 2026-11-03 – Diagnoseplot für Geschwindigkeit und Schallgeschwindigkeit ergänzt

- **Ziel:** Den Abfall des Quotienten `mean_speed/mean_sound_speed` zeitlich auf seine Ursache untersuchen.
- **Geänderte Datei:** `Praktikum_Datenzubereitung/plot_mean_velocity.py`.
- **Darstellung:** Oberes Panel mit typischer Geschwindigkeit `mean_speed` und mittlerer Schallgeschwindigkeit `mean_sound_speed`; unteres Panel mit deren Quotienten und einer Referenzlinie bei `1`.
- **CSV:** Die numerischen Zeitreihen bleiben unverändert verfügbar und enthalten weiterhin `time`, `mean_speed`, `mean_sound_speed` und `speed_to_sound_ratio`.
- **Prüfung:** Statische Syntaxprüfung; keine Simulation und keine GPU-Nutzung durch Codex.

## 2026-11-03 – Neue Vergleichsvariante mit dreifacher Forcierungsamplitude geplant

- **Ziel:** Den Einfluss einer stärkeren Anfangsturbulenz auf die Supernova-Ausbreitung untersuchen.
- **Änderung gegenüber der bisherigen Medium-Referenz:** `forcing_amplitude=3.0` statt `1.0`.
- **Konstant zu haltende Parameter:** `128^3` Zellen, `t_end=5`, `forcing_wavenumber=25`, `correlation_time=1`, `seed=42`, `rho=1`, `p=1`, `gamma=5/3`, `C_cfl=0.4` und periodische Randbedingungen.
- **Geplanter Ablauf:** Neues turbulentes Medium mit vollständigen Snapshots erzeugen, danach die Supernova für `Delta t=0.1` bis `t=5.1` in diesem Zustand simulieren und mit der zweiseitig randverstärkten Dichteskala plotten.
- **Status:** Nur geplant; kein neuer Medium-, Supernova- oder Plotlauf durch Codex gestartet. Die konkrete GPU wird vom Nutzer beim manuellen Start ausgewählt.

## 2026-11-02 – Zeitlupenfaktor auf 2 reduziert

- **Änderung:** Der Standardwert `slowdown_factor` in `plot_supernovae_turbolenz.py` wurde von `4` auf `2` gesetzt. Damit wird die Darstellung ab `t=4.98` doppelt so schnell wie zuvor abgespielt.
- **Abgrenzung:** Die Anzahl der physikalischen Supernova-Snapshots wird dadurch nicht verändert; nur die Wiederholungsdauer der GIF-Frames ändert sich.
- **Prüfung:** Statische Codeänderung; kein Simulationslauf gestartet.

## 2026-11-03 – Cooling-Option für Vergleichsläufe ergänzt

- **Ziel:** Einen adiabatischen Referenzlauf mit `64^3` Zellen und einen zweiten
  Lauf mit `128^3` Zellen und aktiviertem Cooling vorbereiten.
- **Ausgangslage:** `anfangsmedium.py` verwendete bisher ausschließlich die
  adiabatische Konfiguration. Die installierte `astronomix`-Version besitzt
  jedoch einen Cooling-Zweig mit `CoolingConfig`, `CoolingParams` und einer
  einfachen Power-Law-Kühlkurve.
- **Änderung:** `Praktikum_Datenzubereitung/anfangsmedium.py` akzeptiert nun
  `--cooling` sowie die reproduzierbaren Parameter
  `--cooling-factor`, `--cooling-exponent`,
  `--cooling-reference-temperature` und
  `--cooling-floor-temperature`. Das Cooling ist standardmäßig aus und wird
  erst mit `--cooling` aktiviert. Die gewählten Cooling-Parameter werden in
  den Metadaten gespeichert.
- **Modellannahme:** Für die dimensionslosen Projektgrößen wird zunächst eine
  explizite Power-Law-Kurve mit `Lambda(T)=1*T`, `T_ref=1` und einem Boden von
  `10^-6` verwendet. Die astronomix-Standardwerte mit `T_floor=10^4` würden
  in diesem Codeeinheitensystem praktisch kein Cooling auslösen. Diese Wahl
  ist ein numerischer Vergleichsversuch und keine astrophysikalische
  Kalibrierung.
- **Prüfung:** Python-Syntax und CLI-Hilfe erfolgreich geprüft. Es wurde kein
  Simulations- oder GPU-Lauf durch Codex gestartet. Die GPUs 6 und 7 wurden
  vom Nutzer als frei gemeldet; die tatsächliche Auswahl für die manuellen
  Läufe bleibt beim Nutzer.
- **Offen:** Vor dem Lauf sollte bestätigt werden, dass diese dimensionslose
  Cooling-Annahme gewünscht ist. Cooling-Faktor und Cooling-Zeitskala sind
  wissenschaftlich nicht aus den bisherigen Projektparametern festgelegt.

## 2026-11-03 – Geschwindigkeitsplot des 64³-Mediums erzeugt

- **Datensatz:** `Praktikum_Datenzubereitung/turbulent_medium_64cells_t5_kf25_amp3_gpu6.npz`.
- **Auswertung:** `plot_mean_velocity.py` berechnete aus den gespeicherten
  32 Snapshots die typische Geschwindigkeit, die mittlere lokale
  Schallgeschwindigkeit `sqrt(gamma*p/rho)` und deren Quotienten.
- **Ergebnisse:** PNG und CSV wurden unter den Namen
  `turbulent_velocity_statistics_64cells_amp3_gpu6` im Ordner
  `Praktikum_Datenzubereitung` gespeichert. Die typische Geschwindigkeit
  erreicht ungefähr `0.58`, während der Quotient im Verlauf von etwa `0.39`
  auf `0.20` fällt.
- **Prüfung:** Der erzeugte PNG-Plot wurde visuell kontrolliert. Es wurde kein
  Simulations- oder GPU-Lauf gestartet.

## 2026-11-04 – Geschwindigkeitsplot des 128³-Cooling-Mediums erzeugt

- **Datensatz:** `Praktikum_Datenzubereitung/turbulent_medium_128cells_t5_kf25_amp3_cooling_gpu7.npz`.
- **Auswertung:** `plot_mean_velocity.py` wurde CPU-only auf die gespeicherten
  Medium-Snapshots angewendet. Berechnet wurden typische Geschwindigkeit,
  mittlere lokale Schallgeschwindigkeit und deren Quotient.
- **Ergebnisse:** PNG und CSV wurden unter den Namen
  `turbulent_velocity_statistics_128cells_amp3_cooling_gpu7` in
  `Praktikum_Datenzubereitung` gespeichert. Der Quotient erreicht anfänglich
  ungefähr `0.52` und liegt am Ende bei ungefähr `0.39`; damit bleibt das
  Medium in dieser Statistik subsonisch.
- **Prüfung:** Der erzeugte PNG-Plot wurde visuell kontrolliert. Es wurde kein
  Simulations- oder GPU-Lauf gestartet.

## 2026-11-04 – Amp9-Cooling-Lauf bei 68,2 % abgebrochen

- **Versuch:** `128^3` Zellen, `t=0...5`, `gamma=5/3`, `CFL=0.4`, Seed 42,
  `k_f=25`, Forcing-Amplitude `9.0`, explizites Power-Law-Cooling mit
  `factor=1`, `exponent=1`, `T_ref=1` und `T_floor=10^-6` auf GPU 7. Die
  GPU war vom Nutzer als frei angegeben.
- **Verlauf:** Der Nutzer brach den Lauf bei `68,2 %` mit `Ctrl+C` ab. Der
  anschließende `KeyboardInterrupt` innerhalb eines JAX-Callbacks erklärt die
  lange Traceback-Kette; ein eigenständiger numerischer Fehler ist daraus
  nicht abzuleiten.
- **Backend:** astronomix meldete den Fallback auf `NATIVE_JAX`, weil keine
  sichtbare GPU mit Compute Capability `>=8.0` erkannt wurde. Dadurch ist der
  Lauf voraussichtlich deutlich langsamer als mit dem PALLAS/Triton-Backend.
- **Ausgabe:** Es läuft kein Simulationsprozess mehr, und es wurde keine
  vollständige NPZ-Datei erzeugt. Im geposteten Befehl fehlte außerdem
  `--save-snapshots`; für einen späteren Velocity-Plot muss diese Option beim
  Neustart gesetzt werden.
- **Offen:** Der Lauf muss bei Bedarf vollständig neu gestartet werden; ein
  Fortsetzen bei `68,2 %` ist mit dem aktuellen Skript nicht möglich.

## 2026-11-04 – Amp9-Cooling-Lauf erfolgreich durch Codex überwacht

- **Lauf:** Der zuvor abgebrochene Lauf wurde nach ausdrücklicher Freigabe des
  Nutzers erneut auf GPU 7 gestartet: `128^3` Zellen, `t=0...5`, Seed 42,
  `k_f=25`, Forcing-Amplitude `9.0`, `CFL=0.4`, periodische Randbedingungen,
  explizites Power-Law-Cooling mit `factor=1`, `exponent=1`, `T_ref=1` und
  `T_floor=10^-6`.
- **Ausgabe:** `--save-snapshots` war aktiviert. Der Lauf erreichte `100 %`
  ohne Exception und speicherte
  `Praktikum_Datenzubereitung/turbulent_medium_128cells_t5_kf25_amp9_cooling_gpu7.npz`.
  Die Datei enthält `states` in der Form `(5, 128, 128, 128)` sowie die
  erwarteten Cooling-Metadaten.
- **Diagnose:** Die vorherige Abbruchstelle bei `68,2 %` wurde überschritten
  und der Lauf erfolgreich beendet. Damit ist kein reproduzierbarer Fehler in
  der Simulation nachgewiesen; wahrscheinlich war der vorherige Lauf durch
  `Ctrl+C` beziehungsweise das nicht reagierende Frontend beendet worden.
  astronomix verwendete weiterhin den `NATIVE_JAX`-Fallback, weil GPU 7 nicht
  als Compute-Capability-`>=8.0` für PALLAS/Triton erkannt wurde.
- **Laufzeit:** Nicht separat mit einer Stoppuhr erfasst; der überwachte
  Rechenlauf dauerte mehrere Minuten. GPU 7 wurde ausschließlich für diesen
  vom Nutzer freigegebenen Simulationszweck verwendet.

## 2026-11-04 – Ursache des fehlenden supersonischen Langzeitbereichs untersucht

- **Datengrundlage:** Vergleich der gespeicherten `64^3`-Referenz ohne Cooling,
  des `128^3`-Laufs mit Amplitude `3` und Cooling, des `128^3`-Laufs mit
  Amplitude `9` und Cooling sowie des vorhandenen `128^3`-Laufs mit Amplitude
  `3` ohne Cooling.
- **Befund für Amplitude 9 mit Cooling-Faktor 1:** Die typische Geschwindigkeit
  steigt bis ungefähr `1.23` und liegt bei `t=5` noch bei `1.16`. Gleichzeitig
  steigt der mittlere Druck bis ungefähr `9.08`, wodurch die mittlere
  Schallgeschwindigkeit bis ungefähr `3.89` wächst. Das Verhältnis fällt
  deshalb nach dem Anfangspeak bis ungefähr `0.30`.
- **Vergleich mit Amplitude 3 und Cooling:** Bei `t=5` liegen typische
  Geschwindigkeit, Schallgeschwindigkeit und Verhältnis ungefähr bei
  `0.69`, `1.77` und `0.39`; die stärkere Forcierung erhöht also die
  Geschwindigkeit weniger stark als den thermischen Druck.
- **Physikalische Interpretation:** Die periodische OU-Forcierung injiziert
  fortlaufend Energie. Bei größerer Amplitude wird ein größerer Anteil der
  kinetischen Energie durch Kompression und Schocks in innere Energie
  umgewandelt. Die mittlere Dichte bleibt wegen der periodischen Randbedingung
  ungefähr `1`; der Anstieg der Schallgeschwindigkeit stammt daher vor allem
  vom Druck beziehungsweise der Temperatur `p/rho`.
- **Lokale Mach-Zahlen:** Im Amplitude-9-Lauf sind anfangs lokal supersonische
  Zellen vorhanden: bei `t≈0.16` haben etwa `24.7 %` der Zellen `M>1`, bei
  `t≈0.47` etwa `17.2 %`. Bei `t=5` ist keine Zelle mehr supersonisch. Die
  bisher geplottete Größe `<|v|>/<c_s>` ist dagegen ein globaler
  charakteristischer Quotient und nicht der Mittelwert der lokalen Mach-Zahlen.
- **Cooling-Schlussfolgerung:** Für die verwendete Power-Law-Kurve ist der
  `cooling-factor` direkt die Stärke des Kühlterms. Bei Exponent `1` entspricht
  Faktor `1` in der aktuellen Dichteordnung ungefähr einer Cooling-Zeit von
  `3.8` Codezeiteinheiten; Faktor `5` beziehungsweise `10` verkürzt sie grob
  auf `0.76` beziehungsweise `0.38`. Ein stärkerer Cooling-Faktor ist daher
  der direkteste nächste Versuch. Die explizite Zeitintegration muss dabei auf
  mögliche Stabilitätsprobleme kontrolliert werden.
- **Status:** CPU-only-Auswertung, keine neue Simulation gestartet. Als
  nächster Versuch wird ein Cooling-Faktor von `5` empfohlen; für einen
  sichereren supersonischen charakteristischen Quotienten könnte anschließend
  Faktor `10` untersucht werden. Diese Parameterwahl ist ein Experiment und
  keine astrophysikalische Kalibrierung.

## 2026-11-04 – Amp9-Lauf mit Cooling-Faktor 5 erfolgreich durchgeführt

- **Lauf:** Nach ausdrücklicher Nutzerfreigabe wurde der `128^3`-Lauf auf GPU
  7 gestartet. Verwendet wurden `t=0...5`, `gamma=5/3`, `CFL=0.4`, Seed 42,
  `k_f=25`, Forcing-Amplitude `9.0`, periodische Randbedingungen und explizites
  Power-Law-Cooling mit `factor=5`, `exponent=1`, `T_ref=1` und
  `T_floor=10^-6`.
- **Ergebnis:** Der Lauf erreichte `100 %` ohne Exception. Die Datei
  `Praktikum_Datenzubereitung/turbulent_medium_128cells_t5_kf25_amp9_cooling_factor5_gpu7.npz`
  wurde vollständig gespeichert. Sie enthält 32 gespeicherte Snapshots und
  die Zustandsform `(5, 128, 128, 128)` pro Zustand.
- **Metadatenprüfung:** `cooling=true`, `cooling_factor=5.0`,
  `forcing_amplitude=9.0` und `snapshots_saved=true` wurden aus der NPZ-Datei
  bestätigt.
- **Backend:** astronomix verwendete wie zuvor den `NATIVE_JAX`-Fallback;
  der Lauf überschritt die frühere Abbruchstelle bei `68,2 %` problemlos.

## 2026-11-04 – Cooling-Faktor 5 quantitativ bewertet

- **Datensatz:** `turbulent_medium_128cells_t5_kf25_amp9_cooling_factor5_gpu7.npz`.
- **Befund:** Der charakteristische Quotient erreicht kurzzeitig ungefähr
  `0.91` und liegt bei `t=5` bei ungefähr `0.65`. Die typische Geschwindigkeit
  beträgt am Ende ungefähr `1.26`, die mittlere Schallgeschwindigkeit ungefähr
  `1.94` und der mittlere Druck ungefähr `2.29`.
- **Zielabschätzung:** Für einen Endquotienten von `1.5` wäre bei unveränderter
  typischer Geschwindigkeit eine mittlere Schallgeschwindigkeit von ungefähr
  `0.84` beziehungsweise ein Druck von ungefähr `0.42` erforderlich.
- **Nächste Wahl:** Im Faktor-5-Lauf ist der Druck näherungsweise in einem
  Cooling-/Heizgleichgewicht. Eine grobe Skalierung legt deshalb einen Faktor
  von etwa `25...30` für einen Endquotienten um `1.5` nahe. Als kontrollierten
  nächsten Versuch wird zunächst `cooling_factor=25` empfohlen. Diese
  Extrapolation ist unsicher, weil stärkeres Cooling die Geschwindigkeit und
  die Energiebilanz ebenfalls verändern kann.
- **Prüfung:** CPU-only-Auswertung der vorhandenen NPZ-Datei; keine neue
  Simulation gestartet.

## 2026-11-04 – Faktor-25-Lauf erreicht Zielbereich, aber wurde numerisch ungültig

- **Lauf:** Nach ausdrücklicher Nutzerfreigabe wurde auf GPU 7 ein weiterer
  `128^3`-Lauf mit Amplitude `9.0`, explizitem Cooling und Faktor `25.0`
  gestartet. Der Prozess meldete `100 %` und Exit-Code `0`.
- **Datenprüfung:** Die gespeicherte Datei
  `turbulent_medium_128cells_t5_kf25_amp9_cooling_factor25_gpu7.npz` ist
  dennoch nicht als gültiges Ergebnis verwendbar. Die Snapshots bis ungefähr
  `t=1.09` sind endlich; danach werden `time_points` null beziehungsweise
  `NaN`, und der Endzustand enthält `NaN`. Die Ursache ist ein numerisch zu
  aggressiver expliziter Cooling-Term, nicht ein sauberer abgeschlossener
  physikalischer Lauf.
- **Zwischenergebnis:** In den gültigen frühen Snapshots erreicht der
  charakteristische Quotient ungefähr `1.6...1.7`; das Ziel wird also
  grundsätzlich erreicht, aber die Simulation bleibt nicht bis `t=5` stabil.
- **Codeänderung:** `anfangsmedium.py` besitzt nun die Option
  `--cooling-method {explicit,implicit}`. Zusätzlich werden nicht-endliche
  Zustände sowie ungültige oder nicht streng ansteigende Snapshotzeiten vor
  dem Speichern abgewiesen. Die Standardmethode bleibt für Reproduzierbarkeit
  `explicit`; für Faktor 25 soll die implizite Methode getestet werden.
- **Prüfung:** Syntax-/CLI-Prüfung erfolgreich. Die CLI-Prüfung wurde zuerst
  ohne CPU-Einschränkung gestartet und erzeugte CUDA-OOM-Warnungen beim
  Geräteprüfen, startete aber keine Simulation; die eigentliche Validierung
  erfolgt CPU-only. Kein weiterer GPU-Lauf gestartet.

## 2026-11-04 – Impliziter Faktor-25-Lauf ebenfalls instabil

- **Lauf:** Nach erneuter ausdrücklicher Freigabe wurde auf GPU 7 ein Lauf mit
  `128^3` Zellen, Amplitude `9.0`, Cooling-Faktor `25.0` und
  `--cooling-method implicit` gestartet. Snapshots waren aktiviert.
- **Ergebnis:** Die Integration wurde numerisch ungültig, bevor ein gültiger
  Datensatz geschrieben werden konnte. Die neue Validierung brach beim
  `final_state` mit einer verständlichen `RuntimeError`-Meldung ab; es wurde
  keine gültige NPZ-Ausgabe erzeugt.
- **Interpretation:** Der Wechsel von explizitem zu implizitem Cooling löst das
  Problem nicht. Faktor 25 senkt den Druck so stark, dass die anschließende
  Hydrodynamik in sehr stark supersonische Zustände gerät und negative oder
  nicht-endliche thermodynamische Werte erzeugt. Die Stabilitätsgrenze liegt
  daher nicht nur in der Cooling-Zeitintegration.
- **Nächste Optionen:** Cooling-Faktor reduzieren, `CFL` verkleinern und/oder
  astronomix-Positivitätsschutz beziehungsweise flux-preserving Limiting
  aktivieren. Ein weiterer GPU-Lauf wurde nach diesem Fehlschlag nicht
  automatisch gestartet.

## 2026-11-04 – Faktor-25-CFL-0.2-Lauf instabil und Faktor-15-Lauf erfolgreich

- **Faktor 25 mit CFL 0.2:** Der implizite Lauf auf GPU 7 wurde mit halbierter
  CFL-Zahl erneut getestet. Er blieb bis ungefähr `16,5 %` numerisch aktiv,
  wurde danach aber beim finalen Zustand als nicht-endlich erkannt. Die
  Halbierung der CFL-Zahl genügt für Faktor 25 daher nicht.
- **Faktor 15:** Gemäß der vorher vereinbarten nächsten Stufe wurde danach auf
  GPU 7 ein `128^3`-Lauf mit Amplitude `9.0`, `CFL=0.2`, implizitem Cooling,
  Cooling-Faktor `15.0`, Seed 42, `k_f=25` und 32 gespeicherten Snapshots
  durchgeführt.
- **Ergebnis:** Der Faktor-15-Lauf erreichte `100 %`, wurde ohne Exception
  gespeichert und bestand die nachträgliche Validierung vollständig:
  endliche Zustände, endliche und streng ansteigende Zeitpunkte.
- **Datensatz:**
  `Praktikum_Datenzubereitung/turbulent_medium_128cells_t5_kf25_amp9_cooling_factor15_implicit_cfl02_gpu7.npz`.
- **Geschwindigkeitsverhältnis:** Peak ungefähr `1.27`; bei `t=5`
  ungefähr `1.23`. Am Ende liegen typische Geschwindigkeit und mittlere
  Schallgeschwindigkeit bei ungefähr `1.19` beziehungsweise `0.96`, der
  mittlere Druck bei ungefähr `0.59`.
- **Interpretation:** Faktor 15 ist in dieser Konfiguration stabil und führt
  zu einem dauerhaft überschallnahen beziehungsweise supersonischen
  charakteristischen Quotienten, erreicht aber das Ziel `1.5` noch nicht.
- **GPU/Laufzeit:** GPU 7 wurde nach ausdrücklicher Nutzerfreigabe verwendet;
  eine genaue Laufzeit wurde nicht separat gemessen. Die Auswertung danach
  erfolgte CPU-only.

## 2026-11-04 – Cooling aus dem Faktor-15-Medium in die Supernova-Phase übernehmen

- **Ziel:** Die Supernova soll ab dem gültigen turbulenten Zustand bei `t=5`
  bis `t=5.1` simuliert werden, wobei die während der Medium-Erzeugung
  verwendete Cooling-Konfiguration auch in Phase 2 aktiv bleibt.
- **Ausgangsdatensatz:**
  `Praktikum_Datenzubereitung/turbulent_medium_128cells_t5_kf25_amp9_cooling_factor15_implicit_cfl02_gpu7.npz`
  mit Amplitude `9.0`, `CFL=0.2`, implizitem Cooling und Faktor `15.0`.
- **Codeänderung:** `Praktikum_Datenzubereitung/supernovae_turbolenz.py`
  liest Cooling-Modell, Methode, Faktor, Exponent, Referenztemperatur und
  Temperaturboden aus den Medium-Metadaten und verwendet diese Werte für die
  Supernova-Phase. Die OU-Forcierung bleibt in Phase 2 deaktiviert. Zusätzlich
  werden nicht-endliche Zustände und ungültige Snapshot-Zeitpunkte vor dem
  Speichern abgewiesen; die Cooling-Parameter werden in der Ausgabedatei
  dokumentiert.
- **Parameter des vorgesehenen Laufs:** `duration=0.1`, `120` Snapshots,
  Explosionsenergie `1.0`, Explosionsradius `0.01`, Glättung `2.0` Zellen und
  `CFL=0.2`. Die Ausgabe wird unter einem neuen Dateinamen gespeichert, damit
  vorhandene Ergebnisse nicht überschrieben werden.
- **Prüfungen:** Python-Kompilierung, CLI-Hilfe und isolierte Erstellung der
  impliziten Cooling-Konfiguration mit Faktor `15.0` erfolgreich. Es wurde
  keine Simulation und kein GPU-Lauf durch Codex gestartet.
- **Annahme/offener Punkt:** Das Cooling wird während der kurzen
  Supernova-Ausbreitung physikalisch als weiterhin aktiv angenommen. Für einen
  adiabatischen Vergleich müsste ein separater Lauf mit explizit deaktiviertem
  Cooling erzeugt werden.

## 2026-11-04 – Fehler beim Speichern der Cooling-Metadaten korrigiert

- **Fehler:** Die Supernova-Integration mit 120 Snapshots erreichte auf GPU 7
  `100 %`, brach danach aber beim Speichern mit `NameError: cooling_enabled`
  ab. Deshalb wurde keine vollständige Supernova-NPZ-Datei erzeugt.
- **Korrektur:** `cooling_enabled` wird nun explizit an
  `_save_central_slices()` übergeben. Der Fehler betraf nur den Speicherschritt;
  die numerische Integration selbst war bis zum Ende gelaufen.
- **Prüfung:** Python-Kompilierung und `git diff --check` erfolgreich. Kein
  neuer GPU-Lauf durch Codex gestartet. Für die GIF-Erzeugung muss der Nutzer
  die Supernova-Phase einmal erneut mit denselben Parametern starten.

## 2026-11-04 – Physikalische Referenzskalierung für warmes ISM ergänzt

- **Ziel:** Die normierten Simulationen sollen zusätzlich auf einen
  astrophysikalisch plausiblen Referenzfall bezogen werden können.
- **Festgelegte Arbeitsannahme:** warmes interstellares Medium mit
  Wasserstoffzahlendichte `n_H=1 cm^-3`, Temperatur `T=8000 K`, physikalischer
  Boxlänge `100 pc`, `gamma=5/3`, mittlerer Teilchenmasse `1.27` und
  `1.4` Wasserstoffmassen pro Wasserstoffkern. Die Boxlänge ist eine
  Modellannahme und muss bei Bedarf mit dem Tutor angepasst werden.
- **Codeänderung:** Neues Modul
  `Praktikum_Datenzubereitung/physikalische_einheiten.py` berechnet aus diesen
  Referenzgrößen die Dichte-, Druck-, Geschwindigkeits-, Zeit- und
  Energieeinheit. `anfangsmedium.py` kann die Skalierung über neue
  `--physical-*`-Argumente in den Medium-Metadaten speichern.
- **Supernovaenergie:** `supernovae_turbolenz.py` akzeptiert nun
  `--physical-explosion-energy-erg` und rechnet die physikalische Energie
  anhand der gespeicherten Energieeinheit in Codeeinheiten um. Für `100 pc`,
  `n_H=1 cm^-3` und `T=8000 K` entspricht `10^51 erg` ungefähr
  `27.95` Codeenergieeinheiten; die Codezeit beträgt ungefähr `10.51 Myr`.
- **Wichtige Einschränkung:** Die Felder bleiben dimensionslos normiert. Die
  neue Skalierung interpretiert die Ergebnisse physikalisch, ersetzt aber noch
  nicht das vereinfachte Power-Law-Cooling durch eine tabellierte
  astrophysikalische Kühlkurve. Ein erster Lauf mit Cooling-Faktor `15` bleibt
  deshalb ein physikalisch skalierter, aber beim Cooling weiterhin
  vereinfachter Vergleichslauf.
- **Prüfungen:** Python-Kompilierung, beide CLI-Hilfen und ein isolierter
  Roundtrip der Skalierungsmetadaten erfolgreich. Kein GPU-Lauf gestartet.

## 2026-11-05 – Tabellierte radiative Cooling-Kurve ergänzt

- **Ziel:** Das bisherige dimensionslose `simple_power_law`-Cooling soll für
  den physikalisch skalierten Lauf durch eine temperaturabhängige radiative
  Cooling-Funktion ersetzt werden.
- **Modell:** Die vorhandene astronomix-Tabelle nach Schure et al. (2009)
  wird als `schure2009_cie` verwendet. Sie beschreibt eine optisch dünne,
  solar-metallische Plasma-Cooling-Kurve im kollisionalen
  Ionisationsgleichgewicht (CIE) und basiert auf SPEX. Die Tabelle deckt etwa
  `10^3.8...10^8.16 K` ab; für diesen Lauf wird ein Temperaturboden von
  `10^4 K` verwendet.
- **Konsistenzannahme:** CIE wird mit ionisiertem Gas und
  `mean_molecular_weight=0.61` gekoppelt. Eine neutrale WNM-Skalierung mit
  `mu=1.27` wird für dieses Cooling-Modell im Code abgewiesen.
- **Codeänderung:** Neues Modul `physikalische_cooling.py` baut aus der
  physikalischen Referenzskalierung die astronomix-`CodeUnits`, konvertiert
  die Schure-Tabelle in Codeeinheiten und rekonstruiert sie in der
  Supernova-Phase aus den Medium-Metadaten. `anfangsmedium.py` akzeptiert
  `--cooling-model schure2009_cie`; die Metadaten dokumentieren Tabelle,
  Temperaturboden und Quelle.
- **Zeitintegration:** Für die tabellierte Kurve wird explizites Cooling
  verlangt. astronomix markiert den impliziten Townsend-Pfad für diese
  piecewise-Power-Law-Tabelle derzeit als nicht zuverlässig. Die bisherige
  implizite Methode bleibt für das alte Test-Potenzgesetz verfügbar.
- **Normierungskorrektur:** Die Geschwindigkeitseinheit ist jetzt korrekt
  `sqrt(p_0/rho_0)`; die Schallgeschwindigkeit beträgt in Codeeinheiten
  `sqrt(gamma)`. Dadurch stimmen Druck-, Temperatur- und Cooling-Umrechnung
  dimensional konsistent überein.
- **Prüfungen:** Syntaxprüfung erfolgreich. Die Schure-Tabelle wurde CPU-seitig
  mit 110 Temperaturpunkten und endlichen Werten erzeugt; ein heißer Testwert
  kühlte nach einem Quellterm-Schritt ab. Keine GPU-Simulation gestartet.

## 2026-11-05 – Speicherfehler nach Schure-Cooling-Lauf korrigiert

- **Fehler:** Der Medium-Lauf mit Schure-CIE-Cooling erreichte `100 %`, brach
  aber beim Speichern mit `NameError: cooling_floor_temperature_code` ab.
  Die numerische Integration war damit nicht als gültige NPZ-Datei verfügbar.
- **Ursache:** Die in `build_simulation()` berechnete Code-Temperatur des
  Cooling-Bodens war im separaten Speicherschritt nicht erneut verfügbar.
- **Korrektur:** Die Umrechnung wurde in eine gemeinsame Hilfsfunktion verlegt
  und wird nun sowohl beim Aufbau der Simulation als auch beim Speichern aus
  den CLI-Parametern rekonstruiert.
- **Prüfung:** Python-Syntax und kleiner Schure-CIE-Konfigurations-Dry-Run
  erfolgreich. Kein neuer Medium- oder Supernova-GPU-Lauf gestartet. Der
  Medium-Lauf muss nach diesem Fix erneut ausgeführt werden.

## 2026-11-05 – Schure-CIE-Supernova mit explizitem Update instabil

- **Medium-Lauf:** Der korrigierte Lauf auf GPU 7 wurde mit `128^3` Zellen,
  `CFL=0.1`, `n_H=1 cm^-3`, `T=10^4 K`, `100 pc` Box und Schure-2009-CIE-
  Cooling erfolgreich bis `t=5` integriert und als gültige 640-MB-NPZ-Datei
  gespeichert. Die Datei enthält 32 vollständige Snapshots, endliche Zustände
  und monotone Zeiten.
- **Supernova-Lauf:** Die Phase 2 mit `E_SN=10^51 erg`, 120 Snapshots und
  expliziter Schure-CIE-Kühlung brach bei ungefähr `4,2 %` wegen
  nicht-endlicher Snapshots ab. Es wurde keine Supernova-NPZ-Datei gespeichert.
- **Ursache:** Die starke zentrale Aufheizung erzeugt lokal eine zu kurze
  Cooling-Zeitskala für den expliziten Temperatur-Quellterm. Die Tabelle selbst
  konnte CPU-seitig korrekt geladen und ein heißer Einzelwert endlich gekühlt
  werden.
- **Korrektur:** `supernovae_turbolenz.py` besitzt nun
  `--phase-two-cooling-method {metadata,explicit,implicit}`. Für Phase 2 kann
  die gleiche Schure-CIE-Kurve mit dem vorhandenen impliziten
  Fixed-Point-Updatepfad von astronomix integriert werden, ohne das bereits
  erfolgreiche Medium neu zu berechnen. Die tatsächlich verwendete Methode
  wird in den Supernova-Metadaten gespeichert.
- **Prüfung:** Syntaxprüfung und Rekonstruktion der Schure-Kurve mit beiden
  Phase-2-Methoden CPU-seitig erfolgreich. Ein erneuter GPU-Supernova-Lauf mit
  `--phase-two-cooling-method implicit` ist noch offen.

## 2026-11-05 – Stabilisierung der Supernova-Energieeinbringung

- **Ziel:** Die Schure-CIE-Supernova-Phase im gültigen turbulenten Medium erneut
  stabil integrieren.
- **Ausgangslage:** Die vorherigen Phase-2-Läufe mit einem Explosionsradius von
  `0.01` Codeeinheiten und deaktiviertem Positivitätsschutz brachen bei etwa
  `4,2–4,7 %` mit nicht-endlichen Zuständen ab.
- **Änderungen:** Der Standard-Explosionsradius in
  `supernovae_turbolenz.py` wurde auf `0.03` Codeeinheiten erhöht. Für Phase 2
  wird nun `POSITIVITY_REDISTRIBUTE` sowohl während der Integrationsstufen als
  auch pro Zeitschritt verwendet; `vacuum_rest=True` verhindert dabei extreme
  rekonstruierte Geschwindigkeiten in nahezu leeren Zellen. NaN-Werte werden
  nicht stillschweigend ersetzt. Explosionsradius, Glättungsbreite und
  Positivitätsschutz werden in den Ergebnis-Metadaten dokumentiert.
- **Physikalische Konfiguration:** Dasselbe Medium
  `turbulent_medium_128cells_t5_warmism100pc_schure2009_gpu7.npz`, implizites
  Schure-2009-CIE-Cooling, `E_SN=10^51 erg`, `CFL=0.1`; die Gesamtenergie der
  Injektion bleibt unverändert.
- **GPU-Freigabe:** Nutzerfreigabe für GPU 7 lag vor; Zweck war ein kurzer
  Stabilitätstest vor dem vollständigen Lauf.
- **Tests:** Der konservative Positivitätsschutz mit 16 Glättungspässen lief
  bis etwa `32,5 %` ohne numerischen Fehler, war aber sehr langsam und wurde
  kontrolliert abgebrochen. Der schnellere Redistribute-Test lief bis etwa
  `19,7 %` ohne den früheren Abbruch und wurde ebenfalls kontrolliert beendet;
  die kurze Testdatei wurde deshalb noch nicht gespeichert.
- **Nächster Schritt:** Vollständiger Phase-2-Lauf mit Radius `0.03`,
  `--num-snapshots 120` und Redistribute-Positivitätsschutz auf GPU 7.

## 2026-11-05 – Dokumentation der numerischen Schutzmaßnahmen über beide Phasen

- **Wichtige methodische Unterscheidung:** Das Anfangsmedium verwendet in
  `anfangsmedium.py` bereits `TurbulentForcingConfig.vacuum_protection=True`.
  Diese Sicherung gehört zur OU-Forcierung und schützt insbesondere vor sehr
  kleinen Dichten beziehungsweise extremen Geschwindigkeiten, die durch die
  Forcierung entstehen können.
- **Phase 1:** Ein allgemeiner `PositivityConfig` mit
  `per_stage_mode`/`per_step_mode` ist derzeit nicht gesetzt; damit bleiben
  diese Modi auf astronomix-Standard (`NONE`). Die vorhandene Medium-Datei ist
  dennoch endlich und numerisch erfolgreich bis `t=5` integriert worden.
- **Phase 2:** Die Supernova-Simulation verwendet zusätzlich
  `POSITIVITY_REDISTRIBUTE` während der Integrationsstufen und pro Zeitschritt
  sowie `vacuum_rest=True`, weil die plötzliche Energieeinbringung einen viel
  stärkeren Stoß erzeugt und die OU-Forcierung in Phase 2 abgeschaltet ist.
- **Interpretation:** Die beiden Phasen verwenden damit nicht exakt dieselbe
  numerische Schutzkonfiguration. Das ist für den aktuellen Stabilisierungslauf
  bewusst dokumentiert. Eine vollständige methodische Vergleichsstudie müsste
  Phase 1 mit demselben Positivitätsschutz neu erzeugen und anschließend Phase
  2 aus diesem neuen Zustand wiederholen. Dabei wären Zustände, Gesamtmasse,
  Gesamtenergie, Impuls und die Häufigkeit ausgelöster Schutzkorrekturen zu
  vergleichen.
- **Offene Entscheidung:** Vorläufig wird der gültige vorhandene
  `t=5`-Zustand für den laufenden Supernova-Test weiterverwendet. Ob der
  allgemeine Positivitätsschutz auch in Phase 1 als Produktionskonfiguration
  aktiviert wird, bleibt nach dem Stabilitätstest und einer
  Erhaltungsgrößenprüfung zu entscheiden.

## 2026-11-05 – Phase-2-Lauf mit Redistribute-Schutz weiterhin instabil

- **Beobachtung:** Der persistente zellij-Lauf mit Radius `0.03`, implizitem
  Schure-CIE-Cooling, `CFL=0.1` und `REDISTRIBUTE` erreichte den Integrations-
  fortschritt `100 %`, wurde aber beim Speichern verworfen. Die zurückgegebenen
  Snapshots enthielten nicht-endliche Werte; es wurde keine NPZ-Datei erzeugt.
- **Abgrenzung:** Es gab keinen Out-of-Memory-Fehler und keinen zellij-Abbruch.
  Der Fehler entstand in der numerischen Phase-2-Entwicklung. Da
  `runtime_debugging` bisher deaktiviert war, wurde der erste fehlerhafte
  Zeitschritt nicht direkt von astronomix gemeldet.
- **Diagnoseänderung:** `supernovae_turbolenz.py` besitzt nun den Schalter
  `--runtime-debugging`. Bei aktivierten astronomix-Checks soll die Integration
  beim ersten negativen oder nicht-endlichen Zustand abbrechen. Zusätzlich
  berichtet die Snapshot-Prüfung bei einem später erkannten Fehler den ersten
  fehlerhaften Snapshot, die betroffenen Variablen und den Wertebereich des
  letzten gültigen Snapshots.
- **Nächster Versuch:** Zuerst ein kurzer diagnostischer GPU-Lauf mit derselben
  Konfiguration und aktiviertem `--runtime-debugging`, bevor neue Vergleichs-
  mediumläufe gestartet werden. Die dafür verwendete GPU ist noch festzulegen.

## 2026-11-06 – Debug-Lauf auf GPU 5 durch JAX-Plattformkonfiguration abgebrochen

- **Start:** Der diagnostische Phase-2-Lauf wurde mit
  `CUDA_VISIBLE_DEVICES=5` und `JAX_PLATFORMS=cuda` gestartet. astronomix
  bestätigte den PALLAS-Backend auf der GPU; damit war keine zweite GPU
  sichtbar.
- **Fehler:** Der Lauf erreichte die eigentliche numerische Stabilitätsprüfung
  nicht. `--runtime-debugging` verwendet in astronomix einen
  `jax.debug.callback`, der ein lokales CPU-Gerät benötigt. Mit
  `JAX_PLATFORMS=cuda` war dieses Gerät deaktiviert, daher brach JAX mit
  `jax.debug.callback failed to find a local CPU device` ab.
- **Interpretation:** Dieser Abbruch sagt nichts über die physikalische
  Stabilität der Supernova aus. GPU 5 wurde korrekt ausgewählt; für den
  Diagnoselauf muss zusätzlich die CPU-Plattform verfügbar sein:
  `CUDA_VISIBLE_DEVICES=5` zusammen mit `JAX_PLATFORMS=cuda,cpu`. Die
  eigentliche Simulation bleibt dabei auf der einzigen sichtbaren GPU 5; die
  CPU wird nur für den Debug-Callback verwendet.
- **Nächster Schritt:** Diagnoselauf mit `JAX_PLATFORMS=cuda,cpu` und derselben
  GPU-Maske wiederholen. Für einen strikt GPU-only Produktionslauf bleibt
  `JAX_PLATFORMS=cuda` möglich, dann aber ohne `--runtime-debugging`.

## 2026-11-06 – Positivitätsgeschütztes Anfangsmedium benötigt CPU-Callback

- **Lauf:** Das neue Anfangsmedium wurde mit `--positivity-protection`,
  `CUDA_VISIBLE_DEVICES=5` und `JAX_PLATFORMS=cuda` gestartet.
- **Ergebnis:** Der Lauf brach beim Eintritt in die Zeitintegration mit
  `jax.debug.callback failed to find a local CPU device` ab. PALLAS wurde
  korrekt auf der einzigen sichtbaren GPU 5 aktiviert; es entstand kein
  numerischer Nichtendlichkeitsfehler und keine neue NPZ-Datei.
- **Interpretation:** Der allgemeine Positivitätsschutz verwendet in dieser
  astronomix-Konfiguration einen JAX-Callback, der zusätzlich ein lokales
  CPU-Gerät benötigt. Für diesen Lauf muss deshalb
  `JAX_PLATFORMS=cuda,cpu` gesetzt werden. `CUDA_VISIBLE_DEVICES=5` begrenzt
  die GPU-Auswahl weiterhin ausschließlich auf GPU 5; die CPU wird nur für den
  Callback verwendet.
- **Zusätzlicher Bedienfehler:** In der verwendeten Shell-Zeile war die
  `tee`-Umleitung nach `--output` nicht korrekt enthalten. Der korrigierte
  Startbefehl wird mit einer separaten Logdatei dokumentiert beziehungsweise
  erneut ausgegeben.

## 2026-11-06 – Referenzlauf für den Vergleich mit Phase-1-Positivitätsschutz

- **Vorhandener Referenzlauf:** Der frühere erfolgreiche Supernova-Lauf ist
  `supernovae_turbolenz_128cells_amp9_cooling15_implicit_cfl02_120snapshots.npz`.
  Sein zugehöriges Medium ist
  `turbulent_medium_128cells_t5_kf25_amp9_cooling_factor15_implicit_cfl02_gpu7.npz`.
  Dieses Medium wurde vor der allgemeinen Phase-1-Positivitätserweiterung
  erzeugt.
- **Versuchsdesign:** Um ausschließlich den Einfluss des numerischen
  Positivitätsschutzes im Anfangsmedium zu untersuchen, müssen Seed,
  Forcierung, Cooling-Modell, Cooling-Parameter, CFL, Auflösung und alle
  Supernova-Parameter identisch bleiben. Nur `--positivity-protection` in
  Phase 1 darf sich unterscheiden.
- **Korrektur der Planung:** Der aktuell gestartete neue Medium-Lauf verwendet
  Schure-CIE-Cooling und `CFL=0.1`; der vorhandene Referenzlauf verwendet das
  einfache Potenzgesetz mit Cooling-Faktor `15` und `CFL=0.2`. Dieser neue Lauf
  ist deshalb kein isolierter Positivitätsschutz-Vergleich. Für einen sauberen
  Vergleich muss das geschützte Medium mit den exakten Referenzparametern neu
  erzeugt werden.

## 2026-11-06 – Ziel des neuen Mediumlaufs: realistisches Cooling

- **Nutzerentscheidung:** Der aktuelle Lauf mit Schure-2009-CIE-Cooling,
  physikalischer Skalierung und aktiviertem Phase-1-Positivitätsschutz soll
  nicht verworfen werden. Ziel ist zunächst die realistischere Cooling-Methode
  zusammen mit der numerisch stabilisierten Mediumerzeugung zu untersuchen.
- **Einordnung:** Der frühere Cooling-Faktor-15-Lauf bleibt eine historische
  Referenz für die bisherige vereinfachte Cooling-Konfiguration. Unterschiede
  zwischen diesem Lauf und dem neuen Schure-Lauf dürfen nicht ausschließlich
  dem Positivitätsschutz zugeschrieben werden, da sich auch das Cooling-Modell
  und die CFL-Konfiguration unterscheiden.
- **Weiterer Vergleich:** Für eine spätere isolierte Untersuchung des
  Phase-1-Positivitätsschutzes kann das bereits vorhandene Medium
  `turbulent_medium_128cells_t5_warmism100pc_schure2009_gpu7.npz` ohne den
  allgemeinen Schutz als Schure-CIE-Kontrollmedium dienen. Die Supernova-Phase
  müsste dann mit identischen Phase-2-Parametern auf beiden Schure-Medien
  durchgeführt werden.

## 2026-11-06 – Allgemeiner Positivitätsschutz für Phase 1 optional aktiviert

- **Ziel:** Ein Kontrollmedium erzeugen, das mit demselben allgemeinen
  Positivitätsschutz wie die Supernova-Phase entwickelt wird. Der Zufallsseed
  und alle physikalischen beziehungsweise turbulenten Parameter bleiben dabei
  unverändert.
- **Implementierung:** `anfangsmedium.py` besitzt nun den optionalen Schalter
  `--positivity-protection`. Wenn er gesetzt ist, verwendet Phase 1
  `POSITIVITY_REDISTRIBUTE` während der Integrationsstufen und pro Zeitschritt
  sowie `vacuum_rest=True`. Ohne den Schalter bleibt das bisherige Verhalten
  reproduzierbar.
- **Metadaten:** Die NPZ-Datei speichert jetzt unter
  `positivity_protection`, ob und mit welchen Modi der allgemeine Schutz aktiv
  war. Die bereits vorhandene OU-Forcierungs-Vakuumsicherung bleibt unabhängig
  davon aktiviert.
- **Prüfung:** Python-Syntaxprüfung, Git-Diff-Prüfung und CLI-Hilfeprüfung
  erfolgreich. Noch kein neues Medium und kein neuer Supernova-Lauf gestartet.
- **Nächster Schritt:** Das Kontrollmedium mit identischem Seed und
  `--positivity-protection` auf einer ausdrücklich ausgewählten freien GPU
  erzeugen und anschließend mit identischer Phase-2-Konfiguration auswerten.

## 2026-11-06 – Parallele Schure-CIE-Vergleichsläufe auf GPU 0 und GPU 2

- **Nutzerfreigabe:** Der Nutzer gab GPU 0 und GPU 2 ausdrücklich als frei
  und zur Verwendung frei.
- **Lauf 1, GPU 0:** Die Supernova-Phase wurde im zuvor erfolgreich erzeugten
  positivitätsgeschützten Medium
  `turbulent_medium_128cells_t5_warmism100pc_schure2009_positivity_gpu5_run2.npz`
  gestartet. Verwendet werden `E_SN=10^51 erg`,
  `explosion_radius=0.03`, `smoothing_cells=2.0`, `duration=0.1`,
  `num_snapshots=120`, `CFL=0.1` und implizites Schure-2009-CIE-Cooling in
  Phase 2. Ausgabe und Log sind
  `supernovae_turbolenz_128cells_schure2009_protected_medium_gpu0_120snapshots.npz`
  beziehungsweise die gleichnamige `.log`-Datei.
- **Lauf 2, GPU 2:** Parallel wurde mit demselben Seed `42`, derselben
  Auflösung `128^3`, derselben Forcierung (`amplitude=9`,
  `wavenumber=25`), `CFL=0.1` und Schure-2009-CIE-Cooling ein neues
  turbulentes Medium erzeugt. Der Lauf verwendet ausdrücklich
  `--disable-vacuum-protection` und **nicht** `--positivity-protection`; damit
  sind beide numerischen Schutzmodule für diesen Kontrolllauf deaktiviert.
  Ausgabe und Log sind
  `turbulent_medium_128cells_t5_warmism100pc_schure2009_unprotected_gpu2.npz`
  beziehungsweise die gleichnamige `.log`-Datei.
- **Technischer Hinweis:** Für die Supernova-Phase wird
  `JAX_PLATFORMS=cuda,cpu` gesetzt, weil der verwendete astronomix-Positivitäts-
  beziehungsweise Callback-Pfad ein lokales CPU-Gerät benötigt. Durch
  `CUDA_VISIBLE_DEVICES=0` bleibt für die Rechnung ausschließlich GPU 0
  sichtbar. Der ungeschützte Mediumlauf verwendet `JAX_PLATFORMS=cuda` und
  `CUDA_VISIBLE_DEVICES=2`.
- **Status beim Start:** Beide Prozesse wurden in separaten Zellij-Sessions
  gestartet und beim ersten Check auf GPU 0 beziehungsweise GPU 2 als aktive
  Python-Prozesse bestätigt. Der endgültige Erfolg oder ein numerischer
  Abbruch ist zum Zeitpunkt dieses Eintrags noch offen.

## 2026-11-06 – Neustart des ungeschützten Medium-Kontrolllaufs

- **Ursache des Neustarts:** Der erste Lauf auf GPU 2 wurde mit
  `JAX_PLATFORMS=cuda` gestartet und scheiterte bereits in der astronomix-
  Zeitintegration an einem JAX-Callback, der ein lokales CPU-Gerät benötigt.
  Es lag dabei noch kein Hinweis auf einen physikalischen Nichtendlichkeits-
  fehler vor.
- **Neuer Lauf:** Der Kontrolllauf wurde mit unveränderten physikalischen und
  numerischen Parametern erneut gestartet, nun mit
  `JAX_PLATFORMS=cuda,cpu`. `CUDA_VISIBLE_DEVICES=2` stellt weiterhin sicher,
  dass nur GPU 2 verwendet wird. `--disable-vacuum-protection` bleibt gesetzt;
  `--positivity-protection` wird nicht verwendet.
- **Session und Ausgabe:** Der Lauf läuft in
  `medium-unprotected-gpu2-20261106-rerun`. Ausgabe und Log sind
  `turbulent_medium_128cells_t5_warmism100pc_schure2009_unprotected_gpu2_rerun.npz`
  beziehungsweise die gleichnamige `.log`-Datei.
- **Kontrolle:** Der Prozess wurde auf GPU 2 mit ungefähr 98 Prozent
  Auslastung und bei etwa 0,5 Prozent Fortschritt bestätigt. Die Supernova-
  Phase läuft parallel weiterhin auf GPU 0.

## 2026-11-06 – Prüfung von AthenaPK, Finite Volume und PP-Flux-Blending

- **Paketprüfung:** In der verwendeten Installation `astronomix==0.6.6`
  wurde kein Solver oder Backend mit der Bezeichnung AthenaPK gefunden. Das
  Paket stellt die Solvermodi `FINITE_VOLUME` und `FINITE_DIFFERENCE` bereit.
- **Aktueller Projektstand:** `anfangsmedium.py` und
  `supernovae_turbolenz.py` verwenden beide `FINITE_VOLUME`. Die bisherige
  Stabilisierung erfolgt dort über `PositivityConfig` mit
  `per_stage_mode=REDISTRIBUTE`, `per_step_mode=REDISTRIBUTE` und
  `vacuum_rest=True`.
- **PP-Optionen:** `positivity_config.preserving_flux=True` ist im installierten
  Paket im Finite-Differenzen-WENO-Flux-Blending implementiert. Für den
  Finite-Volume-Pfad existiert dagegen der positivitätserhaltende
  Rekonstruktions-Limiter `VAN_ALBADA_PP`. Dieser wurde bisher noch nicht in
  einem Projektlauf verwendet.
- **Entscheidung für den nächsten Vergleich:** Der nächste naheliegende,
  direkt kompatible Test ist `FINITE_VOLUME` mit `VAN_ALBADA_PP` bei ansonsten
  identischen Parametern. Ein separater `FINITE_DIFFERENCE`-Lauf mit
  `preserving_flux=True` wäre ein anderer Solververgleich und muss wegen der
  unterschiedlichen WENO-/Zeitintegrationspfade separat geprüft werden.
- **Status:** Es wurde in diesem Prüfschritt kein GPU-Lauf gestartet. Die für
  den Test zu verwendende GPU ist noch festzulegen. Der zuletzt geplante
  ungeschützte Mediumlauf besitzt zusätzlich einen unabhängigen Fehler beim
  Speichern der Metadaten (`vacuum_protection_enabled` nicht definiert), der
  vor einem neuen Kontrolllauf behoben werden muss.

## 2026-11-06 – PP-WENO als auswählbare Simulationsvariante ergänzt

- **Implementierung:** `anfangsmedium.py` akzeptiert nun `--pp-weno`. Der
  Schalter verwendet `FINITE_DIFFERENCE`, `RK4_SSP` und
  `PositivityConfig(preserving_flux=True)`. Ohne den Schalter bleiben
  `FINITE_VOLUME` und `RK2_SSP` unverändert aktiv.
- **Phase 2:** `supernovae_turbolenz.py` übernimmt PP-WENO automatisch aus
  den Medium-Metadaten oder kann es mit `--pp-weno` explizit aktivieren. Die
  Phase 2 verwendet dann ebenfalls `FINITE_DIFFERENCE`, `RK4_SSP` und
  `preserving_flux=True`, während die bisherige Redistribution zusätzlich
  erhalten bleibt.
- **Metadaten:** Solvermodus und PP-WENO-Aktivierung werden in Medium- und
  Supernova-Ausgaben gespeichert beziehungsweise ausgegeben.
- **Fehlerkorrektur:** Die bisher fehlende lokale Variable
  `vacuum_protection_enabled` beim Speichern des Mediums wurde ergänzt. Damit
  kann ein erfolgreich integrierter ungeschützter Lauf künftig als NPZ-Datei
  gespeichert werden.
- **Prüfungen:** `git diff --check`, Python-Syntaxprüfung, CLI-Hilfeprüfung
  und eine CPU-only-Konfigurationsprüfung waren erfolgreich. Dabei wurde
  `solver_mode=FINITE_DIFFERENCE`, `time_integrator=RK4_SSP` und
  `preserving_flux=True` korrekt durch `finalize_config` bestätigt. Noch kein
  GPU-Lauf gestartet; GPU-Freigabe steht aus.

## 2026-11-06 – PP-WENO ersetzt Redistribution in Phase 2

- **Nutzerentscheidung:** PP-WENO soll als eigenständige numerische
  Schutzmethode getestet werden. Die Supernova-Phase darf bei aktiviertem
  PP-WENO daher nicht zusätzlich `REDISTRIBUTE` verwenden.
- **Änderung:** `supernovae_turbolenz.py` verwendet bei PP-WENO nun ausschließlich
  `PositivityConfig(preserving_flux=True)`. Damit sind
  `per_stage_mode=NONE`, `per_step_mode=NONE` und `vacuum_rest=False` aktiv.
  Ohne PP-WENO bleibt die bisherige Phase-2-Konfiguration mit Redistribution
  und `vacuum_rest=True` unverändert.
- **Metadaten:** Die Supernova-Ausgabe kennzeichnet bei PP-WENO die beiden
  Redistribution-Modi als `none` und `vacuum_rest` als `false`.
- **Status:** Noch kein GPU-Lauf gestartet. Der nächste Lauf ist ein PP-WENO-
  Mediumlauf mit Seed 42 und den bisherigen realistischen Schure-CIE-
  Parametern auf einer ausdrücklich freigegebenen GPU.

## 2026-11-06 – Ungültige Ergebnisse optional zur Fehleranalyse speichern

- **Nutzerentscheidung:** Nicht-endliche oder zeitlich ungültige Ergebnisse
  sollen für die Ursachenanalyse gespeichert werden können, dürfen aber nicht
  als gültige Simulationsdaten behandelt werden.
- **Implementierung:** `anfangsmedium.py` und
  `supernovae_turbolenz.py` besitzen nun `--save-invalid`. Mit diesem Schalter
  werden auch Rohzustände mit NaN/Inf gespeichert; die NPZ-Metadaten enthalten
  dann `valid=false`, `invalid_output_saved=true` und
  `validity_diagnostics`. Ohne den Schalter bleibt das bisherige Verhalten
  bestehen: ungültige Zustände werden verworfen und als Fehler gemeldet.
- **Supernova-Diagnostik:** Bei nicht-endlichen Supernova-Snapshots wird die
  vorhandene Detaildiagnose zum ersten fehlerhaften Snapshot zusätzlich in den
  Metadaten abgelegt.
- **Laufstatus:** Der bereits gestartete PP-WENO-Mediumlauf wurde vor dieser
  Änderung gestartet und verwendet den neuen Schalter noch nicht. Er wurde
  nicht gestoppt; die neue Speicheroption gilt ab dem nächsten Start.

## 2026-11-06 – PP-WENO-Medium und anschließende Supernova verkettet

- **Medium:** Der bereits laufende PP-WENO-Mediumlauf auf der ausdrücklich
  freigegebenen GPU 5 wird nicht dupliziert. Er verwendet Seed `42`,
  `128^3` Zellen, `CFL=0.1`, Forcierungsamplitude `9`, Wellenzahl `25`,
  Schure-2009-CIE-Cooling und aktivierte OU-Vakuumsicherung, aber keinen
  allgemeinen `REDISTRIBUTE`-Schutz. Ausgabe:
  `turbulent_medium_128cells_t5_warmism100pc_schure2009_ppweno_gpu5.npz`.
- **Verkettete Phase 2:** Eine neue Zellij-Session
  `ppweno-supernova-after-medium-gpu5-20261106` wartet auf den Abschluss des
  Mediumprozesses und startet danach automatisch die Supernova auf GPU 5.
  Die Phase 2 verwendet PP-WENO ohne Redistribution, `duration=0.1`,
  `120` Snapshots, `E_SN=10^51 erg`, Explosionsradius `0.03`,
  `smoothing_cells=2.0` und implizites Schure-CIE-Cooling.
- **Fehleranalyse:** Die Supernova-Phase wird mit `--save-invalid` gestartet,
  damit auch nicht-endliche Snapshots für die Diagnose gespeichert werden und
  über die NPZ-Metadaten als `valid=false` erkennbar bleiben.
- **Kontrolle beim Start der Verkettung:** Der Mediumprozess war aktiv auf GPU 5
  bestätigt; die neue Wartesession war aktiv. Die Supernova hatte zu diesem
  Zeitpunkt noch nicht begonnen.

## 2026-11-06 – PP-WENO-Supernova in vorhandenem geschütztem Medium

- **Ziel:** Die neue PP-WENO-Phase 2 unabhängig davon testen, ob auch das
  turbulente Medium mit PP-WENO erzeugt wurde.
- **Eingabe:** Verwendet wird das bereits validierte Schure-CIE-Medium
  `turbulent_medium_128cells_t5_warmism100pc_schure2009_positivity_gpu5_run2.npz`.
  Dieses Medium wurde ursprünglich mit Finite Volume und allgemeinem
  Positivitätsschutz erzeugt.
- **Phase 2:** Der Supernovalauf verwendet auf GPU 6 explizit
  `FINITE_DIFFERENCE/WENO`, `RK4_SSP` und `preserving_flux=True`. Die
  Redistribution und `vacuum_rest` sind in dieser PP-WENO-Variante deaktiviert.
  Verwendet werden `duration=0.1`, `120` Snapshots, `E_SN=10^51 erg`,
  `explosion_radius=0.03`, `smoothing_cells=2.0`, `CFL=0.1` und implizites
  Schure-2009-CIE-Cooling.
- **Fehleranalyse:** `--save-invalid` ist aktiviert. Ausgabe und Log sind
  `supernovae_turbolenz_128cells_schure2009_existing_protected_medium_ppweno_gpu6_120snapshots.npz`
  beziehungsweise die gleichnamige `.log`-Datei. Ungültige Ergebnisse werden
  gespeichert, aber in den Metadaten als `valid=false` markiert.
- **Kontrolle beim Start:** GPU 6 und Prozess PID 888876 wurden bestätigt; der
  Log weist den erwarteten Finite-Difference/WENO-Solver und PP-WENO aus.

## 2026-11-07 – Ergebnis des PP-WENO-Medium- und Supernovatests

- **Medium:** Der PP-WENO-Mediumlauf auf GPU 5 wurde erfolgreich beendet und
  als `turbulent_medium_128cells_t5_warmism100pc_schure2009_ppweno_gpu5.npz`
  gespeichert. Die Metadaten bestätigen `solver_mode=finite_difference`,
  `pp_weno=true`, deaktivierte Redistribution und aktivierte
  OU-Vakuumsicherung. Der Lauf wurde vor der Einführung von `--save-invalid`
  gestartet; deshalb enthält seine ältere Ausgabe noch kein explizites
  `valid`-Feld. Die damalige Speicherlogik hätte nicht-endliche Zustände nicht
  gespeichert, daher ist die gespeicherte Mediumdatei als gültig einzustufen.
- **Supernova:** Die Phase 2 auf GPU 6 wurde mit demselben PP-WENO-Solver,
  ohne Redistribution, abgeschlossen und dank `--save-invalid` gespeichert:
  `supernovae_turbolenz_128cells_schure2009_existing_protected_medium_ppweno_gpu6_120snapshots.npz`.
  Die Metadaten melden `valid=false` und `invalid_output_saved=true`.
- **Numerischer Fehler:** Der erste fehlerhafte Snapshot ist `index=119` mit
  lokaler Zeit `NaN`; betroffen sind Dichte, alle drei Geschwindigkeiten und
  Druck. Die Zeitpunkte sind nicht streng monoton. Die Datei ist damit nur für
  Fehleranalyse geeignet und darf nicht als gültiger Trainings- oder
  Vergleichsdatensatz verwendet werden.
- **GPU-Status:** Beide Python-Prozesse sind beendet; GPU 5 und GPU 6 sind
  wieder frei.

## 2026-11-07 – Supernova im numerisch gültigen PP-WENO-Medium

- **Eingabe:** Für den neuen Lauf wird das numerisch gültige Medium
  `turbulent_medium_128cells_t5_warmism100pc_schure2009_ppweno_gpu5.npz`
  verwendet. Damit sind sowohl die Mediumerzeugung als auch die Supernova-
  Phase auf PP-WENO abgestimmt.
- **Phase 2:** Die Supernova läuft mit `FINITE_DIFFERENCE/WENO`,
  `preserving_flux=True`, ohne Redistribution und ohne `vacuum_rest`. Die
  Parameter sind `duration=0.1`, `120` Snapshots, `E_SN=10^51 erg`,
  `explosion_radius=0.03`, `smoothing_cells=2.0`, `CFL=0.1` und implizites
  Schure-2009-CIE-Cooling.
- **Hardware und Ausgabe:** Der Lauf wurde nach Prüfung einer freien GPU auf
  GPU 1 in der Zellij-Session `supernova-valid-ppweno-medium-gpu1-20261107`
  gestartet. Ausgabe und Log sind
  `supernovae_turbolenz_128cells_schure2009_valid_ppweno_medium_gpu1_120snapshots.npz`
  beziehungsweise die gleichnamige `.log`-Datei. `--save-invalid` bleibt
  aktiviert, um einen möglichen numerischen Fehler diagnostisch zu sichern.
- **Kontrolle beim Start:** Prozess PID 1836259 und GPU 1 wurden bestätigt;
  der Fortschritt lag beim ersten Check bei etwa 0,1 Prozent.

## 2026-11-07 – PP-WENO-Supernova im gültigen PP-WENO-Medium beendet

- **Ergebnis:** Die Supernova-Simulation im numerisch gültigen PP-WENO-Medium
  wurde auf GPU 1 vollständig durchlaufen und als NPZ-Datei gespeichert:
  `supernovae_turbolenz_128cells_schure2009_valid_ppweno_medium_gpu1_120snapshots.npz`.
- **Numerische Gültigkeit:** Die Ausgabe ist trotz erfolgreichem Durchlauf
  ungültig: `valid=false`, `invalid_output_saved=true`. Die Zeitpunkte reichen
  nicht endlich bis `t=5.1`; der erste fehlerhafte Snapshot ist `index=119`
  mit lokaler Zeit `NaN`. Dichte, alle drei Geschwindigkeiten und Druck sind
  im fehlerhaften Snapshot nicht endlich.
- **Interpretation:** PP-WENO ohne Redistribution stabilisiert diesen
  Supernova-Lauf in dem zuvor gültigen Medium nicht bis zum letzten Snapshot.
  Die Datei bleibt für die Fehleranalyse erhalten, darf aber nicht als gültiger
  Simulations- oder Trainingsdatensatz verwendet werden.
- **GPU-Status:** Prozess und GPU-Lauf sind beendet; GPU 1 ist wieder frei.

## 2026-11-08 – PP-WENO mit zusätzlichem Redistribution- und Vacuum-Rest-Schutz

- **Nutzerentscheidung:** Nach dem ungültigen PP-WENO-Lauf soll die Wirkung
  einer kombinierten Schutzkonfiguration getestet werden.
- **Neue Option:** `supernovae_turbolenz.py` akzeptiert nun
  `--pp-weno-with-redistribute`. Diese Variante aktiviert gleichzeitig
  `FINITE_DIFFERENCE/WENO`, `preserving_flux=True`,
  `per_stage_mode=REDISTRIBUTE`, `per_step_mode=REDISTRIBUTE` und
  `vacuum_rest=True`.
- **Abgrenzung:** Der bisherige reine PP-WENO-Modus bleibt unverändert und
  verwendet keine Redistribution. Der normale Finite-Volume-Modus behält
  ebenfalls seine bisherige Redistribution-Konfiguration.
- **Prüfung:** Syntaxprüfung, CLI-Hilfe und Git-Diff-Prüfung erfolgreich. Der
  Vergleichslauf auf GPU 1 steht nach dem Commit noch aus.

## 2026-11-08 – Vergleichslauf PP-WENO plus Redistribution gestartet

- **Start:** Der Vergleichslauf wurde nach Prüfung der freien GPU 1 in der
  Zellij-Session `supernova-ppweno-redistribute-gpu1-20261108` gestartet.
- **Konfiguration:** Eingabe ist das gültige PP-WENO-Medium
  `turbulent_medium_128cells_t5_warmism100pc_schure2009_ppweno_gpu5.npz`.
  Aktiv sind `FINITE_DIFFERENCE/WENO`, `preserving_flux=True`,
  `REDISTRIBUTE`, `vacuum_rest=True`, `CFL=0.1`, 120 Snapshots,
  `E_SN=10^51 erg`, Explosionsradius `0.03`, implizites Schure-CIE-Cooling
  und `--save-invalid`.
- **Ausgabe:**
  `supernovae_turbolenz_128cells_schure2009_ppweno_redistribute_medium_gpu1_120snapshots.npz`
  mit gleichnamigem Log.
- **Kontrolle:** Der Prozess wurde auf GPU 1 bestätigt. Das Log weist beide
  Schutzkomponenten ausdrücklich als aktiv aus; die Supernova-Integration hat
  begonnen.

## 2026-11-08 – Paralleler CFL-Vergleichslauf mit CFL=0.05

- **Versuchsdesign:** Der bestehende Lauf auf GPU 1 verwendet `CFL=0.1`.
  Parallel wurde auf GPU 2 ein ansonsten identischer Lauf mit `CFL=0.05`
  gestartet. Damit wird ausschließlich der Einfluss der kleineren internen
  CFL-Zeitschritte untersucht; `duration=0.1` bleibt bei beiden Läufen gleich.
- **Gemeinsame Konfiguration:** Beide Läufe verwenden das gültige PP-WENO-
  Medium, `FINITE_DIFFERENCE/WENO`, PP-Flux-Blending,
  `REDISTRIBUTE`, `vacuum_rest=True`, implizites Schure-CIE-Cooling,
  `E_SN=10^51 erg`, Explosionsradius `0.03`, `120` Snapshots und
  `--save-invalid`.
- **GPU 2-Ausgabe:**
  `supernovae_turbolenz_128cells_schure2009_ppweno_redistribute_cfl005_medium_gpu2_120snapshots.npz`
  mit gleichnamigem Log.
- **Kontrolle beim Start:** GPU 1 rechnete den CFL-0.1-Lauf weiter; GPU 2
  wurde für den CFL-0.05-Lauf bestätigt. Beide Logs zeigten beim Start keine
  Fehlermeldung.

## 2026-11-09 – Beide CFL-Vergleichsläufe numerisch gültig

- **CFL=0.1:** Der Lauf auf GPU 1 wurde vollständig beendet und in
  `supernovae_turbolenz_128cells_schure2009_ppweno_redistribute_medium_gpu1_120snapshots.npz`
  gespeichert. Die Metadaten melden `valid=true`; alle Zustände und Zeitpunkte
  sind endlich und die Zeitpunkte streng monoton.
- **CFL=0.05:** Der parallele Lauf auf GPU 2 wurde ebenfalls vollständig
  beendet und in
  `supernovae_turbolenz_128cells_schure2009_ppweno_redistribute_cfl005_medium_gpu2_120snapshots.npz`
  gespeichert. Auch hier melden die Metadaten `valid=true`, endliche Zustände
  und streng monotone Zeitpunkte.
- **Interpretation:** Die Halbierung der CFL-Zahl war für die numerische
  Gültigkeit in dieser Konfiguration nicht erforderlich; beide Läufe waren
  erfolgreich. Unterschiede zwischen den Ergebnissen müssen nun über Felder,
  Erhaltungsgrößen und Laufzeiten untersucht werden, nicht nur über den
  Erfolgsstatus.
- **GPU-Status:** Die beiden Simulationsprozesse sind beendet. GPU 2 ist frei;
  auf GPU 1 laufen beim Check noch andere Prozesse, die nicht zu diesen beiden
  Läufen gehören.

## 2026-11-09 – Tutor-Rückmeldung: CFL=0.5, PP-Flux-Blending und Literaturparameter

- **Neue Prüfaufträge:** Für die aktuelle numerische Konfiguration soll ein
  weiterer Supernovalauf mit `CFL=0.5` auf Gültigkeit geprüft werden. Außerdem
  soll der Einfluss von PP-Flux-Blending ohne zusätzliche Redistribution und
  ohne `vacuum_rest` nachvollziehbar ausgewertet werden. Ein solcher reiner
  PP-WENO-Phase-2-Lauf wurde bereits durchgeführt und endete am letzten
  Snapshot mit nicht-endlichen Werten; die Ausgabe ist als `valid=false`
  diagnostisch gespeichert.
- **Literaturrecherche:** Als erste belastbare Referenzen für die Auswahl von
  Trainingsbereichen wurden Kim & Ostriker (2015), Zhang & Chevalier (2018)
  und Gaensler et al. (2011) identifiziert. Kim & Ostriker untersuchen
  mittlere Wasserstoffdichten von `0.1...100 cm^-3` und verwenden für eine
  einzelne Supernova typischerweise `10^51 erg`. Zhang & Chevalier verwenden
  turbulente Machzahlen `M=3, 10, 30` sowie mittlere Dichten von `1, 10, 100
  cm^-3`; für W44 wird `M >= 10` und für W28 `M <= 3` diskutiert. Für warmes
  ionisiertes Medium berichten Gaensler et al. eine niedrige Schallmachzahl
  `M_s <= 2`.
- **Vorläufige Interpretation:** Für Trainingsdaten sollte zunächst ein
  explizit gekennzeichneter Bereich aus mehreren ISM-Regimen statt eines
  einzelnen "realistischen" Punktes abgedeckt werden. Die endgültige Auswahl
  von Dichte-, Machzahl-, Energie-, Temperatur- und Kühlparametern ist noch
  mit der Betreuung abzustimmen.
- **Vorgemerkte Literatur mit Quellen:**
  - Kim, C.-G. & Ostriker, E. C. (2015), *Momentum Injection by Supernovae
    in the Interstellar Medium*, ApJ 802, 99,
    [arXiv:1410.1537](https://arxiv.org/abs/1410.1537). Verwendbar für die
    Referenzenergie einer einzelnen Supernova von ungefähr `10^51 erg`,
    mittlere Wasserstoffdichten `n_H=0.1...100 cm^-3` und die Abhängigkeit
    der Schalenbildung beziehungsweise des Impulseintrags von der Dichte.
  - Zhang, D. & Chevalier, R. A. (2018), *Numerical simulations of supernova
    remnants in turbulent molecular clouds*, MNRAS 477, 4521,
    [arXiv:1807.06603](https://arxiv.org/abs/1807.06603). Verwendbar für
    turbulente Machzahlen `M=3, 10, 30`, mittlere Dichten `n_H=1, 10,
    100 cm^-3` sowie den Vergleich von weniger und stärker turbulenten
    Umgebungen; die Diskussion ordnet W28 ungefähr `M<=3` und W44 ungefähr
    `M>=10` zu.
  - Gaensler, B. M. et al. (2011), *Low–Mach-number turbulence in
    interstellar gas revealed by radio polarization gradients*, Nature 478,
    214–217, [arXiv:1110.2896](https://arxiv.org/abs/1110.2896). Verwendbar
    als Beobachtungsreferenz für warmes ionisiertes Gas mit einer niedrigen
    Schallmachzahl `M_s<=2`.
  - Schure, K. M. et al. (2009), *A new radiative cooling curve based on an
    up to date plasma emission code*, A&A 508, 751,
    [arXiv:0909.5204](https://arxiv.org/abs/0909.5204). Verwendbar als
    Quelle der tabellierten optisch-dünnen CIE-Kühlkurve; Titel, Autoren und
    arXiv-/A&A-Verweis wurden anhand der Originalquelle geprüft.
- **GPU-Status:** Der CFL-0.5-Lauf wurde noch nicht gestartet, da für jeden
  neuen GPU-Lauf zunächst eine freie und ausdrücklich freigegebene GPU
  benötigt wird.

## 2026-11-09 – CFL=0.5-Supernova gestartet

- **Nutzerfreigabe:** GPU 2 wurde ausdrücklich als frei für den Lauf
  freigegeben.
- **Ziel:** Prüfung, ob die bisher erfolgreiche PP-WENO-Konfiguration auch
  mit der größeren CFL-Zahl `0.5` numerisch gültig bleibt.
- **Eingabe:**
  `turbulent_medium_128cells_t5_warmism100pc_schure2009_ppweno_gpu5.npz`.
- **Phase 2:** `FINITE_DIFFERENCE/WENO`, PP-Flux-Blending,
  `REDISTRIBUTE`, `vacuum_rest=True`, implizites Schure-2009-CIE-Cooling,
  `duration=0.1`, `120` Snapshots, `E_SN=10^51 erg`,
  `explosion_radius=0.03`, `smoothing_cells=2.0` und `CFL=0.5`.
- **Hardware und Session:** Der Lauf wurde ausschließlich mit
  `CUDA_VISIBLE_DEVICES=2` in der Zellij-Session
  `supernova-ppweno-redistribute-cfl050-gpu2-20261109` gestartet. Beim
  Start war der Python-Prozess auf GPU 2 bestätigt; die Integration hatte
  begonnen, ein Endergebnis liegt noch nicht vor.
- **Ausgabe:**
  `supernovae_turbolenz_128cells_schure2009_ppweno_redistribute_cfl050_medium_gpu2_120snapshots.npz`
  sowie die gleichnamige `.log`-Datei.

## 2026-11-09 – Parameterbereichs-PDF um Literaturbereiche erweitert

- **Anlass:** Die PDF soll neben den aktuell verwendeten Einzelwerten die
  recherchierten relevanten Bereiche ausdrücklich und getrennt aufführen.
- **Erweiterung:** Eine zusätzliche Übersicht dokumentiert nun die
  Literaturanker und vorgeschlagenen Stützstellen für mittlere Dichte,
  turbulente Machzahl, Einzel- und Mehrfach-Supernovaenergien,
  Temperatur/ISM-Phase, Cooling-Temperaturbereich und Explosionsradius.
  Designbereiche wie `0.5...2 x 10^51 erg` sind ausdrücklich als
  Explorationsvorschlag und nicht als vollständige Beobachtungsverteilung
  gekennzeichnet.
- **Ausgaben:** Aktualisiert wurden
  `parameterbereiche_supernova_trainingsdaten.pdf`,
  `parameterbereiche_supernova_trainingsdaten.tex` und
  `make_parameter_pdf.py`. Die PDF ist eine gültige vierseitige Datei; das
  Erzeugungsskript wurde per Python-Kompilierung geprüft.

## 2026-11-09 – Sternwind-Relevanz und serielles Trainingsdatensatz-Skript

- **Sternwind-Recherche:** Fichtner et al. (2024), *Connecting stellar and
  galactic scales: Energetic feedback from stellar wind bubbles to supernova
  remnants*, [arXiv:2402.11008](https://arxiv.org/abs/2402.11008), berichten
  vor der Supernova von Wind-/Ionisationsblasen mit Größen von mehreren bis
  einigen zehn Parsec. Die Vor-Supernova-Rückkopplung senkt die lokale Dichte
  und kann dadurch radiative Verluste des späteren Supernova-Überrests
  verzögern; die Supernova dominiert trotzdem meist das Gesamtenergiebudget.
  Sternwind ist für unser Problem daher nicht generell irrelevant, sondern
  sollte als eigener Kontrollvergleich mit und ohne Vor-Supernova-Wind
  behandelt werden. Die aktuelle zweiphasige Simulation enthält noch keinen
  Sternwind.
- **Modellierungsentscheidung:** Der bisherige Lauf ohne Sternwind bleibt der
  Basiskontrollfall. Ein Windvergleich benötigt noch festgelegte
  Progenitorparameter (Winddauer, Massenverlust, Windgeschwindigkeit und
  Injektionsradius) sowie eine zeitliche Zuordnung zur physikalischen
  Referenzskalierung. Diese Parameter wurden noch nicht als wissenschaftliche
  Standardwerte festgelegt.
- **Neue Datei:**
  `Praktikum_Datenzubereitung/run_training_dataset.py` führt mehrere
  Experimente seriell aus. Pro Experiment wird zuerst `anfangsmedium.py` und
  anschließend `supernovae_turbolenz.py` aufgerufen. Das Skript streamt die
  Ausgaben in ein Log, schreibt ein JSON-Manifest, prüft die Medium-Metadaten
  und startet die Supernova standardmäßig nicht, wenn das Medium als ungültig
  markiert ist.
- **Konfigurationsvorlage:**
  `Praktikum_Datenzubereitung/training_dataset_config.example.json` enthält
  einen reproduzierbaren Beispielaufbau mit mehreren Experimenten. Der
  Probelauf mit `--dry-run` war erfolgreich und startete keine Simulation.
  GPU-Auswahl erfolgt weiterhin ausschließlich über `CUDA_VISIBLE_DEVICES`;
  das Skript startet keine parallelen Läufe.

## 2026-11-09 – Fehleranalyse des CFL=0.5-PP-WENO-Supernovalaufs

- **Untersuchter Lauf:**
  `supernovae_turbolenz_128cells_schure2009_ppweno_redistribute_cfl050_medium_gpu2_120snapshots.npz`.
  Verwendet wurden PP-WENO/PP-Flux-Blending, `REDISTRIBUTE`,
  `vacuum_rest=True`, implizites Schure-2009-CIE-Cooling, `CFL=0.5`,
  `E_SN=10^51 erg`, `explosion_radius=0.03` und 120 Snapshots.
- **Tatsächlicher Abbruch:** Der Fortschrittsbalken erreichte nur etwa 5.9 %.
  Der letzte tatsächlich beschriebene Snapshot liegt bei lokaler Zeit
  `t≈0.005844`; danach sind die Snapshot-Puffer überwiegend mit Nullen
  vorbelegt und der letzte Eintrag ist `NaN`. Die Ausgabe `100 %` ist hier
  kein erfolgreicher Abschluss: Der astronomix-Fortschrittsbalken setzt bei
  nicht-endlicher Zeit den Anzeigewert künstlich auf 100 %.
- **Numerisches Ergebnis:** `valid=false`, `finite_states=false` und
  `finite_time_points=false`. Die Datei wurde nur wegen `--save-invalid`
  gespeichert und darf nicht als Trainingsdatensatz verwendet werden.
- **Vergleich:** Der gleiche PP-WENO-Solver mit Redistribution war bei
  `CFL=0.1` und `CFL=0.05` gültig. Der reine PP-WENO-Lauf ohne Redistribution
  scheiterte ebenfalls, aber später bei ungefähr `t≈0.0254`. Damit ist
  `CFL=0.5` ein klarer Verstärker der Instabilität, aber nicht die alleinige
  Ursache des grundsätzlichen PP-WENO-Problems.
- **Eingangsmedium:** Das verwendete PP-WENO-Medium ist endlich, enthält aber
  sehr druckarme Zellen: `rho_min≈7.67e-2`, `p_min≈2.40e-5` und eine lokale
  Machzahl bis ungefähr `79`. Zwei Zellen liegen bereits bei `M>10`. Der
  Standard-Positivitätsschutz arbeitet nur mit `minimum_density=1e-14` und
  `minimum_pressure=1e-14`; er verhindert damit negative Werte, begrenzt aber
  keine physikalisch extrem kleinen positiven Drücke oder hohen Machzahlen.
- **Wahrscheinliche Ursache:** Die starke, nur wenige Gitterzellen große
  Supernova-Druckquelle trifft auf diese druckarmen Hoch-Mach-Zellen. Bei
  `CFL=0.5` ist der explizite SSPRK4/WENO-Schritt zu groß, sodass in mindestens
  einer 3D-Zelle die interne Energie beziehungsweise der Druck nicht mehr
  endlich wird. `REDISTRIBUTE` und `vacuum_rest` greifen erst bei den sehr
  niedrigen Dichte-/Positivitätsschwellen und können diesen Hoch-Mach-/Druck-
  overshoot nicht zuverlässig verhindern.
- **Cooling-Bewertung:** Das implizite Schure-Cooling ist als alleinige Ursache
  nicht belegt, weil die identische Phase-2-Cooling-Konfiguration in den
  `CFL=0.1`- und `CFL=0.05`-Redistribution-Läufen gültig war. Es beeinflusst
  jedoch die thermische Entwicklung und kann die druckarmen Bereiche
  verschärfen. Ein exakter erster NaN-Operator wurde in diesem Lauf nicht
  protokolliert, da `--runtime-debugging` nicht aktiviert war.
- **Nächster diagnostischer Schritt:** Ein einzelner kurzer Wiederholungslauf
  mit `--runtime-debugging` und derselben Konfiguration kann den ersten
  fehlerhaften Operator beziehungsweise die erste NaN-/Druckverletzung genauer
  lokalisieren. Dafür ist eine ausdrückliche GPU-Freigabe erforderlich.

## 2026-11-09 – Kritische Nachprüfung: Schutzlücken statt gesicherter Hoch-Mach-Ursache

- **Korrektur der vorigen Interpretation:** Die Hoch-Mach-Zellen im Medium
  sind nachgewiesen, ihre kausale Verbindung mit dem CFL-0.5-Abbruch jedoch
  nicht. Weder der erste fehlerhafte 3D-Ort noch der erste fehlerhafte
  Rechenoperator wurden gespeichert. Die Behauptung, die Supernova habe
  gerade diese Zellen erreicht und dadurch den Abbruch ausgelöst, ist nicht
  belegt. Hohe Machzahl allein ist kein Instabilitätsbeweis. CFL=0.5 ist in
  diesem Versuch gescheitert; eine allgemeine Stabilitätsgrenze folgt daraus
  nicht. Die letzte echte Ausgabe liegt bei lokal t=0.005844, der Logfortschritt
  vor NaN bei etwa 5.9 %; der exakte Ausfallzeitpunkt ist nicht gespeichert.
- **CPU-Funktionstests:** Neu angelegt und erfolgreich ausgeführt:
  `Praktikum_Datenzubereitung/diagnose_ppweno_schutz.py`. Keine Simulation oder
  GPU-Nutzung; JAX meldete ausschließlich CpuDevice(0). Tests beziehen sich
  auf die aktuell installierten astronomix-Quellen im pde-Environment; ein
  vollständiger historischer Quellcode-Hash des ursprünglichen Laufs fehlt.
- **Mehrdimensionale Limiter-Lücke:** `_hydro_step_rhs` begrenzt die Flüsse
  jeder Achse ausgehend vom gleichen Zustand getrennt und addiert danach die
  drei Inkremente. Ein synthetischer Test mit dem echten Limiter ergibt aus
  rho=1 für jedes einzelne Achsenupdate rho=0.4, insgesamt aber rho=-0.8.
  Verwendet wurde dt/dx=0.05 und ein synthetischer Dichtefluss mit Steigung 12.
  Dies beweist eine fehlende allgemeine Garantie für das zusammengesetzte
  Update, nicht dass genau diese Flüsse im Supernova-Lauf vorkamen. Die
  zugrundeliegende Literatur verlangt eine geeignete positive Zerlegung:
  Hu, Adams & Shu, https://arxiv.org/abs/1203.1540 (Methodenteil).
- **Runge-Kutta-Kopplung:** `_integrators/_explicit_rk.py:ssprk4` ruft vor
  vier RHS-Auswertungen `pre_stage` auf, vor der fünften (`rhs(u4, ...)`)
  nicht. Der Hook-Test bestätigt diese Aufrufreihenfolge. Ein unzulässiges
  u4 kann daher vor der abschließenden Korrektur schon in WENO und Cooling
  gelangen. Zusätzlich begrenzt der Limiter U+deltaU, während RK teilweise
  gewichtete Zustände plus deltaU bildet. In der letzten Kombination gibt
  es sogar einen negativen Startzustandskoeffizienten. Bei nichtlinearem
  Flux-Blending/Cooling darf die SSP-Garantie deshalb nicht ungeprüft aus
  der Bezeichnung RK4_SSP übernommen werden.
- **Rundung des Druckbodens:** Gespeicherte Felder sind float32. Beim Aufbau
  E=E_kin+p/(gamma-1) kann der Boden p=1e-14 gegenüber E_kin verloren gehen.
  Beispiel: float32(1000+1.5e-14)-float32(1000)=0. Eine nachgeschaltete
  Subtraktion zur Druckrekonstruktion kann Null oder kleine negative Werte
  liefern. Dies ist ein nachgewiesener arithmetischer Mechanismus, nicht
  die eindeutige Rekonstruktion des konkreten NaN-Auslösers.
- **Validierung bislang unzureichend:** Aus den tatsächlich beschriebenen
  zentralen Schnitten wurden im CFL-0.1-Lauf 7 Zell-Snapshot-Einträge mit
  p<=0 gefunden (globales Schnittminimum -1.820885e-6), im CFL-0.05-Lauf 47
  (Minimum -3.3708284e-6). Beide Dateien melden dennoch valid=true, weil der
  Export nur Endlichkeit und Zeitmonotonie prüft. Vollständiges Durchlaufen
  ist damit nachgewiesen, strikte Positivität und physikalische Gültigkeit
  nicht. Die Zahlen gelten nur für gespeicherte Schnitte, nicht alle 3D-Zellen.
- **Cooling bleibt prüfbedürftig:** Der implizite Pfad führt maximal 50
  ungedämpfte Fixpunktiterationen aus, ohne Konvergenzfehler zu melden.
  Im FD-Pfad wird die Cooling-Energieänderung zu den bereits begrenzten
  hydrodynamischen Inkrementen addiert; der Flux-Limiter prüft diese Summe
  nicht. Der Temperaturboden verwirft Cooling-Updates unterhalb des Bodens
  und behält die alte Temperatur; er hebt bereits kältere Zellen nicht an.
  Eine stabile, positivitätserhaltende Gesamtkopplung ist dadurch nicht
  garantiert. Kleinere CFL-Erfolge schließen Cooling als Mitursache nicht aus.
- **Folgerung und nächste Schritte:** Schutz vor jeder Flussauswertung,
  gemeinsame Positivitätsprüfung von 3D-Flüssen und Quellen sowie eine
  rundungsrobuste Druckrekonstruktion untersuchen. Für kausalen Nachweis
  werden 3D-Diagnosen vor/nach jedem Teilschritt benötigt (erste Verletzung,
  Ort, rho, p, E, kinetische Energie, dt und Cooling-Konvergenz). NaN-Ersetzung
  durch Nullen würde Fehler verdecken. Kein Solver-Fix und kein neuer
  Produktionslauf in dieser Nachprüfung vorgenommen.

## 2026-11-09 – Tutor empfiehlt supersonische Trainingsläufe

- **Nutzerhinweis:** Der Tutor meint, dass nun supersonische Läufe gestartet
  werden können. Als nächster wissenschaftlicher Schritt soll daher eine
  supersonische Turbulenzkonfiguration für die Datenerzeugung untersucht
  werden.
- **Bisherige Ergebnisse, die dabei zu beachten sind:** Es gab bereits
  Amplitude-9-Läufe, deren charakteristisches Verhältnis
  `<|v|>/<c_s>` am Ende bei ungefähr `1.23` lag (Cooling-Faktor 15,
  `CFL=0.2`). Diese Größe ist nicht identisch mit der räumlichen Verteilung
  lokaler Machzahlen. Ein anderer Amplitude-9-Lauf ohne hinreichendes Cooling
  hatte am Ende keine lokal supersonischen Zellen. Daher muss vor der Auswahl
  eines Datensatzes klar sein, ob „supersonisch“ das globale Verhältnis, die
  mittlere lokale Machzahl oder einen Anteil der Zellen mit `M>1` bedeutet.
- **Numerische Vorbehalte:** Die vorherige Fehleranalyse hat offene Fragen
  zur multidimensionalen PP-WENO-Begrenzung, zu den SSPRK-Stufen, zur
  float32-Druckrekonstruktion und zur Kopplung des impliziten Cooling-Terms
  aufgezeigt. Außerdem meldeten CFL-0.1/0.05-Ausgaben `valid=true`, obwohl
  die gespeicherten Schnitte einzelne nichtpositive Drücke enthielten. Ein
  supersonischer Zustand darf deshalb nicht allein anhand endlicher Werte
  oder des Fortschrittsbalkens als gültig eingestuft werden.
- **Planungsstatus:** Noch keine neue Machzahl, Forcing-Amplitude, Cooling-
  Einstellung oder CFL-Zahl festgelegt; keine Simulation gestartet. Für den
  nächsten Versuch sind Zieldefinition für die Machzahl, Auswahl des bereits
  vorhandenen validierten Mediums und Positivitätskriterien vorab festzulegen.

## 2026-11-09 – Validierung im seriellen Trainingsdaten-Runner korrigiert

- **Anlass:** Der Runner las das Metadatenfeld `metadata`, während beide
  Simulationsskripte `metadata_json` in ihren NPZ-Dateien speichern. Außerdem
  wurde Phase 2 allein anhand des Prozess-Exitcodes als abgeschlossen markiert.
- **Änderung:** `run_training_dataset.py` liest jetzt `metadata_json` (mit
  Rückwärtskompatibilität für `metadata`) und verlangt für neu erzeugte Medien
  explizit `valid=true`. Zusätzlich prüft es im Medium `primitive_state` auf
  erwartete Form, Endlichkeit sowie strikt positive Dichte und Druck.
- **Phase-2-Prüfung:** Nach Exit-Code 0 validiert der Runner die
  Supernova-NPZ-Metadaten, Felder, Endlichkeit, Zeitmonotonie sowie strikt
  positive Dichte und Druck in den gespeicherten zentralen Schnitten. Nur dann
  wird der Manifeststatus `complete`; andernfalls `supernova_invalid`. Die
  Prüfung der Supernova-Schnitte ersetzt keine Prüfung des gesamten
  3D-Zustands; die NPZ-Datei enthält hier nur zentrale 2D-Schnitte.
- **Verifikation:** Syntaxprüfung erfolgreich. Isolierte NPZ-Fixtures für
  gültiges Medium/Supernova-Ergebnis, nichtpositiven Mediumdruck,
  nichtpositiven Supernovadruck und `valid=false` bestanden. Ein Dry-Run mit
  temporärem Ausgabeordner gab Phase 1 und Phase 2 seriell aus. Es wurden keine
  Simulationen gestartet.

## 2026-11-09 – Startparameter für Energie- und Dichtesweeps

- **Ziel:** Beim Start des seriellen Datensatz-Runners Energie- und
  Wasserstoffdichtewerte als Sweep angeben können.
- **Eingaben:** Vorgesehen sind Supernovaenergien `0.5, 1, 2 × 10^51 erg`
  sowie `n_H=0.1...100 cm^-3` mit Schrittweite `5 cm^-3`.
- **Änderung:** `run_training_dataset.py` akzeptiert nun
  `--sweep-energy-e51` und `--sweep-nh-range START STOP STEP`. Der Endwert
  wird aufgenommen, auch wenn dafür der letzte Schritt kürzer ausfällt.
  Vorhandene Medium- und Supernova-NPZ-Dateien werden vor einem echten Lauf
  erkannt und nicht überschrieben.
- **Sweepgröße und Dichtefolge:** Der vorgeschlagene Dichtesweep ist
  `0.1, 5.1, 10.1, ..., 95.1, 100 cm^-3`; mit drei Energien ergeben sich
  63 serielle Experimente. Nach aktuellem Runnerdesign wird für jedes
  Energie-Dichte-Paar ein eigenes Medium gerechnet, auch wenn bei gleicher
  Dichte dieselben Mediumparameter verwendet werden.
- **Dokumentation:** Ein Startbeispiel mit `--dry-run` wurde in
  `Praktikum_Datenzubereitung/README.md` ergänzt.
- **Machzahl offen:** `anfangsmedium.py` nimmt eine Forcing-Amplitude an,
  keine direkt einstellbare Ziel-Machzahl. Die Forcing-Amplitude ist nicht
  ohne Kalibrierung mit einer erreichten Machzahl gleichzusetzen. Für die
  geplanten Werte `0.5...1.0` muss daher zunächst festgelegt werden, ob der
  bereits geplottete Kennwert `<|v|>/<c_s>` bei `t=5` angesteuert werden soll
  oder eine andere Machstatistik. Bis dahin wurde keine Machzahl als
  Simulationsparameter implementiert.
- **Prüfungen/Läufe:** Keine GPU-Simulation gestartet. Ein Dry-Run und
  Syntax-/Funktionstests wurden in diesem Arbeitsschritt noch nicht
  ausgeführt.

## 2026-11-10 – Machzahl-Kalibrierung vorbereitet

- **Ziel:** Den Zusammenhang zwischen OU-Forcing-Amplitude, mittlerer
  Wasserstoffdichte und erreichter Machzahl bestimmen, bevor die Zielwerte
  `0.5...1.0` in den Datensatz-Sweep aufgenommen werden.
- **Vorhandene Daten CPU-seitig ausgewertet:** Bei `128^3`, Seed `42`,
  `A=9`, `n_H=1 cm^-3`, `CFL=0.1` und Schure-2009-CIE-Cooling ergab der
  finale Zustand im vorhandenen PP-WENO-Medium ohne allgemeinen
  REDISTRIBUTE-Schutz den bestehenden Kennwert
  `<|v|>/<c_s>=1.0535`. Der Mittelwert der lokalen Machzahlen war `1.0773`,
  und `51.8 %` der Zellen hatten lokal `M>1`.
- **Numerik beeinflusst den Messwert:** Zwei weitere vorhandene Dateien mit
  gleicher Auflösung, Seed, Amplitude, Referenzdichte, CFL und Cooling, aber
  anderer Schutz-/Solverkonfiguration, ergaben für denselben Kennwert `0.7936`
  beziehungsweise `0.7736`. Die finalen Dichte-, Druck- und
  Geschwindigkeitsfelder hatten in diesen Dateien jeweils `2,097,152` von
  `2,097,152` Zellen endlich und positiv. Die älteren Dateien besitzen kein
  explizites `valid`-Metadatenfeld; die Endzustandsprüfung allein beweist
  keine Positivität während aller Zwischenstufen. Kalibrierungen müssen daher
  Solver und Schutzkonfiguration festhalten und konstant lassen.
- **Vorbereitete Auswertung:** Neu angelegt wurde
  `Praktikum_Datenzubereitung/calibrate_mach_forcing.py`. Es startet bei
  manueller Ausführung ausschließlich Mediumläufe mit den Defaults aus der
  Trainingsdaten-Konfiguration, misst den finalen Kennwert
  `<|v|>/<c_s>` und protokolliert zusätzlich mittlere lokale Machzahl sowie
  den Zellanteil mit `M>1`. Ergebnisse und Validitätsstatus werden in CSV und
  JSON-Manifest gespeichert. Ein Dry-Run ist verfügbar; vorhandene Dateien
  werden nicht überschrieben.
- **Pilotpunkte:** Als grober Start sind Amplituden `3, 6, 9` und Dichten
  `0.1, 1, 100 cm^-3` vorgesehen (neun serielle Mediumläufe). Das ist eine
  erste Prüfung der Dichteabhängigkeit, keine vollständige Kalibrierung aller
  21 späteren Dichtewerte. Diese Pilotwerte sind Versuchsplanung, keine
  bestätigte Mach-Amplituden-Zuordnung.
- **Änderte Dateien:** `calibrate_mach_forcing.py`,
  `Praktikum_Datenzubereitung/README.md` und dieses Arbeitsprotokoll. Die
  uncommitteten Änderungen am seriellen Datensatz-Runner aus dem Eintrag vom
  9. November blieben erhalten.
- **Läufe/Prüfungen:** Vorhandene NPZ-Endzustände wurden mit Python/NumPy
  CPU-seitig gelesen; es wurde keine Simulation gestartet. Für die neue
  Kalibrierungsdatei wurden in diesem Schritt keine Tests oder Dry-Runs
  ausgeführt. Keine GPU wurde vom Nutzer für einen neuen Lauf benannt.
- **Nächster Schritt:** Nutzer wählt eine freie GPU und startet den
  dokumentierten Pilotlauf manuell. Danach werden die erreichten Kennwerte
  geprüft und entschieden, welche Amplituden für die Dichten zwischen den
  Pilotpunkten nachgemessen werden müssen.

## 2026-11-10 – GPU-Warteschlange mit autocvd ergänzt

- **Nutzeranforderung:** Der Machzahl-Pilot soll beim Start selbst auf eine
  verfügbare GPU warten und dann automatisch anlaufen, da die geeigneten GPUs
  aktuell belegt sind.
- **Implementierung:** `calibrate_mach_forcing.py` ruft vor dem ersten
  Medium-Unterprozess `autocvd(num_gpus=1)` auf. Die Auswahl wird im
  Elternprozess gesetzt und von allen seriellen Mediumläufen übernommen.
  Standardmäßig wird unbegrenzt gewartet und alle 30 Sekunden neu geprüft;
  optional sind Wartezeit und Prüfintervall über CLI-Optionen einstellbar.
  `--dry-run` wartet nicht und ruft `autocvd` nicht auf.
- **Verfügbarkeitsprüfung:** Die installierte `autocvd==0.2.1`-Implementierung
  nutzt `nvidia-smi` und betrachtet eine GPU als frei, wenn kein Compute-
  Prozess gelistet ist. Das ist keine Scheduler-Reservierung; konkurrierende
  auto-Auswahlen können theoretisch dieselbe GPU wählen.
- **Environment:** `autocvd==0.2.1` ist im Environment
  `/export/home/bgschwen/micromamba/envs/pde` installiert und wurde in
  `Praktikum_Datenzubereitung/requirements_pde_gpu.txt` ergänzt.
- **Geänderte Dateien:** `calibrate_mach_forcing.py`,
  `Praktikum_Datenzubereitung/README.md`,
  `Praktikum_Datenzubereitung/requirements_pde_gpu.txt` und dieses Protokoll.
- **Läufe/Prüfung:** Es wurde keine GPU-Auswahl aufgerufen, keine GPU-Simulation
  gestartet und kein Funktionstest ausgeführt. Die Paketverfügbarkeit wurde
  nur über Python-Metadaten geprüft; damit wurde weder JAX importiert noch eine
  GPU belegt.
- **Nächster Schritt:** Nutzer kann den dokumentierten Befehl starten. Wenn
  keine GPU frei ist, zeigt `autocvd` den Wartezustand und beginnt nach
  erfolgreicher Auswahl mit den neun seriellen Mediumläufen.

## 2026-11-11 – Mach-Kalibrierung ausgewertet und Snapshot-Wiederholung vorbereitet

- **Abschluss des ersten Kalibrierungslaufs:** Der am 10. November gestartete
  Lauf wurde auf GPU 5 mit Exit-Code 0 abgeschlossen. Alle neun Kombinationen
  aus `n_H = 0.1, 1, 100 cm^-3` und Forcing-Amplituden `3, 6, 9` erreichten
  `t=5`; je Lauf waren `2,097,152/2,097,152` Zellen numerisch gültig.
- **Charakteristischer Mach-Kennwert** `<|v|>/<c_s>` (Spalten: Amplitude
  `3, 6, 9`): `n_H=0.1`: `0.589, 0.858, 1.062`; `n_H=1`:
  `0.574, 0.855, 1.041`; `n_H=100`: `0.295, 0.271, 0.252`. Bei
  `n_H=100` stieg die mittlere Geschwindigkeit mit der Amplitude, aber die
  mittlere Schallgeschwindigkeit stieg stärker. Diese Ergebnisse sind
  Endzustandsmessungen, keine zeitlichen Velocity-Verläufe.
- **Ausgangslage für Plots:** Die neun Kalibrierungs-NPZ-Dateien enthalten
  `primitive_state` und Zeitpunkte, aber keine Snapshot-Felder (`states`);
  `snapshots_saved=false`. Damit ist `plot_mean_velocity.py` auf ihnen nicht
  für den Zeitverlauf verwendbar.
- **Änderung für Wiederholung:** `calibrate_mach_forcing.py` setzt nun
  `save_snapshots=true` und erzeugt nach jedem gültigen Lauf mit
  `plot_mean_velocity.py` einen PNG-Zeitplot und eine CSV-Zeitreihe. Läufe
  ohne gespeicherte Zustände werden nicht als vollständige Plot-Kalibrierung
  akzeptiert. Die gemeinsame Machzahl-CSV und das JSON-Manifest bleiben
  erhalten.
- **Geplanter neuer Satz:** Forcing-Amplituden `3, 9, 15`, mittlere
  Wasserstoffdichten `0.1, 1, 10, 100 cm^-3`, zwölf serielle Mediumläufe.
  Ausgabeziel: `Praktikum_Datenzubereitung/mach_calibration_snapshots`; so
  bleiben die ersten Ergebnisse unberührt. Bei `128^3` und 32 Snapshots
  benötigen die NPZ-Zustände voraussichtlich ungefähr 1.2 GB pro Lauf, also
  zusammen etwa 14 GB.
- **Geänderte Dateien:** `Praktikum_Datenzubereitung/calibrate_mach_forcing.py`,
  `Praktikum_Datenzubereitung/README.md` und dieses Protokoll.
- **Laufstatus/Prüfungen:** Für die zwölf neuen Läufe wurde keine GPU
  ausgewählt und keine Simulation gestartet. Es wurden keine Tests oder
  Dry-Runs ausgeführt. Vor dem manuellen Start muss der Nutzer die freie GPU
  bestätigen; die Snapshots können insgesamt etwa 14 GB Speicher belegen.

## 2026-11-11 – Plot-Ausgabe der Snapshot-Wiederholung angepasst

- **Nutzerpräzisierung:** Während der zwölf Läufe sollen Daten gesichert
  werden, damit Velocity-Zeitplots oder Medium-GIFs später gezielt und
  manuell erstellt werden können. Es sollen keine PNG-Plots automatisch beim
  Simulationslauf entstehen.
- **Änderung:** `calibrate_mach_forcing.py` speichert weiterhin die 32
  Zustands-Snapshots im NPZ und schreibt pro gültigem Lauf eine CSV mit `time`,
  `mean_speed`, `mean_sound_speed` und `speed_to_sound_ratio`. Die automatische
  PNG-Erstellung wurde entfernt. `plot_anfangsmedium.py` liest die NPZ-Snapshots;
  `plot_mean_velocity.py` kann später manuell den Velocity-Zeitplot erzeugen.
  Der manuelle Plotlauf wählt einen nummerierten Dateinamen, falls die
  zugehörige CSV bereits existiert, und überschreibt die gespeicherte CSV nicht.
- **Dokumentation:** README entsprechend präzisiert; das vorherige Protokoll
  bleibt chronologisch erhalten und dieser Eintrag hält die Korrektur fest.
- **Geänderte Dateien:** `calibrate_mach_forcing.py`,
  `plot_mean_velocity.py`, `Praktikum_Datenzubereitung/README.md` und dieses
  Protokoll.
- **Prüfungen/Läufe:** Keine GPU-Simulation, kein Dry-Run und keine Tests
  ausgeführt. Die Änderung wurde anhand der beiden Plot-Skripte und des
  Kalibrierungsrunners nachvollzogen.

## 2026-11-11 – NPZ als einzige Plot-Datenquelle bestätigt

- **Klärung:** `plot_anfangsmedium.py` benötigt die räumlichen Snapshot-Felder
  aus der NPZ-Datei. `plot_mean_velocity.py` liest dieselben Snapshots und
  berechnet daraus den Zeitverlauf; beim manuellen Plotaufruf kann es die
  Velocity-Zeitreihen-CSV erzeugen. Einzelne Velocity-CSV-Dateien während der
  Kalibrierung sind daher für die späteren Plots nicht erforderlich.
- **Änderung:** Die automatische Erzeugung von per-run Velocity-CSV-Dateien
  wurde aus `calibrate_mach_forcing.py` entfernt. Die NPZ-Snapshots bleiben
  erhalten; die gemeinsame `mach_calibration.csv` mit den finalen
  Machzahlmessungen wird weiter geschrieben.
- **Geänderte Dateien:** `calibrate_mach_forcing.py`,
  `Praktikum_Datenzubereitung/README.md` und dieses Protokoll.
- **Prüfungen/Läufe:** Kein GPU-Lauf, kein Dry-Run und keine Tests ausgeführt.

## 2026-11-11 – Ursache des Abbruchs auf compgpu14 geklärt

- **Nutzerbestätigung:** `compgpu14` ist bis zum 16.11. für eine
  andere Gruppe reserviert. Der Nutzer wurde vom System entfernt; dadurch
  erhielt der Prozess das Interrupt-Signal und endete mit Exit-Code 130.
- **Korrigierte Einordnung:** Der Abbruch war eine Folge der GPU-/Host-
  Reservierung, kein numerischer oder CUDA-Fehler der Simulation und kein
  absichtlicher manueller Abbruch. Die frühere Ursache war anhand des Logs
  zunächst nicht feststellbar.
- **Nächster Schritt:** Den Kalibrierungslauf nach Ende der Reservierung auf
  einem freigegebenen Host oder einer anderen autorisierten GPU mit einem
  neuen Ausgabeordner erneut starten. Keine Wiederholung wurde gestartet.

## 2026-11-11 – Neustart auf compgpu13 durch vorhandene Ausgaben blockiert

- **Versuch:** Der Nutzer startete den 12er-Runner erneut auf `compgpu13` mit
  demselben Zielordner `mach_calibration_snapshots`.
- **Ergebnis:** Ein Startversuch traf auf eine bereits vorhandene
  `medium_nh0p1_amp3.npz`; der folgende Versuch wurde wegen des bereits
  vorhandenen Kalibrierungsmanifests abgewiesen. Der Schutz vor Überschreiben
  hat den Neustart verhindert. Bei der anschließenden Prüfung lagen im Ordner
  das Manifest und das Teil-Log, aber keine NPZ-Datei; das Manifest enthält
  noch den alten Status `running` von compgpu14.
- **Loghinweis:** Die erneute Verwendung desselben `tee`-Zielnamens hat die
  zentrale Datei `mach_calibration_snapshots.log` durch die kurze
  Fehlermeldung ersetzt. Das ausführlichere Teil-Log
  `medium_nh0p1_amp3.log` im Ausgabeverzeichnis ist erhalten.
- **Nächster Schritt:** Für einen erneuten Versuch einen neuen Ausgabeordner
  und einen eigenen zentralen Lognamen verwenden. Keine Dateien wurden von
  Codex gelöscht oder überschrieben; kein neuer Mediumlauf wurde gestartet.

## 2026-11-11 – Snapshot-Kalibrierung auf compgpu13: VRAM-Erschöpfung

- **Befund zum Ausgabeordner:** `mach_calibration_snapshots_compgpu13_retry1`
  enthielt bereits ein Manifest mit Startzeit `2026-11-11T10:56:29Z` und
  GPU-Auswahl `[0]`. Deshalb wurde der danach erneut eingegebene Startbefehl
  korrekt wegen vorhandener Manifest-/CSV-Ausgaben beziehungsweise NPZ-Pfaden
  abgewiesen.
- **Laufstatus im Manifest:** Bei der Prüfung waren `nh0p1_amp3` und
  `nh0p1_amp9` als `simulation_failed`, Exit-Code 1, vermerkt; `nh0p1_amp15`
  stand noch als `running`. In den Logs zu den drei Kombinationen stehen
  CUDA-`RESOURCE_EXHAUSTED`-/Out-of-memory-Meldungen. JAX konnte unter anderem
  eine weitere Allokation von etwa `1.09 GiB` nicht reservieren.
- **GPU-Kontext:** Das Manifest dokumentiert GPU 0. Bei einer späteren Abfrage
  belegte ein anderer Compute-Prozess den Großteil des GPU-Speichers. Das
  stützt Speicherkonkurrenz als wahrscheinlichen Faktor, beweist aber nicht,
  wann genau der zweite Prozess hinzugekommen ist.
- **Ergebnis:** Für die ersten Kombinationen wurden keine gültigen
  Snapshot-NPZ-Dateien erzeugt. Die letzte Manifest-Angabe `running` ist
  möglicherweise veraltet; der Lauf wurde nicht als vollständig bestätigt.
- **Nächster Schritt:** Nicht denselben Ausgabeordner wiederverwenden. Vor
  einem weiteren Lauf muss eine tatsächlich ausreichend freie GPU verfügbar
  sein; anschließend einen neuen Ausgabeordner und Lognamen wählen. Codex
  startete keinen weiteren Lauf und löschte keine vorhandenen Diagnose-Logs.

## 2026-11-11 – Kalibrierung auf compgpu13 erneut in Warteschlange gestellt

- **Nutzerauftrag:** Den 12er-Snapshotlauf mit Forcing-Amplituden `3, 9, 15`
  und `n_H = 0.1, 1, 10, 100 cm^-3` in der Zellij-Session `gpu13` erneut
  starten und mit `autocvd` auf eine freie GPU warten.
- **Ausgabeziel:** Neuer Ordner
  `Praktikum_Datenzubereitung/mach_calibration_snapshots_compgpu13_retry2`
  und separater Logname `mach_calibration_snapshots_compgpu13_retry2.log`;
  frühere Diagnoseausgaben bleiben erhalten.
- **Status bei Einreihung:** Zellij-Pane
  `mach-calibration-autocvd-retry2` ist aktiv. `autocvd` meldete
  `0/1 GPU verfügbar`; bis zu diesem Zeitpunkt war noch keine GPU ausgewählt
  und keine Mediumsimulation gestartet. GPUs sind auf `compgpu13` aktuell
  belegt. Der Runner wartet mit 30-Sekunden-Abfragen weiter.
- **GPU-Freigabe:** Nutzer hat den Start ausdrücklich angefordert; ausgewählte
  physische GPU und Laufzeit sind noch offen, bis `autocvd` eine Karte findet.

## 2026-11-11 – Retry-2-Lauf auf GPU 6 endete vor dem ersten Snapshot

- **Fortschritt:** Das Manifest bestätigt GPU 6. Der erste Mediumlauf
  `n_H=0.1 cm^-3`, Amplitude `3` wurde in Zellij bis etwa `72.4 %`
  fortgeschrieben.
- **Aktueller Zustand:** Bei der Prüfung war kein Projektprozess mehr in der
  GPU-Prozessliste sichtbar. Der Zellij-Pane stand nach einem erneuten
  Startversuch wegen des vorhandenen Manifests an der Shell. Im Ausgabeordner
  liegt kein NPZ-Snapshot; das Manifest hat daher noch veraltet den Status
  `running`.
- **Fehlerursache:** Das erhaltene Medium-Log endet bei `72.4 %` ohne
  Traceback. Die zentrale Retry-2-Logdatei wurde beim erneuten Aufruf mit
  `tee` durch die `FileExistsError`-Meldung ersetzt. Der Grund, weshalb der
  ursprüngliche Integrationsprozess stoppte, ist deshalb nicht mehr aus den
  vorhandenen Logs bestimmbar. Während des Laufs war GPU 6 mit weiteren
  Prozessen geteilt; ein Zusammenhang ist möglich, aber nicht belegt.
- **Nächster Schritt:** Ausgabeordner `mach_calibration_snapshots_compgpu13_retry2`
  nicht erneut verwenden. Vor einem weiteren Lauf neue GPU-Verfügbarkeit
  sicherstellen und einen frischen Ausgabeordner sowie Lognamen verwenden.

## 2026-11-11 – Verschachtelte Kalibrierungsausgaben aus Git-Status ausnehmen

- **Befund:** Nach dem Code-Commit zeigte Git noch 30 ungetrackte Einträge.
  Sie lagen ausschließlich in den erzeugten `mach_calibration*`-
  Ergebnisordnern (NPZ-Snapshots, Logs, Manifest- und CSV-Dateien). Die
  vorhandenen `.gitignore`-Muster trafen nur Dateien direkt unter
  `Praktikum_Datenzubereitung`, nicht Dateien in diesen Unterordnern.
- **Änderung:** Eine gezielte `.gitignore`-Regel nimmt die
  `Praktikum_Datenzubereitung/mach_calibration*/`-Ergebnisverzeichnisse aus
  Git aus. Die lokalen Ergebnisse werden nicht gelöscht; Quellcode wie
  `calibrate_mach_forcing.py` bleibt weiterhin versionierbar.
- **Lauf:** Retry-3 in Zellij `gpu13` blieb unberührt und läuft separat weiter.

## 2026-11-11 – CSV des Mediumzustands vor der Supernova exportiert

- **Ziel:** Für die Darstellung des Zustands unmittelbar vor dem Einsetzen der Supernova wurde aus dem turbulenten Medium die zentrale z-Schnittfläche des letzten Snapshots exportiert.
- **Quelle:** `Praktikum_Datenzubereitung/turbulent_medium_128cells_t5_warmism100pc_schure2009_ppweno_gpu5.npz`; letzter gespeicherter Zeitpunkt `t=5.0`, 128³ Zellen, Seed 42.
- **Ausgabe:** `Praktikum_Datenzubereitung/turbulent_medium_128cells_t5_schure2009_ppweno_gpu5_z64.csv`, 16.384 Zellen in `z_index=64`. Spalten: Zellmittelpunktkoordinaten `x_code,y_code` sowie `rho_code,vx_code,vy_code,vz_code,p_code`. Werte sind in den dimensionslosen Codeeinheiten der NPZ-Datei.
- **Hinweis:** Das GIF wird aus NPZ-Dateien erzeugt; diese CSV ist ein zusätzlicher Export der Medium-Schnittfläche.

## 2026-11-12 – Velocity-Zeitplots der gültigen Retry-3-Läufe erzeugt

- **Ziel:** Die zeitliche Entwicklung von typischer Geschwindigkeit,
  Schallgeschwindigkeit und dem Quotienten `<|v|>/<c_s>` für alle gültigen
  Läufe der Machzahl-Kalibrierung vergleichen.
- **Datensatz:** `mach_calibration_snapshots_compgpu13_retry3`; verwendet
  wurden die zehn Experimente mit Status `complete` aus dem Manifest. Die
  Läufe `nh0p1_amp15` und `nh100_amp15` mit Status `invalid` wurden ausgelassen.
- **Auswertung:** `plot_mean_velocity.py` las die vorhandenen NPZ-Snapshots
  CPU-only und erzeugte pro gültigem Lauf eine PNG-Grafik und eine CSV-Datei
  im selben Ergebnisordner. Dateinamen folgen
  `mean_velocity_<run_id>.png` beziehungsweise `.csv`.
- **Interpretationshinweis:** Der Plotter schließt pro Snapshot nicht-endliche
  Zellen sowie Zellen mit nichtpositiver Dichte oder Druck aus den Mittelwerten
  aus. Kurven sind damit Mittelwerte über jeweils gültige Zellen.
- **Prüfung:** Alle zehn Plotaufrufe endeten erfolgreich; keine Simulation,
  GPU-Auswertung oder Quellcodeänderung wurde vorgenommen.

## 2026-11-12 – Velocity-Zeitplots der zwei ungültigen Retry-3-Läufe ergänzt

- **Ziel:** Die Zeitverläufe der beiden im Manifest als `invalid` markierten
  Läufe ebenfalls sichtbar machen.
- **Datensätze:** `medium_nh0p1_amp15.npz` und
  `medium_nh100_amp15.npz` aus
  `mach_calibration_snapshots_compgpu13_retry3`.
- **Auswertung:** `plot_mean_velocity.py` konnte beide vorhandenen Snapshot-
  Dateien CPU-only verarbeiten. Für jeden Lauf wurden PNG und CSV erzeugt und
  im VS-Code-Fenster des Projekts geöffnet.
- **Einschränkung:** Der Plotter maskiert pro Snapshot Zellen mit nicht-endlicher
  oder nichtpositiver Dichte beziehungsweise Druck. Die Plots visualisieren
  daher Mittelwerte der jeweils verbleibenden gültigen Zellen; sie heben den
  Status `invalid` der Läufe nicht auf.
- **Prüfung:** Beide Plotaufrufe endeten erfolgreich. Keine Simulation,
  GPU-Auswertung oder Quellcodeänderung wurde vorgenommen.

## 2026-11-12 – Auffällige Velocity-Zeitkurven nachgeprüft

- **Datengrundlage:** CSV-Zeitreihen, gespeicherte interne/kinetische Energie
  und vollständige Zustands-Snapshots der Läufe `nh10_amp3`, `nh100_amp15`
  und `nh0p1_amp15`; CPU-only-Auswertung.
- **`nh10_amp3`:** In keinem gespeicherten Snapshot wurden Zellen durch die
  Positivitätsmaske des Plotters ausgeschlossen. Zwischen `t=3.750` und
  `t=3.906` fällt die mittlere Schallgeschwindigkeit von `2.298` auf `2.062`
  und die interne Energie von `4.752` auf `3.901`; die kinetische Energie
  steigt nur von `0.336` auf `0.360`. Das weist auf einen starken Abbau
  thermischer Energie in diesem Intervall hin. Aktives explizites
  Schure-CIE-Cooling ist eine plausible Erklärung; die genaue Ursache ist
  ohne Teilschritt-Diagnostik nicht belegt.
- **`nh100_amp15`:** Bis `t=4.688` wurden keine Zellen ausgeblendet. Im
  Snapshot `t=5` sind `81.916/2.097.152` Zellen (rund `3,9 %`) nicht positiv
  beziehungsweise ungültig. Gleichzeitig fällt die mittlere
  Schallgeschwindigkeit von `7.068` auf `3.215` und die interne Energie von
  `44.618` auf `9.512`. Der letzte Plotabschnitt ist daher wegen Cooling und
  des bereits ungültigen Endzustands nicht als belastbarer physikalischer
  Verlauf zu interpretieren; die Zellmaske beeinflusst zusätzlich den
  dargestellten Mittelwert.
- **`nh0p1_amp15`:** Im finalen Zustand ist eine Zelle mit negativem Druck;
  das beeinflusst den über rund 2,1 Millionen Zellen gebildeten Mittelwert
  kaum, hebt den Status `invalid` aber nicht auf.
- **Einschränkung:** Snapshotabstände betragen ungefähr `0.156` Codezeiteinheiten
  und zeigen nicht, in welchem Integrations- oder Cooling-Teilschritt die
  Änderung einsetzt. Keine Simulation gestartet und kein Code geändert.

## 2026-11-13 – Velocity-Zeitplots des Machzahl-Sweeps

- **Vorhandene Plots:** Die zwölf Einzelgrafiken `mean_velocity_nh*_amp*.png` lagen bereits im Ordner `Praktikum_Datenzubereitung/mach_calibration_snapshots_compgpu13_retry3`; sie wurden auf Nutzerhinweis wiedergefunden.
- **Zusätzliche Ausgabe:** Bei der Suche wurden irrtümlich zwölf weitere Einzelplots samt Statistik-CSV unter `velocity_plots_20261113/` erzeugt; außerdem eine 4x3-Übersicht `velocity_overview_12runs.png`. Keine Simulation und keine GPU verwendet.
- **Gültigkeit:** Die beiden ungültigen Fälle (`n_H=0.1` und `100 cm^-3`, jeweils Forcing 15) sind in der Übersicht markiert. Die Plotroutine maskiert Zellen mit nichtpositiver Dichte oder nichtpositivem Druck; diese Kurven sind daher nur diagnostisch und stellen keine gültigen Resultate dar.

## 2026-11-16 – Ausgabeumfang für Supernova-Vergleich und Trainingsdaten festgelegt

- **Nächster Vergleich:** Die zehn als gültig markierten turbulenten Medien des Machzahl-Sweeps sollen jeweils als Anfangszustand für eine Supernova-Simulation dienen. Für diese Vergleichsläufe genügen die zentralen 2D-z-Schnitte als Plot-Ausgabe; die numerische Integration bleibt vollständig 3D.
- **Trainingsdaten:** Für spätere Trainingsdatensätze müssen vollständige 3D-Endzustände gespeichert werden. `supernovae_turbolenz.py` speichert derzeit standardmäßig nur zentrale 2D-Schnitte; vor der Trainingsdatenerzeugung ist daher die Ausgabe zu erweitern.
- **Laufparameter als Arbeitsannahme:** Vorgesehene Werte aus `training_dataset_config.example.json`: `E_SN=10^51 erg`, `duration=0.1`, 120 Snapshots, Explosionsradius `0.03`, `CFL=0.1`, PP-WENO mit REDISTRIBUTE/vacuum_rest und implizites Schure-CIE-Cooling. GPU-Freigabe und Start stehen noch aus.

## 2026-11-16 – Zehn Supernova-Läufe auf gültigen Kalibrierungsmedien gestartet

- **Ziel:** Die zehn im Kalibrierungsmanifest als `complete` markierten turbulenten Medien nacheinander als Anfangszustände für Phase 2 verwenden; die zwei als `invalid` markierten Medien werden ausgeschlossen.
- **Runner:** Neu erstellt `Praktikum_Datenzubereitung/run_supernovae_on_calibration_media.py`. Er validiert die zehn Eingaben, lädt die `supernova_defaults` aus `training_dataset_config.example.json`, wählt per `autocvd` genau eine freie GPU und protokolliert jeden Lauf mit Status und Validierung in einem Batch-Manifest.
- **Konfiguration:** `duration=0.1`, `120` Snapshots, `E_SN=10^51 erg`, Radius `0.03`, `smoothing_cells=2`, `CFL=0.1`, PP-WENO plus REDISTRIBUTE/vacuum_rest, implizites Schure-2009-CIE-Cooling und `save_invalid=true`. Phase 1 der Medien hatte explizites Cooling.
- **Ausgabe:** `Praktikum_Datenzubereitung/mach_calibration_snapshots_compgpu13_retry3/supernova_phase2_10valid/`; pro Lauf zentrale z-Schnittdaten, nicht vollständige 3D-Endzustände. Batch-Log: `supernova_phase2_10valid_autocvd.log`.
- **GPU-Freigabe und Start:** Nutzer wies an, autocvd zur automatischen Auswahl einer freien GPU einzusetzen. `autocvd` wählte GPU 4. Der serielle Lauf `nh0p1_amp3` wurde am 2026-11-16 gestartet; das Log bestätigte GPU-Backend und Start der vollständigen 3D-Integration.
- **Prüfung:** CPU-only-Dry-Run fand alle zehn gültigen NPZ-Medien und gab zehn eindeutige Befehle aus. Kein Überschreiben vorhandener Ergebnisse; keine Test-Suite ausgeführt.

## 2026-11-18 – Abschlussstand des Supernova-Vergleichs aus dem Manifest

- Das gespeicherte `supernova_batch_manifest.json` im Ordner `mach_calibration_snapshots_compgpu13_retry3/supernova_phase2_10valid/` weist für alle zehn Läufe einen Endstatus aus. Die Serie endete am 17. November.
- Sechs Fälle haben den Status `complete`. Vier sind `invalid`: nh0p1_amp3, nh0p1_amp9, nh1_amp9 und nh1_amp15. Die Zuordnung basiert auf der dokumentierten Validierung der Phase-2-Ausgaben. Die Ursache der numerischen Fehler bleibt offen.
- Die gespeicherten zentralen z-Schnitte dienen dem Vergleich. Für die geplanten 3D-Trainingspaare fehlt die Ausgabe vollständiger 3D-Endzustände. Für diese Bestandsaufnahme wurden das Manifest und die Dokumentation gelesen, keine Simulation und keine erneute numerische Validierung ausgeführt.

## 2026-11-18 – Wiederholte Verbindungsabbrüche bei VS Code untersucht

- Die persönlichen VS-Code-Serverprotokolle melden um 11:30, 11:47, 12:16, 16:04 und 16:18 Uhr den Verlust der Clientverbindung. Daneben sind reguläre Sitzungsenden verzeichnet. Die Zuordnung zu den vom Nutzer bemerkten Abbrüchen bleibt offen.
- Der VS-Code-Server registrierte um 16:18:17 einen Verbindungsverlust und nahm um 16:18:33 eine neue Verbindung an. Die fünfminütige Aufräumfrist für die alte Sitzung lief um 16:23:33 ab. Zu diesem Zeitpunkt erschien auch die Meldung `Unexpected SIGPIPE`.
- Der untersuchte Host `compgpu14` lief seit dem 10. November. Laut atop waren zwischen 16:10 und 16:20 im Mittel rund 97 % der CPU-Kapazität frei. Bei der Messung um 16:20 waren rund zwei Drittel des Arbeitsspeichers frei.
- Die Serverprotokolle lassen die Ursache des ersten Verbindungsverlusts offen. Das SSH-Systemprotokoll ist mit dem Projektkonto nicht lesbar. Für die Zuordnung fehlt die lokale Remote-SSH-Ausgabe zum selben Zeitpunkt.

## 2026-11-18 – Wiederverbindung mit dem Protokoll des Macs abgeglichen

- Laut lokalem Remote-SSH-Protokoll begann um 16:18:29 ein neuer SSH-Verbindungsaufbau, rund elf Sekunden nach dem vom Server erfassten Abbruch. Die Passwortanmeldung gelang um 16:18:32. Um 16:18:33 stand der Tunnel, und VS Code fand den laufenden Serverprozess.
- Der Ausschnitt erklärt die erneute Passwortabfrage. Er enthält keine Meldung der vorherigen Sitzung zum Zeitpunkt der Trennung. Der ursprüngliche Auslöser bleibt offen.

## 2026-11-19 - Machzahl-Sweep korrigiert und Fortsetzung gestartet

- **Ziel und feste Konfiguration:** Kalibrierung der Machzahl fuer TurbGen-Medien mit solenoidalem Forcing. Geplant sind 20 Kombinationen aus physikalischer Wasserstoffzahldichte `n_H = 0.1, 1, 10, 100 cm^-3` und `driving_velocity = 1, 1.5, 2, 2.5, 3`. Gemeinsam: 128^3 Zellen, Boxgroesse 1, `t_end=2`, 33 Snapshots, `gamma=5/3`, Seed 42, FD-Riemann, implizites Schure-2009-CIE-Cooling, `CFL=1`.
- **Gewuenschte Messgroessen:** Machzahl am letzten gespeicherten Zeitpunkt sowie zeitlicher Mittelwert zwischen `t=1.5` und `t=2` der raeumlich gemittelten lokalen Machzahl. Die Resultate sollen als CSV-Tabelle neben NPZ-Ausgaben und einem Batch-Manifest landen.
- **Erster Versuch und Ursache des Abbruchs:** `nh0p1_dv1` hatte die Integration bis `t=2` beendet und eine NPZ-Datei geschrieben. Die Auswertung brach ab, weil sie `primitive_state` als Zeitreihe behandelte. Im Dateiformat ist `primitive_state` nur der Endzustand mit Form `(5,128,128,128)`; alle gespeicherten Zeitpunkte liegen in `states` mit Form `(33,5,128,128,128)`, die Zeiten in `time_points` mit Form `(33,)`. Die Simulation selbst war nicht der Ausloeser.
- **Korrektur und Wiederaufnahme:** Der korrigierte Fortsetzungslauf liest die Snapshots aus `states`, berechnet daraus die Mach-Statistik und soll den bereits geschriebenen Fall `nh0p1_dv1` wiederverwenden. Die uebrigen Faelle werden fortgesetzt; Manifest und CSV werden laufweise aktualisiert.
- **Aktueller Status:** Der Sweep-Fortsetzungslauf laeuft in Zellij `gpu14`, Pane `terminal_75`, wartet aber bei `autocvd` auf eine freie GPU (`0/1 GPU(s) available`). Eine GPU-Ausfuehrung des korrigierten Sweeps ist noch nicht bestaetigt. Bislang vorhanden sind NPZ und Log fuer `nh0p1_dv1` sowie das Batch-Manifest; Mach-Auswertung/CSV sind noch nicht fertig.

## 2026-11-19 - Supernovae-Kontrolllauf mit einheitlichem Positivitaetsschutz angefordert

- Im Vergleichsordner `fd_riemann_forcing_comparison_batch2/supernova_nh100_protection_cfl_compare_20261119/` liegen zwei beendete, als gueltig gepruefte Simulationen (Metadaten gueltig, endliche Werte, positive Dichte und positiver Druck, monotone Zeiten): CFL 1 mit zusaetzlich gesetztem harten Positivitaets-Floor und Step-Retry sowie CFL 0.1 ohne diese zusaetzlichen Optionen. Beide nutzen FD-Riemann und implizites Schure-CIE-Cooling.
- Auf Nutzerwunsch wird CFL 1 mit denselben Standardoptionen wie bei den anderen Laeufen erneut gerechnet. `PositivityConfig()` bedeutet hier keine zusaetzlichen per-step Floor-/Retry-Optionen; die solverseitige Positivitaetsbehandlung von FD-Riemann bleibt aktiv. Der bisherige CFL-1-Lauf mit Zusatzoptionen bleibt als separater Vergleich erhalten.
- **Aktueller Status:** Der korrigierte CFL-1-Lauf wartet in Zellij `gpu14`, Pane `terminal_73`, bei `autocvd` auf eine freie GPU (`0/1 GPU(s) available`). Start und Abschluss sind noch nicht bestaetigt.

## 2026-11-19 - Stand der Supernova-Animationen

- Fuenf GIFs der gueltigen Supernova-Laeufe liegen unter `Praktikum_Datenzubereitung/supernova_5valid_gifs_fdriemann_implicit_cfl1/`: `nh0p1_amp3`, `nh0p1_amp9`, `nh1_amp9`, `nh1_amp15` und `new_nh0p1_legacy_ou`.
- GIFs der beiden NH100-Vergleichslaeufe liegen unter `fd_riemann_forcing_comparison_batch2/supernova_nh100_protection_cfl_compare_20261119/`. Die bisher erzeugten Animationen zeigen separat Dichte; sie wurden nicht mit dem Projekt-Plotprogramm erstellt, das Dichte, Druck und Geschwindigkeit gemeinsam darstellt. Eine Neuerstellung mit der vorgesehenen Projekt-Plotroutine ist noch offen.

## 2026-11-19 - NH100 CFL=0.1 Kontrolllauf neu gestartet

- Der CFL=1-Lauf mit Standard-Positivitaetskonfiguration wurde validiert: Metadaten melden `valid=False`; Werte sind endlich, aber mindestens Dichte oder Druck ist nicht ueberall positiv. Die Konfiguration hatte keine zusaetzlichen per-step Floors oder Step-Retries.
- Auf Nutzerwunsch wurde derselbe Supernova-Fall mit CFL=0.1 neu gestartet. Geaendert wurde nur der CFL-Wert; beibehalten: dasselbe NH100-Eingabemedium `nh100_legacy_ou_implicit_cfl1_1.npz`, FD-Riemann, implizites Schure-2009-CIE-Cooling, Standard-Positivitaetskonfiguration, Dauer 0.1, 120 Snapshots, Explosionsenergie `1e51 erg`, Radius 0.03 und 2 Glaettungszellen.
- `autocvd` waehlt GPU 4. Beim Start wurden CUDA/PALLAS erkannt; der Lauf begann die 3D-Integration. Beim letzten Statuscheck stand er bei rund 19.5 % (`t=0.0195/0.1`); GPU 4 war mit rund 8.9 GiB und 96 % Auslastung belegt.
- Zellij: neuer Tab `nh100-cfl0p1` in Sitzung `gpu14`, mit nur einem Pane. Ergebnis und Log liegen geplant unter `/tmp/fd_riemann_forcing_comparison_batch2/supernova_nh100_protection_cfl_compare_20261119/nh100_fdriemann_implicit_cfl0p1_default_positivity_rerun.npz` und entsprechendem `.log`.
- Der vorhandene CFL=0.1-Lauf `no_protection` bleibt unveraendert; dieser neue Lauf ist eine separate Wiederholung mit eigenem Ausgabepfad.

- **Fortschritt bei letzter Kontrolle:** Der CFL=0.1-Lauf war bei 57 % (`t=0.0570/0.1`) und lief auf GPU 4 mit rund 8.9 GiB VRAM und 96 % Auslastung. Er war zu diesem Zeitpunkt noch nicht abgeschlossen.

## 2026-11-19 - Validierung der CFL=0.1-Wiederholung

- Die CFL=0.1-Ausgabe ist gueltig: NPZ-Metadaten `valid=True`; Dichte, Druck und Zeiten sind endlich; Dichte und Druck sind ueberall positiv; Zeitpunkte sind streng monoton. Gespeichert sind 120 Snapshots bis zur absoluten Zeit `5.1000000015`.

## 2026-11-19 - Projekt-GIF fuer den gueltigen CFL=0.1-Lauf erstellt

- Verwendet wurde die Projektdatei `Praktikum_Datenzubereitung/plot_supernovae_turbolenz.py`, nicht eine separate Ad-hoc-Animation. Sie kombiniert die Medium-Snapshots mit dem Supernova-Ergebnis und stellt Dichte, Druck sowie Geschwindigkeitsvektoren dar.
- Eingaben: `nh100_legacy_ou_implicit_cfl1_1.npz` als Medium und `nh100_fdriemann_implicit_cfl0p1_default_positivity_rerun.npz` als Supernova.
- Ausgabe: `/tmp/fd_riemann_forcing_comparison_batch2/supernova_nh100_protection_cfl_compare_20261119/projektplot_cfl0p1_default_positivity_rerun.gif` (29 MB). Der Plotter beendete sich ohne Fehler und bestaetigte die gespeicherte kombinierte Animation.

- Eine Kopie fuer den direkten Zugriff im Projekt liegt unter `Praktikum_Datenzubereitung/supernova_nh100_cfl_comparison_20261119/projektplot_cfl0p1_default_positivity_rerun.gif` (29 MB).

## 2026-11-19 - Sechs Supernova-Wiederholungen und Machzahl-Sweep aktualisiert

- Das finale Manifest `supernova_6cases_fdriemann_implicit_cfl1_retry3/batch_manifest.json` weist fuenf gueltige Laeufe aus: `nh0p1_amp3`, `nh0p1_amp9`, `nh1_amp9`, `nh1_amp15` und `new_nh0p1_legacy_ou`. Der sechste Fall `new_nh100_legacy_ou_implicit_cfl1` ist ungueltig: Metadaten, endliche Werte und monotone Zeiten sind gueltig, aber Dichte/Druck sind nicht ueberall positiv. Fuer die fuenf gueltigen Faelle wurden Projekt-GIFs erstellt.
- Der fortgesetzte 20-Fall-Machzahl-Sweep hat in Pane `terminal_75` die ersten 17 Faelle als `complete` ausgewiesen. Noch offen sind die drei driving-velocity-Faelle 2.0, 2.5 und 3.0 bei `n_H=100 cm^-3`; der Runner wartete beim letzten Blick auf eine GPU. Die bisherigen End-Machzahlen reichen ungefaehr von 0.4 bis 2.97.

## 2026-11-19 - GIFs fuer zwoelf gueltige Mediumlaeufe gesammelt

- Mit `Praktikum_Datenzubereitung/plot_anfangsmedium.py` wurden GIFs mit Dichte, Druck und Geschwindigkeitsvektoren erstellt. Alle liegen zusammen in `Praktikum_Datenzubereitung/medium_12_valid_gifs_20261119/`; der Ordner enthaelt zwoelf Dateien mit insgesamt rund 87 MB.
- Die Auswahl umfasst die zehn als gueltig markierten Originalmedien (Amplitude 15 bei `n_H=0.1` und `100 cm^-3` ausgeschlossen) sowie die zwei neu erzeugten gueltigen Ersatzmedien `nh0p1_legacy_ou` und `nh100_legacy_ou`. Die urspruenglichen beiden NPZ-Dateien bleiben im alten Manifest als ungueltig markiert und wurden nicht als gueltige Medien geplottet.

## 2026-11-19 - CFL=0.1-Wiederholungen der drei ungueltigen TurbGen-Medien vorbereitet

- **Ziel:** Den Einfluss der CFL-Zahl auf die drei bei CFL=1 ungueltigen Mediumlaeufe bei `n_H=100 cm^-3` untersuchen.
- **Faelle:** TurbGen-Zielgeschwindigkeiten `2.0`, `2.5` und `3.0`; jeweils `128^3` Zellen, periodische Randbedingungen, Boxlaenge 100 pc, Referenztemperatur 10000 K, `gamma=5/3`, Seed 42, `t=0...2`, 33 Snapshots, solenoidales TurbGen-Forcing mit `k_min=2`, `k_max=6`, `k_driving=0.6`, Amplitudenfaktor 1 und 10 Forcing-Schritte pro Turnover sowie FD-Riemann und implizites Schure-2009-CIE-Cooling. Gegenueber den CFL=1-Laeufen soll nur `CFL=0.1` geaendert werden.
- **Vorgesehene Ausgabe:** Neuer Ordner `Praktikum_Datenzubereitung/mach_turbgen_solenoidal_implicit_cfl0p1_t2_20261119/`; NPZ-Dateien und Logs erhalten je Fall eigene Namen. Die CFL=1-Ergebnisse bleiben unveraendert.
- **Ausfuehrungsstatus:** Noch nicht gestartet. Gemaess der dokumentierten Startregel startet Codex keine Simulation; der Nutzer startet die drei Laeufe manuell auf einer freigegebenen GPU. Der konkrete GPU-Identifier ist offen.
- **Naechster Schritt:** Die drei Laeufe seriell mit `CFL=0.1` ausfuehren und danach Endzustand, Snapshotzeiten, endliche Werte sowie positive Dichte und positiven Druck pruefen. Ein gueltiger Lauf soll anschliessend in einer separaten CSV-/Manifestauswertung als gueltig markiert werden; ungueltige Rohdaten bleiben erhalten.

## 2026-11-19 - CFL=0.1-Lauf gestartet und nach Laufzeithinweis pausiert

- **Startfreigabe:** Der Nutzer bat Codex, die Wiederholungen in Zellij zu starten und bestaetigte, dass `autocvd` eine freie GPU auswaehlen darf. `autocvd` waehlt GPU 4.
- **Erster Startversuch:** Der Lauf lud wegen eines Modulpfadfehlers die alte Projektdatei statt des TurbGen-Overlays und endete vor der Integration. Log und Manifest bleiben in `mach_turbgen_solenoidal_implicit_cfl0p1_t2_20261119/` erhalten; es wurde keine Simulation gestartet.
- **Korrigierter Start:** Das Overlay wurde als Python-Paket aufgeloest; `nh100p0_dv2p0` startete mit PALLAS auf GPU 4 und erreichte `t=0.110044/2` (5,5 %). Wegen der bei CFL 0,1 gegenueber CFL 1 etwa zehnmal kleineren Zeitschritte wurde dem Nutzer ein voraussichtlich mehrstuendiger Gesamtaufwand fuer drei Faelle genannt.
- **Pausierung:** Der Nutzer hatte den Aufwand noch nicht bestaetigt; der Prozess wurde mit Ctrl-C angehalten. Es wurde keine NPZ-Ergebnisdatei geschrieben. GPU 4 war bei der anschliessenden Kontrolle wieder ohne Prozessbelegung.
- **Ausgaben:** Der abgebrochene Versuch ist unter `mach_turbgen_solenoidal_implicit_cfl0p1_t2_20261119_retry2/` dokumentiert. Die Script-Ausgabe fuer eine bestaetigte Fortsetzung ist auf einen neuen, separaten Ordner `..._retry3/` gesetzt, damit keine Logs oder Manifeste ueberschrieben werden.
- **Offen:** Fortsetzung der drei seriellen Laeufe nach Bestaetigung des mehrstuendigen Aufwands.

## 2026-11-19 - CFL=0.1-Wiederholungen nach Aufwandsbestaetigung gestartet

- **Freigabe:** Der Nutzer bestaetigte nach Hinweis auf voraussichtlich rund 35-40 Minuten pro Fall beziehungsweise etwa zwei Stunden fuer drei serielle Laeufe die Fortsetzung. `autocvd` darf eine freie GPU automatisch waehlen.
- **Korrektur nach dem ersten Importfehler:** Das Python-Modul wird nun mit dem TurbGen-Overlay als Arbeitsverzeichnis gestartet. Der fehlgeschlagene Importversuch und der vor Bestaetigung abgebrochene Teilversuch bleiben in den separaten Ordnern `mach_turbgen_solenoidal_implicit_cfl0p1_t2_20261119/` und `..._retry2/` erhalten.
- **Runner:** `Praktikum_Datenzubereitung/run_turbgen_cfl_repeats.py` fuehrt die Faelle `dv=2.0, 2.5, 3.0` seriell mit `CFL=0.1` aus, streamt die Logs in Zellij und schreibt Manifest sowie Machzahl-CSV. Ausgabeordner ist `Praktikum_Datenzubereitung/mach_turbgen_solenoidal_implicit_cfl0p1_t2_20261119_retry3/`.
- **Startstatus:** Zellij-Sitzung `gpu14`, Pane `terminal_90`; `autocvd` waehlt GPU 4 (A100). Der Fall `nh100p0_dv2p0` hat begonnen; beim ersten Statuscheck stand er bei `t=0.000201718/2`. Die beiden anderen Faelle stehen aus. Noch keine NPZ-Ergebnisse oder Machzahlmessungen.
- **Startzeit:** 2026-11-19 14:23:49 UTC laut Manifest.

## 2026-11-19 - Zwoelf gueltige Supernova-GIFs in einem Ordner gesammelt

- Nach Klarstellung des Nutzers ist die gewuenschte Auswahl die zwoelf final gueltigen Supernova-Ergebnisse, nicht die zwoelf Anfangsmedien. Die sechs gueltigen Original-Phase-2-Laeufe, fuenf gueltigen FD-Riemann-Wiederholungen aus `retry3` und der gueltige NH100-CFL=0.1-Standardlauf ergeben zusammen zwoelf Ergebnisse.
- Alle zwoelf GIFs liegen in `Praktikum_Datenzubereitung/supernova_12_valid_gifs_projectplot_20261119/`. Sechs bereits mit dem Projektplotter erzeugte GIFs wurden uebernommen; die sechs fehlenden wurden mit `plot_supernovae_turbolenz.py` gerendert. Jedes zeigt das Medium zusammen mit der Supernova, mit Dichte, Druck und Geschwindigkeitsvektoren.
- Die zuvor erstellten GIFs der zwoelf Anfangsmedien sind eine getrennte Ausgabe und gehoeren nicht zu dieser Supernova-Auswahl.
