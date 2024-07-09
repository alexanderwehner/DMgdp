graph dependency parser eval.txt

Als erstes habe ich den vorimplementierten MLP Parser installiert und mit ganz abgespeckten Parametern über meine Notebook CPU (Intel® Core™ i7-5500U CPU @ 2.40GHz × 4; 2 physische, 4 logische Kerne; Bj. ca. 2015) lokal laufen lassen, nur um zu testen ob dieser überhaupt durchläuft, was funktioniert hat (siehe 2 Screenshot-Dateien "local test mlp model installation (...) .png").

Zum Testen des D&M Parsers habe ich dann zunächst 10 Durchläufe mit etwas niedrigeren Parametern laufen lassen, jedoch mit dem vollen Trainigsdatenset (limit_train auf 13000 und limit_dev auf 2100, trial & error), da der Parser zu diesem Zeitpunkt schon gut lief.

(siehe Screenshot-Datei "DM parser - W&B testing runs data.png") 

Zuerst wollte ich mit den Parametern batchsize und epochs etwas variieren, um zu sehen, wie sich diese beeinflussen und welches Setup die besten Ergebnisse für Laufzeit und UAS liefert.

Ich habe zuerst vier Läufe (in pink) laufen lassen, mit batchsizes {8, 16, 32, 64}, mit jeweils nur zehn Epochen, die alle in etwa gleichschnell trainierten (ein kleines bisschen länger für höhere batchsize, aber vernachlässigbar), jedoch mit steigender batchsize immer leicht schlechtere UAS lieferten. (für Kurven siehe Datei "epoch10.png")

Hier habe ich mir erschlossen, dass für gleichbleibende Anzahl an Epochen geringere batchsize wohl bessere Ergebnisse liefert. 

Auf die selbe Art und Weise habe ich dann noch vier Läufe (in grün) mit batchsizes {8, 16, 32, 64} und jeweils 20 Epochen laufen lassen. Hier zeigt sich im Vergleich der grünen Durchläufe untereinander ein ähnliches Muster wie bei den pinken Läufen zuvor, also bei gleichbleibender Anzahl Epochen liefert niedrigere batchsize bessere Ergebnisse, die Laufzeit blieb wieder in etwa gleich, etwa doppelt so lange wie bei den pinken Läufen. Auch habe ich beobachten können, dass mit steigender batchsize der batch loss besser wurde, und dies mit steigenden Epochen ebenso, nur noch merklicher passiert. (für Kurven siehe Datei "epoch20.png")

Folglich gehe ich davon aus dass, wenn man nun den jeweils pinken und grünen Lauf mit gleicher batchsize, jedoch 10 bzw. 20 Epochen respektive vergleicht (also bspw. grün batchsize 8, epochs 20 vs. pink batchsize 8, epochs 10), die Anzahl der Epochen einen größeren Einfluss auf die resultierende Performanz des Parsers haben muss, und größere batchsizes die Performanz etwas negativ beeinflussen, aber das Training beschleunigen (letzteres vor allem sichtbar bei Vergleich des späteren blauen Laufs mit batch 1, epochs 20 vs. grün batch 8, epochs 20; Zeiteinsparung von ca. Faktor 4).

Auch interessant: pink 10 Epochen mit batchsize 8 war marginal besser als grün 20 Epochen mit batchsize 64. 

Danach habe ich einen Testlauf mit batchsize 8 und 60 Epochen laufen lassen (in grau), der mit einem UAS von fast 86 von diesen zehn Testläufen bisher das beste Ergebnis lieferte, und auch nur etwas mehr als 2/3 der Zeit des vorherigen Laufs brauchte, trotz drei Mal mehr Epochen. 

Als nächstes habe ich einen Lauf (in blau) mit batchsize 1 und 20 Epochen laufen lassen, um zu sehen wie sich das Modell mit batchsize 1 verhält. Hier hat der Parser mit batchsize 1 und nur 20 Epochen schon einen UAS von ca. 84, was okay sein sollte. Zusätzlich war hier auch der batch loss bisher am niedrigsten, mit nur 14%.

Basierend auf diesen Beobachtungen vermute ich, dass es irgendwo im Bereich batchsize 8 bis 16 einen sweet spot geben muss, der die Performanz sowie die Laufzeit optimal gestaltet, und je mehr Epochen das Modell durchläuft, desto besser wird die Performanz, jedoch steigt dabei auch die Laufzeit stark an.

(siehe Datei "compare runs.png")

Für optimale Performanz und tragbare Laufzeiten scheint also batchsize 8 vernünftig. 

Nach ein paar weiteren Verbesserungen und Anpassungen im Code (u.a. Modellauswahl via config.yml, siehe util.py, zum schnelleren Wechsel zwischen D&M Parser und MLP Parser, und viel bugfixing (verschiedene Geräte, verschiedene Probleme)) habe ich noch einen kompletten Lauf mit 1000 Epochen und batchsize 8 laufen lassen, dieser ist in der Tabelle in orange zu sehen. (für Kurve siehe Datei "epoch1000.png")

Unter diesen Parametern lieferte das Modell wie erwartet die beste Performanz, mit UAS über 86 und batch loss von nur 11%, dauerte aber auch aufgrund der hohen Zahl an Epochen fast neun Stunden.

Abschließend muss ich anmerken, dass ich gerne weiter getestet und angepasst hätte, ich jedoch vor vielen Hürden speziell bezüglich des Rechenaufwands stand, da ich zuerst in Google Colab gearbeitet hatte, das Nutzungslimit hier aber bereits sehr schnell aufgebraucht war. Danach hatte ich sehr große Probleme das ganze auf meiner eigenen GPU kompatibel und ausführbar zu bekommen, was letzten Endes nicht geklappt hat, da mein Desktop-PC auf Windows läuft und die Einrichtung des CUDA Toolkits, der CUDNN Bibliothek und der dazu kompatiblen PyTorch Version auf Windows nicht einfach funktioniert und undurchsichtig ist, und meine GPU nach stundenlangem Versuchen immer noch nicht erkannt und zum Rechnen genutzt werden konnte (mein übliches Arbeits-Notebook, mit seiner veralteten CPU und ohne Grafikkarte, ist der Rechenleistung einer Gartenkartoffel ebenbürtig). Ich bin nicht sicher wie ich meine GPU bearbeiten muss, damit sie für mich rechnet (nach dieser Übung am ehesten mit einem Hammer).

Letzten Endes musste ich die ganze Rechenarbeit auf geliehener (leistungsfähiger, auf Linux laufender) Hardware laufen lassen und diese hatte ich nur zeitlich sehr begrenzt zur Verfügung. Ich hoffe, dass die angefügten Daten der elf durchgeführten Läufe ausreichen.
