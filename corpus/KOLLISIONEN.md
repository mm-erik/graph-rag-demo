# Geplante Kollisionen im synthetischen Korpus

Dieses Verzeichnis wird von `graph-rag generate-corpus` erzeugt. Der Korpus ist absichtlich
feindlich gegenüber flacher Retrieval-Logik gebaut. Was geplant wurde und wo es liegt:

## 1. Gleicher Projektname bei zwei verschiedenen Kunden

`Projekt Atlas` existiert bei **Nordlicht Energie AG** und bei
**Brandt & Söhne Maschinenbau GmbH** — in allen Jahren, mit völlig unterschiedlichem Inhalt
(Netzleittechnik gegenüber Produktdatenmanagement).

## 2. Die Brücke zwischen beiden

Dr. Katharina Vogt leitet beide Projekte. Die bei Nordlicht Energie AG entstandene
**Atlas-Referenzarchitektur** (Fassung 2) wird bei Brandt & Söhne Maschinenbau GmbH als methodische Grundlage
wiederverwendet; übernommen wurden Schnittstellenverträge, Namenskonvention der
Ereignistypen und das Fehlerbehandlungsmuster über Apache Kafka.

**Kein Dokument nennt den jeweils anderen Kunden.** Die Verbindung existiert nur über die
gemeinsame Person und das gemeinsame Artefakt — also nur im Graphen. Ein einzelner Chunk
enthält sie nie. Genau das soll die Demo zeigen.

Zu finden in: `*/{Jahr}/Projekt Atlas/projektbericht-*.pdf`, Abschnitt
„Methodisches Vorgehen".

## 3. Zweite, schwächere Brücke

`Projekt Datenschild` existiert bei **Sanitas Klinikverbund gGmbH** und
**Finanzkontor Weser AG**. Milan Horvat trägt das
**Prüfschema Berechtigungsrollen** vom einen zum anderen. Nützlich, um zu zeigen, dass der erste
Treffer kein Einzelfall ist.

## 4. Nahezu identischer Boilerplate

Jeder Statusbericht benutzt dieselben Überschriften und dieselben Satzgerüste; es ändern
sich nur Namen und Zahlen. Zwei Statusberichte pro Projekt und Jahr. Dense Retrieval
verwechselt sie zuverlässig.

## Fragen für die Demo

- „Was verbindet das Projekt Atlas von Nordlicht Energie AG mit dem Projekt Atlas von
  Brandt & Söhne Maschinenbau GmbH?" — die Zielfrage.
- „Woran hat Dr. Katharina Vogt gearbeitet?" — zeigt den kundenübergreifenden Knoten.
- „Wie war der Status von Projekt Atlas in Q2?" — naiv formuliert, trifft flach mit hoher
  Wahrscheinlichkeit den falschen Kunden.

---

# Zweiter Teilkorpus: Maschinen

Ablage: `<Hersteller>/<Maschinentyp>/<Modell>/` — eine Ebene tiefer als der Kundenteil.
Je Modell liegen drei Dokumente: `datenblatt-*.pdf`, `ausstattungsvarianten-*.pdf` und
`wartungsplan-*.pdf`.

Der Kundenteil ist feindlich, weil **Namen kollidieren**. Der Maschinenteil ist feindlich,
weil **Namen nichts bedeuten**: `XTC27`, `B2BC`, `HX-114` sind
undurchsichtige Codes. Die Einbettung einer Frage liegt nirgends in der Nähe des richtigen
Dokuments, und ein lexikalischer Treffer auf `B2BC` ist wertlos, weil
derselbe Code zwei nicht verwandte Maschinen bezeichnet. Nur die Struktur — Hersteller,
Maschinentyp, Modell, Variante — macht den Code eindeutig.

## 5. Gleicher Modellcode unter zwei Maschinentypen

`B2BC` ist ein Modell des XTC27 (Hagedorn Verfahrenstechnik GmbH,
Vakuum-Mischtrockner, 1480 kg Leergewicht) **und** ein Modell des ZM3K
(Pleickner Werke KG, Zahnkranz-Montagezelle, 960 kg). Eine Frage nach
„Modell B2BC" ohne Baureihe ist nicht beantwortbar — die Differenz der
Leergewichte macht eine Antwort aus dem falschen Zweig sofort erkennbar.

## 6. Gleicher Maschinentyp bei zwei Herstellern

`XTC27` ist bei Hagedorn Verfahrenstechnik GmbH ein Vakuum-Mischtrockner und bei
Ostmark Fördertechnik AG ein Rollenbahn-Umsetzer. Zwei Ordner gleichen Namens unter
verschiedenen Herstellern, inhaltlich ohne jede Berührung.

## 7. Gleicher Optionscode, andere Bedeutung

`HK` heißt bei Hagedorn Verfahrenstechnik GmbH **Heizkreis**
(+130 kg) und bei Pleickner Werke KG sowie
Ostmark Fördertechnik AG **Hubkonsole** (+74 kg bzw. +96 kg).
Aufgelöst wird der Code nur im Dokument `ausstattungsvarianten-*.pdf` des jeweiligen
Modells.

## 8. Dritte Brücke: eine Baugruppe über zwei Hersteller

Das **Planetengetriebe HX-114** steckt im XTC27-B2BC von Hagedorn Verfahrenstechnik GmbH
und im QH4L-P6V von Ostmark Fördertechnik AG. Dipl.-Ing. Reinhard Oswald hat es
konstruiert und in die zweite Maschine übernommen. **Kein Dokument nennt den jeweils
anderen Hersteller** — die Verbindung existiert nur über Baugruppe und Person, also nur im
Graphen. Zu finden im Abschnitt „Konstruktive Herkunft der Baugruppen".

## 9. Vierte, schwächere Brücke

Der **Spannsatz KN-30** verbindet den NB7X-D1F von
Pleickner Werke KG mit dem VR88D-K4R von
Hagedorn Verfahrenstechnik GmbH; getragen von Beate Lindqvist.

## 10. Varianten: die Antwort steht nur in einer Zeile

Leergewicht und Gesamthöhe einer Variante ergeben sich **additiv** aus der
Grundausführung und den Optionen. Beispiel: XTC27-B2BC-FW/OW wiegt
1480 kg + 145 kg = **1625 kg** bei einer Gesamthöhe von
2435 mm. Dieser Wert steht in genau einer Tabellenzeile eines einzigen
Dokuments; jede Antwort aus dem Nachbarmodell oder aus der Grundausführung ist um einen
prüfbaren Betrag falsch.

## Fragen für die Demo (Maschinen)

- „Was wiegt ein XTC27 B2BC mit Fahrwerk und
  Ölauffangwanne?" — verlangt Hersteller, Baureihe, Modell und Variante; flach
  beantwortet landet man im ZM3K.
- „Welche Maschinen verwenden das Planetengetriebe HX-114?" — die herstellerübergreifende
  Brücke.
- „Was bedeutet die Option HK?" — ohne Zweig nicht beantwortbar.
- „Welche Modelle gibt es zur Baureihe XTC27?" — zwei Hersteller, zwei
  völlig verschiedene Antwortmengen.
