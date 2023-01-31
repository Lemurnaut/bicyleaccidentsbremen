info = 'Es werden ausschließlich Unfälle mit Fahrradbeteiligung angezeigt. Sofern eine Eingrenzung der ' \
       'Unfallbeteiligung auf Fahrrad über einen Filter möglich ist, werden die Unfälle angezeigt an denen ' \
       'ausschließlich ein oder mehrere Fahrräder beteiligt waren. Die Daten lassen keine Aussage über die ' \
       'Anzahl der Unfallbeteiligten zu. '

dataproc = 'Notwendige und durchgeführte Datenvorverarbeitung zur Darstellung in dieser App:  ' \
           'Zusammenfassen der einzelnen csv Dateien (2016-2021) aus der Datenquelle. Filter nach Bundesland Bremen ' \
           '(ULAND = 4) und ' \
           'ausschließlich Unfällen mit Fahrradbeteiligung (IstRad = 1). Vereinheitlichung von Spaltennamen. ' \
           'Aufarbeitung der Geodaten (Punkt statt Komma, einzelne Spalten für Breiten- und Längengrade). ' \
           'Löschen nicht oder nicht mehr benötigter Daten(OBJECTID, ULAND, UGEMEINDE, UREGBEZ, LINREFX, LINREFY). ' \
           'Einfügen je einer Spalte für Stadtteil und Ortsteil.'

geoproc = 'Zur Darstellung des lokalen Unfallgeschehens bis auf Stadtteil- und Ortsteilebene wurde die ' \
          'Position (Längengrad, Breitengrad) der einzelnen Unfälle mit den Bremer Stadtteil- und Ortsteilgrenzen ' \
          'abgeglichen. Quelle Bremer Verwaltungsgrenzen: [Geoportal Bremen](https://geoportal.bremen.de/geoportal/).'
