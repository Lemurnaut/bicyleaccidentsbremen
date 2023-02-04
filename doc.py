info = 'Es werden ausschließlich Unfälle mit Fahrradbeteiligung angezeigt (IstRad = 1). Sofern eine Eingrenzung der ' \
       'Unfallbeteiligung auf Fahrrad über einen Filter möglich ist, werden ausschließlich die Unfälle angezeigt ' \
       'an denen ein oder mehrere Fahrräder ohne Beteiligung Dritter beteiligt waren. ' \
       'Die Daten lassen keine Aussage über die ' \
       'Anzahl der Unfallbeteiligten zu. Die Ortsteile Borgfeld, Blockland, Seehausen und Strom überschneiden sich' \
       'mit keinem Stadtteil, sie werden in der Stadtteilliste unter Unbekannt geführt. Drei Unfälle an der ' \
       'Stuhrer Landstraße und der Varreler Landstraße sind keinem Ortsteil zugeordnet, sie liegen außerhalb der ' \
       'Landesgrenze. Sie werden in der Ortsteilliste unter Unbekannt geführt.'

dataproc = 'Notwendige und durchgeführte Datenvorverarbeitung zur Darstellung in dieser App:  ' \
           'Zusammenfassen der einzelnen csv Dateien (2016-2021) aus der Datenquelle. Filtern der Unfälle' \
           ' nach Bundesland Bremen (ULAND = 4) und ' \
           'ausschließlich Unfällen mit Fahrradbeteiligung (IstRad = 1). Vereinheitlichung von Spaltennamen. ' \
           'Aufarbeitung der Geodaten (Punkt statt Komma, einzelne Spalten für Breiten- und Längengrade). ' \
           'Löschen nicht oder nicht mehr benötigter Daten(OBJECTID, ULAND, UGEMEINDE, UREGBEZ, LINREFX, LINREFY). ' \
           'Einfügen je einer Spalte für Stadtteil und Ortsteil.'

geoproc = 'Zur Darstellung des lokalen Unfallgeschehens bis auf Stadtteil- und Ortsteilebene wurde die ' \
          'Position (Längengrad, Breitengrad) der einzelnen Unfälle mit den Bremer Stadtteil- und Ortsteilgrenzen ' \
          'abgeglichen. Quelle Bremer Verwaltungsgrenzen: [Geoportal Bremen](https://geoportal.bremen.de/geoportal/).'
