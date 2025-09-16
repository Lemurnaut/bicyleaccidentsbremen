import pydeck
import plotly.express as px
import pandas as pd
import geopandas
import streamlit as st

HEIGHT = 450


def unfallbeteiligung(dataframe):
    data = {'Pkw': len(dataframe.loc[dataframe['IstPKW'] == 1]),
            'Fuss': len(dataframe.loc[dataframe['IstFuss'] == 1]),
            'Krad': len(dataframe.loc[dataframe['IstKrad'] == 1]),
            'GKfz': len(dataframe.loc[dataframe['IstGkfz'] == 1]),
            'Sonstige': len(dataframe.loc[dataframe['IstSonstig'] == 1]),
            'Fahrrad': len(dataframe.loc[(dataframe['IstRad'] == 1) & (dataframe['IstPKW'] == 0) & (dataframe['IstFuss'] == 0) & (
                                       dataframe['IstKrad'] == 0) & (dataframe['IstGkfz'] == 0) & (
                                       dataframe['IstSonstig'] == 0)])
           }

    dataframe = pd.DataFrame.from_dict(data, orient='index')
    dataframe = dataframe.rename(columns={0 : 'Anzahl'})

    fig = px.bar(dataframe, x=dataframe.index, y=dataframe[dataframe.columns[0]].values, text=dataframe[dataframe.columns[0]].values)

    fig.update_layout(xaxis_title='Unfallbeteiligung', yaxis_title='Anzahl', height= HEIGHT)

    fig.update_traces(hovertemplate='Unfallbeteiligung: %{x}<br>' +
                                    'Anzahl: %{y}<br>'
                      )

    return fig, dataframe


def local_districts(dataframe):

    district_names = dataframe['Ortsteil'].unique().tolist()
    tmp_df = pd.DataFrame()

    tmp_df['Ortsteil'] = district_names

    value_list = []

    for name in district_names:
        value_list.append(len(dataframe.loc[dataframe['Ortsteil'] == name]))
    tmp_df['Anzahl'] = value_list
    tmp_df = tmp_df.set_index('Ortsteil')

    fig = px.bar(tmp_df, x=tmp_df.index, y=tmp_df.Anzahl.values, height=HEIGHT, text=tmp_df.Anzahl.values)

    fig.update_layout(xaxis_title='Ortsteil', yaxis_title='Anzahl')

    fig.update_traces(hovertemplate='Stadtteil: %{x}<br>' +
                                    'Anzahl: %{y}<br>'
                      )

    return fig, tmp_df

def month(dataframe):
    by_month_sum = dataframe.groupby(dataframe['UMONAT']).agg(
        {'IstPKW': 'sum', 'IstFuss': 'sum', 'IstKrad': 'sum', 'IstGkfz': 'sum', 'IstSonstig': 'sum'})

    by_month_sum = by_month_sum.rename(columns={
        'IstPKW': 'Pkw', 'IstFuss': 'Fußgänger', 'IstKrad': 'Krad', 'IstGkfz': 'Lkw', 'IstSonstig': 'Sontiges'})

    fig = px.bar(by_month_sum, x=by_month_sum.index, y=by_month_sum.columns.values, height=HEIGHT, text_auto=True)

    fig.update_layout(xaxis_title='Monat',
                      yaxis_title='Anzahl')

    fig.update_traces(hovertemplate='Anzahl: %{y}<br>')

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[2, 4, 6, 8, 10, 12],
            ticktext=['Feb', 'Apr', 'Jun', 'Aug', 'Okt', 'Dez']
        )
    )

    return fig, by_month_sum

def week(dataframe):
    by_week_sum = dataframe.groupby(dataframe['UWOCHENTAG']).agg(
        {'IstPKW': 'sum', 'IstFuss': 'sum', 'IstKrad': 'sum', 'IstGkfz': 'sum', 'IstSonstig': 'sum'})

    by_week_sum = by_week_sum.rename(columns={
        'IstPKW': 'Pkw', 'IstFuss': 'Fußgänger', 'IstKrad': 'Krad', 'IstGkfz': 'Lkw', 'IstSonstig': 'Sonstiges'})

    fig = px.bar(by_week_sum, x=by_week_sum.index, y=by_week_sum.columns.values, height= HEIGHT, text_auto=True)

    fig.update_layout(xaxis_title='Wochentag',
                      yaxis_title='Anzahl')

    fig.update_traces(hovertemplate='Anzahl: %{y}<br>')

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5, 6, 7],
            ticktext=['So', 'Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa']
        )
    )

    return fig, by_week_sum

def hour(dataframe):
    by_hour_sum = dataframe.groupby(dataframe['USTUNDE']).agg(
        {'IstPKW': 'sum', 'IstFuss': 'sum', 'IstKrad': 'sum', 'IstGkfz': 'sum', 'IstSonstig': 'sum'})

    by_hour_sum = by_hour_sum.rename(columns={
        'IstPKW': 'Pkw', 'IstFuss': 'Fußgänger', 'IstKrad': 'Krad', 'IstGkfz': 'Lkw', 'IstSonstig': 'Sonstiges'})

    fig = px.bar(by_hour_sum, x=by_hour_sum.index, y=by_hour_sum.columns.values, height= HEIGHT, text_auto=True)

    fig.update_layout(xaxis_title='Stunde',
                      yaxis_title='Anzahl')

    fig.update_traces(hovertemplate='Anzahl: %{y}<br>')



    return fig, by_hour_sum


def piechart(dataframe):
    data = {'Pkw': len(dataframe.loc[
                           (dataframe['IstRad'] == 1) & (dataframe['IstPKW'] == 1) & (dataframe['IstFuss'] == 0) & (
                                   dataframe['IstKrad'] == 0) & (dataframe['IstGkfz'] == 0) & (
                                   dataframe['IstSonstig'] == 0)]),
            'Fuss': len(dataframe.loc[
                            (dataframe['IstRad'] == 1) & (dataframe['IstPKW'] == 0) & (dataframe['IstFuss'] == 1) & (
                                    dataframe['IstKrad'] == 0) & (dataframe['IstGkfz'] == 0) & (
                                    dataframe['IstSonstig'] == 0)]),
            'Krad': len(dataframe.loc[
                            (dataframe['IstRad'] == 1) & (dataframe['IstPKW'] == 0) & (dataframe['IstFuss'] == 0) & (
                                    dataframe['IstKrad'] == 1) & (dataframe['IstGkfz'] == 0) & (
                                    dataframe['IstSonstig'] == 0)]),
            'GKfz': len(dataframe.loc[
                            (dataframe['IstRad'] == 1) & (dataframe['IstPKW'] == 0) & (dataframe['IstFuss'] == 0) & (
                                    dataframe['IstKrad'] == 0) & (dataframe['IstGkfz'] == 1) & (
                                    dataframe['IstSonstig'] == 0)]),
            'Sonstige': len(dataframe.loc[(dataframe['IstRad'] == 1) & (dataframe['IstPKW'] == 0) & (
                    dataframe['IstFuss'] == 0) & (dataframe['IstKrad'] == 0) & (dataframe['IstGkfz'] == 0) & (
                                                  dataframe['IstSonstig'] == 1)]),
            'Fahrrad': len(dataframe.loc[
                               (dataframe['IstRad'] == 1) & (dataframe['IstPKW'] == 0) & (dataframe['IstFuss'] == 0) & (
                                       dataframe['IstKrad'] == 0) & (dataframe['IstGkfz'] == 0) & (
                                       dataframe['IstSonstig'] == 0)])}

    dataframe = pd.DataFrame.from_dict(data, orient='index')
    dataframe = dataframe.rename(columns={0: 'Anzahl'})

    fig = px.pie(dataframe, names=dataframe.index, values=dataframe[dataframe.columns[0]].values,
                 color_discrete_sequence=px.colors.sequential.Emrld_r, labels={'values': 'Anzahl',
                                                                               'index': 'Unfallbeteiligung'},
                 height= HEIGHT)

    fig.update_traces(textposition='inside', textinfo='percent+label')

    return fig, dataframe


def districts(dataframe):
    district_names = dataframe['Stadtteil'].unique().tolist()
    tmp_df = pd.DataFrame()

    tmp_df['Stadtteil'] = district_names

    value_list = []

    for name in district_names:
        value_list.append(len(dataframe.loc[dataframe['Stadtteil'] == name]))
    tmp_df['Anzahl'] = value_list
    tmp_df = tmp_df.set_index('Stadtteil')

    fig = px.bar(tmp_df, x=tmp_df.index, y=tmp_df.Anzahl.values, height= HEIGHT, text=tmp_df.Anzahl.values)

    fig.update_layout(xaxis_title='Stadtteil', yaxis_title='Anzahl')

    fig.update_traces(hovertemplate='Stadtteil: %{x}<br>' +
                                    'Anzahl: %{y}<br>'
                      )

    return fig, tmp_df


def accidentKat(dataframe):
    kat_data = {'Unfall mit Getöteten': len(dataframe.loc[dataframe['UKATEGORIE'] == 1]),
                'Unfall mit Schwerverletzten': len(dataframe.loc[dataframe['UKATEGORIE'] == 2]),
                'Unfall mit Leichtverletzten': len(dataframe.loc[dataframe['UKATEGORIE'] == 3]),
               }

    dataframe = pd.DataFrame.from_dict(kat_data, orient='index')
    dataframe = dataframe.rename(columns={0 : 'Anzahl'})

    fig = px.bar(dataframe, x=dataframe[dataframe.columns[0]].values, y=dataframe.index,
                 text=dataframe[dataframe.columns[0]].values)

    fig.update_layout(xaxis_title='Unfallbeteiligung',
                      yaxis_title='Anzahl',
                      height=HEIGHT,
                      )

    fig.update_traces(hovertemplate='Unfallverletzungen: %{x}<br>' +
                                        'Anzahl: %{y}<br>'
                     )
    return fig, dataframe


def accidentArt(dataframe):
    art_dict = {1: 'Zusammenstoß mit anfahrendem/ anhaltendem/ruhendem Fahrzeug',
                2: 'Zusammenstoß mit vorausfahrendem / wartendem Fahrzeug',
                3: 'Zusammenstoß mit seitlich in gleicher Richtung fahrendem Fahrzeug',
                4: 'Zusammenstoß mit entgegenkommendem Fahrzeug',
                5: 'Zusammenstoß mit einbiegendem / kreuzendem Fahrzeug',
                6: 'Zusammenstoß zwischen Fahrzeug und Fußgänger',
                7: 'Aufprall auf Fahrbahnhindernis',
                8: 'Abkommen von Fahrbahn nach rechts',
                9: 'Abkommen von Fahrbahn nach links',
                0: 'Unfall anderer Art'
                }

    art_data = {(art_dict[key], len(dataframe.loc[dataframe['UART'] == key])) for key in art_dict}

    dataframe = pd.DataFrame(art_data)
    dataframe = dataframe.rename(columns={0 : 'Merkmal', 1 : 'Anzahl'})
    dataframe = dataframe.set_index('Merkmal')

    fig = px.bar(dataframe, x=dataframe.Anzahl, y=dataframe.index, text=dataframe.Anzahl)

    fig.update_layout(xaxis_title='Anzahl', yaxis_title='Merkmal', height= HEIGHT)

    fig.update_traces(hovertemplate='%{y}<br>' +
                                        'Anzahl: %{x}<br>'
                     )
    return fig, dataframe


def accidentType(dataframe):
    typ_dict = {1: 'Fahrunfall',
                2: 'Abbiegeunfall',
                3: 'Einbiegen / Kreuzen-Unfall',
                4: 'Überschreiten-Unfall',
                5: 'Unfall durch ruhenden Verkehr',
                6: 'Unfall im Längsverkehr',
                7: 'sonstiger Unfall'
                }

    typ_data = {(typ_dict[key], len(dataframe.loc[dataframe['UTYP1'] == key])) for key in typ_dict}

    dataframe = pd.DataFrame(typ_data)
    dataframe = dataframe.rename(columns={0 : 'Merkmal', 1 : 'Anzahl'})
    dataframe = dataframe.set_index('Merkmal')

    fig = px.bar(dataframe, x=dataframe.Anzahl, y=dataframe.index, text=dataframe.Anzahl)

    fig.update_layout(xaxis_title='Anzahl', yaxis_title='Merkmal', height= HEIGHT)

    fig.update_traces(hovertemplate='%{y}<br>' +
                                        'Anzahl: %{x}<br>'
                     )
    return fig, dataframe


def light_conditions(dataframe):
    light_dict = {0: 'Tageslicht',
                  1: 'Dämmerung',
                  2: 'Dunkelheit'
                  }

    light_data = {(light_dict[key], len(dataframe.loc[dataframe['ULICHTVERH'] == key])) for key in light_dict}

    dataframe = pd.DataFrame(light_data)
    dataframe = dataframe.rename(columns={0 : 'Merkmal', 1 : 'Anzahl'})
    dataframe = dataframe.set_index('Merkmal')

    fig = px.bar(dataframe, x=dataframe.Anzahl, y=dataframe.index, text=dataframe.Anzahl)

    fig.update_layout(xaxis_title='Anzahl', yaxis_title='Merkmal', height= HEIGHT)

    fig.update_traces(hovertemplate='%{y}<br>' +
                                        'Anzahl: %{x}<br>'
                     )
    return fig, dataframe


def street_conditions(dataframe):
    condition_dict = {0: 'trocken',
                      1: 'nass/feucht/schlüpfrig',
                      2: 'winterglatt'
                      }

    condition_data = {(condition_dict[key], len(dataframe.loc[dataframe['STRZUSTAND'] == key])) for key in condition_dict}

    dataframe = pd.DataFrame(condition_data)
    dataframe = dataframe.rename(columns={0 : 'Merkmal', 1 : 'Anzahl'})
    dataframe = dataframe.set_index('Merkmal')

    fig = px.bar(dataframe, x=dataframe.Anzahl, y=dataframe.index, text=dataframe.Anzahl)

    fig.update_layout(xaxis_title='Anzahl', yaxis_title='Merkmal', height= HEIGHT)

    fig.update_traces(hovertemplate='%{y}<br>' +
                                        'Anzahl: %{x}<br>'
                     )
    return fig, dataframe


def get_bikeroutes():
    url = 'https://raw.githubusercontent.com/Lemurnaut/BicyleAccidentsBremen/main/data/HB_Hauptrouten.geojson'
    dataframe = geopandas.read_file(url)
    # https://gis.stackexchange.com/questions/433460/extracting-latitude-and-longitude-pairs-as-list-from-linestring-in-geopandas
    dataframe['coords'] = dataframe.geometry.apply(lambda geom: list(geom.coords))
    return dataframe


def get_kitas():
    url = 'https://raw.githubusercontent.com/Lemurnaut/BicyleAccidentsBremen/main/data/kitas_bre.geojson'
    dataframe = geopandas.read_file(url)
    # https://gis.stackexchange.com/questions/433460/extracting-latitude-and-longitude-pairs-as-list-from-linestring-in-geopandas
    dataframe = dataframe.to_crs(4326)
    dataframe['lat'] = dataframe['geometry'].y
    dataframe['lon'] = dataframe['geometry'].x
    return dataframe


def get_schools():
    url = 'https://raw.githubusercontent.com/Lemurnaut/BicyleAccidentsBremen/main/data/schools_bre.geojson'
    dataframe = geopandas.read_file(url)
    # https://gis.stackexchange.com/questions/433460/extracting-latitude-and-longitude-pairs-as-list-from-linestring-in-geopandas
    dataframe = dataframe.to_crs(4326)
    dataframe['lat'] = dataframe['geometry'].y
    dataframe['lon'] = dataframe['geometry'].x
    return dataframe


def map(dataframe):
    '''
    '''
    # load additional infos
    bikeroutes = get_bikeroutes()
    kitas = get_kitas()
    schools = get_schools()

    # select vehicle to highlight in scatter plot
    highlight_list = ['Keine', 'Pkw', 'Fußgänger', 'Motorrad', 'Lkw', 'Sonstige']

    to_highlight = st.radio('Unfallbeteiligung hervorheben', highlight_list, horizontal=True)

    highlight_dict = {'Keine' : 0,
                      'Pkw' : 'IstPKW',
                      'Fußgänger' : 'IstFuss',
                      'Motorrad' : 'IstKrad',
                      'Lkw' : 'IstGkfz',
                      'Sonstige' : 'IstSonstig'
                      }
    highlight = highlight_dict[to_highlight]

    # making colors
    dataframe['color'] = dataframe.apply(lambda x: [10, 0, 180, 100], axis=1)

    if to_highlight != 'Keine':
        mask = dataframe[highlight] == 1
        dataframe.loc[mask, 'color'] = dataframe.apply(lambda x: [255, 0, 0, 100], axis=1)

    # building legend for mousover info by translating numerical values to human readable stuff
    kat_dict = {1: 'Unfall mit Getöteten',
                2: 'Unfall mit Schwerverletzten',
                3: 'Unfall mit Leichtverletzten'
                }

    dataframe['Kategorie'] = dataframe.apply(lambda row: kat_dict.get(row['UKATEGORIE']), axis=1)

    art_dict = {1: 'Zusammenstoß mit anfahrendem/ anhaltendem/ruhendem Fahrzeug',
                2: 'Zusammenstoß mit vorausfahrendem / wartendem Fahrzeug',
                3: 'Zusammenstoß mit seitlich in gleicher Richtung fahrendem Fahrzeug',
                4: 'Zusammenstoß mit entgegenkommendem Fahrzeug',
                5: 'Zusammenstoß mit einbiegendem / kreuzendem Fahrzeug',
                6: 'Zusammenstoß zwischen Fahrzeug und Fußgänger',
                7: 'Aufprall auf Fahrbahnhindernis',
                8: 'Abkommen von Fahrbahn nach rechts',
                9: 'Abkommen von Fahrbahn nach links',
                0: 'Unfall anderer Art'
                }
    dataframe['Art'] = dataframe.apply(lambda row: art_dict.get(row['UART']), axis=1)

    typ_dict = {1: 'Fahrunfall',
                2: 'Abbiegeunfall',
                3: 'Einbiegen / Kreuzen-Unfall',
                4: 'Überschreiten-Unfall',
                5: 'Unfall durch ruhenden Verkehr',
                6: 'Unfall im Längsverkehr',
                7: 'sonstiger Unfall'
                }
    dataframe['Typ'] = dataframe.apply(lambda row: typ_dict.get(row['UTYP1']), axis=1)

    light_dict = {0: 'Tageslicht',
                        1: 'Dämmerung',
                        2: 'Dunkelheit'
                        }
    dataframe['Licht'] = dataframe.apply(lambda row: light_dict.get(row['ULICHTVERH']), axis=1)

    condition_dict = {0: 'trocken',
                        1: 'nass/feucht/schlüpfrig',
                        2: 'winterglatt'
                        }
    dataframe['Strassenzustand'] = dataframe.apply(lambda row: condition_dict.get(row['STRZUSTAND']), axis=1)

    bool_dict = {0: 'Nein',
                        1: 'Ja'
                        }

    dataframe['Pkw'] = dataframe.apply(lambda row: bool_dict.get(row['IstPKW']), axis=1)
    dataframe['Fussgaenger'] = dataframe.apply(lambda row: bool_dict.get(row['IstFuss']), axis=1)
    dataframe['Krad'] = dataframe.apply(lambda row: bool_dict.get(row['IstKrad']), axis=1)
    dataframe['Lkw'] = dataframe.apply(lambda row: bool_dict.get(row['IstGkfz']), axis=1)
    dataframe['Sonstige'] = dataframe.apply(lambda row: bool_dict.get(row['IstSonstig']), axis=1)

    # map
    view = pydeck.data_utils.compute_view(dataframe[['longitude', 'latitude']])
    view.pitch = 55
    view.bearing = 60
    view.zoom = 13
    view.height = 800

    scatterplot_layer = pydeck.Layer(
        'ScatterplotLayer',
        dataframe,
        get_position=['longitude', 'latitude'],
        auto_highlight=True,
        get_radius=10,
        get_fill_color='color',
        pickable=True)
        
    bikeroutes_layer = pydeck.Layer(
        type="PathLayer",
        data=bikeroutes,
        pickable=False,
        get_color='[125, 125, 0, 100]',
        width_scale=5,
        width_min_pixels=2,
        get_path="coords",
        get_width=2,
        )
    
    kitas_layer = pydeck.Layer(
        'ScatterplotLayer',
        kitas,
        get_position=['lon', 'lat'],
        auto_highlight=True,
        get_radius=50,
        get_fill_color='[10, 0, 180, 100]',
        pickable=False)
    
    schools_layer = pydeck.Layer(
        'ScatterplotLayer',
        schools,
        get_position=['lon', 'lat'],
        auto_highlight=True,
        get_radius=50,
        get_fill_color='[10, 0, 180, 100]',
        pickable=False)
    
    # streamlit additional info select
    st_layer_list = ['Keine', 'Hauptrouten', 'Kitas', 'Schulen']

    layer_select = st.radio('zusätzliche Infos', st_layer_list, horizontal=True)

    layer_dict = {'Keine' :  '',
                      'Hauptrouten' : bikeroutes_layer,
                      'Kitas' : kitas_layer,
                      'Schulen' : schools_layer,
                      }
    
    add_layer = layer_dict[layer_select]
    
    layer_list = [scatterplot_layer]

    layer_list.append(add_layer)
    

    tooltip = {
                'html': '<b>Zeit: {UMONAT}/{UJAHR}</b>'
                        '<br><b>Beteiligung</b>'
                        '<br>Pkw: {Pkw}'
                        '<br>Fuss: {Fussgaenger}'
                        '<br>Krad: {Krad}'
                        '<br>Lkw: {Lkw}'
                        '<br>Sonstige: {Sonstige}'
                        '<br>'
                        '<br><b>Kategorie: {Kategorie}</b>'
                        '<br><b>Unfallart: {Art}</b>'
                        '<br><b>Unfalltyp: {Typ}</b>'
                        '<br>'
                        '<br><b>Lichtverhältnisse: {Licht}</b>'
                        '<br><b>Strassenzustand: {Strassenzustand}</b>'
                        '<br>'
                        '<br><b>Position</b>'
                        '<br>Lon: {longitude}'
                        '<br>Lat: {latitude}<br>',
                'style': {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
        }

    map = pydeck.Deck(
                layers=layer_list,
                initial_view_state=view,
                tooltip=tooltip,
                map_style='light'
                #map_provider='mapbox',
                #map_style='mapbox://styles/mapbox/light-v11',
                #api_keys={'mapbox': st.secrets['MAPBOX_KEY']},
                #map_style='mapbox://styles/mapbox/navigation-night-v1',
        )
    return map
