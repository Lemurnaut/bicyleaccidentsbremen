import pydeck
import plotly.express as px
import pandas as pd
import streamlit as st


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

    fig = px.bar(dataframe, x=dataframe.index, y=dataframe[dataframe.columns[0]].values)

    fig.update_layout(xaxis_title='Unfallbeteiligung', yaxis_title='Anzahl', height= 600)

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

    fig = px.bar(tmp_df, x=tmp_df.Ortsteil, y=tmp_df.Anzahl.values, height= 600)

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

    fig = px.bar(by_month_sum, x=by_month_sum.index, y=by_month_sum.columns.values, height= 600)

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

    fig = px.bar(by_week_sum, x=by_week_sum.index, y=by_week_sum.columns.values, height= 600)

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

    fig = px.bar(by_hour_sum, x=by_hour_sum.index, y=by_hour_sum.columns.values, height= 600)

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
                 height= 600)

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

    fig = px.bar(tmp_df, x=tmp_df.Stadtteil, y=tmp_df.Anzahl.values, height= 600)

    fig.update_layout(xaxis_title='Stadtteil', yaxis_title='Anzahl')

    fig.update_traces(hovertemplate='Stadtteil: %{x}<br>' +
                                    'Anzahl: %{y}<br>'
                      )

    return fig, tmp_df

def map(dataframe):
    '''
    '''

    highlight_list = ['Keine', 'Pkw', 'Fußgänger', 'Motorrad', 'Lkw', 'Sonstige']

    to_highlight = st.selectbox('Unfallbeteiligung hervorheben', highlight_list)
    highlight = 0
    if to_highlight == 'Keine':
        highlight = 0
    if to_highlight == 'Pkw':
        highlight = 'IstPKW'
    if to_highlight == 'Fußgänger':
        highlight = 'IstFuss'
    if to_highlight == 'Motorrad':
        highlight = 'IstKrad'
    if to_highlight == 'Lkw':
        highlight = 'IstGkfz'
    if to_highlight == 'Sonstige':
        highlight = 'IstSonstig'


    view = pydeck.data_utils.compute_view(dataframe[['longitude', 'latitude']])
    view.pitch = 55
    view.bearing = 60
    view.zoom = 13
    view.height = 800

    column_layer = pydeck.Layer(
        "ColumnLayer",
        data=dataframe,
        get_position=['longitude', 'latitude'],
        get_elevation=highlight,
        elevation_scale=20,
        radius=10,
        get_color='[255, 25, 255, 100]',
        pickable=True,
        auto_highlight=True,
    )

    scatterplot_layer = pydeck.Layer(
        'ScatterplotLayer',
        dataframe,
        get_position=['longitude', 'latitude'],
        auto_highlight=True,
        get_radius=10,
        get_fill_color='[255, 0, 0, 100]',
        pickable=True)

    tooltip = {
        "html": "<b>Beteiligung</b><br>Pkw: {IstPKW}<br>Fuss: {IstFuss}<br>Krad: {IstKrad}<br>Lkw: {IstGkfz}<br>sonstige: {IstSonstig}<br><b>Kategorie:{UKATEGORIE}</b><br><b>Position</b><br>Lon: {longitude}<br>Lat: {latitude}<br>",
        "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
    }

    map = pydeck.Deck(
        layers=[column_layer, scatterplot_layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_provider='mapbox',
        map_style='mapbox://styles/mapbox/light-v11',
        api_keys={'mapbox': st.secrets['MAPBOX_KEY']},
        #map_style='mapbox://styles/mapbox/navigation-night-v1',
        #api_keys={'mapbox': 'pk.eyJ1Ijoid2FzaTAwNyIsImEiOiJja2lobmFvcjQwOW56MnNtbDc4aTAwcTB5In0.wnT9dpy3BaeDXA7UegN0ng'}
    )

    return map