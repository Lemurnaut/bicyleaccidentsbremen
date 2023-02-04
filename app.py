import streamlit as st
import pandas as pd

import plot
import doc as appdocumentation


# read csv create a dataframe and drop one unused column
url = 'https://raw.githubusercontent.com/Lemurnaut/dibrut/main/data/unfalldaten_2016_2021_localized.csv'

data = pd.read_csv(url, sep=',', header=0, encoding='utf-8-sig')
data = data.drop(columns={'Unnamed: 0'})


def sidebar_year_range_slider():
    year_list = ['2016', '2017', '2018', '2019', '2020', '2021']
    start_year, end_year = st.sidebar.select_slider('Jahr', year_list, value=(year_list[0], year_list[-1]))
    return start_year, end_year


def sidebar_month_range_slider():
    month_dict = {'Januar': 1, 'Februar': 2, 'März': 3, 'April': 4, 'Mai': 5, 'Juni': 6, 'Juli': 7, 'August': 8,
                  'September': 9, 'Oktober': 10, 'November': 11, 'Dezember': 12}
    start_month, end_month = st.sidebar.select_slider('Monat', list(month_dict.keys()), value=('Januar', 'Dezember'))
    return start_month, end_month


def sidebar_weekday_range_slider():
    weekday_dict = {'Sonntag': 1, 'Montag': 2, 'Dienstag': 3, 'Mittwoch': 4, 'Donnerstag': 5, 'Freitag': 6,
                    'Samstag': 7}
    start_weekday, end_weekday = st.sidebar.select_slider('Wochentag', list(weekday_dict.keys()),
                                                          value=('Sonntag', 'Samstag'))
    return start_weekday, end_weekday


def sidebar_hour_range_slider():
    hour_list = list(range(0, 24))
    start_hour, end_hour = st.sidebar.select_slider('Stunde', hour_list, value=(0, 23))
    return start_hour, end_hour


def sidebar_accident_category_dropdown():
    accident_categorie_list = ['Alle Unfallkategorien', 'Unfall mit Getöteten', 'Unfall mit Schwerverletzten',
                               'Unfall mit Leichtverletzten']
    selected_accident_cat = st.sidebar.selectbox('Unfallkategorie', accident_categorie_list)
    return selected_accident_cat


def sidebar_accident_type_dropdown():
    accident_type_dict = {'Alle Unfallarten': 11,
                          'Zusammenstoß mit anfahrendem/ anhaltendem/ruhendem Fahrzeug': 1,
                          'Zusammenstoß mit vorausfahrendem / wartendem Fahrzeug': 2,
                          'Zusammenstoß mit seitlich in gleicher Richtung fahrendem Fahrzeug': 3,
                          'Zusammenstoß mit entgegenkommendem Fahrzeug': 4,
                          'Zusammenstoß mit einbiegendem / kreuzendem Fahrzeug': 5,
                          'Zusammenstoß zwischen Fahrzeug und Fußgänger': 6,
                          'Aufprall auf Fahrbahnhindernis': 7,
                          'Abkommen von Fahrbahn nach rechts': 8,
                          'Abkommen von Fahrbahn nach links': 9,
                          'Unfall anderer Art': 0}
    selected_accident_type = st.sidebar.selectbox('Unfallart', list(accident_type_dict.keys()))
    return selected_accident_type


def sidebar_accident_type_1_dropdown():
    accident_type_1_dict = {'Alle Unfalltypen': 0,
                            'Fahrunfall': 1,
                            'Abbiegeunfall': 2,
                            'Einbiegen / Kreuzen-Unfall': 3,
                            'Überschreiten-Unfall': 4,
                            'Unfall durch ruhenden Verkehr': 5,
                            'Unfall im Längsverkehr': 6,
                            'sonstiger Unfall': 7}
    selected_accident_type_1 = st.sidebar.selectbox('Unfalltyp', list(accident_type_1_dict.keys()))
    return selected_accident_type_1


def sidebar_daylight_dropdown():
    daylight_dict = {'Alle Lichtverhältnisse': 3, 'Tageslicht': 0, 'Dämmerung': 1, 'Dunkelheit': 2}
    selected_daylight = st.sidebar.selectbox('Lichtverhältnisse', list(daylight_dict.keys()))
    daylight_type = daylight_dict[selected_daylight]
    return daylight_type


def sidebar_accident_party_dropdown():
    accident_party_dict = {'Alle': 'Alle',
                           'Fußgänger*in': 'IstFuss',
                           'Pkw': 'IstPKW',
                           'Fahrrad': 'IstRad',
                           'Krad': 'IstKrad',
                           'Lkw': 'IstGkfz',
                           'Sonstige': 'IstSonstig'
                           }

    selected_accident_party = st.sidebar.selectbox('Unfallbeteiligung (Fahrrad und Auswahl)', list(accident_party_dict.keys()))
    accident_party_select = accident_party_dict[selected_accident_party]
    return accident_party_select


def dropdown_city(data):
    '''
    creates a dropdown menu on top of the screen to select a city (HB/BHV)
    filter the raw dataframe by selecrd city


    :param dataframe:
    :return: dataframe:
    '''
    selected_city = col1.selectbox('Stadt', ['Bremen', 'Bremerhaven'])

    if selected_city == 'Bremen':
        dataframe = data.loc[data['UKREIS'] == 11]
    if selected_city == 'Bremerhaven':
        dataframe = data.loc[data['UKREIS'] == 12]
    return dataframe, selected_city


def dropdown_district(dataframe):
    '''
    creates a dropdown menu on top of the screen to select a district ('Mitte', 'Neustadt', etc.)
    filter by selected district

    :param dataframe:
    :return: dataframe:
    '''
    district_list = dataframe['Stadtteil'].unique().tolist()
    district_list.insert(0, 'Alle Stadtteile')
    selected_district = col2.selectbox('Stadtteil', district_list)
    return selected_district


def dropdown_local_district(dataframe, selected_district):
    '''
    creates a dropdown menu on top of the screen to select a local district ('Radio Bremen, etc.)
    filter by selected local district
    :param dataframe:
    :param selected_district:
    :return:
    '''
    dataframe = dataframe.loc[dataframe['Stadtteil'] == selected_district]
    local_district_list = dataframe['Ortsteil'].unique().tolist()
    local_district_list.insert(0, 'Alle Ortsteile')
    selected_local_district = col3.selectbox('Ortsteil', local_district_list)
    return selected_local_district


def years(dataframe, start_year, end_year):
    '''
    - function to select one or more years and filter dataframe by selected month
    - creates a selectslider in sidebar for year input
    - takes a dataframe
    - filter dataframe column 'UJAHR' by input value
    - returns filtered dataframe
    '''
    year_list = dataframe['UJAHR'].unique().astype(str).tolist()

    selected_years = list(map(int, year_list[year_list.index(start_year):year_list.index(end_year) + 1]))

    dataframe = dataframe[dataframe['UJAHR'].isin(selected_years)]

    return dataframe


def months(dataframe, start_month, end_month):
    '''
    - function to select one or more month and filter dataframe by selected months
    - creates a selectslider in sidebar for month input
    - takes a dataframe
    - filter dataframe column 'UMONAT' by input value
    - returns filtered dataframe
    '''
    month_dict = {'Januar': 1, 'Februar': 2, 'März': 3, 'April': 4, 'Mai': 5, 'Juni': 6, 'Juli': 7, 'August': 8,
                  'September': 9, 'Oktober': 10, 'November': 11, 'Dezember': 12}

    selected_months = list(range(month_dict[start_month], month_dict[end_month] + 1))

    dataframe = dataframe[dataframe['UMONAT'].isin(selected_months)]

    return dataframe


def weekdays(dataframe, start_weekday, end_weekday):
    '''
    - function to select one or more weekday and filter dataframe by selected days
    - creates a selectslider in sidebar for weekday input
    - takes a dataframe
    - filter dataframe column 'UWOCHENTAG' by input value
    - returns filtered dataframe
    '''
    weekday_dict = {'Sonntag': 1, 'Montag': 2, 'Dienstag': 3, 'Mittwoch': 4, 'Donnerstag': 5, 'Freitag': 6,
                    'Samstag': 7}

    selected_weekdays = list(range(weekday_dict[start_weekday], weekday_dict[end_weekday] + 1))

    dataframe = dataframe[dataframe['UWOCHENTAG'].isin(selected_weekdays)]

    return dataframe


def hours(dataframe, start_hour, end_hour):
    '''
    - function to select one or a range of hours and filter dataframe by selected hours
    - creates a selectslider in sidebar for input
    - takes a dataframe
    - filter dataframe column 'USTUNDE' by input value
    - returns filtered dataframe
    '''
    hour_list = list(range(0, 24))

    selected_hours = hour_list[hour_list.index(start_hour):hour_list.index(end_hour) + 1]

    dataframe = dataframe[dataframe['USTUNDE'].isin(selected_hours)]

    return dataframe


def accident_categorie(dataframe, selected_accident_cat):
    '''
    - function to select a accident categorie
    - creates a sidebar dropdown for input
    - takes dataframe
    - filter dataframe column 'UKATEGORIE' by input value
    - returns filtered dataframe
    '''
    if selected_accident_cat == 'Unfall mit Getöteten':
        dataframe = dataframe.loc[dataframe['UKATEGORIE'] == 1]
    elif selected_accident_cat == 'Unfall mit Schwerverletzten':
        dataframe = dataframe.loc[dataframe['UKATEGORIE'] == 2]
    elif selected_accident_cat == 'Unfall mit Leichtverletzten':
        dataframe = dataframe.loc[dataframe['UKATEGORIE'] == 3]
    elif selected_accident_cat == 'Alle Unfallkategorien':
        pass
    return dataframe


def accident_type(dataframe, selected_accident_type):
    '''
    - function to select a accident type
    - creates a sidebar dropdown for input
    - takes dataframe
    - filter dataframe column 'UART' by input value
    - returns filtered dataframe
    '''
    accident_type_dict = {'Alle Unfallarten': 11,
                          'Zusammenstoß mit anfahrendem/ anhaltendem/ruhendem Fahrzeug': 1,
                          'Zusammenstoß mit vorausfahrendem / wartendem Fahrzeug': 2,
                          'Zusammenstoß mit seitlich in gleicher Richtung fahrendem Fahrzeug': 3,
                          'Zusammenstoß mit entgegenkommendem Fahrzeug': 4,
                          'Zusammenstoß mit einbiegendem / kreuzendem Fahrzeug': 5,
                          'Zusammenstoß zwischen Fahrzeug und Fußgänger': 6,
                          'Aufprall auf Fahrbahnhindernis': 7,
                          'Abkommen von Fahrbahn nach rechts': 8,
                          'Abkommen von Fahrbahn nach links': 9,
                          'Unfall anderer Art': 0}

    accident_type = accident_type_dict[selected_accident_type]

    if accident_type != 11:
        dataframe = dataframe.loc[dataframe['UART'] == accident_type]
    elif accident_type == 11:
        pass

    return dataframe


def accident_type_1(dataframe, selected_accident_type_1):
    '''
    - funtion to select one accident type (detail)
    - creates a sidebar dropdown for input
    - takes dataframe
    - filter dataframe column 'UTYP' by input value
    - returns dataframe
    '''
    accident_type_1_dict = {'Alle Unfalltypen': 0,
                            'Fahrunfall': 1,
                            'Abbiegeunfall': 2,
                            'Einbiegen / Kreuzen-Unfall': 3,
                            'Überschreiten-Unfall': 4,
                            'Unfall durch ruhenden Verkehr': 5,
                            'Unfall im Längsverkehr': 6,
                            'sonstiger Unfall': 7}

    accident_type_1 = accident_type_1_dict[selected_accident_type_1]

    if accident_type_1 != 0:
        dataframe = dataframe.loc[dataframe['UTYP1'] == accident_type_1]
    elif accident_type_1 == 0:
        pass

    return dataframe


def daylight(dataframe, daylight_type):
    '''
    - function to select kind of daylight
    - creates sidebar dropdown for input
    - takes dataframe
    - filter dataframe column 'ULICHTVERH' by input value
    - returns dataframe
    '''
    if daylight_type != 3:
        dataframe = dataframe.loc[dataframe['ULICHTVERH'] == daylight_type]
    elif daylight_type == 3:
        pass

    return (dataframe)


def accident_party(dataframe, accident_party_select):
    '''
    - function to select one or more vehicles involved in accident
    - creates sidebar multiselect box for input
    - takes dataframe
    - filter dataframe columns 'IstPkw', 'IstFuss', 'IstGkfz', 'IstSonstig' by input values
    '''
    if accident_party_select == 'Alle':
        dataframe_out = dataframe

    elif accident_party_select == 'IstRad':
        dataframe_out = dataframe.loc[(dataframe['IstRad'] == 1) & (dataframe['IstPKW'] == 0) & (dataframe['IstFuss'] == 0) & (
                                       dataframe['IstKrad'] == 0) & (dataframe['IstGkfz'] == 0) & (
                                       dataframe['IstSonstig'] == 0)]

    else:
        dataframe_out = dataframe.loc[dataframe[accident_party_select] == 1]

    return dataframe_out


def select_district(dataframe, selected_city, selected_district):
    '''
    - function to select one city district
    - creates a dropdown to select city districs
    - takes dataframe
    - dict district_name:shape_of_district
    - returns dataframe with values of selected district
    '''

    if selected_district != 'Alle Stadtteile':
        if selected_city == 'Bremen':
            dataframe = dataframe.loc[dataframe['UKREIS'] == 11]
            dataframe = dataframe.loc[dataframe['Stadtteil'] == selected_district]

        elif selected_city == 'Bremerhaven':
            dataframe = dataframe.loc[dataframe['UKREIS'] == 12]
            dataframe = dataframe.loc[dataframe['Stadtteil'] == selected_district]

    elif selected_district == 'Alle Stadtteile':
        pass
    return dataframe


def select_local_district(dataframe, selected_local_district):
    '''

    '''

    if selected_local_district != 'Alle Ortsteile':
        dataframe = dataframe.loc[dataframe['Ortsteil'] == selected_local_district]

    elif selected_local_district == 'Alle Ortsteile':
        pass
    return dataframe


def get_dataframe(dataframe):
    '''
    - function to filter data by selections from sidebar menu inputs
    - takes dataframe
    - filter dataframe by multiple inputs
    - returns dataframe
    '''
    dataframe = years(dataframe, start_year, end_year)
    dataframe = months(dataframe, start_month, end_month)
    dataframe = weekdays(dataframe, start_weekday, end_weekday)
    dataframe = hours(dataframe, start_hour, end_hour)
    dataframe = accident_categorie(dataframe, selected_accident_cat)
    dataframe = accident_type(dataframe, selected_accident_type)
    dataframe = accident_type_1(dataframe, selected_accident_type_1)
    dataframe = daylight(dataframe, daylight_type)
    dataframe = accident_party(dataframe, accident_party_select)
    dataframe = select_district(dataframe, selected_city, selected_district)
    dataframe = select_local_district(dataframe, selected_local_district)

    return dataframe


def maps(dataframe):
    dataframe = get_dataframe(dataframe)
    fig = plot.map(dataframe)
    st.pydeck_chart(fig, use_container_width=True)

    st.write(f'In der Karte angezeigte Unfälle: {len(dataframe)}')
    with st.expander('Daten'):
        st.dataframe(dataframe)

    st.write('Datenquelle: https://unfallatlas.statistikportal.de/_opendata2022.html, '
             'Lizenz: dl-de/by-2-0 | www.govdata.de/dl-de/by-2-0')


def select_analysis_type():
    '''
    '''

    tab1, tab2, tab3 = st.tabs(['Karte', 'Diagramme', 'Info'])
    with tab1:
        maps(dataframe)
    with tab2:
        diagramme(dataframe)
    with tab3:
        st.header('Info')
        st.write(appdocumentation.info)
        with st.expander('Datenvorverarbeitung'):
            st.markdown(appdocumentation.dataproc)
        with st.expander('Verarbeitung Geodaten'):
            st.write(appdocumentation.geoproc)
        st.write('Datenquelle Unfalldaten: https://unfallatlas.statistikportal.de/_opendata2022.html, '
                 'Lizenz: dl-de/by-2-0 | www.govdata.de/dl-de/by-2-0')


# def if 'Diagramme' is selected from sidebar 'Typ' dropdown
def diagramme(dataframe):
    dataframe = get_dataframe(dataframe)

    st.header('Unfallbeteiligung')
    tab1, tab2 = st.tabs(['Absolut', 'Relativ'])
    with tab1:
        fig, dataframe_to_show = plot.unfallbeteiligung(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)

    with tab2:
        fig, dataframe_to_show = plot.piechart(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)


    st.header('Merkmale und Bedingungen')
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Unfallkategorie',  'Unfallart', 'Unfalltyp', 'Lichtverhältnisse',
                                     'Strassenzustand'])
    with tab1:
        fig, dataframe_to_show = plot.accidentKat(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)

    with tab2:
        fig, dataframe_to_show = plot.accidentArt(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)

    with tab3:
        fig, dataframe_to_show = plot.accidentType(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)

    with tab4:
        fig, dataframe_to_show = plot.light_conditions(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)

    with tab5:
        fig, dataframe_to_show = plot.street_conditions(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)


    if selected_district == 'Alle Stadtteile':
        st.header('Unfälle in Stadtteilen')
        fig, dataframe_to_show = plot.districts(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)

    if selected_local_district == 'Alle Ortsteile':
        st.header('Unfälle in Ortsteilen')
        fig, dataframe_to_show = plot.local_districts(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)

    st.header('Unfälle nach Zeit')
    tab1, tab2, tab3 = st.tabs(['Monat', 'Woche', 'Stunde'])
    with tab1:
        st.write('Unfälle nach Monat')
        fig, dataframe_to_show = plot.month(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)

    with tab2:
        st.write('Unfälle nach Wochentag')
        fig, dataframe_to_show = plot.week(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)
    with tab3:
        st.write('Unfälle nach Stunde')
        fig, dataframe_to_show = plot.hour(dataframe)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True, config={'displaylogo': False})

        with st.expander('Daten'):
            st.dataframe(dataframe_to_show)

    st.write('Datenquelle: https://unfallatlas.statistikportal.de/_opendata2022.html, '
             'Lizenz: dl-de/by-2-0 | www.govdata.de/dl-de/by-2-0')

# 'main'

# streamlit layout
menu_items = {
    'About': '''
    ## Fahrraunfälle in Bremen

    made by stefko

    www.moin-stefko.de
    ''',
    'Get help': 'mailto: hallo@moin-stefko.de',
    }

st.set_page_config(layout="wide",
                   menu_items=menu_items
                   )

st.header('Fahrradunfälle in Bremen')

col1, col2, col3 = st.columns(3)

dataframe, selected_city = dropdown_city(data)

start_year, end_year = sidebar_year_range_slider()
start_month, end_month = sidebar_month_range_slider()
start_weekday, end_weekday = sidebar_weekday_range_slider()
start_hour, end_hour = sidebar_hour_range_slider()

selected_accident_cat = sidebar_accident_category_dropdown()
selected_accident_type = sidebar_accident_type_dropdown()
selected_accident_type_1 = sidebar_accident_type_1_dropdown()
daylight_type = sidebar_daylight_dropdown()
accident_party_select = sidebar_accident_party_dropdown()

selected_district = dropdown_district(dataframe)
selected_local_district = dropdown_local_district(dataframe, selected_district)

select_analysis_type()
