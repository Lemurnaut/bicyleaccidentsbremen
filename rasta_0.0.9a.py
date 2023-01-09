import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd

version = '0.0.9_alpha'

# setup streamlit layout
st.set_page_config(layout="wide")
col1, col2 = st.columns(2)


@st.cache
def download_data():
    stationID = ['100044202',
                 '100000575',
                 '100002926',
                 '100002927',
                 '100002930',
                 '100002931',
                 '100002932',
                 '100002933',
                 '100002934',
                 '100002935',
                 '100047799',
                 '100047805',
                 ]
    today = dt.date.today()

    data_startdate = f'2012-01-01'
    data_enddate = str(dt.date.today() - dt.timedelta(1))

    source = f'https://vmz.bremen.de/radzaehler-api/?action=Values&apiFormat=csv&resolution=hourly&startDate={data_startdate}&endDate={data_enddate}'

    IDs = []
    for x in stationID:
        IDs.append('&stationId%5B%5D=' + x)

    IDs = ''.join(IDs)

    URL = source + IDs

    dataframe = pd.read_csv(URL, header=0, sep=';')
    #dataframe = dataframe.drop(columns={dataframe.columns[0]})
    dataframe = dataframe.rename(columns={dataframe.columns[0]: "Datetime"})

    # fix to avoid streamlit timezone/ambiguous errors
    dataframe['Datetime'] = pd.DatetimeIndex(dataframe.Datetime)
    dataframe['Datetime'] = dataframe['Datetime'].dt.tz_localize('UTC')

    dataframe.set_index(dataframe.Datetime, inplace=True)
    dataframe = dataframe.drop(columns={dataframe.columns[0]})
    dataframe.set_index(dataframe.index.floor('H'), inplace=True)
    dataframe = dataframe.sort_index()

    return dataframe

dataframe_source = download_data()

st.write(dataframe_source)


@st.cache
def proceed_data(*,remove_outliers: bool, fill_outliers_with_mean: bool, fill_missing_index: bool, fill_missing_values_with_mean: bool):
    import pandas as pd
    

    column_names = dataframe_source.columns.tolist()
    dataframe_list = []
    for i in column_names:
        dataframe_list.append(pd.DataFrame(dataframe_source[i]))

    dataframe_proceed = []
    for dataframe in dataframe_list:

        dataframe = dataframe.truncate(before=(dataframe.gt(0).idxmax()[0]), copy=True)
        
        if remove_outliers == True:
            from scipy.stats import zscore
            z_score = zscore(dataframe, nan_policy='omit')
            filtered_entries = (z_score < 3).all(axis=1)
            dataframe = dataframe[filtered_entries]
        if remove_outliers == False:
            pass

        if fill_outliers_with_mean == True:
            from scipy.stats import zscore
            z_score = zscore(dataframe, nan_policy='omit')

            filtered_entries = (z_score > 3).all(axis=1)
            dataframe_outliers = dataframe[filtered_entries]

            filtered_entries = (z_score < 3).all(axis=1)
            dataframe_without_outliers = dataframe[filtered_entries]

            means = dataframe_without_outliers.groupby([dataframe_without_outliers.index.year, dataframe_without_outliers.index.month, dataframe_without_outliers.index.day_of_week, dataframe_without_outliers.index.hour]).mean() 

            for i in dataframe_outliers.index:
                if i.hour in means.loc[i.year, i.month, i.day_of_week, ].index:
                    dataframe.loc[i][0] = means.loc[i.year, i.month, i.day_of_week, i.hour]
                else:
                    dataframe.loc[i][0] = dataframe_without_outliers.loc[f'{i.year}-{i.month}'].between_time(f'{i.hour}:00:00', f'{i.hour}:00:00').mean()
        if fill_outliers_with_mean == False:
            pass

        if fill_missing_index == True:
            import pandas as pd
            full_index = pd.date_range(dataframe.index.min(), dataframe.index.max(), freq='H', tz='UTC')
            index_to_fill = full_index.difference(dataframe.index)
            dataframe = dataframe.reindex(dataframe.index.union(index_to_fill))

            means = dataframe.groupby([dataframe.index.year, dataframe.index.month, dataframe.index.day_of_week, dataframe.index.hour]).mean()

            for i in index_to_fill:
                dataframe.loc[i][0] = means.loc[i.year, i.month, i.day_of_week, i.hour]
        if fill_missing_index == False:
            pass

        if fill_missing_values_with_mean == True:
            is_month_nan = dataframe.groupby([dataframe.index.year, dataframe.index.month]).mean()
            
            means = dataframe.groupby([dataframe.index.year, dataframe.index.month, dataframe.index.day_of_week, dataframe.index.hour]).mean()

            for i in dataframe[dataframe.isna().all(axis=1)].index:
                if (i.year, i.month) in list(is_month_nan[is_month_nan.isna().all(axis=1)].index.values):
                    dataframe.loc[i][0] = (means.loc[i.year -1, i.month, i.day_of_week, i.hour] + means.loc[i.year +1, i.month, i.day_of_week, i.hour]) /2
                else:
                    dataframe.loc[i][0] = means.loc[i.year, i.month, i.day_of_week, i.hour]
        if fill_missing_values_with_mean == False:
            pass

        dataframe_proceed.append(dataframe)
    return(dataframe_proceed)

def Niederschlag(id, starttime, endtime):
    from wetterdienst.dwd.observations import (
        DWDObservationParameterSet,
        DWDObservationPeriod,
        DWDObservationResolution,
        DWDObservationStations,
        DWDObservationValues
        )
    import pandas 

    station_data = DWDObservationValues(
        station_id=[id],
        parameter = [DWDObservationParameterSet.PRECIPITATION],
        resolution = DWDObservationResolution.HOURLY,
        start_date= starttime,
        end_date= endtime,
        tidy_data = False
    ).all()

    df = station_data
    pandas.to_datetime(df['DATE'], format='%Y-%m-%d %H:%M:%S').copy()
    df.set_index(pandas.DatetimeIndex(df['DATE']), inplace=True)
    df['PRECIPITATION_HEIGHT'] = df['PRECIPITATION_HEIGHT'].astype('float64', copy=True)
    df = df.rename(columns={'PRECIPITATION_HEIGHT': 'Niederschlag'})
    df = df.drop(columns={'STATION_ID','QN_8', 'PRECIPITATION_INDICATOR', 'PRECIPITATION_FORM', 'DATE'})
    return df

def Temperatur(id, starttime, endtime):
    from wetterdienst.dwd.observations import (
        DWDObservationParameterSet,
        DWDObservationPeriod,
        DWDObservationResolution,
        DWDObservationStations,
        DWDObservationValues
        )
    import pandas 

    station_data = DWDObservationValues(
        station_id=[id],
        parameter = [DWDObservationParameterSet.TEMPERATURE_AIR],
        resolution = DWDObservationResolution.HOURLY,
        start_date= starttime,
        end_date= endtime,
        tidy_data = False
    ).all()

    df = station_data
    pandas.to_datetime(df['DATE'], format='%Y-%m-%d %H:%M:%S').copy()
    df.set_index(pandas.DatetimeIndex(df['DATE']), inplace=True)
    df['TEMPERATURE_AIR_200'] = df['TEMPERATURE_AIR_200'].astype('float64', copy=True)
    df = df.rename(columns={'TEMPERATURE_AIR_200': 'Temperatur'})
    df = df.drop(columns={'STATION_ID','QN_9', 'HUMIDITY', 'DATE'})
    return df

def Schnee(id, starttime, endtime):
    from wetterdienst.dwd.observations import (
        DWDObservationParameterSet,
        DWDObservationPeriod,
        DWDObservationResolution,
        DWDObservationStations,
        DWDObservationValues
        )
    import pandas 

    station_data = DWDObservationValues(
        station_id=[id],
        parameter = [DWDObservationParameterSet.PRECIPITATION_MORE],
        resolution = DWDObservationResolution.DAILY,
        start_date= starttime,
        end_date= endtime,
        tidy_data = False
    ).all()

    df = station_data
    pandas.to_datetime(df['DATE'], format='%Y-%m-%d %H:%M:%S').copy()
    df.set_index(pandas.DatetimeIndex(df['DATE']), inplace=True)
    df = df.rename(columns={'RS' : 'Tägliche Niederschlagshöhe', 'RSF' : 'Niederschlagsform', 'SH_TAG' : 'Schneehöhe'})
    df = df.drop(columns={'STATION_ID','QN_6', 'DATE'})
    return df

def dataframe_6to6(dataframe):
    import pandas as  pd
    import datetime as dt

    datelist = dataframe.index.strftime('%Y-%m-%d').unique().to_list()
    df_final = pd.DataFrame()

    for idx, date in enumerate(datelist):
        loc_to_6to6 = dataframe.loc[f'{date} 06:00:00' : f'{datelist[(idx +1) % len(datelist)]} 06:00:00'].sum()
        df_temp = pd.DataFrame(loc_to_6to6)
        df_final = df_final.append(df_temp)
    
    df_final['Datetime'] = datelist
    df_final = df_final.rename(columns={df_final.columns[0]: dataframe.columns[0]})

    df_final['Datetime'] = pd.DatetimeIndex(df_final.Datetime)
    df_final['Datetime'] = df_final['Datetime'].dt.tz_localize('UTC')
    df_final.set_index(pd.DatetimeIndex(df_final['Datetime']), inplace=True)
    
    df_final = df_final.drop(columns={df_final.columns[1]})
    return df_final

@st.cache
def combine_bike_counter_and_snow(list_of_dataframes):
    import pandas as pd
    import datetime as dt

    schnee = Schnee(691, '2012-01-01', str(dt.date.today() - dt.timedelta(0)))
        
    list_of_combined_dataframes = []
    for dataframe in list_of_dataframes:
        if dataframe.empty:
            pass
        else:
            dataframe = dataframe_6to6(dataframe)
            
            sn = schnee.loc[dataframe.index.min().strftime('%Y-%m-%d'):dataframe.index.max().strftime('%Y-%m-%d')]
                        
            dataframe = pd.merge(dataframe, sn, left_index=True, right_index=True, how='outer')

            list_of_combined_dataframes.append(dataframe)
    return list_of_combined_dataframes

@st.cache
def combine_bike_counter_and_weather(list_of_dataframes):
    import pandas as pd
    import datetime as dt

    niederschlag = Niederschlag(691, '2012-01-01 00:00:00', str(dt.date.today() - dt.timedelta(0)))
    temperatur = Temperatur(691, '2012-01-01 00:00:00', str(dt.date.today() - dt.timedelta(0)))
    
    list_of_combined_dataframes = []
    for dataframe in list_of_dataframes:
        if dataframe.empty:
            pass
        else:
            ns = niederschlag.loc[dataframe.index.min().strftime('%Y-%m-%d %H:%M:%S'):dataframe.index.max().strftime('%Y-%m-%d %H:%M:%S')]
            tmp = temperatur.loc[dataframe.index.min().strftime('%Y-%m-%d %H:%M:%S'):dataframe.index.max().strftime('%Y-%m-%d %H:%M:%S')]
            dataframe = pd.merge(pd.merge(dataframe, ns, left_index=True, right_index=True, how='outer'), tmp, left_index=True, right_index=True, how='outer')

            list_of_combined_dataframes.append(dataframe)
    return list_of_combined_dataframes

def sidebar_date():
    st.sidebar.write('Datum eingrenzen')

    dataframe_startdate = st.sidebar.date_input('vom', value=df_list[0].index.min())
    dataframe_enddate = st.sidebar.date_input('bis', value=df_list[0].index.max())

    if dataframe_startdate == dataframe_enddate:
       dataframe_startdate, dataframe_enddate = selected_dfs[0].loc[str(dataframe_startdate)].index.min(), selected_dfs[0].loc[str(dataframe_startdate)].index.max()
    else:
        pass

    return dataframe_startdate, dataframe_enddate

def Time_Slider():
    dftime= df_list[0].index.strftime('%H:%M:%S').unique().astype(str).tolist()
    dataframe_starttime, dataframe_endtime = st.sidebar.select_slider('Tageszeit eingrenzen', options=dftime, value=('00:00:00', '23:00:00'), key='Time_slider')
    return dataframe_starttime, dataframe_endtime

#######################################
###         plot functions          ###
#######################################

def dataframe_mainstats(dataframe, sum_mean_option, resample_option):
    import pandas as pd

    #dataframe = pd.DataFrame(dataframe.iloc[:, 0])

    if sum_mean_option == 'Summen':
        sum_mean_option = 'sum'
    if sum_mean_option == 'Mittelwerte':
        sum_mean_option = 'mean'
    if resample_option == 'Tag':
        resample_option = 'D'
    if resample_option == 'Monat':
        resample_option = 'M'
    if resample_option == 'Jahr':
        resample_option = 'Y'

    if resample_option == 'Option wählen':
        dataframe = dataframe
        resample_option = 'H'
        
    else: 
        dataframe = dataframe.resample(resample_option).agg({dataframe.columns[0] : sum_mean_option})

    st.subheader(dataframe.columns[0])

    st.line_chart(dataframe)
    details_expander = st.expander(label='Details')
    with details_expander:
        full_index = pd.date_range(dataframe.index.min(), dataframe.index.max(), freq=resample_option, tz='UTC')
        len_full_index = len(full_index)
        st.subheader('Messzeitpunkte')
        st.write(f'Optimale Anzahl von Messzeitpunkten: {len_full_index}')
        missing_in_index = len(full_index.difference(dataframe.index))
        len_index = len(dataframe.index)
        st.write(f'Tatsächliche Anzahl von Messzeitpunkten: {len_index}')
        st.write(f'Fehlende Messzeitpunkte: {missing_in_index}')
        st.subheader('Messwerte')
        st.write(f'Anzahl der Messwerte: {dataframe.iloc[:,0].count()}')
        missing_values = len(dataframe[dataframe.isna().all(axis=1)])
        st.write(f'Fehlende Messwerte: {missing_values}')
        st.subheader('Statistik')
        st.write(f'Minimaler Wert: {dataframe.iloc[:,0].min()}')
        st.write(f'Maximaler Wert: {dataframe.iloc[:,0].max()}, am ' + dataframe.iloc[:,0].idxmax().strftime('%d %B %Y') + ', um ' + dataframe.iloc[:,0].idxmax().strftime('%H') + ' Uhr')
        st.write(f'Mittelwert: {dataframe.iloc[:,0].mean().round(2)}')
        st.write(f'Standardabweichung: {dataframe.iloc[:,0].std().round(2)}')

    daten_expander = st.expander(label='Zeige Daten')
    with daten_expander:
        st.dataframe(dataframe)

def plot_timeline_w_MA(dataframe, window, periods):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.plot(dataframe.iloc[:, 0].resample('D').sum(), linewidth=0.25, label='Werte')
    ax.plot(dataframe.iloc[:, 0].resample('D').sum().rolling(window=window, center=True, min_periods=periods).mean(), linewidth=2, label='gleitender Mittelwert')
    ax.set_xlabel('Jahr')
    ax.set_ylabel('Radverkehrsaufkommen')
    ax.legend()
    return fig

def plot_boxplots_weekday(dataframe):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots()
    sns.boxplot(dataframe.index.day_name(), dataframe.iloc[:,0], showmeans=True, meanprops={'marker':'o','markerfacecolor':'white', 'markeredgecolor':'black','markersize':'10'}, ax=ax, palette="Blues_d")
    sns.set(style="whitegrid")
    sns.despine()
    ax.set_xlabel('Wochentag', size=13)
    ax.set_ylabel('Radverkehrsaufkommen pro Stunde', size=13)
    plt.grid(True)
    return fig

def plot_week_vs_weekend(dataframe, dataframe_startdate, dataframe_enddate):
    import matplotlib.pyplot as plt
    import numpy as np

    dataframe_first_monday = dataframe_startdate - dataframe_startdate.weekday() * dt.timedelta(days=1)
    dataframe_last_sunday = (dataframe_enddate - dataframe_enddate.weekday() * dt.timedelta(days=1)) + dt.timedelta(days=6)

    dataframe = dataframe.loc[dataframe_first_monday : dataframe_last_sunday]

    weekend = np.where(dataframe.index.weekday < 5, 'Weekday', 'Weekend')
    by_time = dataframe.groupby([weekend, dataframe.index.time]).mean()
    
    fig, ax = plt.subplots(1,1)
    hourly_ticks = 4 * 60 * 60 * np.arange(6)
    by_time.xs('Weekday').plot(ax=ax, xticks=hourly_ticks)
    by_time.xs('Weekend').plot(ax=ax, xticks=hourly_ticks)
    ax.legend(['Werktage', 'Wochenende'])
    ax.set_xlabel('Uhrzeit', size=13)
    ax.set_ylabel('Durchschnittliches Radverkehrsaufkommen', size=13)
    return fig, dataframe_first_monday, dataframe_last_sunday

def plot_weekdays(dataframe, dataframe_startdate, dataframe_enddate):
    import matplotlib.pyplot as plt
    import numpy as np

    dataframe_first_monday = dataframe_startdate - dataframe_startdate.weekday() * dt.timedelta(days=1)
    dataframe_last_sunday = (dataframe_enddate - dataframe_enddate.weekday() * dt.timedelta(days=1)) + dt.timedelta(days=6)

    dataframe = dataframe.loc[dataframe_first_monday : dataframe_last_sunday]

    weekday = np.where(dataframe.index.weekday == 0, 'Monday', 'restofweek')
    monday = dataframe.groupby([weekday, dataframe.index.time]).mean()

    weekday = np.where(dataframe.index.weekday == 1, 'Tuesday', 'restofweek')
    tuesday = dataframe.groupby([weekday, dataframe.index.time]).mean()

    weekday = np.where(dataframe.index.weekday == 2, 'Wednesday', 'restofweek')
    wednesday = dataframe.groupby([weekday, dataframe.index.time]).mean()

    weekday = np.where(dataframe.index.weekday == 3, 'Thursday', 'restofweek')
    thursday = dataframe.groupby([weekday, dataframe.index.time]).mean()

    weekday = np.where(dataframe.index.weekday == 4, 'Friday', 'restofweek')
    friday = dataframe.groupby([weekday, dataframe.index.time]).mean()

    weekday = np.where(dataframe.index.weekday == 5, 'Saturday', 'restofweek')
    saturday = dataframe.groupby([weekday, dataframe.index.time]).mean()

    weekday = np.where(dataframe.index.weekday == 6, 'Sunday', 'restofweek')
    sunday = dataframe.groupby([weekday, dataframe.index.time]).mean()

    fig, ax = plt.subplots(1,1)
    hourly_ticks = 4 * 60 * 60 * np.arange(6)
    monday.xs('Monday').plot(ax=ax, xticks=hourly_ticks)
    tuesday.xs('Tuesday').plot(ax=ax, xticks=hourly_ticks)
    wednesday.xs('Wednesday').plot(ax=ax, xticks=hourly_ticks)
    thursday.xs('Thursday').plot(ax=ax, xticks=hourly_ticks)
    friday.xs('Friday').plot(ax=ax, xticks=hourly_ticks)
    saturday.xs('Saturday').plot(ax=ax, xticks=hourly_ticks)
    sunday.xs('Sunday').plot(ax=ax, xticks=hourly_ticks)
    ax.legend(['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag'])
    ax.set_xlabel('Uhrzeit', size=13)
    ax.set_ylabel('Durchschnittliches Radverkehrsaufkommen', size=13)
    return fig, dataframe_first_monday, dataframe_last_sunday

def plot_STL(dataframe,resample_option, robust_option):
    from statsmodels.tsa.seasonal import STL
    import matplotlib.pyplot as plot

    if robust_option == 'robust':
        robust_option = True
    if robust_option == 'non robust':
        robust_option = False

    if resample_option == 'täglich/Summe':
        res_robust = STL(dataframe.iloc[:,0].resample('D').sum(), robust=robust_option).fit()
        res_non_robust = STL(dataframe.iloc[:,0].resample('D').sum(), robust=robust_option).fit()

    if resample_option == 'täglich/Mittelwert':
        res_robust = STL(dataframe.iloc[:,0].resample('D').mean(), robust=robust_option).fit()
        res_non_robust = STL(dataframe.iloc[:,0].resample('D').mean(), robust=robust_option).fit()
    
    if resample_option == 'monatlich/Summe':
        res_robust = STL(dataframe.iloc[:,0].resample('M').sum(), robust=robust_option).fit()
        res_non_robust = STL(dataframe.iloc[:,0].resample('M').sum(), robust=robust_option).fit()
    
    if resample_option == 'monatlich/Mittelwert':
       res_robust = STL(dataframe.iloc[:,0].resample('M').mean(), robust=robust_option).fit()
       res_non_robust = STL(dataframe.iloc[:,0].resample('M').mean(), robust=robust_option).fit()   
    
    if robust_option == True:
        return res_robust.trend, res_robust.seasonal, res_robust.resid
    if robust_option == False:
        return res_non_robust.trend, res_non_robust.seasonal, res_non_robust.resid

def plot_Radverkehr_Niederschlag(dataframe, resample_option):
    import plotly.express as px

    if resample_option == 'Tage':
        dataframe = dataframe.resample('D').agg({dataframe.columns.values[0] : 'sum', dataframe.columns.values[1] : 'sum', dataframe.columns.values[2] : 'mean'})
    else:
        pass
    
    fig = px.scatter_3d(dataframe, x=dataframe.index, z=str(dataframe.columns.values[1]), color='Niederschlag', y=str(dataframe.columns.values[0]),
                        color_continuous_scale=px.colors.sequential.Jet
    )
    fig.update_layout(width=1600, height=850)
    return fig

def plot_Radverkehr_Temperatur(dataframe, resample_option):
    import plotly.express as px

    if resample_option == 'Tage':
        dataframe = dataframe.resample('D').agg({dataframe.columns.values[0] : 'sum', dataframe.columns.values[1] : 'sum', dataframe.columns.values[2] : 'mean'})
    else:
        pass
    
    fig = px.scatter_3d(dataframe, x=dataframe.index, z=str(dataframe.columns.values[2]), color='Temperatur', y=str(dataframe.columns.values[0]),
                        color_continuous_scale=px.colors.sequential.Jet
    )
    fig.update_layout(width=1600, height=850)
    return fig

def plot_Radverkehr_Wetter(dataframe, resample_option):
    import plotly.express as px
    
    if resample_option == 'Tage':
        dataframe = dataframe.resample('D').agg({dataframe.columns.values[0] : 'sum', dataframe.columns.values[1] : 'sum', dataframe.columns.values[2] : 'mean'})
    else:
        pass

    fig = px.scatter_3d(dataframe, x=dataframe.index, z=str(dataframe.columns.values[1]), color='Temperatur', y=str(dataframe.columns.values[0]),
                        color_continuous_scale=px.colors.sequential.Jet
                        )
    fig.update_layout(width=1600, height=850)
    return fig

def plot_Radverkehr_Schnee(dataframe, option):
    import numpy as np
    import plotly.express as px

    if option == 'Schnee(fest)':
        dataframe = dataframe[dataframe['Niederschlagsform'] == 7]

    if option == 'Schnee, Regen und/oder Schneeregen':
        dataframe = dataframe[dataframe['Niederschlagsform'] == 7]

    fig = px.scatter_3d(dataframe, x=dataframe.index, z=str(dataframe.columns.values[3]), color='Schneehöhe', y=str(dataframe.columns.values[0]),
                        color_continuous_scale=px.colors.sequential.Jet,
    )
    fig.update_layout(width=1600, height=850)
    return fig

class surface():
    def year_month(dataframe, calc_option):
        import plotly.graph_objects as go
        
        dataframe = dataframe.groupby([dataframe.index.year,dataframe.index.month]).agg(calc_option)
        dataframe = dataframe.unstack()
    
        fig = go.Figure(data=[go.Surface(z=dataframe.values, hovertemplate='Jahr: %{y}<br>Monat: %{x}<br>Anzahl: %{z}<extra></extra>')])
        fig.update_layout(scene = dict(
                            yaxis = dict(
                                title='Jahr',
                                ticktext = dataframe.index.unique().astype(str).tolist(),
                                tickvals = list(range(len(dataframe.index.unique()))),
                                tickmode = 'array'
                                ),
                            xaxis = dict(
                                title = 'Monat',
                                ticktext = ['JAN','FEB','MÄR','APR','MAI','JUN','JUL','AUG','SEP','OKT','NOV','DEZ'],
                                tickvals = list(range(len(dataframe.columns))),
                                tickmode = 'array'
                                ),
                            zaxis = dict(
                                title = 'Anzahl'
                                ),
                        ),
                        width=1600,
                        height=850,
                        margin=dict(r=10, l=10, b=10, t=10
                        ))
        return fig

    def month_week(dataframe, calc_option):
        import plotly.graph_objects as go

        dataframe = dataframe.groupby([dataframe.index.hour, dataframe.index.month]).sum()
        dataframe = dataframe.unstack()

        fig = go.Figure(data=[go.Surface(z=dataframe.values, hovertemplate='Monat: %{x}<br>Stunde: %{y} Uhr<br>Anzahl: %{z}<extra></extra>')])
        fig.update_layout(scene = dict(
                            xaxis = dict(
                                title = 'Monat',
                                ticktext = ['JAN','FEB','MÄR','APR','MAI','JUN','JUL','AUG','SEP','OKT','NOV','DEZ'],
                                tickvals = list(range(len(dataframe.columns))),
                                tickmode = 'array'
                                ),
      
                            yaxis = dict(
                                title='Stunde',
                                ticktext = dataframe.index.unique().astype(str).tolist(),
                                tickvals = list(range(len(dataframe.index.unique()))),
                                tickmode = 'array'
                                ),
                            zaxis = dict(
                                title = 'Anzahl'
                                ),
                        ),
                        width=1600,
                        height=850,
                        margin=dict(r=10, l=10, b=10, t=10
                        ))
        return fig
        
    def week_hour(dataframe, calc_option):
        import plotly.graph_objects as go

        dataframe = dataframe.groupby([dataframe.index.hour, dataframe.index.isocalendar().week]).sum()
        dataframe = dataframe.unstack()

        fig = go.Figure(data=[go.Surface(z=dataframe.values, hovertemplate='Woche: %{x}<br>Stunde: %{y} Uhr<br>Anzahl: %{z}<extra></extra>')])
        fig.update_layout(scene = dict(
                            xaxis = dict(
                                title = 'Woche',
                                tickvals = list(range(len(dataframe.columns))),
                                tickmode = 'array'
                                ),
      
                            yaxis = dict(
                                title='Stunde',
                                ticktext = dataframe.index.unique().astype(str).tolist(),
                                tickvals = list(range(len(dataframe.index.unique()))),
                                tickmode = 'array'
                                ),
                            zaxis = dict(
                                title = 'Anzahl'
                                ),
                        ),
                        width=1600,
                        height=850,
                        margin=dict(r=10, l=10, b=10, t=10
                        ))
        return fig



# pyplot plot config
plt.rc('figure',figsize=(18.5, 11))
plt.rc('figure', dpi=1200)
plt.rc('font',size=13)

# sidebar Options
st.sidebar.markdown('Optionen')

# sidebar data preprocessing
st.sidebar.markdown('Datenvorverarbeitung')

#remove_outliers_option = st.sidebar.checkbox('Entferne Ausreißer', value=False)
outliers_option = st.sidebar.radio('Außreißer Optionen', ('Behalte Ausreißer','Entferne Ausreißer', 'Ersetze Ausreißer mit Mittelwerten'))

if outliers_option == 'Behalte Ausreißer':
    remove_outliers_option = False
    fill_outliers_option = False
if outliers_option == 'Entferne Ausreißer':
    remove_outliers_option = True
    fill_outliers_option = False
if outliers_option == 'Ersetze Ausreißer mit Mittelwerten':
    remove_outliers_option = False
    fill_outliers_option = True

st.sidebar.write('Fehlende Werte')
fill_missing_index_option = st.sidebar.checkbox('Ersetze fehlende Messzeitpunkte mit Mittelwerten', value=False)
fill_missing_values_option = st.sidebar.checkbox('Ersetze fehlende Messwerte mit Mittelwerten', value= False)

# import data
df_list = proceed_data(remove_outliers=remove_outliers_option, fill_outliers_with_mean=fill_outliers_option, fill_missing_index=fill_missing_index_option, fill_missing_values_with_mean=fill_missing_values_option)

# sidebar Radzählstationen
dict_of_stations = {'Wilhelm-Kaisen-Brücke (Ost)':df_list[0],
                    'Wilhelm-Kaisen-Brücke (West)':df_list[1],
                    'Langemarckstraße (Ostseite)':df_list[2],
                    'Langemarckstraße (Westseite)':df_list[3],
                    'Radweg Kleine Weser':df_list[4],
                    'Graf-Moltke-Straße (Ostseite)':df_list[5],
                    'Graf-Moltke-Straße (Westseite)':df_list[6],
                    'Schwachhauser Ring':df_list[7],
                    'Wachmannstraße auswärts (Süd)':df_list[8],
                    'Wachmannstraße einwärts (Nord)':df_list[9],
                    'Osterdeich':df_list[10],
                    'Hastedter Brückenstraße':df_list[11]}

list_of_stations = [dataframe.columns[0] for dataframe in df_list]

stationID = st.sidebar.multiselect('Radzählstation', list_of_stations, default='Wilhelm-Kaisen-Brücke (Ost)')

selected_dfs = [dict_of_stations.get(i) for i in stationID]

# sidebar Date
dataframe_startdate, dataframe_enddate =  sidebar_date()

# sidebar time
dataframe_starttime, dataframe_endtime = Time_Slider()

switch_graph_menue = st.sidebar.radio('Diagrammtyp',[
    'Info',
    'Allgemeine Übersicht',
    'Zeitreihe mit Mittelwert (gleitend)', 
    'BoxPlots: Wochentage',
    'Tagesverlauf Woche/Wochenende',
    'Tagesverlauf Wochentage (Detail)',
    'Saison-Trend-Zerlegung',
    'Radverkehr und Niederschlag',
    'Radverkehr und Temperatur',
    'Radverkehr und Wetter',
    'Radverkehr und Schneehöhe',
    'Oberflächendiagramm'
    ])

if switch_graph_menue == 'Info':

    st.header('RaStA')
    st.subheader('Das inoffizielle Bremer Radzählstationen Analysetool')
    st.markdown('by M o i n S t e f k o')
    st.write(f'Version {version}')

    src = '<iframe width="100%" height="600px" frameborder="0" allowfullscreen src="//umap.openstreetmap.fr/de/map/unbenannte-karte_570969?scaleControl=false&miniMap=false&scrollWheelZoom=false&zoomControl=true&allowEdit=false&moreControl=true&searchControl=null&tilelayersControl=null&embedControl=null&datalayersControl=true&onLoadPanel=undefined&captionBar=false"></iframe>'
    components.html(src, height=600,scrolling=False)

    st.header('Willkommen')
    st.subheader('Was ist RaStA?')
    st.write('RaStA wurde entwickelt, um interessierten Benutzer:innen einen tieferen Einblick in die Daten der Bremer Radzählstationen (https://vmz.bremen.de/radzaehlstationen/) zu bieten. Dafür stellt RaStA verschiedene Diagrammtypen als Werkzeuge zu Verfügung. Darüber hinaus ist es möglich, die Daten der Radzählstationen mit anderen Datensätzen - zum Beispiel des Deutschen Wetterdienstes (www.dwd.de) - zu verknüpfen.')
    st.subheader('Mit welchen Daten arbeitet RaStA?')
    st.write('RaStA arbeitet möglichst nah an den Rohdaten der Radzähstationen. Das heißt, es findet nur eine minimale automatische Datenvorverarbeitung statt. So verbleiben beispielsweise fehlerhafte Messwerte, wie sie durch technische Fehler der Messung und/oder Ausfall einer Zählstelle auftreten können, weitestgehend im Datensatz. Sie werden nur entfernt, sofern dies für die Verarbeitung und visuelle Darstellung unbedingt erforderlich ist. Der Ansatz der minimalen Datenvorverarbeitung betrifft auch die verknüpften Datensätze.')
    st.write('HINWEIS: Mit Version 0.0.5_Alpha wurden Optionen zur Datenvorverarbeitung implementiert. Der:die Benutzer:in kann nun selbst auf die Datenvorverarbeitung Einfluss nehmen. Zum Beispiel, ob Ausreißer aus dem Datensatz entfernt werden und/oder fehlende Messwerte/Messzeitpunkte mit Mittelwerten gefüllt werden. Das "trimmen" des jeweiligen Datensatzes vom 1. Januar 2012 bis zum ersten Zeitpunkt des Auftretens eines Wertes größer als Null ist als Standard in der Datenvorverarbeitung verankert. Für Mehr Infos siehe "Datenvorverarbeitung im Detail".')
    with st.expander('Datenvorverarbeitung im Detail anzeigen'):
        st.write('- Ausreißer mit einer Standardabweichung größer als 3 werden können optional aus dem Datensatz entfernt oder durch Mittelwerte ersetzt werden.')
        st.write('- Fehlende Daten - wie sie z.B. durch Ausfall einer Radzählstation auftreten können - können mit der Option *Ersetze fehlende Messzeitpunkte/Messwerte mit Mittelwerten* mit Mittelwerten des Datensatzes basierend auf Monat, Tag, Stunde gefüllt werden.')
        st.write('- Der Datensatz jeder Zählstation beginnt am 1. Januar 2012 um 0 Uhr, unabhängig davon, ob zu diesem Zeitpunkt die jeweilige Zählstation bereits Messwerte aufweist. Nur für die beiden Radzählstationen an der Wilhelm-Kaisen-Brücke (Ost und West) liegen ab dem 1. Januar 2012 fortlaufende Messwerte vor. Unter Annahme, dass die Radzählstationen erst nach und nach installiert wurden, und so zu unterschiedlichen Zeitpunkten ihre Tätigkeit aufgenommen haben, werden die Datensätze der jeweiligen Radzählstationen so eingekürzt (getrimmt), dass der erste Wert nach dem 1. Januar 2012 größer als 0 den Beginn des Datensatzes darstellt.')
   
        Stationstabelle = {'Stationsname': ['Wilhelm-Kaisen-Brücke (West)',
                                            'Wilhelm-Kaisen-Brücke (Ost)',
                                            'Langemarckstraße (Ostseite)',
                                            'Langemarckstraße (Westseite)',
                                            'Radweg Kleine Weser',
                                            'Graf-Moltke-Straße (Ostseite)',
                                            'Graf-Moltke-Straße (Westseite)',
                                            'Hastedter Brückenstraße',
                                            'Schwachhauser Ring',
                                            'Wachmannstraße auswärts (Süd)',
                                            'Wachmannstraße einwärts (Nord)',
                                            'Osterdeich',
                                            ],
                'Erster Messwert > 0 am' : ['2012-01-01 00:00:00',
                                            '2012-01-01 00:00:00',
                                            '2012-03-14 11:00:00',
                                            '2012-04-23 13:00:00',
                                            '2012-04-18 11:00:00',
                                            '2012-04-16 13:00:00',
                                            '2012-04-16 11:00:00',
                                            '2012-04-25 12:00:00',
                                            '2012-03-14 05:00:00',
                                            '2012-04-24 15:00:00',
                                            '2012-04-24 12:00:00',
                                            '2012-04-17 11:00:00',
                                            ]}
        stationstabelle = pd.DataFrame(Stationstabelle)
        stationstabelle.set_index(stationstabelle.Stationsname, inplace=True)
        stationstabelle = stationstabelle.drop(columns={stationstabelle.columns[0]})
        st.markdown('- Tabelle Radzählstationen und Zeitpunkt des ersten Wertes größer als Null:')
        st.table(stationstabelle)
       
    st.subheader('Wie wurde RaStA entwickelt?')
    st.write('Das Backend von RaStA wurde in Python3 auf Basis einer Reihe von data science Programmbibliotheken geschrieben. Das Webfrontend basiert auf der Programmbibliothek streamlit (www.streamlit.io).')
    with st.expander('verwendete Programmbibliotheken anzeigen'):
        st.markdown('Datenverarbeitung') 
        st.write('- Pandas: https://pandas.pydata.org')
        st.write('- numpy: https://numpy.org')
        st.write('- datetime: https://docs.python.org/3/library/datetime.html')
        st.write('- scipy: https://www.scipy.org')
        st.write('- statsmodels: https://www.statsmodels.org/stable/index.html')
        st.write('- wetterdienst: https://pypi.org/project/wetterdienst/')
        st.write(' ')
        st.write(' ')
 
        st.markdown('Visualisierung')
        st.write('- Streamlit: https://streamlit.io')
        st.write('- matplotlib: https://matplotlib.org')
        st.write('- seaborn: https://seaborn.pydata.org')
        st.write('- Plotly Express: https://plotly.com/python/plotly-express/')

    st.subheader('Verwendung von RaStA')
    with st.expander('Optionen: Datenvorverarbeitung, Radzählstation, Zeit und Diagrammtyp'):
        st.write('An der linken Bildschirmleiste befindet sich das Optionsmenü. Hier können die Grundeinstellungen zur Datenvorverarbeitung, den Radzählstationen, dem Zeitraum und des Diagrammtyps vorgenommen werden.') 
        st.markdown('Datenvorverarbeitung')
        st.write('Ausreißer Optionen')
        st.write('HINWEIS: Das Verhalten der folgenden Optionen kann durch einblenden der *Details* unter den Diagrammen von *Allgmeine Übersicht* nachverfolgt werden.')
        st.write('- Behalte Aureißer: Der Datensatz bleibt unverändert. Es werden keine Ausreißer entfernt, oder durch Mittelwerte ersetzt.')
        st.write('- Entferne Ausreißer: Entfernt Werte mit einer Standardabweichung größer als drei aus dem Datensatz. Es werden Messzeitpunkt und Messwert gelöscht. Messzeitpunkte zu denen kein Messwert vorhanden ist werden ebenfalls gelöscht. Die Anzahl fehlender Messzeitpunkte erhöht sich dadurch.')
        st.write('- Ersetze Ausreißer mit Mittelwerten: Ersetzt Ausreißer mit Mittelwerten. Die Mittelwerte werden in Abhängigkeit von Uhrzeit, Wochentag, Monat und Jahr berechnet. Beispiel: Der Messzeitpunkt eines Ausreißers ist Montag, 17. Juni 2013, 17:00 Uhr. Der Mittelwert für diesen Ausreißer wird dann aus den 17:00 Uhr Werten aller Montage im Monat Juni des Jahres 2013 berechnet. Sollte ein Mittelwert aufgrund fehlender Datengrundlage nicht berechnet werden können, werden zur Berechnung die entsprechenden stündlichen Werte des gesamten Monats verwendet.')
        st.write('')
        st.write('Fehlende Werte')
        st.write('- Ersetze fehlende Messzeitpunkte mit Mittelwerten: Für nicht vorhandene Messwerte aufgrund fehlender Messzeitpunkte in der Zeitreihe wird ein Mittelwert eingesetzt. Die Berechnung des Mittelwertes verläuft analog zur Berechnung des Mittelwertes der Option *Ersetze Ausreißer mit Mittelwerten*.')
        st.write('- Ersetzte fehlende Messwerte mit Mittelwerten: Für fehlende Messwerte zu vorhandenen Messzeitpunkten in der Zeitreihe wird ein Mittelwert eingesetzt. Die Berechnung des Mittelwertes verläuft analog zur Berechnung des Mittelwertes der Option *Ersetze Ausreißer mit Mittelwerten*. Als Ausnahme gilt, wenn ein Datensatz über einen Zeitraum von einen Monat oder mehr keine Messwerte enthält (z.B. Wachmannstraße (Nord) in 2016) wird ein Mittelwert aus den entsprechenden Mittelwerten des Vorjahres und Folgejahres berechnet.')
        st.write('')
        st.markdown('Radzählstationen')
        st.write('- Mit dem Dropdown Menü *Radzählstationen* können einzelne oder mehrere Radzählstationen ausgewählt werden. Als Voreinstellung ist, nach dem Aufruf von RaStA, hier bereits die Wilhelm-Kaisen-Brücke (Ost) eingetragen.')
        st.write('')
        st.markdown('Datum/Tageszeit eingrenzen')
        st.write('- Über die Eingabefelder *vom/bis* kann der Zeitraum der Anzeige des Datensatzes bestimmt werden. Als Standard ist hier, nach dem Aufruf von RaStA, der Beginn und das Ende des Datensatzes der Radzählstation Wilhelm-Kaisen-Brücke (Ost) eingetragen.')
        st.write('- Mit dem Slider *Tageszeit eingrenzen* lässt sich die Anzeige des Datensatzes weiter definieren, es kann eine Zeitspanne, z.B. 08:00:00 Uhr bis 16:00:00 Uhr, oder ein Zeitpunkt, z.B. 09:00:00 eingestellt werden.')
        st.write('- Nach Auswahl eines *Diagrammtyps* werden die obrigen Parameter auf das jeweilige Diagramm angewendet und je nach Diagrammtyp weitere Parameter verfügbar. Sofern die Parameter *Radzählstation, Datum eingrenzen* und *Tageszeit eingrenzen* nach Auswahl eines Diagrammtyps verändert werden, werden die Diagramme den veränderten Parametern angepasst. Wird auf einen anderen Diagrammtyp gewechselt, werden die Einstellungen zu Datum und Zeit übernommen. Unter *Info* sind weitere Informationen zu den Diagrammtypen sowie deren Parametereinstellungen zu finden.')
    with st.expander('Cache, Running und Rerun'):
        st.markdown('Cache')
        st.write('Um einzelne Berechnungen bei unveränderten Parametern nicht fortlaufend wiederholen zu müssen, speichert RaStA diese in einem Cache. Bei unerwarteten Verhalten von RaStA kann der Cache über das Menü am Bildschirmrand oben rechts geleert werden und wird bei weiterer Verwendung wieder neu angelegt.')    
        st.markdown('Running')
        st.write('RaStA ist unglaublich performant. Jedoch kann es je nach Größe des eingebenen Zeitraums, der Anzahl der angegebenen Radzählstellen und des Diagrammtyps vorkommen, dass RaStA einige Zeit benötigt, bis die Diagramme und Informationen angezeigt werden. Wenn RaStA gerade beschäftigt ist, wird oben rechts ein *RUNNING* Symbol erscheinen.')
        st.markdown('Rerun')
        st.write('Sollte RaStA einmal nicht wie erwartet funktionieren, bzw. eine Fehlermeldung die Weiterarbeit verhindern, so kann RaStA über den Befehl *Rerun* im Menü oben rechts neu gestartet werden.')

    st.subheader('Informationen zu den Diagrammtypen')

    with st.expander('Allgemeine Übersicht'):
        st.write('Ein einfaches interaktives Liniendiagramm. Mit *Details* lassen sich Informationen zu vorhandenen/fehlenden Messzeitpunkten/Messwerten und allgmeine statistische Werte anzeigen. *Zeige Daten* blendet die Datentabelle ein.')
        st.write('Datenbasis: tägliche/monatliche/jährliche Summen oder Mittelwerte (werden aus den stündlichen Messwerten berechnet)')    
    
    with st.expander('Zeitreihe mit gleitender Mittelwert'):
        st.write('Zeigt neben dem tatsächlichen Beobachtungen (blaue Linie) den gleitenden Mittelwert (orange Linie) an. *Zeige Daten* belndet die Datentabelle, ohne Spalte für den gleitenden Mittelwert, ein.')
        st.write('Für den gleitenden Mittelwert lassen sich die Parameter für die Größe des gleitenden Mittelwert Fensters (in Tagen) und erforderliche minimale Anzahl von Beobachtungen im Fenster anpassen.')
        st.write('Datenbasis: tägliche Summen')
        st.write('Hinweis: Das Diagramm sollte auf Daten angewendet werden, die einen größeren Zeitraum umfassen.')
        st.write('Links')
        st.write('https://de.wikipedia.org/wiki/Gleitender_Mittelwert')
        st.write('https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html?highlight=rolling#pandas.DataFrame.rolling')
    
    with st.expander('BoxPlots Wochentage'):
        st.write('Zeigt die Verteilung der Streuungs- und Lagemaße als Kastendiagramm an.')
        st.write('Datenbasis: stündliche Werte')
        st.write('Links')
        st.write('https://de.wikipedia.org/wiki/Box-Plot')

    with st.expander('Tagesverlauf Woche/Wochenende'):
        st.write('Zeigt den Radverkehr im Tagesverlauf der Wochentage im Vergleich zum Tagesverlauf an den Wochenendtagen an.')
        st.write('Datenbasis: stündliche Werte')
        st.write('Hinweis: Die Angaben unter *Datum eingrenzen* werden automatisch so angepasst, dass der Montag in der Woche mit dem *vom* Datum und der Sonntag der Woche mit dem *bis* Datum enthalten ist. Bitte die Infozeile über dem Diagramm beachten.')

    with st.expander('Tagesverlauf Wochentage (Detail'):
        st.write('Zeigt den Radverkehr im Tagesverlauf der einzelnen Wochentage')
        st.write('Datenbasis: stündliche Werte')
        st.write('Hinweis: Die Angaben unter *Datum eingrenzen* werden automatisch so angepasst, dass der Montag in der Woche mit dem *vom* Datum und der Sonntag der Woche mit dem *bis* Datum enthalten ist. Bitte die Infozeile über dem Diagramm beachten.')
    
    with st.expander('Saison-Trend-Zerlegung'):
        st.write('Zerlegt die Zeitreihe in Saison-Komponente, Trend-Komponente und Residuen mittels einer lokalen linearen Kernregression.')
        st.write('Links')
        st.write('https://de.wikipedia.org/wiki/Kernregression#Lokal_lineare_Kernregression')
        st.write('https://www.statsmodels.org/devel/examples/notebooks/generated/stl_decomposition.html')

    with st.expander('Radverkehr und Niederschlag'):
        st.write('Zeigt die Radvekehrsdaten in Verbindung mit den DWD Niederschlagsdaten an.')
        st.write('Datenbasis: Tägliche Summe der stündlichen Werte der jeweiligen Zählstation und durschnittlicher Niederschlag/Tag berechnet aus den stündlichen Niederschlagswerten.')
        st.write('Hinweis: Die Wetterdaten liegen bis max. 48h vor aktuellem Datum vor.')
        st.write('Links')
        st.write('https://pypi.org/project/wetterdienst/')
        st.write('https://www.dwd.de/DE/leistungen/klimadatendeutschland/stationsuebersicht.html')

    with st.expander('Radverkehr und Temperatur'):
        st.write('Zeigt die Radvekehrsdaten in Verbindung mit den DWD Temperaturdaten an.')
        st.write('Datenbasis: Tägliche Summe der stündlichen Werte der jeweiligen Zählstation und durschnittliche Tagestemperatur, berechnet aus den stündlichen Temperaturwerten.')
        st.write('Hinweis: Die Wetterdaten liegen bis max. 48h vor aktuellem Datum vor.')
        st.write('Links')
        st.write('https://pypi.org/project/wetterdienst/')
        st.write('https://www.dwd.de/DE/leistungen/klimadatendeutschland/stationsuebersicht.html')

    with st.expander('Radverkehr und Wetter'):
        st.write('Zeigt die Radvekehrsdaten in Verbindung mit den DWD Wetterdaten Niederschlag und Temperatur an.')
        st.write('Datenbasis: Tägliche Summe der stündlichen Werte der jeweiligen Zählstation und durschnittliche Niederschlagshöhe sowie durchschnittliche Tagestemperatur.')
        st.write('Hinweis: Die Wetterdaten liegen bis max. 48h vor aktuellem Datum vor.')
        st.write('Links')
        st.write('https://pypi.org/project/wetterdienst/')
        st.write('https://www.dwd.de/DE/leistungen/klimadatendeutschland/stationsuebersicht.html')

    with st.expander('Radverkehr und Schnee'):
        st.write('Zeigt die Radvekehrsdaten in Verbindung mit den DWD Wetterdaten Schnee (fest) sowie Regen und Schnee(und/oder Schneeregen) an.')
        st.write('Datenbasis: Tägliche Tägliche Summe der stündlichen Werte der jeweiligen Zählstation und tägliche Schneehöhe an Tagen mit festem Schnee sowie Regen und Schnee und/oder Schneeregen')
        st.write('Hinweis: Das Zeitintervall der täglichen Niederschlagshöhe wird seitens des DWD über den Zeitraum 6 Uhr bis 6 Uhr des Folgetags definiert. Um eine bessere Vergleichbarkeit zu gewährleisten wurden die täglichen Werte der Radzählstationen in diesen Diagrammen dem Zeitraum angepasst. Dadurch können beim Vergleich mit anderen Diagrammen Abweichungen auftreten. Dies trifft nicht auf die Diagrammtypen Radverkehr und Niederschlag/Temperatur/Wetter zu, da hier die täglichen Summen aus den sündlichen DWD Messwerten berechnet werden.')
        st.write('Links: https://opendata.dwd.de/test/CDC/observations_germany/climate/daily/more_precip/historical/BESCHREIBUNG_obsgermany_climate_daily_more_precip_historical_de.pdf')
    with st.expander('Oberflächendiagramm'):
        st.write('Zeigt ein Oberflächendiagramm der Radverkehrsdaten an.')
        st.write('Mittels *Typ* können die X-Achse und Y-Achse definiert werden. Zudem können die Werte als Summen oder Mittelwerte angezeigt werden. Hinweis: Der Typ *Wochen/Stunden* zeigt die  stündlichen Messwerte an. Somit können keine Summen oder Mittelwerte gebildet werden.')

    st.subheader('Changelog')
    with st.expander('Anzeigen'):
        st.subheader('0.0.1_alpha')
        st.write('Initiale Version')

        st.subheader('0.0.2_alpha')
        st.write('- Fix: In der Verarbeitung der Wetterdaten gab es einen Fehler, der dazu führte, dass nicht alle DWD Messerwerte übernommen wurden.')
        st.write('- Bug: "MemoryError: In RendererAgg: Out of memory" in Saison-Trend-Zerlegung, der dazu führt, dass das Diagramm nicht angezeigt wird.')

        st.subheader('0.0.3_alpha')
        st.write('- Geänderte Datenverarbeitung: Die Daten der Radzählstationen werden nun einmal am Tag ( 10:00:00 Uhr UTC) abgerufen.')
        st.write('- Grid zu Boxplot Diagramm hinzugefügt.')
        st.write('- Neue Funktion: Unter *Allgemeine Übersicht* kann nun die Datentabelle angezeigt werden.')
        st.write('- Neue Funktion: Auswahl der Radzählstation(en) nach Name der Station und nicht mehr nach ID der Station.')
        

        st.subheader('0.0.4_alpha (das *Schnee* Update)')
        st.write('- Fix: Zur Vermeidung des "MemoryError: In RendererAgg: Out of memory" Fehlers, werden nun in der Saison-Trend-Zerlegung, Trendkomponente, Saisonkomponente und Residuen getrennt voneinander angezeigt.')
        st.write('- Neue Funktion: In der Saison-Trend-Zerlegung können nun weitere Parameter zu täglichen/monatlichen Summen/Mittelwerten sowie non robust/non robust definiert werden.')
        st.write('- Neue Funktion: Diagrammtyp *Radverkehr und Schnee* hinzugefügt. Diagramm zeigt die Radverkehrsdaten in Zusammenhang mit Schneehöhe.')
        st.write('- Neue Funktion: Diagrammtyp *Allgemeine Übersicht* lässt sich nun nach Summen/Mittelwerte und/oder Tag/Monat/Jahr anpassen.')
        st.write('- Verhalten in der Diagrammerstellung von *Tagesverlauf Wochentage/Wochenendtage* geändert. Für mehr Infos: Siehe *Informationen zu den Diagrammtypen* unter *Info*.')
        st.write('- Verhalten in der Diagrammerstellung von *Tagesverlauf Wochentage(Detail)* geändert. Für mehr Infos: Siehe *Informationen zu den Diagrammtypen* unter *Info*.')
        st.write('- Beschreibung zu RaStA, Datenvorverarbeitung und Datenverarbeitung, Benutzung hinzugefügt.')
        st.write('- Beschreibung zu Diagrammtypen leicht überarbeitet')

        st.subheader('0.0.5_alpha (das *Index* Update)')
        st.write('- Neue Funktion: Neue Optionen zur Datenvorverarbeitung *Entferne Ausreißer* und *Ersetze fehlende Messzeitpunkte mit Mittelwerten* hinzugefügt.')
        st.write('- Neue Funktion: Diagrammtyp *Oberflächendiagramm* hinzugefügt. Für mehr Infos: Siehe *Informationen zu den Diagrammtypen*.')
        st.write('- Fix: Die Datentabellen der Wetterdiagrammen werden nun bei Veränderung der Abtastrate korrekt angezeigt.')
        st.write('- Fix: Datentabelle der Saison-Trend-Zerlegung wird nun bei Veränderung der Abtastrate korrekt angezeigt.')

        st.subheader('0.0.6_alpha')
        st.write('- Fix: Die Anzeige der "Hoverdaten" bei mouseover von Datenpunkten im Diagramm wird nun korrekt dargestellt.')
        st.write('- Komplette Überarbeitung der Datenvorverarbeitung.')
        st.write('- Neue Funktion: *Ersetze fehlende Messwerte mit Mittelwerten* hinzugefügt. Für mehr Infos: Siehe *Optionen: Datenvorverarbeitung, Radzählstation, Zeit und Diagrammtyp*.' )
        st.write('- Informationen der *Details* unter den Diagrammen von *Allgemeine Übersicht* erweitert. Es werden nun Informationen zu vorhandenen/fehlenden Messzeitpunkten und Messwerten angezeigt')
        st.write('- Überarbeitung diverser Infotexte u.a. *Optionen: Datenvorverarbeitung, Radzählstation, Zeit und Diagrammtyp*')
        st.write('- Überarbeitung des Menüs')


    st.write('©2021 www.moin-stefko.de / mail: hallo@moin-stefko.de')

if switch_graph_menue == 'Allgemeine Übersicht':

    st.sidebar.markdown('Datenoptionen')
    sum_mean_option = st.sidebar.radio('Summen oder Mittelwert', ['Summen', 'Mittelwerte']) 
    resample_option = st.sidebar.selectbox('Zeithorizont', ('Option wählen', 'Tag', 'Monat', 'Jahr'))
    
    st.header('Allgemeine Übersicht')


    for dataframe  in selected_dfs:
        st.subheader(dataframe.columns[0])
        st.write(f'vom {dataframe_startdate} bis zum {dataframe_enddate} zwischen {dataframe_starttime} Uhr und {dataframe_endtime} Uhr')

        df = dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime)
        dataframe_mainstats(df, sum_mean_option, resample_option)

if switch_graph_menue == 'Zeitreihe mit Mittelwert (gleitend)':
    
    st.header('Zeitreihe mit gleitender Mittelwert')

    st.sidebar.markdown('Einstellungen für den gleitenden Mittelwert')
    moving_average_options = str([*range(1, df_list[0].count()[0])])
    moving_average_window = st.sidebar.slider('Größe des sich bewegenden Fensters. Enthält die Anzahl der Beobachtungen, die zur Berechnung gleitenden Mittelwertes verwendet werden.', min_value=0, max_value=int(df_list[0].count()[0] / 24))
    moving_average_period = st.sidebar.slider('Erforderliche Mindestanzahl von Beobachtungen im Fenster', min_value=0, max_value=int(df_list[0].count()[0] / 24))
 
    for dataframe  in selected_dfs:
        st.subheader(dataframe.columns[0])
        st.write(f'vom {dataframe_startdate} bis zum {dataframe_enddate} zwischen {dataframe_starttime} Uhr und {dataframe_endtime} Uhr')
    
        df = dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime)
        fig = plot_timeline_w_MA(df, moving_average_window, moving_average_period)
        st.write(fig)
        with st.expander('Zeige Daten'):
            st.write(dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime))

if switch_graph_menue == 'BoxPlots: Wochentage':

    st.header('BoxPlots: Wochentage')

    for dataframe  in selected_dfs:
        st.subheader(dataframe.columns[0])
        st.write(f'vom {dataframe_startdate} bis zum {dataframe_enddate} zwischen {dataframe_starttime} Uhr und {dataframe_endtime} Uhr')
      
        df = dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime)
        fig = plot_boxplots_weekday(df)
        st.write(fig)
        with st.expander('Zeige Daten'):
            st.write(dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime))

if switch_graph_menue == 'Tagesverlauf Woche/Wochenende':

    st.header('Tagesverlauf Woche/Wochenende')

    for dataframe  in selected_dfs:
        st.subheader(dataframe.columns[0])

        df = dataframe.iloc[:,0].between_time(dataframe_starttime, dataframe_endtime)
        fig, dataframe_first_monday, dataframe_last_sunday = plot_week_vs_weekend(df,dataframe_startdate, dataframe_enddate)

        st.write(f'vom {dataframe_first_monday} bis zum {dataframe_last_sunday} zwischen {dataframe_starttime} Uhr und {dataframe_endtime} Uhr')
        
        st.write(fig)
        with st.expander('Zeige Daten'):
            st.write(dataframe.loc[dataframe_first_monday: dataframe_last_sunday].between_time(dataframe_starttime, dataframe_endtime))

if switch_graph_menue == 'Tagesverlauf Wochentage (Detail)':

    st.header('Tagesverlauf Wochentage (Detail)')

    for dataframe  in selected_dfs:
        st.subheader(dataframe.columns[0])
        
        df = dataframe.iloc[:,0].between_time(dataframe_starttime, dataframe_endtime)
        fig, dataframe_first_monday, dataframe_last_sunday = plot_weekdays(df, dataframe_startdate, dataframe_enddate)

        st.write(f'vom {dataframe_first_monday} bis zum {dataframe_last_sunday} zwischen {dataframe_starttime} Uhr und {dataframe_endtime} Uhr')
  
        st.write(fig)
        with st.expander('Zeige Daten'):
            st.write(dataframe.loc[dataframe_first_monday : dataframe_last_sunday].between_time(dataframe_starttime, dataframe_endtime))

if switch_graph_menue == 'Saison-Trend-Zerlegung':

    resample_option = st.sidebar.selectbox('Abtastrate/Funktion', ('täglich/Summe', 'täglich/Mittelwert', 'monatlich/Summe', 'monatlich/Mittelwert'))
    robust_option = st.sidebar.selectbox('Robustheit', ('robust', 'non robust'))

    st.header('Saison-Trend-Zerlegung')

    for dataframe  in selected_dfs:
        st.subheader(dataframe.columns[0])
        st.write(f'vom {dataframe_startdate} bis zum {dataframe_enddate} zwischen {dataframe_starttime} Uhr und {dataframe_endtime} Uhr')
  
        df = dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime)

        trend, seasonal, resid = plot_STL(df, resample_option, robust_option)

        st.markdown('Trendkomponente')
        st.line_chart(trend)

        with st.expander('Zeige Saisonkomponente'):
            st.line_chart(seasonal)
        with st.expander('Zeige Residuen'):
            st.line_chart(resid)
        with st.expander('Zeige Daten'):
            data_to_show = pd.merge(pd.merge(trend, seasonal, left_index=True, right_index=True, how='outer'), resid, left_index=True, right_index=True, how='outer')
            data_to_show[dataframe.columns[0]] = data_to_show.sum(axis=1).round()
            st.write(data_to_show)

if switch_graph_menue == 'Radverkehr und Niederschlag':
    df_list = combine_bike_counter_and_weather(selected_dfs)

    resample_option = st.sidebar.selectbox('Abtastrate', ('Stunden', 'Tage'))

    st.header('Radverkehr und Niederschlag')

    for dataframe in df_list:
        st.subheader(dataframe.columns[0])
        st.write(f'vom {dataframe_startdate} bis zum {dataframe_enddate} zwischen {dataframe_starttime} Uhr und {dataframe_endtime} Uhr')
           
        df = dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime)
        fig = plot_Radverkehr_Niederschlag(df, resample_option)
        st.write(fig)
        with st.expander('Zeige Daten'):
            if resample_option == 'Tage':
                st.write(dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime).resample('D').agg({dataframe.columns.values[0] : 'sum', dataframe.columns.values[1] : 'sum', dataframe.columns.values[2] : 'mean'}))
            if resample_option == 'Stunden':
                st.write(dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime))

if switch_graph_menue == 'Radverkehr und Temperatur':
    df_list = combine_bike_counter_and_weather(selected_dfs)

    resample_option = st.sidebar.selectbox('Abtastrate', ('Stunden', 'Tage'))

    st.header('Radverkehr und Temperatur')

    for dataframe  in df_list:
        st.subheader(dataframe.columns[0])
        st.write(f'vom {dataframe_startdate} bis zum {dataframe_enddate} zwischen {dataframe_starttime} Uhr und {dataframe_endtime} Uhr')
           
        df = dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime)
        fig = plot_Radverkehr_Temperatur(df, resample_option)
        st.write(fig)
        with st.expander('Zeige Daten'):
            if resample_option == 'Tage':
                st.write(dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime).resample('D').agg({dataframe.columns.values[0] : 'sum', dataframe.columns.values[1] : 'sum', dataframe.columns.values[2] : 'mean'}))
            if resample_option == 'Stunden':
                st.write(dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime))

if switch_graph_menue == 'Radverkehr und Wetter':
    df_list = combine_bike_counter_and_weather(selected_dfs)

    resample_option = st.sidebar.selectbox('Abtastrate', ('Stunden', 'Tage'))

    st.header('Radverkehr und Wetter')

    for dataframe  in df_list:
        st.subheader(dataframe.columns[0])
        st.write(f'vom {dataframe_startdate} bis zum {dataframe_enddate} zwischen {dataframe_starttime} Uhr und {dataframe_endtime} Uhr')
           
        df = dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime)
        fig = plot_Radverkehr_Wetter(df, resample_option)
        st.write(fig)
        with st.expander('Zeige Daten'):
            if resample_option == 'Tage':
                st.write(dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime).resample('D').agg({dataframe.columns.values[0] : 'sum', dataframe.columns.values[1] : 'sum', dataframe.columns.values[2] : 'mean'}))
            if resample_option == 'Stunden':
                st.write(dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime))

if switch_graph_menue == 'Radverkehr und Schneehöhe':
    df_list = combine_bike_counter_and_snow(selected_dfs)

    option = st.sidebar.selectbox('Tage mit...', ('Schnee(fest)', 'Schnee, Regen und/oder Schneeregen'))

    st.header('Radverkehr und Schneehöhe')

    for dataframe in df_list:
        st.subheader(dataframe.columns[0])
        st.write(f'vom {dataframe_startdate} bis zum {dataframe_enddate} zwischen {dataframe_starttime} Uhr und {dataframe_endtime} Uhr')

        df = dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime)
        fig = plot_Radverkehr_Schnee(df, option)
        st.write(fig)
        with st.expander('Zeige Daten'):
            st.write(dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime)) 

if switch_graph_menue == 'Oberflächendiagramm':

    plot_option = st.sidebar.selectbox('Typ', ('Jahr/Monat', 'Monat/Wochen','Wochen/Stunden'))

    calc_option = st.sidebar.selectbox('Summen/Mittelwerte', ('Summen', 'Mittelwerte'))
    if calc_option == 'Summen':
        calc_option = 'sum'
    if calc_option == 'Mittelwerte':
        calc_option = 'mean'
    
    for dataframe in selected_dfs:
        st.subheader(dataframe.columns[0])
        st.write(f'vom {dataframe_startdate} bis zum {dataframe_enddate} zwischen {dataframe_starttime} Uhr und {dataframe_endtime} Uhr')

        df = dataframe.loc[dataframe_startdate : dataframe_enddate].between_time(dataframe_starttime, dataframe_endtime)

        if plot_option == 'Jahr/Monat':
            fig = surface.year_month(df, calc_option)
            st.write(fig)
        if plot_option == 'Monat/Wochen':
            fig = surface.month_week(df, calc_option)
            st.write(fig)
        if plot_option == 'Wochen/Stunden':
            fig = surface.week_hour(df, calc_option)
            st.write(fig)

