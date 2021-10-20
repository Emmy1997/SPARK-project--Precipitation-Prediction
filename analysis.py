import geopandas as gpd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pyodbc


def connect_to(db_name):
    conn = pyodbc.connect(
        DRIVER='{SQL Server};',
        SERVER='technionddscourse.database.windows.net;',
        DATABASE=db_name,
        UID=db_name,
        PWD='Qwerty12!')
    return conn


def data_analysis_temporal():
    """
    Analysis by temporal spase. Considering months January and August in stations in Germany during the years.
    Checks the correlation between max tempreture and prcp
    """
    conn = connect_to('ilanit0sobol')
    cursor = conn.cursor()
    query = """SELECT DISTINCT W.StationId as StationId, W.Year as Y, W.AvgPrcp as PRCP, W.AvgTmax as T, W.month as M
                FROM Weather W
                WHERE W.stationId in (select DISTINCT TOP 3 W1.StationId
                                        from Weather W1 
                                        WHERE W1.FIPS_code = 'GM')
                and (W.month = '1' or W.month='8')
                """

    #dfiMX - i is the index od the station, M - for month, X- type of parameter
    df = pd.read_sql(query, conn)
    df = df.sort_values("Y")
    Stations = list(set(df['StationId']))
    #df station 1
    df1 = df[df['StationId']==Stations[0]].drop(axis='columns', columns='StationId', inplace=False)
    df11T = np.array(df1[df1['M'] == 1].drop(axis='columns', columns=['PRCP', 'M'], inplace=False))
    df11P = np.array(df1[df1['M'] == 1].drop(axis='columns', columns=['T', 'M'], inplace=False))
    df18T = np.array(df1[df1['M'] == 8].drop(axis='columns', columns=['PRCP', 'M'], inplace=False))
    df18P = np.array(df1[df1['M'] == 8].drop(axis='columns', columns=['T', 'M'], inplace=False))
    # df station 2
    df2 = df[df['StationId']==Stations[1]].drop(axis='columns', columns='StationId', inplace=False)
    df21T = np.array(df2[df2['M'] == 1].drop(axis='columns', columns=['PRCP', 'M'], inplace=False))
    df21P = np.array(df2[df2['M'] == 1].drop(axis='columns', columns=['T', 'M'], inplace=False))
    df28T = np.array(df2[df2['M'] == 8].drop(axis='columns', columns=['PRCP', 'M'], inplace=False))
    df28P = np.array(df2[df2['M'] == 8].drop(axis='columns', columns=['T', 'M'], inplace=False))
    # df station 3
    df3 = df[df['StationId']==Stations[2]].drop(axis='columns', columns='StationId', inplace=False)
    df31T = np.array(df3[df3['M'] == 1].drop(axis='columns', columns=['PRCP', 'M'], inplace=False))
    df31P = np.array(df3[df3['M'] == 1].drop(axis='columns', columns=['T', 'M'], inplace=False))
    df38T = np.array(df3[df3['M'] == 8].drop(axis='columns', columns=['PRCP', 'M'], inplace=False))
    df38P = np.array(df3[df3['M'] == 8].drop(axis='columns', columns=['T', 'M'], inplace=False))

    ##PLOT ALL STATIONS IN JANUARY
    fig, ax = plt.subplots()
    twin1 = ax.twinx()
    p1, = ax.plot(df11P[:, 0], df11P[:, 1], "b-", label="PRCP station 1", marker ='o')
    p2, = ax.plot(df21P[:, 0], df21P[:, 1], "c-", label="PRCP station 2", marker ='o')
    p3, = ax.plot(df31P[:, 0], df31P[:, 1], "g-", label="PRCP station 3", marker ='o')
    t1, = twin1.plot(df11T[:, 0], df11T[:, 1]/10, "r-", label="Temp station 1", marker ='o')
    t2, = twin1.plot(df21T[:, 0], df21T[:, 1]/10, "m-", label="Temp station 2", marker ='o')
    t3, = twin1.plot(df31T[:, 0], df31T[:, 1]/10, "o-", label="Temp station 3")

    ax.set_ylim(0, 100)
    twin1.set_ylim(-20, 40)
    ax.set_xlabel("YEARS")
    ax.set_ylabel("PRCP")
    twin1.set_ylabel("Temperature")
    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(t1.get_color())
    ax.tick_params(axis='x')
    ax.tick_params(axis='y', colors=p1.get_color())
    twin1.tick_params(axis='y', colors=t1.get_color())
    ax.legend(handles=[p1, p2, p3, t1, t2, t3])
    plt.title('PRCP vs Temp in Germany 3 stations during January over the Years')
    plt.show()

    #January plot
    fig, ax = plt.subplots()
    twin1 = ax.twinx()
    p1, = ax.plot(df31P[:, 0], df31P[:, 1], "b-", label="PRCP", marker ='o')
    p2, = twin1.plot(df31T[:, 0], df31T[:, 1]/10, "r-", label="Temperature", marker ='o')
    ax.set_ylim(0, 100)
    twin1.set_ylim(-20, 40)
    ax.set_xlabel("YEARS")
    ax.set_ylabel("PRCP")
    twin1.set_ylabel("Temperature")
    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    ax.tick_params(axis='x')
    ax.tick_params(axis='y', colors=p1.get_color())
    twin1.tick_params(axis='y', colors=p2.get_color())
    ax.legend(handles=[p1, p2])
    plt.title('PRCP vs Temp in Germany station during January over the Years')
    plt.show()

    # August plot
    fig, ax = plt.subplots()
    twin1 = ax.twinx()
    p1, = ax.plot(df38P[:, 0], df38P[:, 1], "b-", label="PRCP", marker='o')
    p2, = twin1.plot(df38T[:, 0], df38T[:, 1] / 10, "r-", label="Temperature", marker='o')
    ax.set_ylim(0, 100)
    twin1.set_ylim(-20, 40)
    ax.set_xlabel("YEARS")
    ax.set_ylabel("PRCP")
    twin1.set_ylabel("Temperature")
    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    ax.tick_params(axis='x')
    ax.tick_params(axis='y', colors=p1.get_color())
    twin1.tick_params(axis='y', colors=p2.get_color())
    ax.legend(handles=[p1, p2])
    plt.title('PRCP vs Tempreture in Germany station during August over the Years')
    plt.show()


def data_analysis_spatial():
    """
    Analysis by spatial spase. Considering PRCP in months February and August in all station in
    Germany, Brazil and China.
    :return: Scatter plot on the map of each country
    """
    conn = connect_to('ilanit0sobol')
    dict_FIPS = {'CH': 'China', 'BR': 'Brazil', 'GM': 'Germany'}
    dict_months = {2: 'February', 8: 'August'}
    for country in dict_FIPS.keys():
        for month in dict_months.keys():
            query = """select StationId, LONGITUDE, LATITUDE, AvgPrcp
                        from Weather
                        where FIPS_code = ?
                        and Year = 2000 and Month = ?
                        group by StationId, LONGITUDE, LATITUDE, AvgPrcp"""
            df = pd.read_sql(query, conn, params=[country, month])
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']))
            ax = 0
            if country == 'CH':
                ax = world[world.name == dict_FIPS[country]].plot(color='white', edgecolor='black')
            elif country == 'BR':
                ax = world[world.name == dict_FIPS[country]].plot(color='white', edgecolor='black')
            elif country == 'GM':
                ax = world[world.name == dict_FIPS[country]].plot(color='white', edgecolor='black')
            gdf.plot(column='AvgPrcp', ax=ax, marker='o', markersize=4, legend=True,
                     legend_kwds={'label': "Average PRCP", 'shrink': 1})
            ax.set_title(dict_FIPS[country] + ' ' + dict_months[month] + ', 2000')
            plt.show()


def data_analysis_spatialANDtemporal():
    """
    Analysis by temporal&spatial spase. Considering PRCP in 3 different countries during 3 different years
    """
    conn = connect_to('ilanit0sobol')
    cursor = conn.cursor()
    query = """SELECT DISTINCT W.StationId as StationId, W.month as M, W.AvgPrcp as PRCP, W.Year as Y
                FROM Weather W
                WHERE (W.stationId in (select DISTINCT TOP 1 W1.StationId
                                        from Weather W1 
                                        WHERE W1.FIPS_code = 'GM')
                                or (W.stationId in (select DISTINCT TOP 1 W1.StationId
                                        from Weather W1 
                                         WHERE W1.FIPS_code = 'GR'))
                                or (W.stationId in (select DISTINCT TOP 1 W1.StationId
                                        from Weather W1 
                                        WHERE W1.FIPS_code = 'CH')))
                and (W.Year= 1995 or W.Year = 1985 or W.Year= 1975)
             """
    df = pd.read_sql(query, conn)
    S = list(set(df['StationId']))

    #plot for 1995
    df_95 = df[df["Y"] == 1995]
    df1_95 = np.array(df_95[df_95['StationId'] == S[0]].drop(axis='columns', columns=['StationId','Y'], inplace=False))
    df2_95 = np.array(df_95[df_95['StationId'] == S[1]].drop(axis='columns', columns=['StationId','Y'], inplace=False))
    df3_95 = np.array(df_95[df_95['StationId'] == S[2]].drop(axis='columns', columns=['StationId','Y'], inplace=False))
    x = list(set(df['M']))
    plt.plot(df1_95[:,0], df1_95[:,1], '-ob', label=S[0])
    plt.plot(df2_95[:,0], df2_95[:,1], '-ok', label=S[1])
    plt.plot(df3_95[:,0], df3_95[:,1], '-or', label=S[2])
    plt.xticks(x)
    plt.xlabel("MONTHS")
    plt.ylabel("PRCP")
    plt.legend()
    plt.title('PRCP in 3 Countries during 1995')
    plt.show()

    #plot for 2000
    df_05 = df[df["Y"] == 1985]
    df1_05 = np.array(df_05[df_05['StationId'] == S[0]].drop(axis='columns', columns=['StationId','Y'], inplace=False))
    df2_05 = np.array(df_05[df_05['StationId'] == S[1]].drop(axis='columns', columns=['StationId','Y'], inplace=False))
    df3_05 = np.array(df_05[df_05['StationId'] == S[2]].drop(axis='columns', columns=['StationId','Y'], inplace=False))
    x = list(set(df['M']))
    plt.plot(df1_05[:,0], df1_05[:,1], '-ob', label=S[0])
    plt.plot(df2_05[:,0], df2_05[:,1], '-ok', label=S[1])
    plt.plot(df3_05[:,0], df3_05[:,1], '-or', label=S[2])
    plt.xticks(x)
    plt.xlabel("MONTHS")
    plt.ylabel("PRCP")
    plt.legend()
    plt.title('PRCP in 3 Countries during 1985')
    plt.show()

    # plot for 2015
    df_15 = df[df["Y"] == 1975]
    df1_15 = np.array(df_15[df_15['StationId'] == S[0]].drop(axis='columns', columns=['StationId','Y'], inplace=False))
    df2_15 = np.array(df_15[df_15['StationId'] == S[1]].drop(axis='columns', columns=['StationId','Y'], inplace=False))
    df3_15 = np.array(df_15[df_15['StationId'] == S[2]].drop(axis='columns', columns=['StationId','Y'], inplace=False))

    x = list(set(df['M']))
    plt.plot(df1_15[:,0], df1_15[:,1], '-ob', label=S[0])
    plt.plot(df2_15[:,0], df2_15[:,1], '-ok', label=S[1])
    plt.plot(df3_15[:,0], df3_15[:,1], '-or', label=S[2])
    plt.xticks(x)
    plt.xlabel("MONTHS")
    plt.ylabel("PRCP")
    plt.legend()
    plt.title('PRCP in 3 Countries during 1975')
    plt.show()


if __name__ == '__main__':
    print("Temporal data analysis")
    data_analysis_temporal()
    print("Spatial data analysis")
    data_analysis_spatial()
    print("Spatial and Temporal data analysis")
    data_analysis_spatialANDtemporal()
