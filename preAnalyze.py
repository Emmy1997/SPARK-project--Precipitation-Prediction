import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pre_analyze():
    data = pd.read_fwf('ghcnd-inventory.txt', header=None, names=["ID", "LATITUDE", "LONGITUDE", "VARIABLE",
                                                                  "FIRSTYEAR", "LASTYEAR"], sep=' ')
    data= data[(data["FIRSTYEAR"].astype("int") >= 1970) | (data["LASTYEAR"].astype("int")> 1970)]
    data.drop(axis='columns', columns=["LATITUDE", "LONGITUDE", "LASTYEAR", "FIRSTYEAR"], inplace=True)
    parameters_dict = {'Brazil': {}, 'China': {}, 'Greece':{},
                   'Germany': {}, 'Israel': {}}
    station_dict = {'Brazil': 0, 'China': 0, 'Greece': 0,
                    'Germany': 0, 'Israel': 0}
    reg_list = [r'^CH(?!$)', r'^GM(?!$)', r'^GR(?!$)', r'^BR(?!$)', r'^IS(?!$)' ]

    #compute count of non null parameters for every country
    for c, r in zip(parameters_dict.keys(), reg_list):
        df = data[data['ID'].astype(str).str.contains(r)]
        station_dict[c] = len(list(set(df["ID"])))
        variables = list(set(df["VARIABLE"]))
        sorted_variables = {v: df[df['VARIABLE'] == v].shape[0] for v in variables}
        parameters_dict[c] = dict(sorted(sorted_variables.items(), key=lambda kv: kv[1], reverse=True)[:5])
        print("num of stations in " + str(c) + ': ' + str(station_dict[c]))

    #plot station number
    plt.title('Stations count by Country')
    plt.pie(np.array(list(station_dict.values())), labels=station_dict.keys())
    plt.show()

    def func(pct, allvals):
        """compute precents
        """
        absolute = int(round(pct / 100. * sum(list(allvals))))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    #plot non null paremeters across the countries
    for country, parameters in parameters_dict.items():
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
        plt.title('Not Nulls Paremeter count of ' + country)
        names, values = parameters.keys(), parameters.values()
        wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values), textprops=dict(color="w"))
        ax.legend(wedges, names, title="Variables", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=6, weight="bold")
        plt.show()
        plt.pause(0.1)


if __name__ == '__main__':
    pre_analyze()