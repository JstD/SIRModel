"""
Collect data from git reposistory
$python regionize_data [region]
"""

import pandas as pd
import numpy as np
import os
import sys


# covid-19
COVID_19_DATA_URL = "https://github.com/CSSEGISandData/COVID-19"

COVID_19_DATA_DIR = "covid19"

COVID_19_CONFIRMED_FILE = "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
COVID_19_DEATH_PATH_FILE = "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
COVID_19_RECOVERED_FILE = "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"


# population
POPULATION_DATA_URL = "https://github.com/CSSEGISandData/COVID-19"

POPULATION_DATA_DIR = "covid19"

POPULATION_FILE = "csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv"


# country to region mapping
COUNTRY_TO_REGION_MAPPING_URL = "https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes"

COUNTRY_TO_REGION_MAPPING_DIR = "country2region"

COUNTRY_TO_REGION_MAPPING_FILE = "all/all.csv"


def fetchRepo():
    """
    Get latest raw data from repositories.

    Returns
    -------
    None.

    """
    # get covid19 data
    if os.path.isdir(COVID_19_DATA_DIR):
        os.system(f"cd {COVID_19_DATA_DIR} & git pull")
    else:
        os.system(f"git clone {COVID_19_DATA_URL} {COVID_19_DATA_DIR}")
    
    # get populaion data
    if os.path.isdir(POPULATION_DATA_DIR):
        os.system(f"cd {POPULATION_DATA_DIR} & git pull")
    else:
        os.system(f"git clone {POPULATION_DATA_URL} {POPULATION_DATA_DIR}")
        
    # get country to region map
    if os.path.isdir(COUNTRY_TO_REGION_MAPPING_DIR):
        os.system(f"cd {COUNTRY_TO_REGION_MAPPING_DIR} & git pull")
    else:
        os.system(f"git clone {COUNTRY_TO_REGION_MAPPING_URL} {COUNTRY_TO_REGION_MAPPING_DIR}")
    

def convertRawData(model='SIR'):
    """
    Convert raw data to a specified model's data format.

    Parameters
    ----------
    model : string, optional
        The model to convert raw data to.
        The default is 'SIR'.
        Can recognize 'SIR', 'SIRD'.
        None for Raw/Original.

    Returns
    -------
    out : tuple of pd.Series-es
        Data in the specified model's format.

    """
    
    # get latest raw data
    fetchRepo()
    
    # open all raw data file
    confirmed = pd.read_csv(COVID_19_DATA_DIR + "/" + COVID_19_CONFIRMED_FILE)
    death = pd.read_csv(COVID_19_DATA_DIR + "/" + COVID_19_DEATH_PATH_FILE)
    recovered = pd.read_csv(COVID_19_DATA_DIR + "/" + COVID_19_RECOVERED_FILE)
    population = pd.read_csv(POPULATION_DATA_DIR + "/" + POPULATION_FILE)
    country2region = pd.read_csv(COUNTRY_TO_REGION_MAPPING_DIR + "/" + COUNTRY_TO_REGION_MAPPING_FILE)
    
    # define unncessary columns
    excess_columns = {
        "confirmed" : ["Province/State", "Lat", "Long"],
        "death" : ["Province/State", "Lat", "Long"],
        "recovered" : ["Province/State", "Lat", "Long"],
        "population" : ["UID", "iso2", "iso3", "code3", "FIPS", "Admin2", "Province_State", "Lat", "Long_", "Combined_Key"],
        "country2region" : ["alpha-2", "alpha-3", "country-code", "iso_3166-2", "region-code", "sub-region-code", "intermediate-region-code"]
        }
    
    # drop unncessary columns
    confirmed = confirmed.drop(columns=excess_columns["confirmed"])
    death = death.drop(columns=excess_columns["death"])
    recovered = recovered.drop(columns=excess_columns["recovered"])
    population = population.drop(columns=excess_columns["population"])
    country2region = country2region.drop(columns=excess_columns["country2region"])
    
    # convert country2region from dataframe to a map mapping from a lower-level region to the next higher
    country2region_map = {}
    for i, row in country2region.iterrows():
        # nan handling
        name = row["name"]
        intermediate_region = row["intermediate-region"] if pd.notna(row["intermediate-region"]) else name + "1"
        sub_region = row["sub-region"] if pd.notna(row["sub-region"]) else name + "2"
        region = row["region"] if pd.notna(row["region"]) else name + "3"
        
        # actual mapping
        country2region_map[name] = intermediate_region
        country2region_map[intermediate_region] = sub_region
        country2region_map[sub_region] = region
    del country2region
    
    # group provinces/states by country
    confirmed = confirmed.groupby("Country/Region", as_index=False).sum()
    death = death.groupby("Country/Region", as_index=False).sum()
    recovered = recovered.groupby("Country/Region", as_index=False).sum()
    population = population.groupby("Country_Region", as_index=False).sum().astype({"Population" : np.int64})
    
    # from country/region, form bigger regions
    confirmed_copy = confirmed
    death_copy = death
    recovered_copy = recovered
    population_copy = population
    for i in range(3): # no of loops = no  of levels of regions defined 
        # replace by mapped value
        confirmed_copy = confirmed_copy.replace({"Country/Region" : country2region_map})
        death_copy = death_copy.replace({"Country/Region" : country2region_map})
        recovered_copy = recovered_copy.replace({"Country/Region" : country2region_map})
        population_copy = population_copy.replace({"Country_Region" : country2region_map})
            
        # group
        confirmed_copy = confirmed_copy.groupby("Country/Region", as_index=False).sum()
        death_copy = death_copy.groupby("Country/Region", as_index=False).sum()
        recovered_copy = recovered_copy.groupby("Country/Region", as_index=False).sum()
        population_copy = population_copy.groupby("Country_Region", as_index=False).sum()
        
        # concat to *_res
        confirmed = pd.concat([confirmed_copy, confirmed])
        death = pd.concat([death_copy, death])
        recovered = pd.concat([recovered_copy, recovered])
        population = pd.concat([population_copy, population])
    del confirmed_copy, death_copy, recovered_copy, population_copy, country2region_map
    
    # remove dummy rows
    confirmed = confirmed[confirmed["Country/Region"].map(lambda x: str(x)[-1] not in ['1', '2', '3']) ]
    death = death[death["Country/Region"].map(lambda x: str(x)[-1] not in ['1', '2', '3'])]
    recovered = recovered[recovered["Country/Region"].map(lambda x: str(x)[-1] not in ['1', '2', '3'])]
    population = population[population["Country_Region"].map(lambda x: str(x)[-1] not in ['1', '2', '3'])]
    
    # use country/region as index
    confirmed = confirmed.groupby("Country/Region").last()
    death = death.groupby("Country/Region").last()
    recovered = recovered.groupby("Country/Region").last()
    population = population.groupby("Country_Region").last()
    
    
    model = model.upper()
    
    # make data suited to use in SIR model
    if model == 'SIR':
        recovered = recovered + death
        infectious = confirmed - recovered
        susceptible = pd.concat([population] * len(infectious.columns), axis=1)
        susceptible.columns = infectious.columns
        susceptible = susceptible - confirmed
        
        return susceptible, infectious, recovered
    
    # make data suited to use in SIRD model
    if model == 'SIRD':
        infectious = confirmed - recovered - death
        susceptible = pd.concat([population] * len(infectious.columns), axis=1)
        susceptible.columns = infectious.columns
        susceptible = susceptible - confirmed
        
        return susceptible, infectious, recovered, death
    
    return confirmed, death, recovered


def regionize(region='Europe', model='SIR'):
    """
    Fetch data for a specific model for a specific region.

    Parameters
    ----------
    region : string, optional
        The country/region to fetch data about.
        The default is 'Europe'.
        None for All/Global.    
    model : string, optional
        The model to convert raw data to.
        The default is 'SIR'.
        Can recognize 'SIR', 'SIRD'.
        None for Raw/Original.

    Returns
    -------
    out : tuple of pd.Series-es
        Data of the specified region in the specified model's format.

    """
    
    model = model.upper()
    
    if model == 'SIR':
        susceptible, infectious, recovered = convertRawData(model='SIR')
    
        if region:
            susceptible = susceptible.loc[region]
            infectious = infectious.loc[region]
            recovered = recovered.loc[region]
        else:
            susceptible = susceptible.sum()
            infectious = infectious.sum()
            recovered = recovered.sum()
            
        return susceptible, infectious, recovered
    
    if model == 'SIRD':
        susceptible, infectious, recovered, death = convertRawData(model='SIRD')
    
        if region:
            susceptible = susceptible.loc[region]
            infectious = infectious.loc[region]
            recovered = recovered.loc[region]
            death = death.loc[region]
        else:
            susceptible = susceptible.sum()
            infectious = infectious.sum()
            recovered = recovered.sum()
            death = death.sum()
            
        return susceptible, infectious, recovered, death

    return convertRawData(model=None)


# if __name__ == '__main__':
#     # if len(sys.argv) > 1:
#     #     for df in regionize(sys.argv[1]):
#     #         print(df)
#     # else:
#     #     for df in regionize(None):
#     #         print(df)
    
#     import plot
    
#     s, i, r, d = regionize(model='sird')
#     plot.plot_data(s, i, r)
#     print(d)