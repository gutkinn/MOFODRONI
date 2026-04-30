import geopandas as gpd
import os
import glob
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import math
import seaborn as sns
from osgeo import gdal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.iolib.summary2 import summary_col

def rearrange_plotvals(in_plots,year):
    plots = in_plots.copy()
    plots['Drought degree'] = plots['C9']
    plots.drop(plots.loc[plots[f'?INC{year}'] == 'NIL'].index, inplace=True)
    plots[f'change{year}'] = plots[f'?INC{year}'].astype(float)

    plots[f'bin{year}'],bins = pd.qcut(plots[f'change{year}'],q=5,retbins=True,labels=False)
    bin_labels = [str(np.round(bins[0],1)) + ' to ' + str(np.round(bins[1],1)),
        str(np.round(bins[1],1)) + ' to ' + str(np.round(bins[2],1)),
        str(np.round(bins[2],1)) + ' to ' + str(np.round(bins[3],1)),
        str(np.round(bins[3],1)) + ' to ' + str(np.round(bins[4],1)),
        str(np.round(bins[4],1)) + ' to ' + str(np.round(bins[5],1))]

    plots[f'bin{year}'],bins = pd.qcut(plots[f'change{year}'],q=5,retbins=True,labels=bin_labels)
    
    return plots

def flatten(xss):
    return [x for xs in xss for x in xs]

def arrange_dt(dt_str):
    return dt_str[0:4]+'-'+dt_str[4:6]+'-'+dt_str[6:]
    
def obs_plot(obs_df, out_file):
    ax = obs_df.plot()
    ax.set_ylabel('Num. drought observations')
    ax.set_xlabel('Upper threshold (%)')
    plt.savefig(out_file)
    
def rework_riceplots(rice_plots):
    rice_plots = rice_plots.set_crs('4326')
    rice_plots['area'] = rice_plots.to_crs('3587').geometry.area / 10000

    rice_plots['2019 t/h'] = rice_plots['2019 M/T'].astype('float32') / rice_plots['area']
    rice_plots['2020 t/h'] = rice_plots['2020 M/T'].astype('float32') / rice_plots['area']
    rice_plots['2021 t/h'] = rice_plots['2021 M/T'].astype('float32') / rice_plots['area']
    rice_plots['2022 t/h'] = rice_plots['2022 M/T'].astype('float32') / rice_plots['area']
    rice_plots['2023 t/h'] = rice_plots['2023 M/T'].astype('float32') / rice_plots['area']
    rice_plots['2024 t/h'] = rice_plots['2024 M/T'].astype('float32') / rice_plots['area']
    
    rice_plots['change2024'] = rice_plots['2024 t/h'].astype('float32') - rice_plots['2023 t/h'].astype('float32')
    rice_plots['change2023'] = rice_plots['2023 t/h'].astype('float32') - rice_plots['2022 t/h'].astype('float32')
    rice_plots['change2022'] = rice_plots['2022 t/h'].astype('float32') - rice_plots['2021 t/h'].astype('float32')
    rice_plots['change2021'] = rice_plots['2021 t/h'].astype('float32') - rice_plots['2020 t/h'].astype('float32')
    rice_plots['change2020'] = rice_plots['2020 t/h'].astype('float32') - rice_plots['2019 t/h'].astype('float32')

    return rice_plots

def gen_pca_comp(in_vars,num):
    PCA_input = in_vars
    X_std = StandardScaler().fit_transform(PCA_input)
    pca = PCA(n_components=PCA_input.shape[1])
    principalComponents = pca.fit_transform(X_std)# Plot the explained variances
    features = range(pca.n_components_)
    PCA_components = pd.DataFrame(principalComponents)
    return PCA_components[[col for col in range(num)]]

def extract_indices_year(spindices_year,rice_plots,VICI_folder,SM_folder):
    remove_ids = []
    id_month_dict = {}
    VICI_df = pd.DataFrame(columns = ['id','dt','VICI'])
    #SM_df = pd.DataFrame(columns = ['id','dt','SM'])
    
    for i,row in rice_plots.iterrows():
        cent = row.geometry.centroid
    
        #set start and end months
        start_m = row['RCvty1TRP'].split('/')[0]
        end_m = row['RCvty1HVT'].split('/')[-1]
    
        #remove invalid data points
        if (start_m == 'NIL') or (end_m == 'NIL'):
            remove_ids.append(row['id'])
            continue
        if row[f'{year} M/T'] == 'NIL':
            remove_ids.append(row['id'])
            continue
        if row[f'?INC{year}'] == 'NIL':
            remove_ids.append(row['id'])
            continue
        
        season_months = [months_dict[start_m], months_dict[end_m]]
        season_months = [f'{i:02}' for i in range(int(season_months[0]),int(season_months[1])+1)]
        id_month_dict[row['id']] = season_months
        subset_VICI = flatten([sorted(glob.glob(os.path.join(VICI_folder,f'VICI-{year}-{month}*.tif'))) for month in season_months])
        
        for dek in subset_VICI:
            
            with rasterio.open(dek) as src:
                val = [x for x in src.sample([(cent.x,cent.y)])][0][0]
            VICI_df = pd.concat([VICI_df,pd.DataFrame({'id':[row.id],'dt':[dek.split('/')[-1][5:-4]],'VICI':[val]})])
            
        #subset_SM = flatten(sorted([glob.glob(os.path.join(SM_folder,f'*{year}{month}*.tif')) for month in season_months]))
        
        #for dek in subset_SM:
            
            #with rasterio.open(dek) as src:
            #   val = int([x for x in src.sample([(cent.x,cent.y)])][0][0])
            #SM_df = pd.concat([SM_df,pd.DataFrame({'id':[row.id],'dt':[arrange_dt(dek.split('/')[-1][12:20])],'SM':[val]})])
            
    #SM_df = SM_df.astype({'SM':'int16'})
    
    spindices_year = spindices_year[spindices_year['id'].isin(list(set(range(rice_plots['id'].min(),rice_plots['id'].max()+1)) & set(id_month_dict.keys())))]
    spindices_year = spindices_year[spindices_year.apply(lambda x: x['dt'][5:7] in id_month_dict[x['id']], axis=1)]
    spindices_year = spindices_year.merge(VICI_df, on=['id', 'dt'])
    #spindices_year = spindices_year.merge(SM_df, on=['id', 'dt'])
    subset = extract_stat_subset(spindices_year)
  
    return subset

def extract_stat_subset(spindices_year):
    subset_max = spindices_year.drop(columns='dt').groupby('id').max()
    subset_max.rename(columns={col:col+'_max' for col in subset_max.columns},inplace=True)
    subset_median = spindices_year.drop(columns='dt').groupby('id').median()
    subset_median.rename(columns={col:col+'_median' for col in subset_median.columns},inplace=True)
    subset_min = spindices_year.drop(columns='dt').groupby('id').min()
    subset_min.rename(columns={col:col+'_min' for col in subset_min.columns},inplace=True)
    subset_std = spindices_year.drop(columns='dt').groupby('id').std()
    subset_std.rename(columns={col:col+'_std' for col in subset_std.columns},inplace=True)
    subset_count = spindices_year.drop(columns='dt').groupby('id').count()
    subset_countnz = spindices_year.fillna(0).drop(columns='dt').groupby('id').agg(lambda x: x.ne(0).sum())
    subset_countnz = subset_countnz.drop(columns=[col for col in subset_countnz.columns if 'VICI' not in col])
    subset_countnz.rename(columns={'VICI':'VICI_count'},inplace=True)
    subset = pd.concat([subset_max,subset_median,subset_min,subset_std,subset_countnz],axis=1)

    return subset

def merge_plots_indices(rice_plots,subset,year):
    yield_field = f'change{year}'
    merged = rice_plots[['id',yield_field] + list(rice_plots.columns[48:])].merge(subset,left_on='id',right_on='id')
    
    merged['yield'] = merged[yield_field].astype('float32')
    initial_len = len(merged)
    iqr = rice_plots[yield_field].quantile(0.75) - rice_plots[yield_field].quantile(0.25)
    #merged = merged[merged['yield'] < merged['yield'].quantile(0.9)]
    """if year == 2020:
        merged.drop([122,123], inplace=True)
    if year == 2021:
        merged.drop([56], inplace=True)
    if year == 2022:
        merged.drop([38,136], inplace=True)"""
    
    merged.dropna(subset='yield',inplace=True)
    merged = merged[merged['yield'] != 0]
    print(f'{initial_len - len(merged)} removed')
    return merged

months_dict = {'MAR':'03',
               'APR':'04',
               'MAY':'05',
               'JUN':'06',
               'JUL':'07',
               'SEP':'09',
               'AUG':'08',
               'OCT':'10',
               'NOV':'11',
               'DEC':'12',
               'NIL':'NIL'
              }
rice_plots = gpd.read_file(r'/home/eoafrica/shared/MOFODRONI/shapefiles/field_data/RICE_PLOTS_POLYGONS_RF1_fix.shp')

out_images = r'/home/eoafrica/shared/MOFODRONI/Output_noharmonics/images'

main_dir = Path(os.getcwd()).parent

nigeria_shp = os.path.join(main_dir, 'shapefiles','nigeria_country.shp')
nigeria_vector = gpd.read_file(nigeria_shp)['geometry']

#BoundingBox - Set values to the desired bbox
minX = 2.6 #2.6
maxX = 14.5 #14.5
minY = 4.2 #4.2
maxY = 13.8 #13.8
bbox = [minX,minY,maxX,maxY]

print('Imported plots.')

spindices = pd.read_csv('/home/eoafrica/shared/MOFODRONI/shapefiles/rice_plots_spi_spei_data.csv')
spindices.drop(columns = spindices.columns[list(spindices.columns).index('id')+1:],inplace=True)
spindices['dt'] = spindices.apply(lambda x: x['time'].split(' ')[0],axis=1)
spindices.drop(columns = ['time','lat','lon'], inplace=True)

SM_folder = r'/home/eoafrica/shared/MOFODRONI/SM'

rice_plots = rework_riceplots(gpd.read_file(r'/home/eoafrica/shared/MOFODRONI/shapefiles/field_data/RICE_PLOTS_POLYGONS_RF1_fix.shp'))
# create dummy variables
rice_plots = pd.merge(rice_plots,pd.get_dummies(rice_plots[['SOILtype','RICEvarty']],dtype=int),left_index=True, right_index=True)

out_mod_dir = r'/home/eoafrica/shared/MOFODRONI/Output/out_model_analysis_noSM'

zones = [5,10,20,30,40,50,60,70,80,90,100,110,120]
thresholds = [15,20,25,30,35,40,45,50]
years = [2020,2024]
for zone in zones:
    print(f'Running {zone} zones')
    zone_dict = {'zones':[zone]}
    for u_thresh in thresholds:
        print(f'Running {u_thresh} threshold')
        VICI_folder = f'/home/eoafrica/shared/MOFODRONI/Output/analysis/{zone:02d}zones/analysis_05_{u_thresh}/VICI'
        
        for year in years:
            print(f'Processing {year}')
            subset = extract_indices_year(spindices[spindices['dt'].str.contains(str(year))],
                                                  rice_plots,
                                                  VICI_folder,
                                                  SM_folder)
            merged_year = merge_plots_indices(rice_plots,subset,year)
            if year == years[0]:
                merged = merged_year[merged_year.columns.drop(f'change{year}')]
            if year != years[0]:
                merged = pd.concat([merged,merged_year[merged_year.columns.drop(f'change{year}')]])

        in_cols = [col for col in merged.columns[1:] if 'yield' not in col]
        in_cols = [col for col in in_cols if 'VICI' not in col]
        pred_col = merged['yield']
        print(f'in_cols: {in_cols}')
        simple_reg = OLS(pred_col,merged[in_cols],missing='drop')
        r2_score = simple_reg.fit().rsquared
        #r2_score = np.sum(merged['VICI_count'])
        zone_dict[u_thresh]=[r2_score]
        print(f'Model for {zone} zones and p{u_thresh} R²: {r2_score}')
    if zone == zones[0]:
        zone_df = pd.DataFrame(zone_dict)
    else:
        zone_df = pd.concat([zone_df,pd.DataFrame(zone_dict)])
    zone_df.to_csv(os.path.join(out_mod_dir,'noVICI',f'{years}_{zone}_update.csv'),sep=';')