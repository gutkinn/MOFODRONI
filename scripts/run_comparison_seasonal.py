import geopandas as gpd
import os
import glob
import pandas as pd
from pathlib import Path
import rasterio
from statsmodels.regression.linear_model import OLS

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

def flatten(xss):
    return [x for xs in xss for x in xs]


def rework_riceplots(rice_plots):
    """Rework the rice plots GeoDataFrame by calculating the area and yield per hectare for each year, as well as the change in yield compared to the previous year."""
    rice_plots = rice_plots.set_crs('4326')
    rice_plots['area'] = rice_plots.to_crs('3857').geometry.area / 10000

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

def extract_indices_year(spindices_year,rice_plots,VICI_folder,SWIA_folder,year):
    """Extract the relevant indices for the specified year from the provided SPI/SPEI DataFrame, rice plots GeoDataFrame, and folders containing VICI and SWIA data."""
    remove_ids = []
    id_month_dict = {}
    VICI_df = pd.DataFrame(columns = ['id','dt','VICI'])
    SWIA_df = pd.DataFrame(columns = ['id','dt','SWIA'])
    
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
    
        
        season_months = [months_dict[start_m], months_dict[end_m]]
        season_months = [f'{i:02}' for i in range(int(season_months[0]),int(season_months[1])+1)]
        id_month_dict[row['id']] = season_months
        subset_SWIA = flatten([sorted(glob.glob(os.path.join(SWIA_folder,f'SWIA-{year}-{month}*.tif'))) for month in season_months])
        subset_VICI = flatten([sorted(glob.glob(os.path.join(VICI_folder,f'VICI-{year}-{month}*.tif'))) for month in season_months])
        
        for dek in subset_VICI:
            
            with rasterio.open(dek) as src:
                val = [x for x in src.sample([(cent.x,cent.y)])][0][0]
            VICI_df = pd.concat([VICI_df,pd.DataFrame({'id':[row.id],'dt':[dek.split('/')[-1][5:-4]],'VICI':[val]})])
        
        for dek in subset_SWIA:
            
            with rasterio.open(dek) as src:
                val = [x for x in src.sample([(cent.x,cent.y)])][0][0]
            SWIA_df = pd.concat([SWIA_df,pd.DataFrame({'id':[row.id],'dt':[dek.split('/')[-1][5:-4]],'SWIA':[val]})])
            
    spindices_year = spindices_year[spindices_year['id'].isin(list(set(range(rice_plots['id'].min(),rice_plots['id'].max()+1)) & set(id_month_dict.keys())))]
    spindices_year = spindices_year[spindices_year.apply(lambda x: x['dt'][5:7] in id_month_dict[x['id']], axis=1)]
    subset_indices = extract_stat_subset(spindices_year)

    subset_VICI = extract_stat_subset(VICI_df)
    subset_SWIA = extract_stat_subset(SWIA_df)

    subset = subset_indices.merge(subset_VICI, on=['id'])
    subset = subset.merge(subset_SWIA, on=['id'])
  
    return subset

def extract_stat_subset(spindices_year):
    """Extract statistical features (max, median, min, std, count) from the provided SPI/SPEI DataFrame for each plot ID."""
    subset_max = spindices_year.drop(columns='dt').groupby('id').max()
    subset_max.rename(columns={col:col+'_max' for col in subset_max.columns},inplace=True)
    subset_median = spindices_year.drop(columns='dt').groupby('id').median()
    subset_median.rename(columns={col:col+'_median' for col in subset_median.columns},inplace=True)
    subset_min = spindices_year.drop(columns='dt').groupby('id').min()
    subset_min.rename(columns={col:col+'_min' for col in subset_min.columns},inplace=True)
    subset_std = spindices_year.drop(columns='dt').groupby('id').std()
    subset_std.rename(columns={col:col+'_std' for col in subset_std.columns},inplace=True)
    subset_count_swia = spindices_year.fillna(0).drop(columns='dt').groupby('id').agg(lambda x: x.ne(0).sum())
    subset_count_swia = subset_count_swia.drop(columns=[col for col in subset_count_swia.columns if 'SWIA' not in col])
    subset_count_swia.rename(columns={'SWIA':'SWIA_count'},inplace=True)
    subset_count_vici = spindices_year.fillna(0).drop(columns='dt').groupby('id').agg(lambda x: x.ne(0).sum())
    subset_count_vici = subset_count_vici.drop(columns=[col for col in subset_count_vici.columns if 'VICI' not in col])
    subset_count_vici.rename(columns={'VICI':'VICI_count'},inplace=True)
    subset = pd.concat([subset_max,subset_median,subset_min,subset_std,subset_count_swia,subset_count_vici],axis=1)

    return subset

def merge_plots_indices(rice_plots,subset,year):
    """Merge the rice plots GeoDataFrame with the extracted indices DataFrame for the specified year, and filter out outliers based on yield change values."""
    yield_field = f'change{year}'
    merged = rice_plots[['id',yield_field] + list(rice_plots.columns[48:])].merge(subset,left_on='id',right_on='id')
    print(f'initial: {len(merged)}')
    merged['yield'] = merged[yield_field].astype('float32')
    initial_len = len(merged)

    merged = merged[merged['yield'].between(merged['yield'].quantile(0.05),merged['yield'].quantile(0.95))]

    merged.dropna(subset='yield',inplace=True)
    print(f'{initial_len - len(merged)} removed')
    return merged
    
def run_calc(year, zones, thresholds, rem_cols, mode):
    
    seas_mode = 'seasonal'
    main_dir = Path(os.getcwd()).parent
    
    rice_plots_file = gpd.read_file(os.path.join(main_dir,'shapefiles','field_data','RICE_PLOTS_POLYGONS_RF1_fix.shp'))
    
    print('Imported plots.')
    
    spindices = pd.read_csv(os.path.join(main_dir,'shapefiles','rice_plots_spi_spei_data.csv'))
    spindices.drop(columns = spindices.columns[list(spindices.columns).index('id')+1:],inplace=True)
    spindices['dt'] = spindices.apply(lambda x: x['time'].split(' ')[0],axis=1)
    spindices.drop(columns = ['time','lat','lon'], inplace=True)
    
    rice_plots = rework_riceplots(rice_plots_file)

    # create dummy variables
    rice_plots = pd.merge(rice_plots,pd.get_dummies(rice_plots[['SOILtype','RICEvarty']],dtype=int),left_index=True, right_index=True)
    
    out_mod_dir = os.path.join(main_dir,'Output','model_analysis2')
    
    os.makedirs(out_mod_dir, exist_ok=True)

    for zone in zones:
        print(f'Running {zone} zones')
        r2_dict = {'zones':[zone]}
        mse_dict = {'zones':[zone]}
        p_dict = {'zones':[zone]}
        for u_thresh in thresholds:
            print(f'Running {u_thresh} threshold')
            SWIA_folder = os.path.join(main_dir,'Output','analysis_swia',f'{zone:02d}zones/analysis_05_{u_thresh}','SWIA')
            NDVIA_folder = os.path.join(main_dir,'Output','analysis',f'{zone:02d}zones/analysis_05_{u_thresh}','VICI')
            summary_folder = os.path.join(out_mod_dir,'raw_data',seas_mode,'summaries')
            model_vars_folder = os.path.join(out_mod_dir,'raw_data',seas_mode,mode)
            
            os.makedirs(summary_folder, exist_ok=True)
            os.makedirs(model_vars_folder, exist_ok=True)

            print(f'Processing {year}')
            subset = extract_indices_year(spindices[spindices['dt'].str.contains(str(year))],
                                          rice_plots.copy(),
                                          NDVIA_folder,
                                          SWIA_folder,
                                          year)
                
            merged_year = merge_plots_indices(rice_plots.copy(),subset,year)
            merged = merged_year[merged_year.columns.drop(f'change{year}')]
                    
            print(f'plots used: {len(merged)}')
            in_cols = [col for col in merged.columns[1:] if 'yield' not in col]
            print(f'in_cols: {in_cols}')
            for rem_col in list(rem_cols):
                print(rem_col)
                in_cols = [col for col in in_cols if rem_col not in col]
                print(in_cols)
            
            pred_col = merged['yield']
            print(f'in_cols: {in_cols}')
            simple_reg = OLS(pred_col,merged[in_cols],missing='drop')
            acc_mets = simple_reg.fit()
    
            with open(os.path.join(summary_folder,f'{year}_{zone}_{u_thresh}_{mode}_summary.csv'),'w') as fh:
                fh.write(acc_mets.summary().as_csv())

            r2_dict[u_thresh]=[acc_mets.rsquared_adj]
            mse_dict[u_thresh]=[acc_mets.mse_model]
            p_dict[u_thresh]=[acc_mets.f_pvalue]

            print(f'Model for {zone} zones and p{u_thresh} R²: {acc_mets.rsquared_adj} MSE: {acc_mets.mse_model} pval: {acc_mets.f_pvalue}')
        if zone == zones[0]:
            r2_df = pd.DataFrame(r2_dict)
            mse_df = pd.DataFrame(mse_dict)
            p_df = pd.DataFrame(p_dict)
        else:
            r2_df = pd.concat([r2_df,pd.DataFrame(r2_dict)])
            mse_df = pd.concat([mse_df,pd.DataFrame(mse_dict)])
            p_df = pd.concat([p_df,pd.DataFrame(p_dict)])
        
        r2_df.to_csv(os.path.join(model_vars_folder,f'R2_SWIANDVIA{year}_{zone}_update.csv'),sep=';')
        mse_df.to_csv(os.path.join(model_vars_folder,'all_dek_seasons',f'MSE_SWIANDVIA{year}_{zone}_update.csv'),sep=';')
        p_df.to_csv(os.path.join(model_vars_folder,'all_dek_seasons',f'p_SWIANDVIA{year}_{zone}_update.csv'),sep=';')

        
if __name__ == '__main__':
    import configparser
    import argparse
    import json
    import ast
    
    config = configparser.ConfigParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required = True)
    args = parser.parse_args()
    config.read(args.config)

    run_calc(config['SETTINGS']['year'], json.loads(config.get('SETTINGS','zones')), json.loads(config.get('SETTINGS','thresholds')),
            ast.literal_eval(config.get('VARIABLES','cols_to_remove')), config.get('VARIABLES','mode'))
