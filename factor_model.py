import numpy as np
import pandas as pd

fema_county_data = pd.read_csv('fema_data/NRI_Table_Counties.csv')



winter_storm_preffixes = ['AVLN','CWAV','ISTM','WNTW'] # Avalanche, Cold Wave, Ice Storm, Winter Weather
convective_storm_preffixes = ['HAIL','LTNG','SWND','TRND'] # Hail, Lighting, Strong Wind, Tornado
historical_loss_ratio_suffix = 'HLRB' # Historical Loss Ratio (Loss/Exposure) for buildings
annualized_frequency_suffix = 'AFREQ' # Number of events / number of years

#winter_storm_preffixes + historical_loss_ratio_suffix

#fema_county_data.columns[43:]

all_perils_LR = [column for column in fema_county_data.columns[43:] if historical_loss_ratio_suffix in column] # Columns for historical loss ratio (including all perils)
all_perils_freq = [column for column in fema_county_data.columns[43:] if annualized_frequency_suffix in column] # Columns for annualized frequency (including all perils)

winter_storm_LR = [preffix + '_' + historical_loss_ratio_suffix for preffix in winter_storm_preffixes] # Columns for winter storm historical loss ratio
winter_storm_freq = [preffix + '_' + annualized_frequency_suffix for preffix in winter_storm_preffixes] # Columns for winter storm annualized frequency

convective_storm_LR = [preffix + '_' + historical_loss_ratio_suffix for preffix in convective_storm_preffixes] # Columns for convective storm historical loss ratio
convective_storm_freq = [preffix + '_' + annualized_frequency_suffix for preffix in convective_storm_preffixes] # Columns for convective storm annualized frequency

def combine_perils_for_metric(data,perils,start_index=43):
    # data input should be the raw fema data
    # perils should be the perils that we are trying to combine (preffix of columns)
    # metric should be the metric we are interested in

    all_peril_hlrb = [column for column in data[start_index:] if 'HLRB' in column]
    all_perils = [peril.split('_')[0] for peril in all_peril_hlrb]


    final_data = data.copy()
    my_peril_ealb_cols,my_peril_freq_cols = [],[]
    other_peril_ealb_cols,other_peril_freq_cols = [],[]

    for peril in all_perils:
        ealb_col = peril + '_' + 'EALB'
        freq_col = peril + '_' + 'AFREQ'
        rate_col = peril + '_' + 'RATE'
        #final_data[rate_col] = final_data[hlrb_col] * final_data[freq_col]
        if peril in perils:
            my_peril_ealb_cols.append(ealb_col)
            my_peril_freq_cols.append(freq_col)
        else:
            other_peril_ealb_cols.append(ealb_col)
            other_peril_freq_cols.append(freq_col)

    final_data['MY_PERIL_TOTAL_AFREQ'] = final_data[my_peril_freq_cols].sum(axis=1)
    final_data['OTHER_PERIL_TOTAL_AFREQ'] = final_data[other_peril_freq_cols].sum(axis=1)
    final_data['MY_PERIL_WEIGHTED_AVG_HLRB'] = final_data[my_peril_ealb_cols].sum(axis=1) / (final_data['MY_PERIL_TOTAL_AFREQ'] * final_data['BUILDVALUE'])
    final_data['OTHER_PERIL_WEIGHTED_AVG_HLRB'] = final_data[other_peril_ealb_cols].sum(axis=1) / (final_data['OTHER_PERIL_TOTAL_AFREQ'] * final_data['BUILDVALUE'])

    return final_data

conv_storm_expected = combine_perils_for_metric(fema_county_data,convective_storm_preffixes)
wint_storm_expected = combine_perils_for_metric(fema_county_data,winter_storm_preffixes)

conv_county_membership = pd.read_csv('convective_storm_county.csv')
wint_county_membership = pd.read_csv('winter_storm_county.csv')

def split_last(x):
    y = x.split(' ')[:-1]
    z = ''
    for e in y:
        blank = ' ' if len(z) > 0 else ''
        z = z + blank + e
    return z


set(list(conv_county_membership['NAMELSAD'].apply(split_last))) - set(list(conv_storm_expected['COUNTY']))

conv_county_membership['COUNTY'] = conv_county_membership['NAMELSAD'].apply(split_last)
wint_county_membership['COUNTY'] = wint_county_membership['NAMELSAD'].apply(split_last)

conv_storm_expected_merge_ready = conv_storm_expected[['COUNTY','MY_PERIL_TOTAL_AFREQ','OTHER_PERIL_TOTAL_AFREQ','MY_PERIL_WEIGHTED_AVG_HLRB','OTHER_PERIL_WEIGHTED_AVG_HLRB']]
wint_storm_expected_merge_ready = wint_storm_expected[['COUNTY','MY_PERIL_TOTAL_AFREQ','OTHER_PERIL_TOTAL_AFREQ','MY_PERIL_WEIGHTED_AVG_HLRB','OTHER_PERIL_WEIGHTED_AVG_HLRB']]



## Convective Storm Event Index ##
conv_cat_map0 = pd.DataFrame({'probability': [0],
                         'category': ['N/S']})
conv_cat_map1 = pd.DataFrame({'probability': np.arange(1,7)*.1,
                         'category': ['TSTM','MRGL','SLGT','ENH','MDT','HIGH']})
conv_cat_map = pd.concat([conv_cat_map0,conv_cat_map1],ignore_index=True)
conv_cat_map_pivoted = conv_cat_map.set_index('category').T

## Winter Storm Index ##
wint_cat_map0 = pd.DataFrame({'probability': [0],
                         'category': ['N/S']})
wint_cat_map1 = pd.DataFrame({'probability': np.arange(1,6)*.2,
                         'category': ['LIMITED','MINOR','MODERATE','MAJOR','EXTREME']})
wint_cat_map = pd.concat([wint_cat_map0,wint_cat_map1],ignore_index=True)
wint_cat_map_pivoted = wint_cat_map.set_index('category').T

df['weighted_sum'] = df.apply(compute_weighted_sum, axis=1, kwargs={'weight1': 0.5, 'weight2': 0.5})
def factor_computation(row,category,country_wide_median,num_days_in_season=180,num_days_forecast=3):
    numerator_edit_peril = (row['MY_PERIL_TOTAL_AFREQ']*(1 - num_days_forecast/num_days_in_season) +
                            row[category] * 5) * row['MY_PERIL_WEIGHTED_AVG_HLRB']
    denominator_edit_peril = row['MY_PERIL_TOTAL_AFREQ'] * row['MY_PERIL_WEIGHTED_AVG_HLRB']
    constant_peril = row['OTHER_PERIL_TOTAL_AFREQ'] * row['OTHER_PERIL_WEIGHTED_AVG_HLRB']

    return (numerator_edit_peril + constant_peril)/(denominator_edit_peril + constant_peril)

def get_storm_factors(expected_data,category_probs,num_days_in_season=180,num_days_forecast=3):
    category_probs_pivot = category_probs.set_index('category').T # transpose
    storm_factors = conv_storm_expected_merge_ready.assign(**category_probs_pivot.iloc[0].to_dict()) # add values from category_probs_pivot
    country_wide_median = storm_factors['MY_PERIL_TOTAL_AFREQ'].median()

    for category in list(category_probs['category']):
        storm_factors[category+'_fac'] = storm_factors.apply(factor_computation,axis=1,category=category,country_wide_median=country_wide_median)

    return storm_factors

test = get_storm_factors(conv_storm_expected_merge_ready,conv_cat_map)

conv_cat_map['dummy'] = 'A'
other_df.assign(**correct_pivot_df.iloc[0].to_dict())

conv_storm_expected_merge_ready


conv_storm_expectedconv_storm_expected_merge_ready.rename(columns={'MY_PERIL_TOTAL_AFREQ':'CONV_AFREQ','OTHER_PERIL_TOTAL_AFREQ':'OTHR_AFREQ',
                                                'MY_PERIL_WEIGHTED_AVG_HLRB':'CONV_HLRB','OTHER_PERIL_TOTAL_HLRB':'OTHR_HLRB'})

test2 = combine_perils_for_metric(fema_county_data,convective_storm_preffixes,annualized_frequency_suffix,'EALT')
test['MY_PERIL_WEIGHTED_METRIC'].max()