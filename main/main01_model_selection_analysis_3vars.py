

import pandas as pd

file_path = 'C:\\Users\\dqi\\Documents\\Output\\MarketMaking\\2016-09-23-14-57-18rolling_model_selection_12M_predict_1M_normalized_by_12M_add_ma_add_high_order\\'
file_name = 'result_all.csv'
vars_num = 6
output_name = 'var_categories_' + str(vars_num) + 'vars.txt'
output_name2 = 'var_categories_' + str(vars_num) + 'vars.csv'
output_name3 = 'var_categories_' + str(vars_num) + 'vars_only_categories.csv'
output_name4 = 'var_categories_' + str(vars_num) + 'vars_only_vars.csv'


def clean_one_name(one_name):
    to_del = [' ', '(', ')', '\.', '\'']

    new_name = one_name
    for td in to_del:
        new_name = new_name.replace(td, '')

    return new_name


data_raw_from_csv = pd.read_csv(file_path + file_name)
data_raw_selected_num = data_raw_from_csv[data_raw_from_csv['model_length'] == vars_num]
data_raw = data_raw_selected_num.groupby(['time_period'], as_index=False).apply(lambda chunk: chunk.sort_values('rsquared_oos').iloc[-1, :])
# data_raw = data_raw.groupby('time_period', as_index=False, group_keys=False).agg({'model_length': None, 'time_period': None, 'vars_name': None, 'rsquared_oos': 'max'})

data_raw.to_csv(file_path + output_name2, index=None)

vars_list = list(data_raw['var_names'])
vars_new = [[clean_one_name(var) for var in vars_.split(',') if len(clean_one_name(var))!=0] for vars_ in vars_list]


categories_list = [
    'buy_volume_lag', 'sell_volume_lag',
    'asize_change', 'bsize_change', 'asize', 'bsize',
    'price_return', 'index_return', 'intraday_pattern',
    'volatility', 'volume_index'
]


def categories_map_func(this_name):
    mapping_list = [  # (startswith, cat)
        ('asize1_change', 'asize_change'),
        ('asize', 'asize'),
        ('bsize1_change', 'bsize_change'),
        ('bsize', 'bsize'),
        ('buyvolume', 'buy_volume_lag'),
        ('ret_index', 'index_return'),
        ('ret_hs300', 'index_return'),
        ('ret_sh50', 'index_return'),
        ('buy_vol_', 'intraday_pattern'),
        ('sell_vol_', 'intraday_pattern'),
        ('mid_px_ret', 'price_return'),
        ('sellvolume', 'sell_volume_lag'),
        ('spread', 'spread'),
        ('volume_index', 'volume_index'),
        ('volatility', 'volatility'),
    ]

    for to_start, cat in mapping_list:
        if this_name.startswith(to_start):
            this_cat = cat
            return this_cat
    else:
        print(this_name)
        raise LookupError

var_result_record = {}
cat_record = {}
for vars_ in vars_new:
    cat_record_this_vars_ = []
    for one_var in vars_:
        cat_this_var = categories_map_func(one_var)
        if cat_this_var not in cat_record_this_vars_:
            cat_record_this_vars_.append(cat_this_var)
        if one_var in var_result_record.keys():
            var_result_record[one_var] += 1
        else:
            var_result_record[one_var] = 1
    for one_cat in cat_record_this_vars_:
        if one_cat in cat_record.keys():
            cat_record[one_cat] += 1
        else:
            cat_record[one_cat] = 1

to_report_dict = {}
for key in var_result_record.keys():
    this_cat = categories_map_func(key)
    if this_cat in to_report_dict.keys():
        to_report_dict[this_cat].append(key)
    else:
        to_report_dict[this_cat] = [key]

to_report_str = ''
for key in sorted(to_report_dict.keys()):
    num_this_cat = cat_record[key]
    to_report_str += key + ': ' + str(num_this_cat) + '\n'
    vars_in_this_cat = to_report_dict[key]
    for var_ in sorted(vars_in_this_cat):
        num_this_var = var_result_record[var_]
        to_report_str += '\t' + var_ + ': ' + str(num_this_var) + '\n'


with open(file_path + output_name, 'w') as f_out:
    f_out.write(to_report_str)
