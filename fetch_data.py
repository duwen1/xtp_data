import baostock as bs
import pandas as pd
import time

def get_new_data(bs, stockid):
    rs = bs.query_history_k_data_plus(stockid,
                                      "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                                      start_date='2021-08-01', end_date='2021-08-31',
                                      frequency="d", adjustflag="1")  # frequency="d"取日k线，adjustflag="3"默认不复权, 1后复权，2前复权

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    return result

if __name__ == '__main__':
    #### 登陆系统 ####
    lg = bs.login()
    his_data = pd.read_csv('.\\result\\close.csv').drop(columns=['Unnamed: 0'])
    his_data.set_index('date', inplace=True)
    stock_list = list(his_data.columns)
    all_df = pd.DataFrame()
    for stockid in stock_list:
        print(stockid)
        df = get_new_data(bs, stockid)
        all_df = all_df.append(df)
        time.sleep(0.2)

        all_df.to_csv('stock_new.csv')
    #### 登出系统 ####
    bs.logout()