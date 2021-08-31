import pandas as pd


def update_ashare_data():
    all_new_data = pd.read_csv('stock_new.csv').drop(columns=['Unnamed: 0'])
    new_data = all_new_data.groupby('date')
    for type in ['open', 'high', 'low', 'close','volume','turn','peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']:
        his_df = pd.read_csv(f'.\\result\\{type}.csv').drop(columns=['Unnamed: 0'])
        his_df.set_index('date', inplace=True)
        new_dict = {}
        for k, v in new_data:
            v.set_index('code', inplace=True)
            df = v[type]
            new_dict[k] = df.to_dict()
        new_df = pd.DataFrame(new_dict).T
        first_date = new_df.index[0]
        his_df = his_df.reset_index()
        his_df = his_df[his_df['date']<first_date]
        his_df.set_index('date', inplace=True)
        new_df = his_df.append(new_df)
        new_df = new_df.reset_index()
        new_df.rename(columns={'index':'date'}, inplace=True)
        new_df.to_csv(f'.\\new_results\\{type}.csv')


if __name__ == '__main__':
    update_ashare_data()