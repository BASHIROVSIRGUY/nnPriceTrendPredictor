import time

import pandas as pd


WINDOW_SIZE = 500
STEP_ROWS = 3
file_path = "/home/dyadya/PycharmProjects/trade/nnPriceTrendPredictor/Data/_dataset_csv/train_data.csv"


def get_count_lines(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)


def get_columns(filename):
    return pd.read_csv(filename, nrows=0).columns


def load_item_df(filename, columns, num):
    if num == 0: num += 1
    return pd.read_csv(filename, skiprows=num, nrows=WINDOW_SIZE, names=columns)


count_lines = get_count_lines(file_path)
print("count_lines: ", count_lines)

column_names = get_columns(file_path)

current_tick = ''

start_time = time.time()

for i in range(1, count_lines//STEP_ROWS):
    df_i = load_item_df(file_path, column_names, i)

    if df_i.ticker[0] != current_tick:
        current_time = time.time()
        print(df_i.head())
        print("time: ", current_time - start_time)

    current_tick = df_i['ticker'][0]

current_time = time.time()
print("time: ", current_time - start_time)


# нормализация данных: по относительному отклонению по одной свече
# классификация: 5 классов, кластеризировать k-means, брать
# Уинская 15а 2подъезд 2этаж 219 0-калитка