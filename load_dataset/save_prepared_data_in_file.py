import os

import pandas as pd


data_path = "/home/dyadya/PycharmProjects/trade/nnPriceTrendPredictor/Data"
save_train_path = os.path.join(data_path, "_dataset_csv", "train_data.csv")
save_val_path = os.path.join(data_path, "_dataset_csv", "validate_data.csv")

train_tikers = ['LTC', 'DOGE', "BTC", "ADA", "ETH", "XLM", "SOL", "BCH", "AVAX", "APT", "DOT", "GALA", "CRV", "ZEC", "BNB", "XRP"]
val_tikers = ["HBAR", "LINK", "NEAR", "OP", "UNI", "AAVE"]


if __name__ == "__main__":
    if os.path.exists(save_train_path):
        os.remove(save_train_path)
    if os.path.exists(save_val_path):
        os.remove(save_val_path)

    for folder in os.listdir(data_path):
        money_name = folder.split("_")[0]

        if money_name in train_tikers:
            save_path = save_train_path
        elif money_name in val_tikers:
            save_path = save_val_path
        else:
            continue

        df = pd.read_csv(os.path.join(data_path, folder, f"{money_name}USDT_60m.csv"))
        df['ticker'] = money_name

        if os.path.exists(save_path):
            df.to_csv(save_path, mode='a', index=False, header=False)
        else:
            df.to_csv(save_path, mode='w+', index=False)

    df = pd.read_csv(save_train_path)
    print("train: ", len(df))

    df = pd.read_csv(save_val_path)
    print("val: ", len(df))