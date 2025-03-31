import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.data_preprocessing import load_data, clean_data, train_test_split
from utils.feature_engineering import add_features
from utils.model_utils import DeliveryTimeModel, train_model, evaluate_model
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "orders_autumn_2020.csv")
    df = load_data(data_path)
    df = clean_data(df)

    df = add_features(df)

    features = [
        "ITEM_COUNT", "DISTANCE_KM",
        "HOUR_OF_DAY", "DAY_OF_WEEK", "MONTH",
        "TEMPERATURE_SCALED", "WIND_SPEED_SCALED",
        "RAIN_INDICATOR"
    ]

    target = "ESTIMATED_DELIVERY_MINUTES"

    df["DELIVERY_DEVIATION"] = df["ACTUAL_DELIVERY_MINUTES"] - df["ESTIMATED_DELIVERY_MINUTES"]

    EPOCHS = 20
    HIDDEN_DIM = 128

    train_df, test_df = train_test_split(df, test_ratio=0.2)

    X_train = train_df[features].values.astype(np.float32)
    y_train = train_df[target].values.astype(np.float32)

    X_test = test_df[features].values.astype(np.float32)
    y_test = test_df[target].values.astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = DeliveryTimeModel(input_dim=len(features), hidden_dim=HIDDEN_DIM)
    train_model(model, train_loader, epochs=EPOCHS, lr=1e-3)

    train_model(model, train_loader, epochs=10, lr=1e-3)

    mse, mae = evaluate_model(model, test_loader)
    print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

    model_save_path = os.path.join(script_dir, "..", "models", "trained_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")



if __name__ == "__main__":
    main()