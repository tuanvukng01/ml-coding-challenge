import matplotlib.pyplot as plt
import seaborn as sns

def plot_delivery_deviation(df):
    deviation_col = "DELIVERY_DEVIATION"
    if deviation_col not in df.columns:
        df[deviation_col] = df["ACTUAL_DELIVERY_MINUTES"] - df["ESTIMATED_DELIVERY_MINUTES"]

    plt.figure(figsize=(8, 6))
    sns.histplot(df[deviation_col], bins=30, kde=True)
    plt.title("Distribution of Delivery Deviation (Actual - Estimated)")
    plt.xlabel("Delivery Deviation (minutes)")
    plt.ylabel("Frequency")
    plt.show()

def plot_distance_vs_deviation(df):
    deviation_col = "DELIVERY_DEVIATION"
    if deviation_col not in df.columns:
        df[deviation_col] = df["ACTUAL_DELIVERY_MINUTES"] - df["ESTIMATED_DELIVERY_MINUTES"]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="DISTANCE_KM", y=deviation_col, data=df)
    plt.title("Distance vs. Delivery Deviation")
    plt.xlabel("Distance (km)")
    plt.ylabel("Delivery Deviation (minutes)")
    plt.show()