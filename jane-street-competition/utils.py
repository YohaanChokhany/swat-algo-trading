import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
import os


def fetch_data(full_data: bool = False) -> pd.DataFrame:
    """
    Fetches training data.

    Args:
        full_data (bool, optional): If True, concatenates all parquet files in the directory.
                                    If False, reads only the first parquet file. Defaults to False.

    Returns:
        pd.DataFrame: The fetched training data as a pandas DataFrame.
    """
    training_data = []
    for root, dirs, files in os.walk("data/train_data"):
        for file in files:
            if file.endswith(".parquet"):
                training_data.append(os.path.join(root, file))

    if full_data:
        data = pd.concat([pd.read_parquet(file) for file in training_data])
    else:
        data = pd.read_parquet(sorted(training_data)[0])

    return data


def split_dataset(df: pd.DataFrame, context: int = 1) -> tuple:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        context (int, optional): Number of time steps to combine. Defaults to 1.

    Returns:
        tuple: The training, validation, and test sets.
    """
    if context > 1:
        df = combine_features(df, context)

    target_cols = ["weight"]
    for col in df.columns:
        if "responder" in col:
            target_cols.append(col)

    train_df, temp_df = train_test_split(df, train_size=0.9)
    val_df, test_df = train_test_split(temp_df, test_size=0.5)

    train_df.replace(np.nan, 0, inplace=True)
    val_df.replace(np.nan, 0, inplace=True)
    test_df.replace(np.nan, 0, inplace=True)

    return (
        train_df.drop(columns=target_cols),
        train_df[target_cols],
        val_df.drop(columns=target_cols),
        val_df[target_cols],
        test_df.drop(columns=target_cols),
        test_df[target_cols],
    )


def combine_features(df: pd.DataFrame, context: int = 5) -> pd.DataFrame:
    """
    Combines features based on time steps.

    Args:
        df (pd.DataFrame): The input DataFrame.
        context (int, optional): Number of time steps to combine. Defaults to 5.

    Returns:
        pd.DataFrame: The DataFrame with combined features.
    """
    symbol_rows = []
    for symbol_id in sorted(df["symbol_id"].unique()):
        combined_features = []
        for i in range(context):
            shifted_df = (
                df[df["symbol_id"] == symbol_id]
                .shift(i)
                .drop(
                    columns=[
                        "date_id",
                        "time_id",
                        "symbol_id",
                        "weight",
                        "responder_0",
                        "responder_1",
                        "responder_2",
                        "responder_3",
                        "responder_4",
                        "responder_5",
                        "responder_6",
                        "responder_7",
                        "responder_8",
                    ]
                )
                .add_suffix(f"_{i}")
            )
            combined_features.append(shifted_df)
        final_df = pd.concat(
            [
                df[df["symbol_id"] == symbol_id][
                    ["date_id", "time_id", "symbol_id", "weight"]
                ]
            ]
            + combined_features
            + [
                df[df["symbol_id"] == symbol_id][
                    [
                        "responder_0",
                        "responder_1",
                        "responder_2",
                        "responder_3",
                        "responder_4",
                        "responder_5",
                        "responder_6",
                        "responder_7",
                        "responder_8",
                    ]
                ]
            ],
            axis=1,
        )
        symbol_rows.append(final_df)
    return pd.concat(symbol_rows)


@torch.no_grad()
def r2_loss(gt: pd.DataFrame, predictions: pd.DateFrame) -> int:
    """
    Get the loss of the training, validation or test split.

    Args:
        gt (pd.DataFrame): The ground truth data.
        predictions (torch.Tensor): The model predictions.

    Returns:
        int: the r2 score.
    """

    weights = torch.tensor(gt["weight"].values)
    y_true = torch.tensor(gt["responder_6"].values)
    predictions = predictions.squeeze()
    numerator = torch.sum(weights * (y_true - predictions) ** 2)
    denominator = torch.sum(weights * y_true**2)
    r2_loss = 1 - numerator / denominator
    return r2_loss.item()
