import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def onehot_encode(series: pd.Series) -> pd.DataFrame:
    series = series.astype(str)
    unique_categories = [
        category for category in sorted(series.unique()) if category not in ["nan"]
    ]
    records = series.map(
        lambda category: {
            f"{series.name}: {unique_category}": int(unique_category == category)
            for unique_category in unique_categories
        }
    ).tolist()
    dataframe = pd.DataFrame.from_records(records)
    return dataframe[list(sorted(dataframe.columns))].astype(float)


def multihot_encode(series: pd.Series) -> pd.DataFrame:
    all_categories = set()
    series.map(
        lambda categories: [
            all_categories.add(categories)
            for categories in (categories if isinstance(categories, list) else [])
        ]
    )
    all_categories = list(sorted(all_categories))
    records = series.map(
        lambda categories: {
            f"{series.name}: {unique_category}": unique_category
            in (categories if isinstance(categories, list) else [])
            for unique_category in all_categories
        }
    ).tolist()
    dataframe = pd.DataFrame.from_records(records)
    return dataframe[list(sorted(dataframe.columns))].astype(float)


def lambda_onehot_encode(series: pd.Series, fn: Callable[[Any], int]) -> pd.DataFrame:
    series = series.astype(float)
    min_value = series.min()
    max_value = series.max()
    max_lambda_value = fn(max_value)
    return onehot_encode(
        series.map(lambda value: min(fn(value - min_value), max_lambda_value))
    )


def tfidf_vectorize(
    series: pd.Series,
    ngram_ranges: Sequence[tuple[int, int]],
    num_components: int = 512,
):
    matrices = []
    for ngram_range in ngram_ranges:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        vectorizer.fit(series.tolist())
        matrix = vectorizer.transform(series.tolist()).toarray()
        matrices.append(matrix)
    matrix = np.concatenate(matrices, axis=-1)
    pca = PCA(min(*matrix.shape, num_components))
    matrix = pca.fit_transform(matrix)
    return pd.DataFrame(
        data=matrix,
        columns=[
            f"text_{repr(ngram_range)}_{pca.n_components}_{i}"
            for i in range(matrix.shape[-1])
        ],
    )


def count_vectorize(series: pd.Series):
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(series.tolist()).toarray()
    matrix = PCA(64).fit_transform(matrix)
    matrix = StandardScaler().fit_transform(matrix).clip(min=-3, max=3)
    return pd.DataFrame(
        data=matrix, columns=[f"text_{i}" for i in range(matrix.shape[-1])]
    )


# ST = SentenceTransformer("BAAI/bge-base-en-v1.5")


# def sentence_transform(series: pd.Series):
#     matrix = ST.encode(series.tolist())
#     matrix = PCA(64).fit_transform(matrix)
#     matrix = StandardScaler().fit_transform(matrix).clip(min=-3, max=3)
#     return pd.DataFrame(
#         data=matrix, columns=[f"text_{i}" for i in range(matrix.shape[-1])]
#     )


def sqrt_onehot_encode(series: pd.Series) -> pd.DataFrame:
    return lambda_onehot_encode(series, fn=lambda x: int(x**0.5))


def log_onehot_encode(series: pd.DataFrame) -> pd.DataFrame:
    return lambda_onehot_encode(series, fn=lambda x: int(math.log(x)))


def z_score_scale(series: pd.Series) -> pd.DataFrame:
    transformed = StandardScaler().fit_transform(series.values.reshape(-1, 1))
    return pd.DataFrame(data=transformed, columns=[f"{series.name} zscore"])


def min_max_scale(
    series: pd.Series,
    lower: int = None,
    higher: int = None,
) -> pd.DataFrame:
    feature_range = None
    if lower is not None and higher is not None:
        feature_range = [lower, higher]
    transformed = MinMaxScaler(feature_range).fit_transform(
        series.values.reshape(-1, 1)
    )
    return pd.DataFrame(data=transformed, columns=[f"{series.name} minmax"])


def concat_encodings(dataframes: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([e.reset_index(drop=True) for e in dataframes], axis=1)


def encode_continuous_values(series: pd.Series, n_bins):
    values = series.values
    values_range = series.max() - series.min()
    arr = np.arange(n_bins)[None] < np.floor(n_bins * values / values_range)[:, None]
    arr = arr.astype(float)
    extra = (values % (values_range / n_bins)) / (values_range / n_bins)
    extra_mask = (
        np.arange(n_bins)[None] <= np.floor(n_bins * values / values_range)[:, None]
    ) - arr
    arr = arr + extra_mask * extra[:, None]
    return pd.DataFrame(
        data=arr, columns=[f"{series.name}_{idx}" for idx in range(n_bins)]
    )


def binary_encode(series: pd.Series, n_bits: int = None):
    # Calculate the binary representation of each number
    values = series.values[:, None].astype(int).copy()
    values -= values.min()
    values += 1

    if n_bits is None:
        n_bits = np.max((np.log2(values) + 1).astype(int))

    binary_array = ((values & (1 << np.arange(n_bits))) > 0).astype(int)

    # Reverse the order to match the typical binary format (most significant bit first)
    binary_array = binary_array[:, ::-1]
    return pd.DataFrame(
        data=binary_array, columns=[f"{series.name}_{idx}" for idx in range(n_bits)]
    )


def text_encoding(series: pd.Series):

    hasher = TfidfVectorizer(
        max_df=0.5,
        min_df=0.02,
        max_features=64,
        ngram_range=(2, 8),
        lowercase=False,
        token_pattern=r'(?u)\b\w[\w@#$%^&*()_+{}\[\]:;"\'<>,.?/~`!|\\-]*\b',
    )
    X = hasher.fit_transform(series.values)

    df = pd.DataFrame(X.toarray(), columns=hasher.get_feature_names_out())
    return df
