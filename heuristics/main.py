import json
import random
from pprint import pprint

import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

with open("data/data/gen3/species.npy", "rb") as f:
    SPECIES_ONEHOT = np.load(f)

with open("data/data/gen3/abilities.npy", "rb") as f:
    ABILITY_ONEHOT = np.load(f)

with open("data/data/gen3/items.npy", "rb") as f:
    ITEM_ONEHOT = np.load(f)

with open("data/data/gen3/moves.npy", "rb") as f:
    MOVE_ONEHOT = np.load(f)


def to_id(string: str) -> str:
    return "".join(c for c in string if c.isalnum()).lower()


def main():
    with open("heuristics/allSets.json", "r") as f:
        pokemon_sets = json.load(f)

    with open("data/data/data.json", "r") as f:
        indicies = json.load(f)

    arr = []
    for pokemon_set in pokemon_sets:
        arr.append(
            [
                indicies["species"][to_id(pokemon_set["name"])],
                indicies["abilities"][to_id(pokemon_set["ability"])],
                indicies["items"][to_id(pokemon_set["item"])],
                *[indicies["moves"][to_id(move)] for move in pokemon_set["moves"]],
                *[
                    indicies["moves"]["!PAD!"]
                    for _ in range(4 - len(pokemon_set["moves"]))
                ],
            ]
        )
    arr = np.array(arr)

    features = 64

    species_pca = PCA(min(np.unique(arr[..., 0]).size, features)).fit(
        SPECIES_ONEHOT[np.unique(arr[..., 0])]
    )
    abilities_pca = PCA(min(np.unique(arr[..., 1]).size, features)).fit(
        ABILITY_ONEHOT[np.unique(arr[..., 1])]
    )
    item_pca = PCA(min(np.unique(arr[..., 2]).size, features)).fit(
        ITEM_ONEHOT[np.unique(arr[..., 2])]
    )
    move_pca = PCA(min(np.unique(arr[..., 3:].flatten()).size, features)).fit(
        MOVE_ONEHOT[np.unique(arr[..., 3:].flatten())]
    )

    species_vectors = species_pca.transform(SPECIES_ONEHOT[arr[..., 0]])
    abilities_vectors = abilities_pca.transform(ABILITY_ONEHOT[arr[..., 1]])
    item_vectors = item_pca.transform(ITEM_ONEHOT[arr[..., 2]])
    move_vectors = (
        move_pca.transform(MOVE_ONEHOT[arr[..., 3:].flatten()])
        .reshape(arr.shape[0], 4, -1)
        .mean(1)
    )

    entity_vectors = np.concat(
        (species_vectors, abilities_vectors, item_vectors, move_vectors), axis=-1
    )

    rand_index = random.randint(0, len(pokemon_sets) - 1)
    pprint(pokemon_sets[rand_index])
    scores = entity_vectors[rand_index][None] @ entity_vectors.T
    max_indices = np.argsort(scores.flatten())
    for i in reversed(max_indices[-11:-1]):
        pprint(pokemon_sets[i])

    # decomp = PCA(3).fit_transform(entity_vectors)

    # labels = KMeans(32).fit_predict(entity_vectors)

    # fig = px.scatter(
    #     x=decomp[..., 0],
    #     y=decomp[..., 1],
    #     # z=decomp[..., 2],
    #     hover_name=[json.dumps(s) for s in pokemon_sets],
    #     color=labels.astype(str),
    # )

    # fig.show()


if __name__ == "__main__":
    main()
