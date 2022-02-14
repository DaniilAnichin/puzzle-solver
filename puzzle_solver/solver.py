import itertools
from pprint import pprint

import numpy as np


Shape = tuple[int, int]


def make_tile(shape: Shape) -> np.array:
    return np.ones(shape)


def make_border(shape: Shape) -> np.array:
    return -make_tile(shape)


def fit_at(border_shape: Shape, tile: np.array, x: int, y: int) -> np.array:
    return np.pad(tile, (
        (x, border_shape[0] - tile.shape[0] - x),
        (y, border_shape[1] - tile.shape[1] - y),
    ), mode='constant')


def solve_puzzle(border: np.array, tiles: list[np.array], positions: list = None) -> list[list[Shape]]:
    if not tiles:
        yield positions

    if not positions:
        positions = []
    x, y = border.shape
    for i, j in itertools.product(range(x), range(y)):
        if border[i, j] != -1:
            continue

        tile_x, tile_y = tiles[0].shape
        if tile_x + i > x or tile_y + j > y:
            continue

        fit_attempt = border + fit_at(border.shape, tiles[0], i, j)
        if 1 not in fit_attempt:
            yield from solve_puzzle(fit_attempt, tiles[1:], positions + [(i, j)])


def prepare_border():
    border_shape = (10, 10)
    border = make_border(border_shape)
    fixed_tiles = [
        fit_at(border_shape, make_tile((3, 3)), 7, 0),
        fit_at(border_shape, make_tile((3, 3)), 0, 7),
    ]
    for tile in fixed_tiles:
        border += tile
    pprint(border)
    return border


def main():
    border = prepare_border()
    free_tiles = [
        (5, 3),
        (2, 7),
        (2, 4),
        (6, 4),
        (7, 3),
    ]
    prepared_tiles = [
        make_tile(shape)
        for shape in free_tiles
    ]
    pprint(list(solve_puzzle(
        border,
        prepared_tiles,
    )))


if __name__ == '__main__':
    main()
