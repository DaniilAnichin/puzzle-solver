import itertools
from pprint import pprint

import numpy as np


Shape = tuple[int, int]
last_attempts = 0
total_attempts = 0


def make_tile(shape: Shape) -> np.array:
    return np.ones(shape)


def make_border(shape: Shape) -> np.array:
    return -make_tile(shape)


def fit_at(border_shape: Shape, tile: np.array, x: int, y: int) -> np.array:
    return np.pad(tile, (
        (x, border_shape[0] - tile.shape[0] - x),
        (y, border_shape[1] - tile.shape[1] - y),
    ), mode='constant')


def print_tile(tile: np.array):
    block = '\u2588' * 2
    space = ' ' * 2
    for row in tile:
        print(''.join([block if el else space for el in row]))


def solve_puzzle(border: np.array, tiles: list[np.array], positions: list = None) -> list[list[Shape]]:
    global last_attempts
    global total_attempts

    if not positions:
        positions = []
        last_attempts = 0
        total_attempts = 0

    total_attempts += 1
    if not total_attempts % 1000:
        print(f'Made {total_attempts} total attempts to fit tile..')

    if len(tiles) == 1:
        last_attempts += 1
        if not last_attempts % 100:
            print(f'Made {last_attempts} attempts to fit last tile..')

    if not tiles:
        yield positions
        return

    x, y = border.shape
    for i, j in itertools.product(range(x), range(y)):
        first_tile, *other_tiles = tiles

        if first_tile[0, 0] == 1 and border[i, j] != -1:
            continue

        tile_x, tile_y = first_tile.shape
        if tile_x + i > x or tile_y + j > y:
            continue

        fit_attempt = border + fit_at(border.shape, first_tile, i, j)
        if 1 not in fit_attempt:
            yield from solve_puzzle(fit_attempt, other_tiles, positions + [(i, j)])


def prepare_border():
    border_shape = (10, 10)
    border = make_border(border_shape)
    fixed_tiles = [
        fit_at(border_shape, make_tile((3, 3)), 7, 0),
        fit_at(border_shape, make_tile((3, 3)), 0, 7),
    ]
    for tile in fixed_tiles:
        border += tile
    print_tile(border)
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
