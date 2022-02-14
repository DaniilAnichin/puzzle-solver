import click

from .solver import solve_puzzle, make_border, make_tile, fit_at


class PointParamType(click.ParamType):
    name = "point"

    def __init__(self, dims: int = 2):
        self.dims = dims

    def convert(self, value, param, ctx):
        if isinstance(value, tuple):
            return value

        try:
            parts = value.split(',')
            if len(parts) != self.dims:
                self.fail(f"Expected {self.dims} dimensions, got {len(parts)}")

            return tuple(int(part) for part in parts)
        except ValueError:
            self.fail(f"{value!r} is not a valid point", param, ctx)


@click.command()
@click.option('-b', '--border', required=True, help='Border shape, comma separated', type=PointParamType(2))
@click.option('-f', '--fixed', 'fixed_tiles', multiple=True, help='Fixed tiles of the border', type=PointParamType(4))
@click.option('-t', '--tile', 'tiles', multiple=True, help='Free tiles of the puzzle', type=PointParamType(2))
def main(border, fixed_tiles, tiles):
    border = make_border(border)
    for tile in fixed_tiles:
        border += fit_at(border.shape, make_tile(tile[:2]), *tile[2:])

    initial_tile_positions, sorted_tiles = zip(*sorted(
        enumerate(tiles),
        key=lambda x: x[1][0] * x[1][1],  # Shape's area
        reverse=True,
    ))

    prepared_tiles = [
        make_tile(shape)
        for shape in sorted_tiles
    ]

    for i, solution in enumerate(solve_puzzle(border, prepared_tiles), start=1):
        print(f'Solution {i}:')
        for tile, position in zip(sorted_tiles, solution):
            print(f'{tile[0]},{tile[1]}: {position[0]},{position[1]}')
        print()


if __name__ == '__main__':
    main()
