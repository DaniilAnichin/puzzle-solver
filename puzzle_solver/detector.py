import json
from pprint import pprint

import click
import cv2
import numpy as np
import imutils

from puzzle_solver import solver


def prepare_window():
    cv2.namedWindow('cv2', flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cv2', 1200, 700)
    cv2.moveWindow('cv2', 50, 50)


def window_exists() -> bool:
    try:
        if cv2.getWindowProperty('cv2', cv2.WND_PROP_VISIBLE) < 1:
            return False
    except:
        return False
    return True


def imshow(img, title: str = 'Shape'):
    if not window_exists():
        prepare_window()

    cv2.setWindowTitle('cv2', title)
    cv2.imshow('cv2', img)
    while True:
        if not window_exists():
            break

        if cv2.waitKey(100) == 27:
            break
    # sleep(delay)  # Backup because of WSL+Linux UI problems


def get_tilemap_mask(img, offset=3):
    tilemap_color = np.array([61, 63, 73])
    return cv2.inRange(img, tilemap_color - offset, tilemap_color + offset)


def get_tiles_mask(img, offset=10):
    bottom_color = np.array([161, 170, 183])
    return cv2.inRange(img, bottom_color - offset, bottom_color + offset)


def get_tiles_border_mask(img, offset=3):
    border_color = np.array([175, 189, 195])
    return cv2.inRange(img, border_color - offset, border_color + offset)


def smooth(mask, debug: bool = False):
    masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    masks.append(mask)
    mask = cv2.erode(mask, kernel, iterations=1)
    masks.append(mask)
    mask = cv2.dilate(mask, kernel, iterations=3)
    masks.append(mask)
    mask = cv2.erode(mask, kernel, iterations=2)
    masks.append(mask)

    if debug:
        imshow(np.concatenate(
            (
                np.concatenate((masks[0], masks[1]), axis=1),
                np.concatenate((masks[2], masks[3]), axis=1),
            ),
            axis=0,
        ), 'Masks')
    return mask


def crop(*images, offset=0, with_position=False):
    mask = images[0]
    y, x, w, h = cv2.boundingRect(mask)
    x += offset
    y += offset
    w -= offset * 2
    h -= offset * 2
    result = [
        image[x:x + h, y:y + w]
        for image in images
    ]
    if with_position:
        result.append((x, y))
    return result


def get_possible_shape_sizes(
        x: int,
        y: int, *,
        atol: float = 0.01,
        area: float = -1,
        area_rtol: float = 0.2,
        area_atol: float = 1.5,
        debug: bool = False,
) -> list[tuple[int, int]]:
    mask_ratio = round(x / y, 5)
    if debug:
        print(x, y, mask_ratio)

    checks = []
    if np.isclose(mask_ratio, 1, atol=atol):
        checks.append(lambda i, j: i == j)

    if area > 0:
        area_delta = min(area * area_rtol, area_atol)
        # print(area, area_delta, area - area_delta, area + area_delta)
        checks.append(lambda i, j: np.isclose(i * j, area, atol=area_delta))

    if mask_ratio > 1:
        checks.append(lambda i, j: np.isclose(j / i, 1 / mask_ratio, atol=atol))
    else:
        checks.append(lambda i, j: np.isclose(i / j, mask_ratio, atol=atol))

    return [
        (i, j)
        for i in range(10, 0, -1)
        for j in range(10, 0, -1)
        if all(check(i, j) for check in checks)
    ]


def get_shape(shape_mask, area: float = -1, debug: bool = False):
    possible_sizes = get_possible_shape_sizes(*shape_mask.shape, area=area, debug=debug)
    if not possible_sizes:
        # print('Bad luck with defaults: ', shape_mask.shape, area)
        possible_sizes = get_possible_shape_sizes(*shape_mask.shape, area=area, atol=0.07)
    if not possible_sizes:
        # print('Bad luck with defaults: ', shape_mask.shape, area)
        possible_sizes = get_possible_shape_sizes(*shape_mask.shape, area=area, area_rtol=0.3)
    if not possible_sizes:
        # print('Bad luck with defaults: ', shape_mask.shape, area)
        possible_sizes = get_possible_shape_sizes(*shape_mask.shape, area=area, atol=0.07, area_rtol=0.3, area_atol=2)

    differences = {}
    shapes = {}
    shape_mask = np.clip(shape_mask, 0, 1)
    for x, y in possible_sizes:
        shapes[(x, y)] = cv2.resize(shape_mask, (y, x), interpolation=cv2.INTER_AREA)
        rescaled = cv2.resize(
            shapes[(x, y)],
            shape_mask.shape[::-1],
            interpolation=cv2.INTER_AREA,
        )
        difference = cv2.bitwise_xor(shape_mask, rescaled)
        differences[(x, y)] = round(np.mean(difference), 5)

        if debug:
            imshow(difference * 255, title=f'Downscale to {x} * {y}, error is {differences[(x, y)] * 100}%')

    best_fitting_size = min(possible_sizes, key=differences.__getitem__)
    return shapes[best_fitting_size].astype(np.float64)


def extract_tile_from_contour(img, tiles_mask, contour, color_offset=15):
    moments = cv2.moments(contour)
    center_x = int(moments["m10"] / (moments["m00"] + 1))
    center_y = int(moments["m01"] / (moments["m00"] + 1))
    color = img[center_y, center_x]

    _, tile_mask, tile_img = crop(contour, tiles_mask, img, offset=1)

    final_tile_mask = tile_mask
    # inner_mask = cv2.inRange(tile_img, color - color_offset, color + color_offset)
    # final_tile_mask = np.clip(crop(inner_mask)[0], 0, 1)
    return color, final_tile_mask


def separate_tiles(img, tiles_mask, tilemap_area, debug=False):
    tiles_mask = np.clip(tiles_mask, 0, 1)
    img = img.copy()
    contours = cv2.findContours(
        tiles_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = imutils.grab_contours(contours)

    colors = []
    tile_masks = []
    tiles_area = 0
    for contour in contours:
        color, tile_mask = extract_tile_from_contour(img, tiles_mask, contour)
        tile_masks.append(tile_mask)
        tiles_area += np.sum(tile_mask)
        colors.append(color)

    area_ratio = tilemap_area / tiles_area

    shapes = []
    for i, (contour, tile_mask) in enumerate(zip(contours, tile_masks), 1):
        approx_area = round(np.sum(tile_mask) * area_ratio, 5)
        mask_area = round(tile_mask.shape[0] * tile_mask.shape[1] * area_ratio, 5)
        if debug:
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 1)
            step_by_step = img.copy()
            cv2.drawContours(step_by_step, [contour], -1, (0, 0, 255), 1)
            imshow(step_by_step, title=f'Contour #{i}, approx area: {approx_area}, mask area: {mask_area}')

        shape = get_shape(tile_mask, area=mask_area, debug=False)
        shapes.append(shape)

    if debug:
        # print(len(contours))
        pprint(shapes)
        # print(colors)
        imshow(img, title='Contours')
    return shapes, colors


def mask_to_image(mask, scale=50):
    small_img = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return imutils.resize(small_img, width=small_img.shape[1] * scale)


def draw_solution(
        border,
        sorted_tiles,
        sorted_colors,
        solution,
):
    # border_color = np.array([190, 200, 215], dtype=np.uint8)
    # border_color = np.array([255, 255, 255], dtype=np.uint8)
    border_color = np.array([0, 0, 0], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_result = cv2.erode(mask_to_image(1 + border), kernel) * border_color
    for color, tile, position in zip(sorted_colors, sorted_tiles, solution):
        pad_tile = solver.fit_at(border.shape, tile, *position)
        colored_tile = mask_to_image(pad_tile) * color
        tile_with_border = cv2.erode(colored_tile, kernel)
        img_result += tile_with_border

    imshow(img_result, 'Solution result:')


@click.command()
@click.option('-i', '--image', required=True, help='Screenshot location', type=click.Path())
def main(image: str):
    img = cv2.imread(image, flags=-1)

    tilemap_mask = get_tilemap_mask(img)
    cropped_tilemap_mask, tilemap_position = crop(tilemap_mask, with_position=True)
    smol_tilemap_mask = get_shape(cropped_tilemap_mask)
    area = np.sum(smol_tilemap_mask)
    border = -smol_tilemap_mask
    # Tilemap generation ends here

    bottom_line = tilemap_position[0] + cropped_tilemap_mask.shape[0]
    bottom = img[bottom_line + 10:, ...]

    # Mask, crop #1
    tiles_mask = get_tiles_mask(bottom)
    cropped_tiles_mask, img_tiles = crop(tiles_mask, bottom, offset=5)
    cropped_tiles_mask = cv2.bitwise_not(cropped_tiles_mask)
    border_mask = get_tiles_border_mask(img_tiles)
    bordered_tiles_mask = cv2.bitwise_xor(cropped_tiles_mask, border_mask)

    # Smooth, crop #2
    smooth_mask = smooth(bordered_tiles_mask, debug=False)
    final_mask, img_tiles_final = crop(smooth_mask, img_tiles)

    tiles, colors = separate_tiles(img_tiles_final, final_mask, area, debug=False)

    print('Area: ', area)
    print('Tiles area: ', sum(np.sum(tile) for tile in tiles))
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print('Starting to solve;')
    sorted_colors, sorted_tiles = zip(*sorted(
        zip(colors, tiles),
        key=lambda x: np.sum(x[1]),  # Shape's area
        reverse=True,
    ))

    # solutions = list(solver.solve_puzzle(border, sorted_tiles))
    solutions_iter = solver.solve_puzzle(border, sorted_tiles)

    try:
        solution = next(solutions_iter)
        draw_solution(border, sorted_tiles, sorted_colors, solution)
    except StopIteration:
        print('No solution found(')

    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    main()
