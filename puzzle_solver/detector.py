import click
import cv2
import numpy as np
import imutils
from time import sleep


def prepare_window():
    cv2.namedWindow('cv2')
    cv2.moveWindow('cv2', 50, 50)


def imshow(img, delay: int = 10, title: str = 'Shape'):
    cv2.setWindowTitle('cv2', title)
    cv2.imshow('cv2', img)
    cv2.waitKey(0)
    sleep(delay)  # Backup because of WSL+Linux UI problems


def get_tilemap_mask(img):
    tilemap_color = np.array([61, 63, 73])
    return cv2.inRange(img, tilemap_color - 3, tilemap_color + 3)


def get_tiles_mask(img):
    # TODO Blur & smooth for masks
    bottom_color = np.array([160, 170, 182])
    return cv2.inRange(img, bottom_color, bottom_color + 1)


def get_possible_shape_sizes(x: int, y: int) -> list[tuple[int, int]]:
    mask_ratio = round(x / y, 5)
    if np.isclose(mask_ratio, 1, atol=0.01):
        return [(i, i) for i in range(10, 0, -1)]

    return [
        (i, j)
        for i in range(10, 0, -1)
        for j in range(10, 0, -1)
        if np.isclose(i / j, mask_ratio, atol=0.01)
    ]


def get_shape(shape_mask, debug: bool = False):
    possible_sizes = get_possible_shape_sizes(*shape_mask.shape)
    average_pixels = {}
    shapes = {}
    for x, y in possible_sizes:
        shape = imutils.resize(shape_mask, height=x, inter=cv2.INTER_LINEAR)
        rescaled = imutils.resize(
            shape,
            height=shape_mask.shape[0],
            width=shape_mask.shape[1],
        )
        difference = cv2.bitwise_xor(shape_mask, rescaled)
        average_pixel = np.mean(difference)
        average_pixels[(x, y)] = average_pixel
        shapes[(x, y)] = shape

        if debug:
            imshow(difference, delay=1, title=f'Downscale to {x} * {y}, sum is {average_pixel}')

    best_fitting_size = min(possible_sizes, key=average_pixels.__getitem__)
    return np.clip(shapes[best_fitting_size], 0, 1)


@click.command()
@click.option('-i', '--image', required=True, help='Screenshot location', type=click.Path())
def main(image: str):
    img = cv2.imread(image, flags=-1)
    cv2.destroyAllWindows()
    prepare_window()

    mask = get_tilemap_mask(img)
    x, y, w, h = cv2.boundingRect(mask)
    crop_mask = mask[y:y + h, x:x + w]
    smol_mask = get_shape(crop_mask)
    print(smol_mask)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
