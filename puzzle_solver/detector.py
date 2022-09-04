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
    return cv2.inRange(img, bottom_color - 15, bottom_color + 15)


def smooth(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def crop_mask(mask):
    y, x, w, h = cv2.boundingRect(mask)
    return (x, y), mask[x:x + h, y:y + w]


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
    differences = {}
    shapes = {}
    shape_mask = np.clip(shape_mask, 0, 1)
    for x, y in possible_sizes:
        shapes[(x, y)] = imutils.resize(shape_mask, height=x, inter=cv2.INTER_LINEAR)
        rescaled = cv2.resize(
            shapes[(x, y)],
            shape_mask.shape[::-1],
            interpolation=cv2.INTER_AREA,
        )
        difference = cv2.bitwise_xor(shape_mask, rescaled)
        differences[(x, y)] = round(np.mean(difference), 5)

        if debug:
            imshow(difference * 255, delay=1, title=f'Downscale to {x} * {y}, error is {differences[(x, y)] * 100}%')

    best_fitting_size = min(possible_sizes, key=differences.__getitem__)
    return shapes[best_fitting_size]


def separate_tiles(img, tile_mask):
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # tile_mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(tile_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sd = ShapeDetector()
    for c in cnts:
        print(type(c))
        print(c)
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        # cX = int((M["m10"] / M["m00"]) * ratio)
        # cY = int((M["m01"] / M["m00"]) * ratio)
        # shape = sd.detect(c)
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        # c = c.astype("float")
        # c *= ratio
        # c = c.astype("int")
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        # cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (255, 255, 255), 2)
        # show the output image
    imshow(img, title='Contours')


@click.command()
@click.option('-i', '--image', required=True, help='Screenshot location', type=click.Path())
def main(image: str):
    img = cv2.imread(image, flags=-1)
    cv2.destroyAllWindows()
    prepare_window()

    tilemap_mask = get_tilemap_mask(img)
    tilemap_position, cropped_tilemap_mask = crop_mask(tilemap_mask)
    # imshow(cropped_tilemap_mask, title='Tilemap mask')
    smol_tilemap_mask = get_shape(cropped_tilemap_mask)
    # print(smol_tilemap_mask)
    bottom_line = tilemap_position[0] + cropped_tilemap_mask.shape[0]

    bottom = img[bottom_line + 10:, ...]
    tiles_mask = get_tiles_mask(bottom)
    (x, y), cropped_tiles_mask = crop_mask(tiles_mask)
    tiles_mask_inv = cv2.bitwise_not(cropped_tiles_mask)
    img_tiles = bottom[x:x + cropped_tiles_mask.shape[0], y:y + cropped_tiles_mask.shape[1]]
    # print(np.histogram(tiles_mask_inv, bins=2))
    # imshow(cv2.bitwise_and(img_tiles, img_tiles, mask=(tiles_mask_inv // 2) + 120), title='Tiles masked')
    img_tiles_2 = np.copy(img_tiles)
    img_tiles[cropped_tiles_mask > 0] //= 2
    # imshow(img_tiles, delay=3, title='Tiles masked')

    smooth_mask = smooth(cropped_tiles_mask)
    # img_tiles_2[smooth_mask > 0] //= 2
    # imshow(img_tiles_2, delay=3, title='Tiles masked, smooth')
    separate_tiles(img_tiles_2, cv2.bitwise_not(smooth_mask))

    # imshow(cropped_tiles_mask, title='Tiles mask')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
