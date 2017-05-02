import cv2
import math
import os
import sys
import getopt
import numpy as np

# границы для Canny
g_canny_thr0 = 100
g_canny_thr1 = 300
# Делители, которые позволяют задать минимальный размер прямоугольных контуров
# эти делители используются для фильтрация мелких прямоугольных контуров - шумов.
# Значение используется в знаменателе (widht//contour_rect_thr_w),
# где width - ширина изображения. Если ширина контура больше получившегося значение
# то контур допустимый. То же самое, для contour_rect_thr_h и высоты изображения
# Большее значение приводит к большему проценту ложных срабатываний
g_contour_rect_thr_w = 10
g_contour_rect_thr_h = 10
# параметр, который задаёт радиус при схлопывании точек похожих прямоугольных контуров
g_near_radius = 20
# счётчик для отладки, чтобы генерить уникальные имена файлов
g_dbg_dir = '_debug'
g_dbg_on = False
g_dbg_counter = 0


def dbg_img(name, img):
    global g_dbg_dir
    global g_dbg_on
    global g_dbg_counter
    if not g_dbg_on:
        return
    img_fname = '{}_{}.jpg'.format(g_dbg_counter, name)
    g_dbg_counter += 1
    print(img_fname)
    if not os.path.exists(g_dbg_dir):
        os.makedirs(g_dbg_dir)
    cv2.imwrite(os.path.join(g_dbg_dir, img_fname), img)


def dbg_rects(name, img, rects):
    global g_dbg_on
    if not g_dbg_on:
        return
    img_copy = np.zeros_like(img)
    for rect in rects:
        cv2.rectangle(img_copy, (rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), 255, 1)
    dbg_img(name, img_copy)


def dbg_contours(name, img, contours):
    global g_dbg_on
    if not g_dbg_on:
        return
    img_copy = np.zeros_like(img)
    cv2.drawContours(img_copy, contours, contourIdx=-1, color=255, thickness=1)
    dbg_img(name, img_copy)


# На входе должно быть одноканальное изображение
def process_single_channel(img):
    img_canny = cv2.Canny(img, threshold1=g_canny_thr0, threshold2=g_canny_thr1, apertureSize=3, L2gradient=True)
    dbg_img('dbg_canny', img_canny)
    img_maxx, img_maxy = img_canny.shape
    # строим дополнительный контур по периметру, чтобы морфология и контурирование работали лучше
    img_canny = cv2.rectangle(img_canny, pt1=(0, 0), pt2=(img_maxy, img_maxx), color=255, thickness=5)
    dbg_img('dbg_canny_otmst_rect', img_canny)
    # ядрa для морфологического поиска
    morph_kernel_a = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (img_maxy // 10, 1))
    morph_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_maxx // 10))
    # морфологический поиск А. TODO: разобрать зачем он нужен, с ним работает намного лучше
    img_canny = cv2.morphologyEx(img_canny, op=cv2.MORPH_GRADIENT, kernel=morph_kernel_a)
    dbg_img('dbg_canny_morph_A', img_canny)
    # морфологический поиск Б. Вертикали
    img_canny_v = cv2.erode(img_canny, morph_kernel_v)
    dbg_img('dbg_canny_morph_B_erode', img_canny_v)
    img_canny_v = cv2.dilate(img_canny_v, morph_kernel_v)
    dbg_img('dbg_canny_morph_B_dilate', img_canny_v)
    # морфологический поиск В. Горизонтали
    img_canny_h = cv2.erode(img_canny, morph_kernel_h)
    dbg_img('dbg_canny_morph_C_erode', img_canny_h)
    img_canny_h = cv2.dilate(img_canny_h, morph_kernel_h)
    dbg_img('dbg_canny_morph_C_dilate', img_canny_h)
    # объединяем результат
    img_canny = cv2.add(img_canny_v, img_canny_h)
    dbg_img('dbg_canny_out', img_canny)
    return img_canny


def is_valid_contour(img, rect):
    r, c = img.shape
    return (None not in rect.values()
            and (rect['maxx'] - rect['minx']) > c // g_contour_rect_thr_w
            and (rect['maxy'] - rect['miny']) > r // g_contour_rect_thr_h)


# На входе должно быть одноканальное изображение
# На выходе: массив точек - противоположных углов прямоугольных контуров
def process_contours(img):
    # находим контуры
    (_, contours, _) = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dbg_contours('raw_contours', img, contours)
    ret_points = list()
    # iterate over top level contours
    for contour in contours:
        rect = {'minx': None, 'miny': None, 'maxx': None, 'maxy': None}
        for cpoint in contour:
            y, x = tuple(cpoint[0])
            if rect['minx'] is None or x < rect['minx']:
                rect['minx'] = x
            if rect['miny'] is None or y < rect['miny']:
                rect['miny'] = y
            if rect['maxx'] is None or x > rect['maxx']:
                rect['maxx'] = x
            if rect['maxy'] is None or y > rect['maxy']:
                rect['maxy'] = y
        if is_valid_contour(img, rect):
            ret_points.append(((rect['miny'], rect['minx']), (rect['maxy'], rect['maxx'])))
    dbg_rects('contours', img, ret_points)
    return ret_points


def is_near(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1]) < g_near_radius


def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def square(rect):
    return (rect[1][0]-rect[0][0])*(rect[1][1]-rect[0][1])


def rect_sort_func(r):
    # Упорядочиваем сначала по расстоянию до центра прямоугольника
    # а потом по углу
    r_center = [(r[0][0] + r[1][0]) / 2, (r[0][1] + r[1][1]) / 2]
    return dist([0, 0], r_center), r_center[0]/r_center[1]


# На входе имеем array формы:
# [[[   2    2] [1023 1050]]
#  [[  10  890] [ 201 1041]]
#  [[  10  728] [ 201  879]]
#  [[  10  566] .....
def reduce_rects(rects):
    # Упорядочиваем по расстоянию от начала координат до центра прямоугольника
    rects = sorted(rects, key=rect_sort_func)
    # среди всех контуров, левые углы которых находятся в радиусе rad друг от друга
    # нужно найти крайнюю левую точку и крайнюю правую точку эти прямоугольников
    # начинаем с начала координат
    rect_otmst_set = list()
    lmin_x = rects[0][0][0]
    lmin_y = rects[0][0][1]
    lmax_x = rects[0][1][0]
    lmax_y = rects[0][1][1]
    for rect in rects:
        if is_near(rect[0], [lmin_x, lmin_y]) and is_near(rect[1], [lmax_x, lmax_y]):
            if rect[0][0] < lmin_x:
                lmin_x = rect[0][0]
            if rect[0][1] < lmin_y:
                lmin_y = rect[0][1]
            if rect[1][0] > lmax_x:
                lmax_x = rect[1][0]
            if rect[1][1] > lmax_y:
                lmax_y = rect[1][1]
        else:
            rect_otmst_set.append([[lmin_x, lmin_y], [lmax_x, lmax_y]])
            lmin_x = rect[0][0]
            lmin_y = rect[0][1]
            lmax_x = rect[1][0]
            lmax_y = rect[1][1]
    rect_otmst_set.append([[lmin_x, lmin_y], [lmax_x, lmax_y]])
    return rect_otmst_set


def process(rgb_img):
    # Фильтруем с помощью bilateral filter
    #
    # http://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html
    # http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
    #
    # Параметры:
    # d          - diameter of each pixel neighborhood that is used during filtering.
    #              If it is non-positive, it is computed from sigmaSpace
    # sigmaColor - Filter sigma in the color space. A larger value of the parameter means that farther colors
    #              within the pixel neighborhood (see sigmaSpace ) will be mixed together,
    #              resulting in larger areas of semi-equal color
    # sigmaSpace - Filter sigma in the coordinate space. A larger value of the parameter means that farther
    #              pixels will influence each other as long as their colors are close enough (see sigmaColor ).
    #              When d>0 , it specifies the neighborhood size regardless of sigmaSpace .
    #              Otherwise, d is proportional to sigmaSpace
    img = cv2.bilateralFilter(rgb_img, d=1, sigmaColor=5, sigmaSpace=30)
    # делим на отдельные каналы
    img_split = cv2.split(img)
    img_r_edges = process_single_channel(img_split[0])
    rects_r = process_contours(img_r_edges)
    img_g_edges = process_single_channel(img_split[1])
    rects_g = process_contours(img_g_edges)
    img_b_edges = process_single_channel(img_split[2])
    rects_b = process_contours(img_b_edges)
    rects_0 = rects_r + rects_g + rects_b
    dbg_rects('contours_all', img, rects_0)
    # схлопываем "похожие" контуры в один
    reduced_rects = reduce_rects(rects_0)
    dbg_rects('contours_reduced', img, reduced_rects)
    return reduced_rects


def do_tile(img, rects, prefix, outdir):
    i = 0
    print('Rects: [x1-x2] - [y1-y2]')
    for rect in rects:
        print('[{}-{}] - [{}-{}]'.format(rect[0][1], rect[1][1], rect[0][0], rect[1][0]))
        img_rect = img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        outpath = os.path.join(outdir, prefix + str(i) + '.jpg')
        cv2.imwrite(outpath, img_rect)
        i = i+1


def main(argv):
    opts, args = getopt.getopt(argv, 'i:p:o:d', ['input=', 'prefix=', 'outdir=', 'debug'])
    input_img = None
    prefix = 'out_'
    outdir = '.'
    global g_dbg_on
    for opt, arg in opts:
        if opt in ('-i', '--input'):
            input_img = arg
        elif opt in ('-p', '--prefix'):
            prefix = arg
        elif opt in ('-o', '--outdir'):
            outdir = arg
        elif opt in ('d', '--debug'):
            g_dbg_on = True
    print('Input: ', input_img)
    print('Preifx: ', prefix)
    print('Outdir: ', outdir)
    if not os.path.exists(input_img):
        print('ERROR: No input file found')
        return
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Загружаем изображение
    g_img_orig = cv2.imread(input_img, cv2.IMREAD_COLOR)
    # Обрабатываем и получаем на выходе прямоугольные контуры
    reduced_rects = process(g_img_orig)
    # Упорядочиваем по площади контуров. Самый большой по площади - первый с индексом 0
    reduced_rects = sorted(reduced_rects, key=lambda r: -square(r))
    # Разбиваем на отдельные изображения
    do_tile(g_img_orig, reduced_rects, prefix, outdir)


if __name__ == '__main__':
    main(sys.argv[1:])