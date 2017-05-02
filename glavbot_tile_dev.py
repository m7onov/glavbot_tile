import cv2
import math
import os
import sys
import getopt
import numpy as np

g_canny_thr0 = 100
g_canny_thr1 = 300
g_contour_rect_thr_w = 10
g_contour_rect_thr_h = 10
g_near_radius = 20

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
    img_canny = cv2.rectangle(img_canny, pt1=(0, 0), pt2=(img_maxy, img_maxx), color=255, thickness=2)
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


def is_valid_contour2(img, minx, miny, maxx, maxy):
    r, c = img.shape
    if minx is None or miny is None or maxx is None or maxy is None:
        return False
    return (is_near(minx, miny) and is_near(maxx, maxy)
            and (maxx[0]-maxy[0]) > c // g_near_radius
            and (maxy[0]-miny[0]) > c // g_near_radius);


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
    img = cv2.bilateralFilter(rgb_img, d=1, sigmaColor=5, sigmaSpace=30)
    # делим на отдельные каналы
    img_split = cv2.split(img)
    img_r_edges = process_single_channel(img_split[0])
    rects_r = process_contours(img_r_edges)
    img_g_edges = process_single_channel(img_split[1])
    rects_g = process_contours(img_g_edges)
    img_b_edges = process_single_channel(img_split[2])
    rects_b = process_contours(img_b_edges)
    # прямоугольные контуры для отдельных каналов
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


def main(input_img, prefix, outdir):
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

    # plt.rcParams['image.cmap'] = 'gray'
    # g_img_blank = np.zeros_like(g_img_orig)
    # # axis_color = 'lightgoldenrodyellow'
    # fig = plt.figure()
    #
    # ax1 = fig.add_subplot(1, 1, 1)
    # for p in reduced_rects:
    #     # cv2.circle(g_img_blank, (p[0][0], p[0][1]), 3, (255, 0, 0), 1)
    #     # cv2.circle(g_img_blank, (p[1][0], p[1][1]), 3, (0, 255, 0), 1)
    #     cv2.rectangle(g_img_blank, (p[0][0], p[0][1]), (p[1][0], p[1][1]), (255, 0, 0), 3)
    #
    # ax1.imshow(g_img_blank)
    # plt.subplots_adjust(left=0.08, right=0.92, top=1, bottom=0.1, wspace=0, hspace=0.02)
    # plt.show()

    # Разбиваем на отдельные изображения
    do_tile(g_img_orig, reduced_rects, prefix, outdir)


def prepare_args(argv):
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
    return input_img, prefix, outdir


if __name__ == '__main__':
    g_input_img, g_prefix, g_outdir = prepare_args(sys.argv[1:])
    print('Input: ', g_input_img)
    print('Preifx: ', g_prefix)
    print('Outdir: ', g_outdir)
    i = 1
    # main('_badpics/Screenshot_7.png', str(i) + '_' + g_prefix, g_outdir); i += 1
    # main('_badpics/ssmv_05.jpg', str(i) + '_' + g_prefix, g_outdir); i += 1
    # main('_badpics/ssmv_08.jpg', str(i) + '_' + g_prefix, g_outdir); i += 1
    # main('_badpics/ssmv_09.jpg', str(i) + '_' + g_prefix, g_outdir); i += 1
    main('_testpics/1024px-Red_Slate_Mountain_1.jpg', str(i) + '_' + g_prefix, g_outdir); i += 1
    # main('_testpics/example.jpg', g_prefix, g_outdir);
    # main('_testpics/montage.jpg', g_prefix, g_outdir);
    # main('_testpics/montage2.jpg', g_prefix, g_outdir);
    # main('_testpics/montage3.jpg', g_prefix, g_outdir);
    # main('_testpics/opencv_logo.png', g_prefix, g_outdir);

