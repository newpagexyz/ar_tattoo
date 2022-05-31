import sys
import cv2 as cv
import numpy as np

# Диапазон цветов по HSV модели
lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

# OpenCV использует BGR вместо RGB
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

tattoo = cv.imread("tattoo_example.jpg")
tattoo = cv.flip(tattoo, 0)


def find_cm(hull):
    moments = cv.moments(hull)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    return (cx, cy)


# Набор точек в 3d - модель руки
# Пирамида с основанием 720x720 и вершиной в (360, 360, 200)
points_3d = np.float32([
    [0.0, 720.0, 0.0],
    [720.0, 720.0, 0.0],
    [720.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [360.0, 360.0, 360.0]]).reshape(-1, 3)

# Набор точек в 3d - фигура для построения (паралеллипипед 720x720x400 поднят на 100) - над плоскостью
projection_3d = np.float32([
    [0.0, 720.0, 500.0],
    [720.0, 720.0, 500.0],
    [720.0, 0.0, 500.0],
    [0.0, 0.0, 500.0],
    [0.0, 720.0, 100.0],
    [720.0, 720.0, 100.0],
    [720.0, 0.0, 100.0],
    [0.0, 0.0, 100.0]]).reshape(-1, 3)


def find_orthogonal(vector):
    (a, b) = vector
    c = b**2/(a**2+b**2)
    d = 1-c**2
    return (int(np.sqrt(c)*300), int(np.sqrt(d)*300))


def distance(start, end):
    # Расстояние между двумя точками
    return np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)


def process_frame(frame, points_2d_old, points_2d_int_old):
    contours_dislay_frame = frame.copy()
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Фильтруем по цветам, чтобы найти руку
    hsv_hand = cv.inRange(hsv_frame, lower, upper)
    _, thresh = cv.threshold(hsv_hand, 170, 255, cv.THRESH_BINARY)

    # Ищем контур с наибольшей площадью
    contours, _ = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
    contours = max(contours, key=lambda x: cv.contourArea(x))

    # И обводим его полигоном
    hull = cv.convexHull(contours)

    center_of_mass = find_cm(contours)

    cv.drawContours(contours_dislay_frame, [
                    contours], -1, color=green, thickness=3)
    cv.circle(contours_dislay_frame, center_of_mass,
              radius=5, color=blue, thickness=5)

    nodes = []
    max_angle = 0
    bottom_triangle = []
    for edge in hull:
        x, y = edge[0][0], edge[0][1]
        nodes.append((x, y))
    node_A = nodes[0]
    for i in range(-1, len(nodes)):
        node_B = nodes[i]
        # Рассмотрим треугольник node_A node_B center_of_mass
        c = distance(node_A, node_B)
        a = distance(node_B, center_of_mass)
        b = distance(center_of_mass, node_A)

        angle = np.arccos((a**2 + b**2 - c**2) /
                          (2 * a * b))  # теорема косинусов
        if angle < np.pi / 2 and angle > max_angle:
            max_angle = angle
            bottom_triangle = [node_A, node_B]
        node_A = node_B

    cv.line(contours_dislay_frame, center_of_mass,
            bottom_triangle[0], color=blue)
    cv.line(contours_dislay_frame, center_of_mass,
            bottom_triangle[1], color=blue)

    vector_a = (center_of_mass[0] - bottom_triangle[0]
                [0], center_of_mass[1] - bottom_triangle[0][1])
    vector_b = (center_of_mass[0] - bottom_triangle[1]
                [0], center_of_mass[1] - bottom_triangle[1][1])
    main_vector = (vector_a[0]+vector_b[0],
                   vector_a[1]+vector_b[1])
    main_point = (center_of_mass[0] + main_vector[0],
                  center_of_mass[1] + main_vector[1])
    opposite_point = (center_of_mass[0] - main_vector[0],
                      center_of_mass[1] - main_vector[1])
    orthogonal_vector = find_orthogonal(main_vector)
    # фикс отображения
    d_fix = 1 if orthogonal_vector[0] > orthogonal_vector[1] else -1

    orthogonal_vector_point = (
        center_of_mass[0] + d_fix * orthogonal_vector[0], center_of_mass[1] + orthogonal_vector[1])
    orthogonal_vector_point_opposite = (
        center_of_mass[0] - d_fix * orthogonal_vector[0], center_of_mass[1] - orthogonal_vector[1])

    orthogonal_vector_point_2 = (center_of_mass[0] + d_fix * int(main_vector[0] / 3) + d_fix * orthogonal_vector[0],
                                 center_of_mass[1] - d_fix * int(main_vector[1] / 3) + orthogonal_vector[1])
    orthogonal_vector_point_2_opposite = (center_of_mass[0] + d_fix * int(
        main_vector[0] / 3) - d_fix * orthogonal_vector[0],
        center_of_mass[1] - d_fix * int(main_vector[1] / 3) - orthogonal_vector[1])

    cv.line(contours_dislay_frame, main_point,
            opposite_point, color=red)
    cv.line(contours_dislay_frame, orthogonal_vector_point_opposite,
            orthogonal_vector_point, color=blue, thickness=3)
    cv.line(contours_dislay_frame, orthogonal_vector_point_2_opposite,
            orthogonal_vector_point_2, color=blue, thickness=3)

    blank = np.zeros(frame.shape[0:2])
    img1 = cv.drawContours(blank.copy(), [contours], 0, 1)
    img2 = cv.line(blank.copy(), orthogonal_vector_point_opposite,
                   orthogonal_vector_point, color=(255, 255, 255), thickness=1)
    intersection_image = cv.bitwise_and(img1, img2)
    intersections_1 = cv.findNonZero(intersection_image)

    img3 = cv.line(blank.copy(), orthogonal_vector_point_2_opposite,
                   orthogonal_vector_point_2, color=(255, 255, 255), thickness=1)
    intersection_image_2 = cv.bitwise_and(img1, img3)
    intersections_2 = cv.findNonZero(intersection_image_2)

    if intersections_1 is not None and intersections_2 is not None and len(intersections_1) == 2 and len(intersections_2) == 2:
        points_2d = np.float32([
            [intersections_1[0][0][0], intersections_1[0][0][1]],
            [intersections_1[1][0][0], intersections_1[1][0][1]],
            [intersections_2[1][0][0], intersections_2[1][0][1]],
            [intersections_2[0][0][0], intersections_2[0][0][1]],
            [center_of_mass[0], center_of_mass[1]]
        ])

        points_2d_int = np.int32([
            [int(intersections_1[0][0][0]), int(intersections_1[0][0][1])],
            [int(intersections_1[1][0][0]), int(intersections_1[1][0][1])],
            [int(intersections_2[1][0][0]), int(intersections_2[1][0][1])],
            [int(intersections_2[0][0][0]), int(intersections_2[0][0][1])],
            [center_of_mass[0], center_of_mass[1]]
        ])
    else:
        points_2d = points_2d_old
        points_2d_int = points_2d_int_old

    if np.sum(points_2d) != 0:
        # for i in points_2d:
        #     cv.circle(contours_dislay_frame, (int(i[0]), int(
        #         i[1])), color=red, radius=5, thickness=5)

        fx = 0.5 + cv.getTrackbarPos('focal', 'contours') / 50.0
        h, w = frame.shape[:2]
        K = np.float64([[fx*w, 0, 0.5*(w-1)],
                        [0, fx*w, 0.5*(h-1)],
                        [0.0, 0.0,      1.0]])
        dist_coef = np.zeros(4)
        # M = cv.getPerspectiveTransform(points_3d, points_2d)
        # d_frame = cv.warpPerspective(tattoo, M, (h, w))
        # blank_full = np.zeros((h, w, 3), dtype=np.uint8)
        # blank_black = np.zeros((h, w, 1), dtype=np.uint8)
        # mask = cv.fillConvexPoly(blank_black, points_2d_int,
        #  color=255)
        # mask_inv = cv.bitwise_not(mask)
        # d_frame = cv.bitwise_and(d_frame, frame, mask=mask)
        # frame_without_mask = cv.bitwise_or(d_frame, frame, mask=mask)
        # vis = cv.bitwise_or(d_frame, frame, mask=mask_inv)
        # d_frame = cv.add(vis, d_frame)
        # cv.imshow("d_frame", d_frame)
        ret, rvecs, tvecs = cv.solvePnP(
            points_3d, points_2d, K, dist_coef, flags=cv.SOLVEPNP_EPNP)

        imgpts, jac = cv.projectPoints(
            projection_3d, rvecs, tvecs, K, dist_coef)
        for i in range(-1, len(imgpts)):
            pt = imgpts[i]
            pt1 = imgpts[i-1]
            cv.line(contours_dislay_frame,  (int(pt1[0][0]), int(pt1[0][1])),
                    (int(pt[0][0]), int(pt[0][1])), color=red, thickness=10)
        # print(imgpts)
        # cent = (int(imgpts[0][0][0]), int(
        # imgpts[0][0][1]))
        # print(cent)
        # cv.line(contours_dislay_frame, center_of_mass,
        # cent, color=red, thickness=4)
        # cv.circle(contours_dislay_frame,
        #   (int(imgpts[0][0][0]), int(imgpts[0][0][1])), radius=10, thickness=10, color=blue)
        # print(imgpts[0][0][0])
        # cv.circle(contours_dislay_frame,
        #   (int(imgpts[0][0][0]), int(imgpts[0][0][1])), 5, blue)

    # for i in points_2d:
    #     cv.circle(contours_dislay_frame, i, 5, color=red, thickness=5)
    # for i in intersections_2:
    #     cv.circle(contours_dislay_frame, i[0], 5, color=red, thickness=5)

    cv.imshow("contours", contours_dislay_frame)
    return points_2d_int, points_2d


def main_loop(filename):
    video = cv.VideoCapture(filename)
    points_2d_old = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
    points_2d_int_old = np.int32([0, 0, 0, 0, 0])
    while True:
        _, frame = video.read()

        if frame is None:
            break
        frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        points_2d_int_old, points_2d_old = process_frame(
            frame, points_2d_old, points_2d_int_old)

        key = cv.waitKey(50)
        if key == 27:
            cv.destroyAllWindows()
            break


def on_trackbar(val):
    pass


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # main_loop("sample4.mp4")
        print("Video file is not specified")
    else:
        cv.namedWindow("contours")
        cv.createTrackbar('focal', 'contours', 25, 50, on_trackbar)
        main_loop(sys.argv[1])
