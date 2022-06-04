import sys
import cv2 as cv
import numpy as np


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


points_3d = np.float32([
    [0.0, 720.0],
    [720.0, 720.0],
    [720.0, 0.0],
    [0.0, 0.0]
]).reshape(-1, 2)


def find_orthogonal(vector):
    (a, b) = vector
    c = b**2/(a**2+b**2)
    d = 1-c**2
    return (int(np.sqrt(c)*300), int(np.sqrt(d)*300))


def distance(start, end):
    # Расстояние между двумя точками
    return np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)


kernel = np.ones((3, 3), np.uint8)


def apply_sensibility(avg_color, newHSens, newSSens, newVSens, maxSensibility):
    """
    Applies sensibility values for each value of HSV, taking into account the maximum sensibility possible.
    It analyses the parameters and executes the hand detection accordingly.
    Parameters
    ----------
    avg_color : array
      The average of HSV values to be detected
    newHSens : int
      Percentage of sensibility to apply to Hue
    newSSens : int
      Percentage of sensibility to apply to Saturation
    newVSens : int
      Percentage of sensibility to apply to Value
    maxSensibility : array
      The maximum error margin of HSV values to be detected
    """
    hSens = (newHSens * maxSensibility[0]) / 100
    SSens = (newSSens * maxSensibility[1]) / 100
    VSens = (newVSens * maxSensibility[2]) / 100
    lower_bound_color = np.array(
        [avg_color[0] - hSens, avg_color[1] - SSens, avg_color[2] - VSens])
    upper_bound_color = np.array(
        [avg_color[0] + hSens, avg_color[1] + SSens, avg_color[2] + VSens])
    return np.array([lower_bound_color, upper_bound_color])


def captureCamera(camera):
    outerRectangleXIni = 300
    outerRectangleYIni = 50
    outerRectangleXFin = 550
    outerRectangleYFin = 300
    innerRectangleXIni = 400
    innerRectangleYIni = 150
    innerRectangleXFin = 450
    innerRectangleYFin = 200

    while True:
        _, frame = camera.read()
        # frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # frame = cv.flip(frame, 1)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.rectangle(frame, (outerRectangleXIni, outerRectangleYIni),
                     (outerRectangleXFin, outerRectangleYFin), (0, 255, 0), 0)
        cv.rectangle(frame, (innerRectangleXIni, innerRectangleYIni),
                     (innerRectangleXFin, innerRectangleYFin), (255, 0, 0), 0)
        cv.putText(frame, 'Please center your hand in the square', (0, 35),
                   font, 1, (255, 0, 0), 3, cv.LINE_AA)
        cv.imshow('Camera', frame)

        key = cv.waitKey(1)
        if key == ord('v'):
            roi = frame[innerRectangleYIni +
                        1:innerRectangleYFin, innerRectangleXIni +
                        1:innerRectangleXFin]
            break

    hsvRoi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    lower = np.array(
        [hsvRoi[:, :, 0].min(), hsvRoi[:, :, 1].min(), hsvRoi[:, :, 2].min()])
    upper = np.array(
        [hsvRoi[:, :, 0].max(), hsvRoi[:, :, 1].max(), hsvRoi[:, :, 2].max()])
    h = hsvRoi[:, :, 0]
    s = hsvRoi[:, :, 1]
    v = hsvRoi[:, :, 2]
    hAverage = np.average(h)
    sAverage = np.average(s)
    vAverage = np.average(v)

    hMaxSensibility = max(abs(lower[0] - hAverage), abs(upper[0] - hAverage))
    sMaxSensibility = max(abs(lower[1] - sAverage), abs(upper[1] - sAverage))
    vMaxSensibility = max(abs(lower[2] - vAverage), abs(upper[2] - vAverage))

    cv.destroyAllWindows()

    return np.array([[hAverage, sAverage, vAverage],
                     [hMaxSensibility, sMaxSensibility, vMaxSensibility]])


def process_frame(frame, points_2d_old, points_2d_int_old, lower, upper):
    contours_dislay_frame = frame.copy()
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Фильтруем по цветам, чтобы найти руку
    thresh = cv.inRange(hsv_frame, lower, upper)
    thresh = cv.dilate(thresh, kernel, iterations=3)
    thresh = cv.erode(thresh, kernel, iterations=3)
    thresh = cv.GaussianBlur(thresh, (5, 5), 90)
    # _, thresh = cv.threshold(hsv_hand, 170, 255, cv.THRESH_BINARY)

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

        ])

        points_2d_int = np.int32([
            [int(intersections_1[0][0][0]), int(intersections_1[0][0][1])],
            [int(intersections_1[1][0][0]), int(intersections_1[1][0][1])],
            [int(intersections_2[1][0][0]), int(intersections_2[1][0][1])],
            [int(intersections_2[0][0][0]), int(intersections_2[0][0][1])],

        ])
    else:
        points_2d = points_2d_old
        points_2d_int = points_2d_int_old

    if np.any(points_2d_int != np.int32([[0 ,0], [0 ,0], [0, 0], [0, 0]])):

        # можно впринцепето и убрать
        for i in points_2d:
            cv.circle(contours_dislay_frame, (int(i[0]), int(
                i[1])), color=red, radius=5, thickness=5)

        h, w = frame.shape[:2]

        M = cv.getPerspectiveTransform(points_3d, points_2d)
        d_frame = cv.warpPerspective(tattoo, M, (w, h))

        blank_black = np.zeros((h, w, 1), dtype=np.uint8)
        mask = cv.fillConvexPoly(blank_black, points_2d_int,
                                 color=255)
        mask_inv = cv.bitwise_not(mask)
        d_frame = cv.bitwise_and(d_frame, frame, mask=mask)

        vis = cv.bitwise_or(d_frame, frame, mask=mask_inv)
        # contours_dislay_frame = cv.add(vis, d_frame)
        d_frame = cv.add(vis, d_frame)

        cv.imshow("d_frame", d_frame)

    cv.imshow("contours", contours_dislay_frame)
    return points_2d_int, points_2d


def main_loop():
    camera = cv.VideoCapture(0)
    points_2d_old = np.float32([0.0, 0.0, 0.0, 0.0])
    points_2d_int_old = np.int32([[0 ,0], [0 ,0], [0, 0], [0, 0]])

    hSensibility = 100
    sSensibility = 100
    vSensibility = 100

    avg_color, max_sensibility = captureCamera(camera)

    cv.namedWindow("contours")
    cv.createTrackbar('HSensb', 'contours', hSensibility, 100, on_trackbar)
    cv.createTrackbar('SSensb', 'contours', sSensibility, 100, on_trackbar)
    cv.createTrackbar('VSensb', 'contours',
                      vSensibility, 100, on_trackbar)

    while True:

        _, frame = camera.read()
        newHSens = cv.getTrackbarPos('HSensb', 'contours')
        newSSens = cv.getTrackbarPos('SSensb', 'contours')
        newVSens = cv.getTrackbarPos('VSensb', 'contours')
        lower_bound_color, upper_bound_color = apply_sensibility(
            avg_color, newHSens, newSSens, newVSens, max_sensibility)

        # frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        points_2d_int_old, points_2d_old = process_frame(
            frame, points_2d_old, points_2d_int_old, lower_bound_color, upper_bound_color)

        key = cv.waitKey(1)
        if key == 27:
            cv.destroyAllWindows()
            camera.release()
            break


def on_trackbar(val):
    pass


if __name__ == "__main__":

    main_loop()
