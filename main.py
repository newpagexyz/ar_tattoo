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


def find_cm(hull):
    moments = cv.moments(hull)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    return (cx, cy)


def distance(start, end):
    # Расстояние между двумя точками
    return np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)


def process_frame(frame):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Фильтруем по цветам, чтобы найти руку
    hsv_hand = cv.inRange(hsv_frame, lower, upper)
    _, thresh = cv.threshold(hsv_hand, 170, 255, cv.THRESH_BINARY)

    cv.imshow("threshold", thresh)

    # Ищем контур с наибольшей площадью
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))
    # И обводим его полигоном
    hull = cv.convexHull(contours)

    # У полигона цетр масс ниже чем у контура, это хорошо
    center_of_mass = find_cm(hull)

    contours_dislay_frame = frame.copy()
    cv.drawContours(contours_dislay_frame, [
                    hull], -1, color=red, thickness=3)
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
    for i in range(1, len(nodes)):
        node_B = nodes[i]
        # Рассматрим треугольник node_A node_B center_of_mass
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
            bottom_triangle[0], color=blue, thickness=3)
    cv.line(contours_dislay_frame, center_of_mass,
            bottom_triangle[1], color=blue, thickness=3)

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
    cv.line(contours_dislay_frame, center_of_mass,
            main_point, color=red, thickness=3)
    cv.line(contours_dislay_frame, center_of_mass,
            opposite_point, color=red, thickness=3)

    cv.imshow("countours", contours_dislay_frame)


def main_loop(filename):
    video = cv.VideoCapture(filename)
    while True:
        _, frame = video.read()
        if frame is None:
            break
        process_frame(frame)

        key = cv.waitKey()
        if key == 27:
            cv.destroyAllWindows()
            break


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Video file is not specified")
    else:
        main_loop(sys.argv[1])
