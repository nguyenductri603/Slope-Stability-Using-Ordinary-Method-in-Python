import numpy as np
from sympy import Symbol, solve
from scipy import integrate

class Slope:
    def __init__(self, height, length, gamma, phi, c):
        self.height = height
        self.length = length
        self.c = c
        self.gamma = gamma
        self.phi = phi

class Circle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

def analysis(slope, circle, n):  # n: number of slices
    h = slope.height
    l = slope.length

    # Circle parameters
    x_r = circle.x
    y_r = circle.y
    r = circle.radius

    x_eq = Symbol('x_eq')
    x_re = np.array(solve((x_eq - x_r) ** 2 + (-h / l * x_eq - y_r) ** 2 - r ** 2, x_eq), dtype=np.float64)
    x_left = min(x_re)
    x_right = max(x_re)

    point_1 = 0
    point_2 = 0

    if x_left < -l:
        point_1 = x_r - np.sqrt(r ** 2 - (h - y_r) ** 2)
    else:
        point_1 = x_left

    if x_right > 0:
        point_2 = x_r + np.sqrt(r ** 2 - y_r ** 2)
    else:
        point_2 = x_right

    x_c = np.linspace(point_1, point_2, 100)
    y_c = -np.sqrt(r ** 2 - (x_c - x_r) ** 2) + y_r

    # Slice parameters
    slice_width = (point_2 - point_1) / n
    x_slice = np.zeros(n - 1)
    for i in range(n - 1):
        x_slice[i] = point_1 + slice_width * (i + 1)

    y_s1 = np.zeros(n - 1)
    y_s2 = np.zeros(n - 1)
    y_eq = Symbol('y_eq')

    for i in range(n - 1):
        y_re = np.array(solve((x_slice[i] - x_r) ** 2 + (y_eq - y_r) ** 2 - r ** 2, y_eq), dtype=np.float64)
        y_s1[i] = min(y_re)
        if x_slice[i] < -l:
            y_s2[i] = h
        elif x_slice[i] <= 0:
            y_s2[i] = -h / l * x_slice[i]
        else:
            y_s2[i] = 0

    x_s = np.concatenate((x_slice[np.newaxis].T, x_slice[np.newaxis].T), axis=1)
    y_s = np.concatenate((y_s1[np.newaxis].T, y_s2[np.newaxis].T), axis=1)

    # Slice properties
    slice_boundary = np.zeros([n, 2])
    for i in range(n):
        if i == 0:
            slice_boundary[i, 0] = point_1
            slice_boundary[i, 1] = x_slice[i]
        elif 1 <= i <= n - 2:
            slice_boundary[i, 0] = x_slice[i - 1]
            slice_boundary[i, 1] = x_slice[i]
        else:
            slice_boundary[i, 0] = x_slice[i - 1]
            slice_boundary[i, 1] = point_2

    area, x_cg = slice_area(slope, circle, slice_boundary)

    alpha = np.zeros(n)
    for i in range(n):
        alpha[i] = np.degrees(np.arcsin((x_r - x_cg[i]) / r))

    # Results
    fs, mr, md = ordinary_method(slope, area, alpha, slice_width)

    print('Resisting Moment = %.2f' % mr, 'kN.m/m')
    print('Driving Force = %.2f' % md, 'kN.m/m')
    print('Factor of Safety (Fs) = %.2f' % fs)

    return fs, mr, md, slice_boundary, area, x_cg, alpha


def slice_area(slope, circle, slice_x):
    h = slope.height
    l = slope.length
    x_r = circle.x
    y_r = circle.y
    r = circle.radius
    area = np.zeros(len(slice_x))
    x_cg = np.zeros(len(slice_x))  # center of gravity coordinate

    for i in range(len(slice_x)):
        if slice_x[i, 0] <= -l and slice_x[i, 1] <= -l:
            # Area
            f = lambda x: h - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r)
            result = integrate.quad(f, slice_x[i, 0], slice_x[i, 1])
            area[i] = result[0]
            # Center of gravity (x coordinate)
            f_cg = lambda x: x * (h - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r))
            result_cg = integrate.quad(f_cg, slice_x[i, 0], slice_x[i, 1])
            x_cg[i] = result_cg[0] / area[i]

        elif slice_x[i, 0] < -l < slice_x[i, 1]:
            # Area
            f1 = lambda x: h - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r)
            result1 = integrate.quad(f1, slice_x[i, 0], -l)
            f2 = lambda x: -h / l * x - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r)
            result2 = integrate.quad(f2, -l, slice_x[i, 1])
            area[i] = result1[0] + result2[0]
            # Center of gravity (x coordinate)
            f1_cg = lambda x: x * (h - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r))
            result1_cg = integrate.quad(f1_cg, slice_x[i, 0], -l)
            f2_cg = lambda x: x * (-h / l * x - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r))
            result2_cg = integrate.quad(f2_cg, -l, slice_x[i, 1])
            x_cg[i] = (result1_cg[0] + result2_cg[0]) / area[i]

        elif -l <= slice_x[i, 0] and slice_x[i, 1] <= 0:
            # Area
            f = lambda x: -h / l * x - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r)
            result = integrate.quad(f, slice_x[i, 0], slice_x[i, 1])
            area[i] = result[0]
            # Center of gravity (x coordinate)
            f_cg = lambda x: x * (-h / l * x - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r))
            result_cg = integrate.quad(f_cg, slice_x[i, 0], slice_x[i, 1])
            x_cg[i] = result_cg[0] / area[i]

        elif slice_x[i, 0] < 0 < slice_x[i, 1]:
            # Area
            f1 = lambda x: -h / l * x - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r)
            result1 = integrate.quad(f1, slice_x[i, 0], 0)
            f2 = lambda x: 0 - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r)
            result2 = integrate.quad(f2, 0, slice_x[i, 1])
            area[i] = result1[0] + result2[0]
            # Center of gravity (x coordinate)
            f1_cg = lambda x: x * (-h / l * x - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r))
            result1_cg = integrate.quad(f1_cg, slice_x[i, 0], 0)
            f2_cg = lambda x: x * (0 - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r))
            result2_cg = integrate.quad(f2_cg, 0, slice_x[i, 1])
            x_cg[i] = (result1_cg[0] + result2_cg[0]) / area[i]

        else:
            # Area
            f = lambda x: 0 - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r)
            result = integrate.quad(f, slice_x[i, 0], slice_x[i, 1])
            area[i] = result[0]
            # Center of gravity (x coordinate)
            f_cg = lambda x: x * (0 - (-np.sqrt(r ** 2 - (x - x_r) ** 2) + y_r))
            result_cg = integrate.quad(f_cg, slice_x[i, 0], slice_x[i, 1])
            x_cg[i] = result_cg[0] / area[i]

    return area, x_cg

def ordinary_method(slope, area, alpha, width):
    weight = slope.gamma * area
    sin_alpha = np.sin(np.deg2rad(alpha))
    cos_alpha = np.cos(np.deg2rad(alpha))
    delta_ln = width / cos_alpha
    weight_sin_alpha = weight * sin_alpha
    weight_cos_alpha = weight * cos_alpha
    resisting_moment = np.sum(delta_ln * slope.c) + np.sum(weight_cos_alpha * np.tan(np.deg2rad(slope.phi)))
    driving_force = np.sum(weight_sin_alpha)
    fs = resisting_moment / driving_force
    return fs, resisting_moment, driving_force


