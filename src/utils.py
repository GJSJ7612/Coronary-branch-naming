import numpy as np

def catmull_rom_spline(points, num_points=100):
    """
    使用Catmull-Rom样条平滑路径
    points: 控制点列表，形状为(N, 3)
    num_points: 每段样条插值的点数
    返回平滑后的路径点列表
    """
    smoothed_points = []
    n = len(points)

    for i in range(n - 1):
        p0 = points[i - 1] if i - 1 >= 0 else points[i]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 < n else points[i + 1]

        for t in np.linspace(0, 1, num_points):
            t2 = t * t
            t3 = t2 * t

            x = 0.5 * ((2 * p1[0]) +
                       (-p0[0] + p2[0]) * t +
                       (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)

            y = 0.5 * ((2 * p1[1]) +
                       (-p0[1] + p2[1]) * t +
                       (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)

            z = 0.5 * ((2 * p1[2]) +
                       (-p0[2] + p2[2]) * t +
                       (2*p0[2] - 5*p1[2] + 4*p2[2] - p3[2]) * t2 +
                       (-p0[2] + 3*p1[2] - 3*p2[2] + p3[2]) * t3)

            smoothed_points.append([x, y, z])

    return np.array(smoothed_points)
