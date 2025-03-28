import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import casadi

u = casadi.SX.sym("u")

# CODE CREDIT: https://uuvsimulator.github.io/packages/uuv_simulator/docs/features/jupyter_notebooks/2d_dubins_path/


def get_circle_pnt(u, pos, radius, offset=0.0):
    x = pos[0] + radius * np.cos(2 * np.pi * u + offset)
    y = pos[1] + radius * np.sin(2 * np.pi * u + offset)
    return x, y


def get_circle(pos, radius, offset=0.0):
    u = np.linspace(0, 1, 50)
    return get_circle_pnt(u, pos, radius, offset)


def get_frame(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


radius = 5
wp_1 = np.array([2, 3])
heading_1 = 10 * np.pi / 180
wp_2 = np.array([20, 32])
heading_2 = 130 * np.pi / 180

frame_1 = get_frame(heading_1)
frame_2 = get_frame(heading_2)

center_1 = dict(
    R=wp_1 - radius * frame_1[:, 1].flatten(), L=wp_1 + radius * frame_1[:, 1].flatten()
)

center_2 = dict(
    R=wp_2 - radius * frame_2[:, 1].flatten(), L=wp_2 + radius * frame_2[:, 1].flatten()
)


def getCenter(wp, radius, heading):
    frame = get_frame(heading)
    return dict(
        R=wp - radius * frame[:, 1].flatten(), L=wp + radius * frame[:, 1].flatten()
    )


def plot_base():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Plot first waypoint's L and R circles
    ax.plot(
        [wp_1[0], wp_1[0] + 5 * np.cos(heading_1)],
        [wp_1[1], wp_1[1] + 5 * np.sin(heading_1)],
        color="xkcd:neon pink",
        linewidth=3,
    )
    ax.plot([wp_1[0]], [wp_1[1]], ".r", markersize=10)

    x, y = get_circle(center_1["L"], radius)
    ax.plot(x, y, "-.r")
    x, y = get_circle(center_1["R"], radius)
    ax.plot(x, y, "--r")

    # Plot second waypoint's L and R circles
    x, y = get_circle(wp_2, radius)
    ax.plot(
        [wp_2[0], wp_2[0] + 5 * np.cos(heading_2)],
        [wp_2[1], wp_2[1] + 5 * np.sin(heading_2)],
        color="xkcd:neon pink",
        linewidth=3,
    )
    ax.plot([wp_2[0]], [wp_2[1]], ".b", markersize=10)

    x, y = get_circle(center_2["L"], radius)
    ax.plot(x, y, "-.b")
    x, y = get_circle(center_2["R"], radius)
    ax.plot(x, y, "--b")

    ax.axis("equal")
    ax.grid(True)

    # plt.show()

    return ax


# plot_base()

# print("RSR")


def get_tangents(
    center_1,
    radius_1,
    heading_1,
    delta_1,
    center_2,
    radius_2,
    heading_2,
    delta_2,
    plot=False,
):
    """
    Compute the tangents between two circles and optionally plot the results.
    This function calculates the tangents between two circles defined by their
    centers, radii, headings, and deltas. It also provides an option to visualize
    the tangents, circles, and other relevant geometric elements.
    Parameters:
    ----------
    center_1 : array-like
        Coordinates of the center of the first circle [x, y].
    radius_1 : float
        Radius of the first circle.
    heading_1 : float
        Heading angle of the first circle in radians.
    delta_1 : float
        Direction multiplier for the first circle (+1 for counterclockwise, -1 for clockwise).
    center_2 : array-like
        Coordinates of the center of the second circle [x, y].
    radius_2 : float
        Radius of the second circle.
    heading_2 : float
        Heading angle of the second circle in radians.
    delta_2 : float
        Direction multiplier for the second circle (+1 for counterclockwise, -1 for clockwise).
    plot : bool, optional
        If True, plots the circles, tangents, and other geometric elements (default is False).
    Returns:
    -------
    output : dict
        A dictionary containing the following keys:
        - "C1": List of points along the first circle up to the tangent point.
        - "S": List of points representing the straight-line segment between the tangent points.
        - "C2": List of points along the second circle from the tangent point onward.
    Raises:
    ------
    ValueError
        If no valid path is found due to incorrect tangent calculations.
    Notes:
    ------
    - The function uses CasADi for symbolic computation and NumPy for numerical operations.
    - The plotting functionality requires a compatible plotting library (e.g., Matplotlib).
    Example:
    -------
    >>> center_1 = [0, 0]
    >>> radius_1 = 5
    >>> heading_1 = 0
    >>> delta_1 = 1
    >>> center_2 = [10, 0]
    >>> radius_2 = 5
    >>> heading_2 = 0
    >>> delta_2 = -1
    >>> result = get_tangents(center_1, radius_1, heading_1, delta_1, center_2, radius_2, heading_2, delta_2, plot=True)
    """

    output = dict()

    u1 = casadi.SX.sym("u1")
    u2 = casadi.SX.sym("u2")

    phi_1 = 2 * np.pi * u1 * delta_1 + heading_1 - delta_1 * np.pi / 2
    phi_2 = 2 * np.pi * u2 * delta_2 + heading_2 - delta_2 * np.pi / 2

    u1_func = lambda angle: (angle - heading_1 + delta_1 * np.pi / 2) / (
        delta_1 * 2 * np.pi
    )
    u2_func = lambda angle: (angle - heading_2 + delta_2 * np.pi / 2) / (
        delta_2 * 2 * np.pi
    )
    # Make tangents vector functions
    tan_1 = casadi.cross(
        np.array([0, 0, 1]),
        np.array(
            [delta_1 * radius * np.cos(phi_1), delta_1 * radius * np.sin(phi_1), 0]
        ),
    )[0:2]
    tan_2 = casadi.cross(
        np.array([0, 0, 1]),
        np.array(
            [delta_2 * radius * np.cos(phi_2), delta_2 * radius * np.sin(phi_2), 0]
        ),
    )[0:2]

    # Make circle functions
    circle_1_func = center_1 + radius_1 * np.array([np.cos(phi_1), np.sin(phi_1)])
    circle_2_func = center_2 + radius_2 * np.array([np.cos(phi_2), np.sin(phi_2)])

    # Plot the circles

    if plot:
        ax = plot_base()

        ## Plot the circle center points
        ax.plot(
            [center_1[0]],
            [center_1[1]],
            marker=".",
            color="xkcd:baby blue",
            markersize=10,
        )
        ax.plot(
            [center_2[0]],
            [center_2[1]],
            marker=".",
            color="xkcd:baby blue",
            markersize=10,
        )

        ## Plot a couple of tangent vectors
        for i in np.linspace(0.2, 0.8, 5):
            c1 = casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [i])
            t1 = casadi.substitute(tan_1, casadi.vertcat(*[u1]), [i])
            t1 *= 5 / casadi.norm_2(t1)

            c1_np = np.array(casadi.DM(c1)).flatten()
            t1_np = np.array(casadi.DM(t1)).flatten()
            ax.plot(
                [c1_np[0], c1_np[0] + t1_np[0]], [c1_np[1], c1_np[1] + t1_np[1]], "k"
            )

            c2 = casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [i])
            t2 = casadi.substitute(tan_2, casadi.vertcat(*[u2]), [i])
            t2 *= 5 / casadi.norm_2(t2)
            c2_np = np.array(casadi.DM(c2)).flatten()
            t2_np = np.array(casadi.DM(t2)).flatten()
            ax.plot(
                [c2_np[0], c2_np[0] + t2_np[0]], [c2_np[1], c2_np[1] + t2_np[1]], "k"
            )

        ## Plot line connecting the circle centers

        ax.plot(
            [center_1[0], center_2[0]],
            [center_1[1], center_2[1]],
            linestyle="--",
            color="xkcd:lavender",
        )

        ## Plot the starting point of each circle
        c1 = casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [0])
        c2 = casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [0])

        # Convert to numpy arrays for plotting
        c1 = np.array(casadi.DM(c1)).flatten()
        c2 = np.array(casadi.DM(c2)).flatten()
        ax.plot(
            [c1[0]], [c1[1]], marker="o", color="xkcd:reddish orange", markersize=15
        )
        ax.plot(
            [c2[0]], [c2[1]], marker="o", color="xkcd:reddish orange", markersize=15
        )

    # Compute the line connecting the circle's centers
    d = center_2 - center_1
    # Calculate normal vector to the connecting line
    n = np.dot(get_frame(np.pi / 2), d / np.linalg.norm(d))

    if plot:
        ## Plotting the normal vectors
        ax.plot(
            [center_1[0], center_1[0] + radius_1 * n[0]],
            [center_1[1], center_1[1] + radius_1 * n[1]],
            linestyle="--",
            color="xkcd:hot pink",
        )
        ax.plot(
            [center_2[0], center_2[0] + radius_2 * n[0]],
            [center_2[1], center_2[1] + radius_2 * n[1]],
            linestyle="--",
            color="xkcd:hot pink",
        )

    ##########################################################
    # Compute the first tangent
    ## Compute the normal vector's angle
    n_angle = np.arctan2(n[1], n[0])
    ## Compute the parameter for the tangent points on both circles
    u1_opt = u1_func(n_angle)
    if u1_opt < 0:
        u1_opt = u1_opt + 1
    u2_opt = u2_func(n_angle)
    if u2_opt < 0:
        u2_opt = u2_opt + 1

    ## Compute the points on the circles for the first tangent
    c1 = casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt])
    c2 = casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt])

    tangent_1 = c2 - c1
    tangent_1 /= casadi.norm_2(tangent_1)

    if plot:
        # Convert to numpy arrays for plotting
        c1 = np.array(casadi.DM(c1)).flatten()
        c2 = np.array(casadi.DM(c2)).flatten()
        ax.plot(
            [c1[0], c2[0]], [c1[1], c2[1]], linestyle="--", color="xkcd:kelly green"
        )

    ## Compute the tangent vectors on the circles
    t1 = casadi.substitute(tan_1, casadi.vertcat(*[u1]), [u1_opt])
    t1 /= casadi.norm_2(t1)
    t2 = casadi.substitute(tan_2, casadi.vertcat(*[u2]), [u2_opt])
    t2 /= casadi.norm_2(t2)

    diff = float(casadi.norm_2(tangent_1 - t1) + casadi.norm_2(tangent_1 - t2))

    if np.isclose(diff, 0):
        u = np.arange(0, u1_opt, 0.001)
        output["C1"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [ui]) for ui in u
        ]
        output["S"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt]),
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt]),
        ]
        u = np.arange(u2_opt, 1, 0.001)
        output["C2"] = [
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [ui]) for ui in u
        ]

    if plot:
        ## Plot the tangent vectors on the circles that are parallel to the first tangent
        ax.plot(
            [c1[0], c1[0] + radius_1 * t1[0]],
            [c1[1], c1[1] + radius_1 * t1[1]],
            linestyle="-",
            color="xkcd:bright purple",
        )
        ax.plot(
            [c2[0], c2[0] + radius_2 * t2[0]],
            [c2[1], c2[1] + radius_2 * t2[1]],
            linestyle="-",
            color="xkcd:bright purple",
        )

    ##########################################################
    # Compute the second tangent
    n_angle = np.arctan2(-n[1], -n[0])
    u1_opt = u1_func(n_angle)
    if u1_opt < 0:
        u1_opt = u1_opt + 1
    u2_opt = u2_func(n_angle)
    if u2_opt < 0:
        u2_opt = u2_opt + 1

    c1 = casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt])
    c2 = casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt])

    tangent_2 = c2 - c1
    tangent_2 /= casadi.norm_2(tangent_2)

    if plot:
        ## Plotting the second tangent
        # Convert to numpy arrays for plotting
        c1 = np.array(casadi.DM(c1)).flatten()
        c2 = np.array(casadi.DM(c2)).flatten()
        ax.plot(
            [c1[0], c2[0]], [c1[1], c2[1]], linestyle="--", color="xkcd:kelly green"
        )

    ## Compute the tangent vectors on the circles
    t1 = casadi.substitute(tan_1, casadi.vertcat(*[u1]), [u1_opt])
    t1 /= casadi.norm_2(t1)
    t2 = casadi.substitute(tan_2, casadi.vertcat(*[u2]), [u2_opt])
    t2 /= casadi.norm_2(t2)

    diff = float(casadi.norm_2(tangent_2 - t1) + casadi.norm_2(tangent_2 - t2))

    if np.isclose(diff, 0):
        u = np.arange(0, u1_opt, 0.001)
        output["C1"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [ui]) for ui in u
        ]
        output["S"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt]),
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt]),
        ]
        u = np.arange(u2_opt, 1, 0.001)
        output["C2"] = [
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [ui]) for ui in u
        ]

    if plot:
        ## Plot the tangent vectors on the circles that are parallel to the second tangent
        ax.plot(
            [c1[0], c1[0] + radius_1 * t1[0]],
            [c1[1], c1[1] + radius_1 * t1[1]],
            linestyle="-",
            color="xkcd:bright purple",
        )
        ax.plot(
            [c2[0], c2[0] + radius_2 * t2[0]],
            [c2[1], c2[1] + radius_2 * t2[1]],
            linestyle="-",
            color="xkcd:bright purple",
        )

    ##########################################################
    # Computing inner tangents
    # Calculate the intersection point of the two tangent lines
    xp = (center_1[0] * radius_1 + center_2[0] * radius_2) / (radius_1 + radius_2)
    yp = (center_1[1] * radius_1 + center_2[1] * radius_2) / (radius_1 + radius_2)

    if plot:
        ax.plot([xp], [yp], ".r", markersize=10)
    # Third and fourth tangents
    xt1 = (
        radius_1**2 * (xp - center_1[0])
        + radius_1
        * (yp - center_1[1])
        * np.sqrt((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2 - radius_1**2)
    ) / ((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2) + center_1[0]
    xt2 = (
        radius_1**2 * (xp - center_1[0])
        - radius_1
        * (yp - center_1[1])
        * np.sqrt((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2 - radius_1**2)
    ) / ((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2) + center_1[0]

    yt1 = (
        (radius_1**2 * (yp - center_1[1]))
        - radius_1
        * (xp - center_1[0])
        * np.sqrt((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2 - radius_1**2)
    ) / ((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2) + center_1[1]
    yt2 = (
        (radius_1**2 * (yp - center_1[1]))
        + radius_1
        * (xp - center_1[0])
        * np.sqrt((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2 - radius_1**2)
    ) / ((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2) + center_1[1]

    if plot:
        ## Plotting the tangent points on the first circle
        ax.plot([xt1, xt2], [yt1, yt2], ".r", markersize=10)

    xt3 = (
        radius_2**2 * (xp - center_2[0])
        + radius_2
        * (yp - center_2[1])
        * np.sqrt((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2 - radius_2**2)
    ) / ((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2) + center_2[0]
    xt4 = (
        radius_2**2 * (xp - center_2[0])
        - radius_2
        * (yp - center_2[1])
        * np.sqrt((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2 - radius_2**2)
    ) / ((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2) + center_2[0]

    yt3 = (
        (radius_2**2 * (yp - center_2[1]))
        - radius_2
        * (xp - center_2[0])
        * np.sqrt((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2 - radius_2**2)
    ) / ((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2) + center_2[1]
    yt4 = (
        (radius_2**2 * (yp - center_2[1]))
        + radius_2
        * (xp - center_2[0])
        * np.sqrt((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2 - radius_2**2)
    ) / ((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2) + center_2[1]

    if plot:
        ## Plotting the tangent points on the second circle
        ax.plot([xt3, xt4], [yt3, yt4], ".r", markersize=10)

    # Third tangent
    u1_opt = u1_func(np.arctan2(yt1 - center_1[1], xt1 - center_1[0]))
    if u1_opt < 0:
        u1_opt = u1_opt + 1
    u2_opt = u2_func(np.arctan2(yt3 - center_2[1], xt3 - center_2[0]))
    if u2_opt < 0:
        u2_opt = u2_opt + 1

    c1 = casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt])
    c2 = casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt])

    t1 = casadi.substitute(tan_1, casadi.vertcat(*[u1]), [u1_opt])
    t1 /= casadi.norm_2(t1)
    t2 = casadi.substitute(tan_2, casadi.vertcat(*[u2]), [u2_opt])
    t2 /= casadi.norm_2(t2)

    if plot:
        ## Plot the tangent vectors on the circles that are parallel to the third tangent
        # Convert to numpy arrays for plotting
        c1 = np.array(casadi.DM(c1)).flatten()
        c2 = np.array(casadi.DM(c2)).flatten()
        t1 = np.array(casadi.DM(t1)).flatten()
        t2 = np.array(casadi.DM(t2)).flatten()
        ax.plot(
            [c1[0], c1[0] + radius_1 * t1[0]],
            [c1[1], c1[1] + radius_1 * t1[1]],
            linestyle="-",
            color="xkcd:bright purple",
        )
        ax.plot(
            [c2[0], c2[0] + radius_2 * t2[0]],
            [c2[1], c2[1] + radius_2 * t2[1]],
            linestyle="-",
            color="xkcd:bright purple",
        )

    tangent_3 = np.array([xt3 - xt1, yt3 - yt1])
    tangent_3 /= np.linalg.norm(tangent_3)

    diff = float(casadi.norm_2(tangent_3 - t1) + casadi.norm_2(tangent_3 - t2))

    if np.isclose(diff, 0):
        u = np.arange(0, u1_opt, 0.001)
        output["C1"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [ui]) for ui in u
        ]
        output["S"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt]),
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt]),
        ]
        u = np.arange(u2_opt, 1, 0.001)
        output["C2"] = [
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [ui]) for ui in u
        ]

    # Fourth tangent
    u1_opt = u1_func(np.arctan2(yt2 - center_1[1], xt2 - center_1[0]))
    if u1_opt < 0:
        u1_opt = u1_opt + 1
    u2_opt = u2_func(np.arctan2(yt4 - center_2[1], xt4 - center_2[0]))
    if u2_opt < 0:
        u2_opt = u2_opt + 1

    c1 = casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt])
    c2 = casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt])

    t1 = casadi.substitute(tan_1, casadi.vertcat(*[u1]), [u1_opt])
    t1 /= casadi.norm_2(t1)
    t2 = casadi.substitute(tan_2, casadi.vertcat(*[u2]), [u2_opt])
    t2 /= casadi.norm_2(t2)

    if plot:
        ## Plot the tangent vectors on the circles that are parallel to the fourth tangent
        # Convert to numpy arrays for plotting
        c1 = np.array(casadi.DM(c1)).flatten()
        c2 = np.array(casadi.DM(c2)).flatten()
        t1 = np.array(casadi.DM(t1)).flatten()
        t2 = np.array(casadi.DM(t2)).flatten()
        ax.plot(
            [c1[0], c1[0] + radius_1 * t1[0]],
            [c1[1], c1[1] + radius_1 * t1[1]],
            linestyle="-",
            color="xkcd:bright purple",
        )
        ax.plot(
            [c2[0], c2[0] + radius_2 * t2[0]],
            [c2[1], c2[1] + radius_2 * t2[1]],
            linestyle="-",
            color="xkcd:bright purple",
        )

    tangent_4 = np.array([xt4 - xt2, yt4 - yt2])
    tangent_4 /= np.linalg.norm(tangent_4)

    diff = float(casadi.norm_2(tangent_4 - t1) + casadi.norm_2(tangent_4 - t2))

    if np.isclose(diff, 0):
        u = np.arange(0, u1_opt, 0.001)
        output["C1"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [ui]) for ui in u
        ]
        output["S"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt]),
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt]),
        ]
        u = np.arange(u2_opt, 1, 0.001)
        output["C2"] = [
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [ui]) for ui in u
        ]

    if plot:
        ax.plot([xt1, xt3], [yt1, yt3], "--c")
        ax.plot([xt2, xt4], [yt2, yt4], "--c")

        #########################################################
        # Plot the path
        # Convert to numpy arrays for plotting
        output["C1"] = [np.array(casadi.DM(c)).flatten() for c in output["C1"]]
        output["S"] = [np.array(casadi.DM(c)).flatten() for c in output["S"]]
        output["C2"] = [np.array(casadi.DM(c)).flatten() for c in output["C2"]]
        if len(output["C1"]) == 0 or len(output["C2"]) == 0:
            raise ValueError("No valid path found. Check the tangent calculations.")
        ax.plot(
            [x[0] for x in output["C1"]],
            [x[1] for x in output["C1"]],
            color="xkcd:golden yellow",
            linewidth=3,
        )
        ax.plot(
            [x[0] for x in output["S"]],
            [x[1] for x in output["S"]],
            color="xkcd:vermillion",
            linewidth=3,
        )
        ax.plot(
            [x[0] for x in output["C2"]],
            [x[1] for x in output["C2"]],
            color="xkcd:bright magenta",
            linewidth=3,
        )

    return output


def get_path_length(path):
    """Calculate the total length of the path"""

    length = 0.0

    for key in path:
        # Convert to numpy if necessary
        if isinstance(path[key], list):
            path[key] = [np.array(casadi.DM(c)).flatten() for c in path[key]]
        else:
            path[key] = np.array(casadi.DM(path[key])).flatten()

        segment = path[key]
        if len(segment) < 2:
            continue

        for i in range(1, len(segment) - 1):
            p1 = segment[i - 1]
            p2 = segment[i]
            length += np.linalg.norm(np.array(p2) - np.array(p1))

    return length


def plotPath(path, title="Dubins Path"):
    # Convert from Casadi to numpy
    # Note: Casadi returns a DM object, which we need to convert to numpy arrays

    for key in path:
        if isinstance(path[key], list):
            path[key] = [np.array(casadi.DM(c)).flatten() for c in path[key]]
        else:
            path[key] = np.array(casadi.DM(path[key])).flatten()

    C1 = np.asarray(path["C1"])
    S = np.asarray(path["S"])
    C2 = np.asarray(path["C2"])

    print(C1)
    print(S)
    print(C2)

    if len(C1) > 0:
        plt.plot(
            C1[:, 0], C1[:, 1], label="C1", color="xkcd:golden yellow", linewidth=3
        )

    if len(S) > 0:
        plt.plot(S[:, 0], S[:, 1], label="S", color="xkcd:vermillion", linewidth=3)

    if len(C2) > 0:
        plt.plot(
            C2[:, 0], C2[:, 1], label="C2", color="xkcd:bright magenta", linewidth=3
        )

    plt.legend()
    plt.title(title)
    # Configure matplotlib to use a square aspect ratio
    plt.gca().set_aspect("equal", adjustable="box")

    plt.show()


class Pose:
    def __init__(self, pos_x, pos_y, yaw):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.yaw = yaw


def pathIsValid(path):
    """
    Check if the path is valid.
    A path is considered valid if it has at least one segment in C1, S, and C2.
    """
    if "C1" not in path or "S" not in path or "C2" not in path:
        return False

    if len(path["C1"]) == 0 or len(path["S"]) == 0 or len(path["C2"]) == 0:
        return False

    return True


def trimPath(path, goal_x, goal_y, distance_threshold=0.1):
    # Combine all points from C1, S, C2 into a single list
    full_path = []
    assert "C1" in path, "Paths must contain C1"
    assert "S" in path, "Paths must contain S"
    assert "C2" in path, "Paths must contain C2"

    full_path.extend(path["C1"])
    full_path.extend(path["S"])
    full_path.extend(path["C2"])

    full_length = len(full_path)

    # Now trim any parts that extend beyond the goal point
    trimmed_path = []
    for idx, point in enumerate(full_path):
        trimmed_path.append(point)
        # Check if the point is close to the goal
        if (
            np.linalg.norm(np.array(point) - np.array([goal_x, goal_y]))
            < distance_threshold
        ):
            print(f"Stopping at index {idx}/{full_length}")
            break

    return np.asarray(trimmed_path)


def getShortestDubbinsPath(radius, start_x, start_y, start_yaw, end_x, end_y, end_yaw):

    start_center = getCenter([start_x, start_y], radius, start_yaw)
    end_center = getCenter([end_x, end_y], radius, end_yaw)

    # Print center1 contents
    print("Center 1:")
    print(f"  R: {start_center['R']}")
    print(f"  L: {start_center['L']}")

    min_length = float("inf")
    shortest_path = None

    # RSR
    u1 = casadi.SX.sym("u1")
    u2 = casadi.SX.sym("u2")
    path_rsr = get_tangents(
        start_center["R"], radius, start_yaw, -1, end_center["R"], radius, heading_2, -1
    )
    length = get_path_length(path_rsr)
    print(f"RSR Length: {length:.2f} units")
    # plotPath(path_rsr, f"RSR Length: {length:.2f} units")

    if length < min_length and pathIsValid(path_rsr):
        min_length = length
        shortest_path = path_rsr
        print("Shortest path so far is RSR")

    # LSR
    u1 = casadi.SX.sym("u1")
    u2 = casadi.SX.sym("u2")
    path_lsr = get_tangents(
        start_center["L"], radius, start_yaw, 1, end_center["R"], radius, heading_2, -1
    )
    length = get_path_length(path_lsr)
    print(f"LSR Length: {length:.2f} units")
    # plotPath(path_lsr, f"LSR Length: {length:.2f} units")

    if length < min_length and pathIsValid(path_lsr):
        min_length = length
        shortest_path = path_lsr
        print("Shortest path so far is LSR")

    # RSL
    u1 = casadi.SX.sym("u1")
    u2 = casadi.SX.sym("u2")
    path_rsl = get_tangents(
        start_center["R"], radius, start_yaw, -1, end_center["L"], radius, heading_2, 1
    )
    length = get_path_length(path_rsl)
    print(f"RSL Length: {length:.2f} units")
    # plotPath(path_rsl, f"RSL Length: {length:.2f} units")

    if length < min_length and pathIsValid(path_rsl):
        min_length = length
        shortest_path = path_rsl
        print("Shortest path so far is RSL")

    # LSL
    u1 = casadi.SX.sym("u1")
    u2 = casadi.SX.sym("u2")
    path_lsl = get_tangents(
        start_center["L"], radius, start_yaw, 1, end_center["L"], radius, heading_2, 1
    )
    length = get_path_length(path_lsl)
    print(f"LSL Length: {length:.2f} units")
    # plotPath(path_lsl, f"LSL Length: {length:.2f} units")

    if length < min_length and pathIsValid(path_lsl):
        min_length = length
        shortest_path = path_lsl
        print("Shortest path so far is LSL")

    return shortest_path, min_length


shortest_path, min_length = getShortestDubbinsPath(
    radius=1, start_x=0, start_y=0, start_yaw=0, end_x=-5, end_y=-10, end_yaw=np.pi
)

print(f"Minimum Length: {min_length:.2f} units")
# plotPath(shortest_path)

final_path = trimPath(shortest_path, goal_x=-5, goal_y=-10, distance_threshold=0.1)

print(final_path)

plt.plot(
    final_path[:, 0],
    final_path[:, 1],
    label="Trimmed Path",
    color="xkcd:orange",
    linewidth=3,
)
plt.show()
