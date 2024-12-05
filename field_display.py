
from base import Disc, Throw, huckit
import numpy as np
import matplotlib.pyplot as plt

def plotfield(ax):
    FIELD_LENGTH = 110
    FIELD_WIDTH = 40
    END_ZONE_LENGTH = 20
    BRICK_MARK_DISTANCE = 20

    TARGET_X = 100
    TARGET_Y = 30
    # endzones
    endzone_x = [0, 0, END_ZONE_LENGTH, END_ZONE_LENGTH, 0]
    endzone_y = [0, FIELD_WIDTH, FIELD_WIDTH, 0, 0]
    endzone_z = [0.01]*5
    ax.plot(endzone_x, endzone_y, endzone_z, 'blue', linewidth=3)
    endzone_x = [FIELD_LENGTH - END_ZONE_LENGTH, FIELD_LENGTH - END_ZONE_LENGTH, FIELD_LENGTH, FIELD_LENGTH, FIELD_LENGTH - END_ZONE_LENGTH]
    endzone_y = [0, FIELD_WIDTH, FIELD_WIDTH, 0, 0]
    endzone_z = [0.01]*5
    ax.plot(endzone_x, endzone_y, endzone_z, 'blue', linewidth=3)
    #sidelines
    ax.plot([0, FIELD_LENGTH], [0, 0], [0.01, 0.01], 'red', linewidth=3)
    ax.plot([0, FIELD_LENGTH], [FIELD_WIDTH, FIELD_WIDTH], [0.01, 0.01], 'red', linewidth=3)

def plot_x_marker(ax, x_center, y_center):
    ax.scatter(x_center, y_center, 0, color='red', s=30, marker='x')

def update_throw(speed=35, spin=60, launch_angle=7, nose_angle=7, roll_angle=36, translation_angle=-24):

    disc = Disc("Frisbee")
    disc.diam = 0.274
    disc.jxy = 7.69e-3
    disc.jz = 1.01e-2
    disc.mass = 0.175
    disc.aoarange = np.array([-1.745329252,-1.658062789,-1.570796327,-1.483529864,-1.396263402,-1.308996939,-1.221730476,-1.134464014,-1.047197551,-0.959931089,-0.174532925,-0.157079633,-0.13962634,-0.122173048,-0.104719755,-0.087266463,-0.06981317,-0.052359878,-0.034906585,-0.017453293,0,0.017453293,0.034906585,0.052359878,0.06981317,0.087266463,0.104719755,0.122173048,0.13962634,0.157079633,0.174532925,0.191986218,0.20943951,0.226892803,0.244346095,0.261799388,0.27925268,0.296705973,0.314159265,0.331612558,0.34906585,0.366519143,0.383972435,0.401425728,0.41887902,0.436332313,0.453785606,0.471238898,0.488692191,0.506145483,0.523598776,0.541052068,0.558505361,0.575958653,0.593411946,0.610865238,0.628318531,0.645771823,0.663225116,0.680678408,0.698131701,0.715584993,0.733038286,0.750491578,0.767944871,0.785398163,0.802851456,0.820304748,0.837758041,0.855211333,0.872664626,0.959931089,1.047197551,1.134464014,1.221730476,1.308996939,1.396263402,1.483529864,1.570796327,1.658062789,1.745329252])
    disc.cl = np.array([0.15942029,0.096618357,0.009661836,-0.077294686,-0.144927536,-0.217391304,-0.299516908,-0.357487923,-0.434782609,-0.492753623,-0.234509466,-0.204388985,-0.148450947,-0.126936317,-0.096815835,-0.083907057,-0.058089501,-0.027969019,0.023666093,0.075301205,0.118330465,0.182874355,0.238812392,0.303356282,0.376506024,0.432444062,0.496987952,0.55292599,0.613166954,0.673407917,0.729345955,0.776678141,0.836919105,0.875645439,0.922977625,0.983218589,1.021944923,1.07788296,1.129518072,1.172547332,1.219879518,1.275817556,1.318846816,1.396299484,1.44363167,1.486660929,1.512478485,1.589931153,1.620051635,1.667383821,1.688898451,1.710413081,1.740533563,1.762048193,1.813683305,1.850258176,1.886833046,1.929862306,1.972891566,2.007314974,2.063253012,2.093373494,2.106282272,2.136402754,2.151462995,2.149311532,1.146729776,1.133820998,1.133820998,1.09939759,1.082185886,1.019323671,0.8647343,0.724637681,0.589371981,0.45410628,0.299516908,0.140096618,0.004830918,-0.140096618,-0.280193237])
    disc.cd = np.array([0.81906226,0.843658724,0.848578017,0.828900846,0.811683321,0.806764028,0.789546503,0.760030746,0.748962337,0.710837817,0.162962963,0.145679012,0.12345679,0.10617284,0.10617284,0.096296296,0.088888889,0.088888889,0.086419753,0.088888889,0.09382716,0.101234568,0.113580247,0.125925926,0.133333333,0.151851852,0.166666667,0.190123457,0.216049383,0.239506173,0.264197531,0.286419753,0.309876543,0.341975309,0.367901235,0.397530864,0.427160494,0.459259259,0.498765432,0.530864198,0.572839506,0.602469136,0.641975309,0.681481481,0.72345679,0.754320988,0.787654321,0.832098765,0.864197531,0.90617284,0.920987654,0.95308642,0.975308642,1.002469136,1.032098765,1.071604938,1.103703704,1.135802469,1.182716049,1.232098765,1.295061728,1.340740741,1.380246914,1.437037037,1.491358025,1.530864198,1.009876543,1.017283951,1.039506173,1.04691358,1.064197531,1.097002306,1.146195234,1.185549577,1.212605688,1.247040738,1.269177556,1.269177556,1.29377402,1.286395081,1.276556495])
    disc.cm = np.array([0.031216649,0.013607257,0.000266809,-0.01547492,-0.031216649,-0.046691569,-0.060565635,-0.073372465,-0.085645678,-0.095517609,-0.038247863,-0.036538462,-0.030982906,-0.027564103,-0.023504274,-0.021581197,-0.017307692,-0.014102564,-0.010042735,-0.008333333,-0.006837607,-0.008119658,-0.00982906,-0.006837607,-0.006623932,-0.004059829,-0.002350427,-0.001068376,-0.001068376,0.000641026,0.003205128,0.007478632,0.010683761,0.013461538,0.016452991,0.021153846,0.025854701,0.031410256,0.034401709,0.04017094,0.043376068,0.048931624,0.053632479,0.061111111,0.068589744,0.07542735,0.083119658,0.088888889,0.096367521,0.103846154,0.111324786,0.115811966,0.125854701,0.137820513,0.144017094,0.152564103,0.160042735,0.171794872,0.180769231,0.18974359,0.2,0.208760684,0.216239316,0.223931624,0.227991453,0.22542735,0.019871795,0.018376068,0.018162393,0.018162393,0.016452991,0.016275347,0.017609392,0.021611526,0.021611526,0.020010672,0.015741729,0.008271078,0.002401281,-0.007203842,-0.011472785])


    FIELD_LENGTH = 110
    FIELD_WIDTH = 40
    END_ZONE_LENGTH = 20
    BRICK_MARK_DISTANCE = 20

    TARGET_X = 100
    TARGET_Y = 30

    throw = Throw('Throw 1')
    throw.spindir = 1
    throw.label = 'throw 1'

    throw.speed = speed
    throw.spin = spin * 2 * np.pi  # Convert to rad/s
    throw.launch_angle = launch_angle * (np.pi/180)  # Convert to rad
    throw.nose_angle = nose_angle * (np.pi/180)  # Convert to rad
    throw.roll_angle = roll_angle * (np.pi/180)  # Convert to rad

    result = huckit(disc, throw)

    # Convert position data from meters to yards
    x = result.pos_g[:,0] * 3.28 / 3  # Convert to yards
    y = result.pos_g[:,1] * 3.28 / 3  # Convert to yards
    z = result.pos_g[:,2] * 3.28 / 3  # Convert to yards

    # Apply translation rotation around the Z-axis
    translation_angle_rad = translation_angle * (np.pi / 180)  # Convert to radians
    x_rot = x * np.cos(translation_angle_rad) - y * np.sin(translation_angle_rad)
    y_rot = x * np.sin(translation_angle_rad) + y * np.cos(translation_angle_rad)

    # Apply the offset: move the trajectory to start at (20, 40)
    x_rot += 20  # Shift the entire trajectory 20 yards along the x-axis
    y_rot += 30  # Shift the entire trajectory 40 yards along the y-axis

    color = throw.rot_s[:,0] * (180/np.pi)  # Convert to deg/s
    colorMax = np.max(abs(color))
    colorMin = -colorMax

    plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')

    # BUILD FIELD
    plotfield(ax)
    plot_x_marker(ax, TARGET_X, TARGET_Y)
    # plot_x_marker(ax, 104.10466984109286, 32.25924173915902)

    # PLOT OTHER STUFF
    ax.scatter(x_rot, y_rot, z, c=color, cmap='coolwarm', vmin=colorMin, vmax=colorMax, alpha=1, s=40)
    ax.plot(x_rot, y_rot, '-k', zdir='z', alpha=0.3)

    annot = (throw.label + ' \n{:0.0f} ft ').format(throw.getDistance()*3.28)
    ax.text(x_rot[-1], y_rot[-1], 0, annot,
            horizontalalignment='left',
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xlabel('Forward Travel (yd)', fontsize=14)
    ax.set_ylabel('Lateral Travel (yd)', fontsize=14)
    ax.set_zlabel('Elevation (yd)', fontsize=14)
    ax.set_xlim3d(left=0)
    ax.set_zlim3d(bottom=0)
    max_height = np.max(z)
    ax.set_zlim3d(0, max_height * 1.2)  # Add 20% padding above max height

    ax.set_box_aspect([FIELD_LENGTH/FIELD_LENGTH,
                      FIELD_WIDTH/FIELD_LENGTH,
                      max_height/FIELD_LENGTH])

    ax.xaxis.set_pane_color((0.3, 0.5, 0.8, 0.15))
    ax.yaxis.set_pane_color((0.3, 0.5, 0.8, 0.15))
    ax.zaxis.set_pane_color((0.3, 0.8, 0.3, 0.15))

    ax.view_init(elev=30, azim=-90)
    plt.show()

def plot_disc_trajectory(la: float, ra: float, extra_params: dict = None) -> None:
    """
    Plot disc trajectory for given launch angle and roll angle
    la: launch angle in degrees 
    ra: roll angle in degrees
    extra_params: optional dict with speed, spin, nose_angle, translation_angle
    """
    params = {
        'speed': 35,
        'spin': 60,
        'nose_angle': 7,
        'translation_angle': -24
    }
    if extra_params:
        params.update(extra_params)
        
    update_throw(
        speed=params['speed'],
        spin=params['spin'],
        launch_angle=la, 
        nose_angle=params['nose_angle'],
        roll_angle=ra,
        translation_angle=params['translation_angle']
    )


