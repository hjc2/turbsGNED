import asyncio
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple


import numpy as np
import matplotlib.pyplot as plt
from base import Disc, Throw, huckit


# import base.Disc as Disc

disc = Disc("Frisbee")
disc.diam = 0.274
disc.jxy = 7.69e-3
disc.jz = 1.01e-2
disc.mass = 0.175
disc.aoarange = np.array([-1.745329252,-1.658062789,-1.570796327,-1.483529864,-1.396263402,-1.308996939,-1.221730476,-1.134464014,-1.047197551,-0.959931089,-0.174532925,-0.157079633,-0.13962634,-0.122173048,-0.104719755,-0.087266463,-0.06981317,-0.052359878,-0.034906585,-0.017453293,0,0.017453293,0.034906585,0.052359878,0.06981317,0.087266463,0.104719755,0.122173048,0.13962634,0.157079633,0.174532925,0.191986218,0.20943951,0.226892803,0.244346095,0.261799388,0.27925268,0.296705973,0.314159265,0.331612558,0.34906585,0.366519143,0.383972435,0.401425728,0.41887902,0.436332313,0.453785606,0.471238898,0.488692191,0.506145483,0.523598776,0.541052068,0.558505361,0.575958653,0.593411946,0.610865238,0.628318531,0.645771823,0.663225116,0.680678408,0.698131701,0.715584993,0.733038286,0.750491578,0.767944871,0.785398163,0.802851456,0.820304748,0.837758041,0.855211333,0.872664626,0.959931089,1.047197551,1.134464014,1.221730476,1.308996939,1.396263402,1.483529864,1.570796327,1.658062789,1.745329252])
disc.cl = np.array([0.15942029,0.096618357,0.009661836,-0.077294686,-0.144927536,-0.217391304,-0.299516908,-0.357487923,-0.434782609,-0.492753623,-0.234509466,-0.204388985,-0.148450947,-0.126936317,-0.096815835,-0.083907057,-0.058089501,-0.027969019,0.023666093,0.075301205,0.118330465,0.182874355,0.238812392,0.303356282,0.376506024,0.432444062,0.496987952,0.55292599,0.613166954,0.673407917,0.729345955,0.776678141,0.836919105,0.875645439,0.922977625,0.983218589,1.021944923,1.07788296,1.129518072,1.172547332,1.219879518,1.275817556,1.318846816,1.396299484,1.44363167,1.486660929,1.512478485,1.589931153,1.620051635,1.667383821,1.688898451,1.710413081,1.740533563,1.762048193,1.813683305,1.850258176,1.886833046,1.929862306,1.972891566,2.007314974,2.063253012,2.093373494,2.106282272,2.136402754,2.151462995,2.149311532,1.146729776,1.133820998,1.133820998,1.09939759,1.082185886,1.019323671,0.8647343,0.724637681,0.589371981,0.45410628,0.299516908,0.140096618,0.004830918,-0.140096618,-0.280193237])
disc.cd = np.array([0.81906226,0.843658724,0.848578017,0.828900846,0.811683321,0.806764028,0.789546503,0.760030746,0.748962337,0.710837817,0.162962963,0.145679012,0.12345679,0.10617284,0.10617284,0.096296296,0.088888889,0.088888889,0.086419753,0.088888889,0.09382716,0.101234568,0.113580247,0.125925926,0.133333333,0.151851852,0.166666667,0.190123457,0.216049383,0.239506173,0.264197531,0.286419753,0.309876543,0.341975309,0.367901235,0.397530864,0.427160494,0.459259259,0.498765432,0.530864198,0.572839506,0.602469136,0.641975309,0.681481481,0.72345679,0.754320988,0.787654321,0.832098765,0.864197531,0.90617284,0.920987654,0.95308642,0.975308642,1.002469136,1.032098765,1.071604938,1.103703704,1.135802469,1.182716049,1.232098765,1.295061728,1.340740741,1.380246914,1.437037037,1.491358025,1.530864198,1.009876543,1.017283951,1.039506173,1.04691358,1.064197531,1.097002306,1.146195234,1.185549577,1.212605688,1.247040738,1.269177556,1.269177556,1.29377402,1.286395081,1.276556495])
disc.cm = np.array([0.031216649,0.013607257,0.000266809,-0.01547492,-0.031216649,-0.046691569,-0.060565635,-0.073372465,-0.085645678,-0.095517609,-0.038247863,-0.036538462,-0.030982906,-0.027564103,-0.023504274,-0.021581197,-0.017307692,-0.014102564,-0.010042735,-0.008333333,-0.006837607,-0.008119658,-0.00982906,-0.006837607,-0.006623932,-0.004059829,-0.002350427,-0.001068376,-0.001068376,0.000641026,0.003205128,0.007478632,0.010683761,0.013461538,0.016452991,0.021153846,0.025854701,0.031410256,0.034401709,0.04017094,0.043376068,0.048931624,0.053632479,0.061111111,0.068589744,0.07542735,0.083119658,0.088888889,0.096367521,0.103846154,0.111324786,0.115811966,0.125854701,0.137820513,0.144017094,0.152564103,0.160042735,0.171794872,0.180769231,0.18974359,0.2,0.208760684,0.216239316,0.223931624,0.227991453,0.22542735,0.019871795,0.018376068,0.018162393,0.018162393,0.016452991,0.016275347,0.017609392,0.021611526,0.021611526,0.020010672,0.015741729,0.008271078,0.002401281,-0.007203842,-0.011472785])

throw = Throw('Throw 1')
throw.speed = 10
throw.spin = 60*2*np.pi # Convert to rad/s
throw.spindir = 1
throw.launch_angle = 20*(np.pi/180) # Convert to rad
throw.nose_angle = 20*(np.pi/180) # Convert to rad
throw.roll_angle = 45*(np.pi/180) # Convert to rad
throw.label = 'throw 1'

global FIELD_LENGTH
FIELD_LENGTH = 110
global FIELD_WIDTH
FIELD_WIDTH = 40
global END_ZONE_LENGTH
END_ZONE_LENGTH = 20
global BRICK_MARK_DISTANCE
BRICK_MARK_DISTANCE = 20

global TARGET_X
global TARGET_Y
TARGET_X = 100
TARGET_Y = 30


"""# New Section"""


# Keep all other parameters fixed
speed = 35
spin = 60
nose_angle = 7
translation_angle = 24

# Target coordinates (fixed for now)

rotation_angle_rad = -1 * np.radians(translation_angle)

x_goal = TARGET_X
y_goal = TARGET_Y

# Prepare data to plot the phase space
# launch_angles = np.arange(0, 45, 0.75)  # Launch angles from 0 to 45 degrees
# roll_angles = np.arange(15, 60, 0.75)  # Roll angles from -45 to 45 degrees
# launch_angles = np.arange(10, 11, 1)  # Launch angles from 0 to 45 degrees
# roll_angles = np.arange(30, 31, 1)  # Launch angles from 0 to 45 degrees

# launch_angles = np.arange(0, 45, 0.50)  # Launch angles from 0 to 45 degrees
# roll_angles = np.arange(0, 60, 0.50)  # Roll angles from -45 to 45 degrees
launch_angles = np.arange(0, 45,1)  # Launch angles from 0 to 45 degrees
roll_angles = np.arange(0, 60, 1) 

totalIter = len(launch_angles) * len(roll_angles)
print(f"iterations {totalIter}")
i = 0

print(launch_angles)
print(roll_angles)

# Prepare a list to store the distance results
distance_results = []
outdist_results = []

def check_inbounds(x, y):
  if x < 0 or x > FIELD_LENGTH or y < 0 or y > FIELD_WIDTH:
      return False
  return True

@dataclass
class SimParams:
    speed: float
    spin: float
    nose_angle: float
    translation_angle: float
    launch_angle: float
    roll_angle: float
    
async def simulate_throw(params: SimParams, disc, throw) -> Tuple[float, float, float, float, float]:
    # Set throw parameters from SimParams
    throw.launch_angle = params.launch_angle * (np.pi / 180)
    throw.nose_angle = params.nose_angle * (np.pi / 180)
    throw.roll_angle = params.roll_angle * (np.pi / 180)
    throw.spin = params.spin * 2 * np.pi
    throw.speed = params.speed
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        try:
            result = await loop.run_in_executor(pool, huckit, disc, throw)
            
            x = result.pos_g[:,0] * 3.28 / 3
            y = result.pos_g[:,1] * 3.28 / 3
            z = result.pos_g[:,2] * 3.28 / 3

            x_trans = x * np.cos(rotation_angle_rad) - y * np.sin(rotation_angle_rad)
            y_trans = x * np.sin(rotation_angle_rad) + y * np.cos(rotation_angle_rad)

            x_rot = x_trans + 20
            y_rot = y_trans + 30

            print(f"Debug - x: {x_rot[-1]}, y: {y_rot[-1]}, z: {z[-1]}, la: {params.launch_angle}, ra: {params.roll_angle}")
            return x_rot[-1], y_rot[-1], z[-1], params.launch_angle, params.roll_angle
        except Exception as e:
            print(f"Error in simulate_throw: {e}")
            return None

async def process_batch(params_list: List[SimParams], disc, throw, chunk_size=1):
    tasks = []
    results = []
    
    for i in range(0, len(params_list), chunk_size):
        chunk = params_list[i:i + chunk_size]
        chunk_tasks = [simulate_throw(p, disc, throw) for p in chunk]
        chunk_results = await asyncio.gather(*chunk_tasks)
        results.extend(chunk_results)
        
        print(f"Processed {len(results)}/{len(params_list)} simulations")
        
    return results

async def main():
    # Your existing parameters
    params_list = [
        SimParams(
            speed=speed,
            spin=spin,
            nose_angle=nose_angle,
            translation_angle=translation_angle,
            launch_angle=la,
            roll_angle=ra
        )
        for la in launch_angles
        for ra in roll_angles
    ]

    results = await process_batch(params_list, disc, throw)

    distance_results = []
    outdist_results = []
    
    for x, y, z, la, ra in results:
        if check_inbounds(x, y):
            dist = np.sqrt((x - TARGET_X)**2 + (y - TARGET_Y)**2)
            distance_results.append((la, ra, dist))
        else:
            outdist_results.append((la, ra))

    # Save results
    np.save('distance.npy', np.array(distance_results))
    np.save('outdist.npy', np.array(outdist_results))
    
    return np.array(distance_results), np.array(outdist_results)

if __name__ == "__main__":
    distance_results, outdist_results = asyncio.run(main())
    
