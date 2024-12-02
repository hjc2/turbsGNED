# -*- coding: utf-8 -*-
"""Hugh GNed.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Q7d7Wr_OGcGoDPF-BCnsLrNSQ7YDbAPG

#<font color=red>Before you use the simulator, press **Ctrl-F9** to execute all code on the sheet</font>

#Introduction

HuckIt is a rigid body dynamics solver that predicts the flight path of a flying disc based a given set of throw parameters. It uses lift and drag coefficient data collected from wind tunnel measurements to calculate the forces on a disc in flight. The body forces are then used to calculate the changes in velocity, position and roll angle of the disc. After executing the code on the sheet, you can skip down to the User Interface to experiment with throw parameters and see the effects on the flight path.

# Physics Simulator Code

##References

The simulator is based on the method used in Simulation of a spin stabilized sports disc, Crowther and Potts (2007)
A slightly different coordinate system is used in this model so some of the sign conventions may not match up.
The formulation assumes the following
 - Spin rate is constant and does not decay throughout the flight
 - Aerodynamic side force on the disc is negligible
 - Aerodynamic rolling moment is negligible (center of pressure is centered left to right on the disc)
 - Disc is thrown without wobble

Four coordinate systems are used in the model;
- Ground: From the thrower's position, positive $x$ is forward, positive $y$ is to the left, positive $z$ is up. The position and orientation of the disc is calculated in this coordinate system.
- Disc: The coordinate system that moves with the disc. The $x$-axis runs from tail to nose of the disc, $y$-axis from right to left, and $z$-axis is normal to the flight plate pointing up.
- Zero side-slip: A coordinate system that is used to transition to the wind coordinates. The only difference between this and the disc coordinate system is a rotation about the $z$-axis by the side-slip angle, beta. The rolling rate is calculated in this coordinate system.
- Wind: This coordinate system points directly into the oncoming airflow, and is meant to replicate the coordinate system in a wind tunnel. The only difference between this and the zero side-slip coordinate system is a rotation about the $y$-axis by the angle of attack, alpha. The aerodynamic forces (lift, drag, and pitching moment) and body accelerations are calculated in this coordinate system.

## Setup
"""

import numpy as np

"""## Define Coordinate Transform Functions"""

def T_gd(angles):
  phi = angles[0]
  theta = angles[1]
  psi = angles[2]
  return np.array([[np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
                   [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
                   [-np.sin(theta),            np.sin(phi)*np.cos(theta),                                       np.cos(phi)*np.cos(theta)                                      ]])

def T_dg(angles):
  phi = angles[0] 
  theta = angles[1]
  psi = angles[2]
  return np.array([[np.cos(theta)*np.cos(psi),                                      np.cos(theta)*np.sin(psi),                                      -np.sin(theta)            ],
                    [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.sin(phi)*np.cos(theta)],
                    [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(theta)]])

def T_ds(beta):
  return np.array([[np.cos(beta), -np.sin(beta), 0],
                   [np.sin(beta),  np.cos(beta), 0],
                   [0,             0,            1]])

def T_sd(beta):
  return np.array([[np.cos(beta),  np.sin(beta), 0],
                   [-np.sin(beta), np.cos(beta), 0],
                   [0,             0,            1]])

def T_sw(alpha):
  return np.array([[np.cos(alpha), 0, -np.sin(alpha)],
                   [0,             1, 0            ],
                   [np.sin(alpha), 0, np.cos(alpha)]])

def T_ws(alpha):
  return np.array([[np.cos(alpha), 0, np.sin(alpha)],
                   [0,             1, 0            ],
                   [-np.sin(alpha), 0, np.cos(alpha)]])

"""## Define Discs"""

'''
Define Disc class
Property definitions
- name: Name of disc as listed by the manufacturer
- aoarange: Angle of Attack at which the aerodynamic coefficients have been measured in rad. Must be sorted in ascending order.
- cl: Coefficient of lift at various AoA
- cd: Coefficient of drag at various AoA
- cm: Coefficient of pitching moment at various AoA
- jxy: Normalized mass moment of inertia about the roll/pitch axis in m^2. Multiply by mass to get true MMOI.
- jz: Normalized mass moment of inertia about the spin axis in m^2. Multiply by mass to get true MMOI.
- diam: Disc diameter in m
- mass: Mass of disc in kg

Method definitions
- getCl: Returns the coefficient of lift at the specified Angle of Attack
- getCd: Returns the coefficient of drag at the specified Angle of Attack
- getCm: Returns the coefficient of moment at the specified Angle of Attack
'''

discList = [(0,0)]

class Disc:
  def __init__(self, name):
    self.name = name
    self.cl = [0]
    self.cd = [0]
    self.cm = [0]
    self.jxy = 0
    self.jz = 0
    self.diam = 0

  def getCl(self, aoa):
    return np.interp(aoa, self.aoarange, self.cl)
  def getCd(self, aoa):
    return np.interp(aoa, self.aoarange, self.cd)
  def getCm(self, aoa):
    return np.interp(aoa, self.aoarange, self.cm)

'''
Define discs
Aerodynamic coefficients for frisbee sourced from Frisbee Aerodynamics, Potts and Crowther (2002)
'''
frisbee = Disc("Frisbee")
frisbee.diam = 0.274
frisbee.jxy = 7.69e-3
frisbee.jz = 1.01e-2
frisbee.aoarange = np.array([-1.745329252,-1.658062789,-1.570796327,-1.483529864,-1.396263402,-1.308996939,-1.221730476,-1.134464014,-1.047197551,-0.959931089,-0.174532925,-0.157079633,-0.13962634,-0.122173048,-0.104719755,-0.087266463,-0.06981317,-0.052359878,-0.034906585,-0.017453293,0,0.017453293,0.034906585,0.052359878,0.06981317,0.087266463,0.104719755,0.122173048,0.13962634,0.157079633,0.174532925,0.191986218,0.20943951,0.226892803,0.244346095,0.261799388,0.27925268,0.296705973,0.314159265,0.331612558,0.34906585,0.366519143,0.383972435,0.401425728,0.41887902,0.436332313,0.453785606,0.471238898,0.488692191,0.506145483,0.523598776,0.541052068,0.558505361,0.575958653,0.593411946,0.610865238,0.628318531,0.645771823,0.663225116,0.680678408,0.698131701,0.715584993,0.733038286,0.750491578,0.767944871,0.785398163,0.802851456,0.820304748,0.837758041,0.855211333,0.872664626,0.959931089,1.047197551,1.134464014,1.221730476,1.308996939,1.396263402,1.483529864,1.570796327,1.658062789,1.745329252])
frisbee.cl = np.array([0.15942029,0.096618357,0.009661836,-0.077294686,-0.144927536,-0.217391304,-0.299516908,-0.357487923,-0.434782609,-0.492753623,-0.234509466,-0.204388985,-0.148450947,-0.126936317,-0.096815835,-0.083907057,-0.058089501,-0.027969019,0.023666093,0.075301205,0.118330465,0.182874355,0.238812392,0.303356282,0.376506024,0.432444062,0.496987952,0.55292599,0.613166954,0.673407917,0.729345955,0.776678141,0.836919105,0.875645439,0.922977625,0.983218589,1.021944923,1.07788296,1.129518072,1.172547332,1.219879518,1.275817556,1.318846816,1.396299484,1.44363167,1.486660929,1.512478485,1.589931153,1.620051635,1.667383821,1.688898451,1.710413081,1.740533563,1.762048193,1.813683305,1.850258176,1.886833046,1.929862306,1.972891566,2.007314974,2.063253012,2.093373494,2.106282272,2.136402754,2.151462995,2.149311532,1.146729776,1.133820998,1.133820998,1.09939759,1.082185886,1.019323671,0.8647343,0.724637681,0.589371981,0.45410628,0.299516908,0.140096618,0.004830918,-0.140096618,-0.280193237])
frisbee.cd = np.array([0.81906226,0.843658724,0.848578017,0.828900846,0.811683321,0.806764028,0.789546503,0.760030746,0.748962337,0.710837817,0.162962963,0.145679012,0.12345679,0.10617284,0.10617284,0.096296296,0.088888889,0.088888889,0.086419753,0.088888889,0.09382716,0.101234568,0.113580247,0.125925926,0.133333333,0.151851852,0.166666667,0.190123457,0.216049383,0.239506173,0.264197531,0.286419753,0.309876543,0.341975309,0.367901235,0.397530864,0.427160494,0.459259259,0.498765432,0.530864198,0.572839506,0.602469136,0.641975309,0.681481481,0.72345679,0.754320988,0.787654321,0.832098765,0.864197531,0.90617284,0.920987654,0.95308642,0.975308642,1.002469136,1.032098765,1.071604938,1.103703704,1.135802469,1.182716049,1.232098765,1.295061728,1.340740741,1.380246914,1.437037037,1.491358025,1.530864198,1.009876543,1.017283951,1.039506173,1.04691358,1.064197531,1.097002306,1.146195234,1.185549577,1.212605688,1.247040738,1.269177556,1.269177556,1.29377402,1.286395081,1.276556495])
frisbee.cm = np.array([0.031216649,0.013607257,0.000266809,-0.01547492,-0.031216649,-0.046691569,-0.060565635,-0.073372465,-0.085645678,-0.095517609,-0.038247863,-0.036538462,-0.030982906,-0.027564103,-0.023504274,-0.021581197,-0.017307692,-0.014102564,-0.010042735,-0.008333333,-0.006837607,-0.008119658,-0.00982906,-0.006837607,-0.006623932,-0.004059829,-0.002350427,-0.001068376,-0.001068376,0.000641026,0.003205128,0.007478632,0.010683761,0.013461538,0.016452991,0.021153846,0.025854701,0.031410256,0.034401709,0.04017094,0.043376068,0.048931624,0.053632479,0.061111111,0.068589744,0.07542735,0.083119658,0.088888889,0.096367521,0.103846154,0.111324786,0.115811966,0.125854701,0.137820513,0.144017094,0.152564103,0.160042735,0.171794872,0.180769231,0.18974359,0.2,0.208760684,0.216239316,0.223931624,0.227991453,0.22542735,0.019871795,0.018376068,0.018162393,0.018162393,0.016452991,0.016275347,0.017609392,0.021611526,0.021611526,0.020010672,0.015741729,0.008271078,0.002401281,-0.007203842,-0.011472785])
discList[0] = (frisbee.name, frisbee)

'''
Aerodynamic coefficients for aviar, roc and wraith sourced from Dynamics and Performance of Flying Discs, Kamaruddin (2011)
The curves have been extrapolated to cover a wider range of AoA. Extreme or unusual flight paths may not be as accurate
Normalized MMOI sourced from 3D CAD models of similar discs
'''
aviar = Disc("Aviar")
aviar.aoarange =  np.array([-1.570796327,-0.5235987756,-0.0872664626,-0.06981317008,-0.05235987756,-0.03490658504,-0.01745329252,0,0.01745329252,0.03490658504,0.05235987756,0.06981317008,0.0872664626,0.1047197551,0.1221730476,0.1396263402,0.1570796327,0.1745329252,0.1919862177,0.2094395102,0.2268928028,0.2443460953,0.2617993878,0.7853981634,0.872664626,1.570796327])
aviar.cl = np.array([0,-1,-0.088,-0.049,-0.009,0.034,0.093,0.154,0.21,0.256,0.304,0.343,0.383,0.426,0.468,0.508,0.549,0.591,0.631,0.672,0.702,0.74,0.78,1.6,0.8,0])
aviar.cd = np.array([0.4,0.188,0.076,0.071,0.07,0.072,0.072,0.084,0.088,0.085,0.102,0.117,0.133,0.141,0.157,0.174,0.189,0.203,0.216,0.226,0.245,0.266,0.281,0.7,0.5,0.6])
aviar.cm = np.array([0,-0.08,-0.015,-0.016,-0.011,-0.01,-0.013,-0.018,-0.018,-0.017,-0.014,-0.014,-0.011,-0.008,-0.005,0,0.005,0.009,0.011,0.02,0.024,0.032,0.039,0.23,0.02,0])
aviar.jxy = 4.23e-3
aviar.jz = 8.46e-3
aviar.diam = 0.21
discList.append((aviar.name, aviar))

roc = Disc("Roc")
roc.aoarange = np.array([-1.570796327,-0.5235987756,-0.0872664626,-0.06981317008,-0.05235987756,-0.03490658504,-0.01745329252,0,0.01745329252,0.03490658504,0.05235987756,0.06981317008,0.0872664626,0.1047197551,0.1221730476,0.1396263402,0.1570796327,0.1745329252,0.1919862177,0.2094395102,0.2268928028,0.2443460953,0.2617993878,0.7853981634,0.872664626,1.570796327])
roc.cl = np.array([0,-0.5,-0.121,-0.088,-0.061,-0.017,0.014,0.053,0.091,0.142,0.182,0.235,0.285,0.336,0.38,0.423,0.464,0.509,0.551,0.59,0.637,0.68,0.724,1.5,0.75,0])
roc.cd = np.array([0.4,0.25,0.065,0.054,0.054,0.058,0.063,0.067,0.076,0.076,0.086,0.095,0.105,0.109,0.124,0.13,0.143,0.159,0.17,0.184,0.198,0.215,0.233,0.7,0.5,0.6])
roc.cm = np.array([0,-0.1,-0.028,-0.024,-0.023,-0.018,-0.016,-0.015,-0.013,-0.013,-0.012,-0.011,-0.007,-0.003,-0.002,0.002,0.007,0.013,0.018,0.026,0.029,0.035,0.041,0.2,0.02,0])
roc.jxy = 4.06e-3
roc.jz = 8.15e-3
roc.diam = 0.21
discList.append((roc.name, roc))

wraith = Disc("Wraith")
wraith.aoarange = np.array([-1.570796327,-0.5235987756,-0.0872664626,-0.06981317008,-0.05235987756,-0.03490658504,-0.01745329252,0,0.01745329252,0.03490658504,0.05235987756,0.06981317008,0.0872664626,0.1047197551,0.1221730476,0.1396263402,0.1570796327,0.1745329252,0.1919862177,0.2094395102,0.2268928028,0.2443460953,0.2617993878,0.7853981634,0.872664626,1.570796327])
wraith.cl = np.array([0,-0.5,-0.034,0.007,0.045,0.076,0.108,0.15,0.179,0.212,0.25,0.286,0.323,0.357,0.405,0.463,0.507,0.547,0.591,0.637,0.68,0.73,0.775,2,1,0])
wraith.cd = np.array([0.2,0.2,0.057,0.051,0.053,0.051,0.052,0.056,0.06,0.062,0.073,0.075,0.084,0.089,0.102,0.118,0.127,0.141,0.153,0.167,0.185,0.199,0.22,0.7,0.5,0.6])
wraith.cm = np.array([0,-0.15,-0.059,-0.048,-0.04,-0.035,-0.026,-0.02,-0.017,-0.01,-0.004,0.001,0.009,0.018,0.02,0.024,0.031,0.038,0.05,0.057,0.064,0.071,0.081,0.24,0.02,0])
wraith.jxy = 3.86e-3
wraith.jz = 7.70e-3
wraith.diam = 0.21
discList.append((wraith.name, wraith))

"""## Define Throw Class"""

'''Define Throw class'''
class Throw:
  def __init__(self, name):
    self.name = name
    self.cl = [0]
    self.cd = [0]
    self.cm = [0]

  def getDistance(self):
    distance = (self.pos_g[-1,0]**2 + self.pos_g[-1,1]**2)**0.5
    return distance

"""## Simulation Function"""

def huckit(disc, throw):
  '''
  huckit receives a throw object and a disc object
  '''

  '''Simulation controls'''
  dt = 0.01 # Time step length in s
  step = 0 # Time step number
  maxSteps = 1000 # Maximum number of steps allowed for simulation
  vectorArraySize = [maxSteps+1, 3]
  scalarArraySize = maxSteps+1
  t = np.zeros(scalarArraySize) # Time in s

  '''Ground coordinate system'''
  pos_g = np.zeros(vectorArraySize) # Disc position in m
  vel_g = np.zeros(vectorArraySize) # Disc velocity in m/s
  acl_g = np.zeros(vectorArraySize) # Disc acceleration in m/s^2
  ori_g = np.zeros(vectorArraySize) # Disc roll, pitch, yaw angle in rad
  rot_g = np.zeros(vectorArraySize) # Disc roll, pitch, yaw rate in rad/s

  '''Disc coordinate system'''
  acl_d = np.zeros(vectorArraySize)
  vel_d = np.zeros(vectorArraySize)
  rot_d = np.zeros(vectorArraySize)

  '''Side-slip coordinate system'''
  acl_s = np.zeros(vectorArraySize)
  vel_s = np.zeros(vectorArraySize)
  rot_s = np.zeros(vectorArraySize)
  beta = np.zeros(scalarArraySize)

  '''Wind coordinate system'''
  acl_w = np.zeros(vectorArraySize)
  vel_w = np.zeros(vectorArraySize)
  alpha = np.zeros(scalarArraySize)

  '''Aerodynamic forces'''
  drag = np.zeros(scalarArraySize)
  lift = np.zeros(scalarArraySize)
  mom = np.zeros(scalarArraySize)

  '''Define disc orientation and velocity from inputs'''
  ori_g[step] = np.array([throw.roll_angle, throw.nose_angle, 0])
  vel_g[step] = np.array([throw.speed*np.cos(throw.launch_angle), 0, throw.speed*np.sin(throw.launch_angle)])
  launch_angle_d = np.matmul(T_gd(ori_g[step]), [0, throw.launch_angle, 0])
  ori_g[step] += launch_angle_d

  '''Define environmental constants'''
  rho = 1.18 # Air density in kg/m^3
  g = 9.81 # Gravitational acceleration in m/s^2
  throw.launch_height = 1.5 # Initial height of throw in m
  pos_g[step] = np.array([[0, 0, throw.launch_height]])

  '''Define derived constants'''
  mass = disc.mass
  diam = disc.diam # Diameter of disc in m
  ixy = disc.jxy*mass # Rotational moment of inertia of disc about roll axis in kg-m^2
  iz = disc.jz*mass # Rotational moment of inertia of disc about spin axis in kg-m^2
  area = np.pi*(0.5*diam)**2 # Planform area of disc in m^2
  omega = throw.spin*throw.spindir # Assign spin direction
  weight = g*mass # Gravitational force acting on the disc center of mass in N

  '''Loop until disc hits the ground, z-position=0'''
  while pos_g[step][2] > 0:
    if step >= maxSteps: # Safety valve in case the disc never returns to earth
      break

    ii=0
    while 1:
      '''Transform ground velocity to wind coordinate system'''
      vel_d[step] = np.matmul(T_gd(ori_g[step]), vel_g[step]) # Transform ground velocity to disc coordinate system
      beta[step] = -np.arctan2(vel_d[step][1], vel_d[step][0]) # Calculate side slip angle
      vel_s[step] = np.matmul(T_ds(beta[step]), vel_d[step]) # Transform velocity to zero side-slip coordinate system
      alpha[step] = -np.arctan2(vel_s[step][2], vel_s[step][0]) # Calculate the angle of attack
      vel_w[step] = np.matmul(T_sw(alpha[step]), vel_s[step]) # Transform velocity to wind coordinate system where aerodynamic calculations can be made

      '''Transform gravity loads to wind coordinate system'''
      grav_d = np.matmul(T_gd(ori_g[step]), [0, 0, -weight])
      grav_s = np.matmul(T_ds(beta[step]), grav_d)
      grav_w = np.matmul(T_sw(alpha[step]), grav_s)

      '''Calculate aerodynamic forces on the disc'''
      drag[step] = 0.5*rho*(vel_w[step][0]**2)*area*disc.getCd(alpha[step]) # Calculate drag force in N
      lift[step] = 0.5*rho*vel_w[step][0]**2*area*disc.getCl(alpha[step]) # Calculate lift force in N
      mom[step] = 0.5*rho*vel_w[step][0]**2*area*diam*disc.getCm(alpha[step]) # Calculate pitching moment in N-m

      '''Calculate body accelerations from second law and force balances'''
      acl_w[step,0] = (-drag[step] + grav_w[0]) / mass # Calculate deceleration due to drag
      acl_w[step,2] = (lift[step] + grav_w[2]) / mass # Calculate acceleration due to lift
      acl_w[step,1] = grav_w[1] / mass # Calculate acceleration due to side loading (just gravity)
      rot_s[step,0] = -mom[step]/(omega*(ixy - iz)) # Calculate roll rate from pitching moment

      '''Tranform disc acceleration to ground coordinate system'''
      acl_s[step] = np.matmul(T_ws(alpha[step]), acl_w[step])
      acl_d[step] = np.matmul(T_sd(beta[step]), acl_s[step])
      acl_g[step] = np.matmul(T_dg(ori_g[step]), acl_d[step])

      '''Transform roll rate from zero side-slip to ground coordinate system'''
      rot_d[step] = np.matmul(T_sd(beta[step]), rot_s[step])
      rot_g[step] = np.matmul(T_dg(ori_g[step]), rot_d[step])

      '''Perform one inner iteration to refine speed and position vectors'''
      if step==0: # Do not run inner iterations for initial time step
        break
      if ii>=1: # Only run one inner iteration
        break

      '''Calculate average accelerations and rotation rates between current and previous time steps'''
      avg_acl_g = (acl_g[step-1] + acl_g[step])/2
      avg_rot_g = (rot_g[step-1] + rot_g[step])/2

      '''Calculate new velocity, position and orientation for current time step'''
      vel_g[step] = vel_g[step-1] + avg_acl_g*dt
      pos_g[step] = pos_g[step-1] + vel_g[step-1]*dt + 0.5*avg_acl_g*dt**2
      ori_g[step] = ori_g[step-1] + avg_rot_g*dt

      ii+=1

    '''Estimate disc velocity, position, and orientation at next time step'''
    vel_g[step+1] = vel_g[step] + acl_g[step]*dt
    pos_g[step+1] = pos_g[step] + vel_g[step]*dt + 0.5*acl_g[step]*dt**2
    ori_g[step+1] = ori_g[step] + rot_g[step]*dt

    '''Update simulation variables'''
    t[step+1] = t[step] + dt
    step += 1

  '''Remove unused steps from simulation data arrays and assign them to throw object'''
  throw.t = np.resize(t,step)
  throw.lift = np.resize(lift,step)
  throw.drag = np.resize(drag,step)
  throw.mom = np.resize(mom,step)
  throw.alpha = np.resize(alpha,step)
  throw.beta = np.resize(beta,step)
  throw.pos_g = np.resize(pos_g,[step,3])
  throw.vel_g = np.resize(vel_g,[step,3])
  throw.acl_g = np.resize(acl_g,[step,3])
  throw.ori_g = np.resize(ori_g,[step,3])
  throw.rot_g = np.resize(rot_g,[step,3])
  throw.vel_d = np.resize(vel_d,[step,3])
  throw.vel_w = np.resize(vel_w,[step,3])
  throw.acl_w = np.resize(acl_w,[step,3])
  throw.rot_s = np.resize(rot_s,[step,3])

  return throw

"""# Custom UI

"""

import matplotlib.pyplot as plt
import ipywidgets as widgets

fig1Output = widgets.Output()


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

def plotfield(ax):
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
    """
    Plots a small 'X' marker at the specified (x_center, y_center) on the ground (z=0).

    Parameters:
    - ax: The 3D axis to plot the X marker on.
    - x_center: The x-coordinate for the center of the X.
    - y_center: The y-coordinate for the center of the X.
    - x_length: The length of the arms of the X (default is 2 yards).
    - color: The color of the X (default is red).
    """
    # print(x_center, y_center)
    # Coordinates for the two lines making up the "X"
    ax.scatter(x_center, y_center, 0, color='red', s=30, marker='x')

from ipywidgets import interact, widgets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@interact(
    speed=(5, 60, 0.5),
    spin=(0, 120, 5),
    launch_angle=(0, 45, 1),
    nose_angle=(0, 45, 1),
    roll_angle=(-45, 45, 1),
    translation_angle=(-180, 180, 1)  # New slider for translation angle
)
def update_throw(speed=35, spin=60, launch_angle=7, nose_angle=7, roll_angle=36, translation_angle=-24):
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

    plt.figure(figsize=(20,8))
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
    # max_height = np.max(z)
    # ax.set_zlim3d(0, max_height * 1.2)  # Add 20% padding above max height

    # ax.set_box_aspect([FIELD_LENGTH/FIELD_LENGTH,
    #                   FIELD_WIDTH/FIELD_LENGTH,
    #                   max_height/FIELD_LENGTH])

    ax.xaxis.set_pane_color((0.3, 0.5, 0.8, 0.15))
    ax.yaxis.set_pane_color((0.3, 0.5, 0.8, 0.15))
    ax.zaxis.set_pane_color((0.3, 0.8, 0.3, 0.15))

    ax.view_init(elev=30, azim=-90)
    # plt.show()