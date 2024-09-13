"""Utility functions used by either model or codes using functions in model
Contains:
smallest_positive_quadroot
wrap_angle
plot_distance_func
plot_current_positions
plot_trajectories
distances_to_point
weight_function
local_speed_current
area_occupied
occupancy
current_average_speed
average_speed
"""

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------#
#So far no speed gained by jit-ing this function
# @jit(double(double,double,double))
def smallest_positive_quadroot(a2,b,d):
    """ Significantly aster than numpy.roots for this case
    Returns the smallest positive root of the quadratic equation a*x^2 + b*x + c = 0.
    If there is no positive root then -1 is returned
    a2 = 2*a, d = b^2-4*a*c"""


    b2a = -b/a2
    if d < 0:
        return -1
    if d == 0:
        root = b2a
        if root > 0:
            return root
        else:
            return -1
    else:
        # sqrt_d2a = math.sqrt(d)/a
        # root1 = b2a + sqrt_d2a
        # root2 = b2a - sqrt_d2a

        sqrt_d = np.sqrt(d)
        root1 = (-b + sqrt_d) / (a2)
        root2 = (-b - sqrt_d) / (a2)

        if root1 > 0:
            if root2 > 0:
                if root1 < root2:
                    return root1
                else:
                    return root2
            return root1
        elif root2 > 0:
            return root2
        else:
            return -1
#------------------------------------------------------------------------------#
def wrap_angle(angle):
    """ Wraps values of angles to be in [-pi,pi]
    Can take an array as an arguement as well """
    return ( ( angle + np.pi) % (2. * np.pi ) - np.pi )
#------------------------------------------------------------------------------#

def plot_f_alpha(i, alphas, f_alphas):
    """Plot the function f(alpha) over values of alpha for i"""
    plt.figure()
    plt.plot(alphas,f_alphas)
    plt.title("f(alpha) for i = %d" %(i))
#------------------------------------------------------------------------------#

def plot_distance_func(i, alphas, distances):
    """Plot the distance over values of alpha for i"""
    plt.figure()
    plt.title("Distance for i = %d" %(i))
    plt.plot(alphas,distances,'x')
    plt.figure()
#------------------------------------------------------------------------------#
def plot_current_positions(x,r,t,fig_name, colors = None):
    """ Plots current positions of persons """
    #If colors are not inputted for the persons then blue is chosen by default
    n = r.size
    if colors is None: colors = ['blue'] * n
    for i in range(n):
        circle = plt.Circle( (x[0][i], x[1][i]), r[1], color = colors[i])
        fig_name.gca().add_artist(circle)
    plt.title('Time = %.3f' %(t))
#------------------------------------------------------------------------------#
def plot_trajectories(x_full):
    """ Plots trajectories taken of persons """
    n = x_full.shape[1]
    for i in range(n):
        plt.plot(x_full[0,i,:],x_full[1,i,:],'k')
#------------------------------------------------------------------------------#
def distances_to_point(x,xp,yp):
    """Computes the distances from a point (xp,yp) to all the pedestrians"""
    x_vals = x[0,:] - xp
    y_vals = x[1,:] - yp
    return np.linalg.norm( [x_vals,y_vals], axis = 0)
#------------------------------------------------------------------------------#
def weight_function(dist):
    """Computes the distance based weight function for an inputted value/array"""

    R = 0.7
    return np.exp( - np.multiply( dist, dist) / (R * R)) / ( np.pi * R * R)
#------------------------------------------------------------------------------#
def local_speed_current(x,v,point):
    """Computes the local speed at point x in the current iteraion of the model"""

    dist = distances_to_point(x,point[0],point[1])
    wf = weight_function( dist)
    sum_wf = np.sum(wf)
    if sum_wf == 0:
        return 0

    return np.sum( np.multiply( np.linalg.norm( v,axis = 0), wf)) / np.sum( wf)
#------------------------------------------------------------------------------#
def area_occupied(r):
    """Computes total area the pedestrians occupy - not accounting for overlaps"""

    # Sum over i of [ pi * r_i * r_i ]
    return np.pi * np.sum( np.multiply( r, r))
#------------------------------------------------------------------------------#
def occupancy(r,bounding_area):
    """Computes occupancy given bounding area - not accounting for overlaps"""

    return area_occupied(r) / bounding_area
#------------------------------------------------------------------------------#
def occupancy_accurate(rad,center_coor,bounding_area):
    """Computes occupancy given bounding area - accounting for overlaps"""
    from collections import namedtuple
    Circle = namedtuple("Circle", "x y r")

    circles = []
    for i in range(len(rad)):
        circles.append(Circle(center_coor[0,i],center_coor[1,i],rad[i]))

    x_min = -4.
    x_max = 4.
    y_min = -1.5
    y_max = 1.5
    box_side = 500  

    dx = (x_max - x_min) / box_side
    dy = (y_max - y_min) / box_side
    count = 0
    totalcount = 0

    for r in range(box_side):
        y = y_min + r * dy
        for c in range(box_side):
            x = x_min + c * dx
            totalcount += 1
            if any((x-circle.x)**2 + (y-circle.y)**2 <= (circle.r ** 2)
                   for circle in circles):
                count += 1 


    area = count * dx * dy 
    print(area/ bounding_area)



    return area / bounding_area
#------------------------------------------------------------------------------#
def current_average_speed(v):
    """Computes the average speed in the current iteration of the model """

    return np.average( np.linalg.norm( v, axis = 0))
#------------------------------------------------------------------------------#


def average_speed(v_full,x = None):
    """Computes the average speed in the current iteration of the model """
    return np.average( np.linalg.norm( v_full[:,:,int(v_full.shape[2]/3):], axis = 0))


#------------------------------------------------------------------------------#
def average_x_speed(v_full,x = None):
    """Comvputes the average x speed in the current iteration of the model """
    return np.average( np.abs( v_full[0,:,int(v_full.shape[2]/3):]))
   # return np.average( np.linalg.norm( v_full[:,:,500:], axis = 0))


#------------------------------------------------------------------------------#
def animate_xfull(x_full):
    """ Display sequence of pedestrian positions
    """

    xmin = x_full[0].min()
    xmax = x_full[0].max()
    ymin = x_full[1].min()
    ymax = x_full[1].max()

    fig,ax = plt.subplots()
    for i in range(x_full.shape[-1]):
        ax.plot(x_full[0,:,i],x_full[1,:,i],'o')
        ax.axis([xmin,xmax,ymin,ymax])
        plt.title(i)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pause(0.05)
        ax.clear()


    return None
