import numpy as np
import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
import ped_utils as putils

import time
import math
from matplotlib import cm
import scipy.spatial.distance as scidist
from joblib import Parallel, delayed
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = r'E:\\ffmpeg\\ffmpeg.exe'
from celluloid import Camera
import multiprocessing
num_cores = multiprocessing.cpu_count()
from pf1b import ped_funcs as pf 


def initialize_global_parameters():
    """ To intialize global parameters that are dependent on initial conditions or default settings preffered """
    global variablesready
    if variablesready:
        #Makes sure the scopes of the variables are global to the module
        global alpha_0, x_full, gap, H, alpha_current, alpha_des, f_alpha_des, v_des, contact_p, contact_w, n_walls
        global r, mass, v, v_0, v_full, rsum, d_h

        #angle to destination
        alpha_0 = np.arctan2((o[1]-x[1]),(o[0]-x[0]))
        #Initalize the array that stores movement values over time
        x_full = np.copy(x)
        gap = np.zeros((n,n))
        #Field of Vision for each of the pedestrians
        #H = np.random.uniform(H_min,H_max,n)
        H = H_min*np.ones(n)
        #set initial alpha_direction to alpha_0
        alpha_current = np.copy(alpha_0)
        alpha_des = np.zeros(n)
        f_alpha_des = np.zeros(n)
        #Array to store v_des
        v_des = np.zeros(n)

        d_h = np.zeros(n)

        #Array to store speed
        pf.pf_speed = np.zeros(n)

        #Store information about persons in contact with people and walls

        if n_walls is None:
            n_walls = 0
        contact_p = np.zeros((n,n))
        contact_w = np.zeros((n,n_walls))

        if n_walls>0:
            pf.pf_walls = walls
            pf.pf_contact_w = contact_w

        if np.shape(mass) != (n,):
            mass = np.random.uniform(60,100,n)
        #Radius, r = mass/320
        r = mass/320
        rsum = np.add.outer(r,r) #ri+rj
        #If starting starting velocities are not specified then its assumed that they are zero for all people
        if np.shape(v) != (2,n):
            v = np.zeros((2,n))
        v_full = np.copy(v)

        if np.shape(v_0) != (n,):
            v_0 = 1.3*np.ones(n)

        #For clarity
        variablesinitialized = True
        if instructions: print ("%d cores in use" %(num_cores))
    else:
        if instructions: print ("Not all required variables initialized and checked. To not avoid checking manually configure variablesready to True")
#------------------------------------------------------------------------------#
def check_model_ready():
    """ Make sure all neccessary parameters for the model are initalized properly and allows user to call initialize_global_parameters() """

    global variablesready

    if n is None:
        if instructions: print("value of n not given")
    else:
        variablesready = True

        if x is None or np.shape(x) != (2,n):
            if instructions: print("position values array, x, not initalized or not in the right shape (2xn)")
            variablesready = False

        if o is None or np.shape(o) != (2,n):
            if instructions: print("destination values array, o, not initalized or not in the right shape (2xn)")
            variablesready = False

        if mass is None or np.shape(mass) != (n,):
            if instructions: print("mass array not initialized or not with correct shape (n). It will be initailized with default values when initalizing global parameters - randomly uniform values between 60 and 100")

        if v_0 is None or np.shape(v_0) != (n,):
            if instructions: print("comfortable walking speed array, v_0, not initialized or not with correct shape (n). It will be initailized with default values of 1.3m/s when initalizing global parameters")

        if v is None or np.shape(v) != (2,n):
            if instructions: print("initial velocity array, v, not initialized or not with correct shape (2xn). It will be initailized with default values of zeros when initalizing global parameters")

        if n_walls is None:
            if instructions: print("number of walls, n_walls, not initalized. It will be assumed to be 0 when initalizing global parameters")
        else:
            if walls is None or np.shape(walls) != (7,n_walls):
                if instructions: print("numbers of walls initalized but array to store information about the walls not initialized or not with correct shape (5xn)")
                variablesready = False

    if variablesready:
        if instructions: print("All necessary variables have been initalized. Call initialize_global_parameters() to initaize dependent parameters")
    else:
        if instructions: print("Model is not ready. Please initialize required parameters")
#------------------------------------------------------------------------------#
def print_model_parameters():
    print ("tau = %4.2f, angular resolution in degrees = %4.2f, d_max = %4.2f, k = %4.2e, t = %4.2f" %( tau, math.degrees(ar), d_max, k, t ) )
#------------------------------------------------------------------------------#
def reset_model():
    """ Resets all inital conditions and sets model parameters to their default values """

    global variablesready, tau, ar, d_max, k, t, H_min, H_max, instructions, n, x, o, mass, v_0, v, n_walls, walls, color_p

    #Parameters of the model
    variablesready = False
    tau = 0.5 #second heurostic constant
    ar = math.radians(0.5) #angular resolution
    d_max = 10. #Horizon distance
    k = 5e3 #body collision constant
    t = 0 #Initial time set to 0
    H_min = math.radians(75)
    H_max = math.radians(75)
    instructions = False
    time_step = 0.05

    #Neccessary variables that that need to be initalized properly
    n = None #integer
    x = None #array of size 2xn
    o = None #array of size 2xn
    #Optional - default values initialized if not done so manually in func above
    mass = None #array of size n
    v_0 = None #array of size n
    v = None #array of size 2xn
    n_walls = None #integer
    walls = None #array of size 5xn - a,b,c,startwal, endwal
    #Optional - Not initalized if not specified as it has limited use
    color_p = None

    pf.pf_speed = None

#------------------------------------------------------------------------------#
#To profile speed of the code line by line: @profile
def compute_alpha_des_par(i, params):
    """ Compute the minimum distance function to find alpha_des over
    the horizon of alpha values"""

    rsum_i,dx_i,dy_i,quad_C2_i,in_field_i = params

    alpha_out,f_alpha_out,d_h_i  = \
    pf.compute_alpha_des(n,n_walls,i,rsum_i,dx_i,dy_i,quad_C2_i,in_field_i, \
    x[:,i],v,o[:,i],gap[i,:],d_max,v_0[i], \
    r[i],alpha_0[i],ar,alpha_current[i],H[i])

    if n_walls>0: contact_w[i,:] = pf.pf_contact_w[i,:]
    d_h[i] = min(d_h[i],d_h_i)
    #Format and return output
    result = [alpha_out, f_alpha_out, contact_w[i,:]]
    return result

#------------------------------------------------------------------------------#
def compute_bodycollision_acceleration(i):
    """ Returns acceleration in [x,y] directions caused by body collisions
        for person i current positions of persons """

    axt = 0
    ayt = 0
    ri = r[i]
    #Collisions due to persons
    for j in range(n):
        if contact_p[i][j] != 0:
            kg = k * (ri + r[j] - contact_p[i][j])
            nx = x[0][i] - x[0][j]
            ny = x[1][i] - x[1][j]
            size_n = math.hypot(nx,ny)
            nx = nx / size_n
            ny = ny / size_n
            fx = kg * nx
            fy = kg * ny
            ax = fx / mass[i]
            ay = fy / mass[i]
            axt = axt + ax
            ayt = ayt + ay
    #Collisions due to walls
    for w in range(n_walls):
        if contact_w[i][w] != 0:
            kg = k * (ri - contact_w[i][w])
            #find normal direction to wall
            wall = walls[:,w]
            a = wall[0]
            b = wall[1]
            c = wall[2]
            wall_start = wall[3] - ri
            wall_end = wall[4] + ri
            nx,ny = wall[-2],wall[-1]

            fx = kg * nx
            fy = kg * ny
            ax = fx / mass[i]
            ay = fy / mass[i]
            axt = axt + ax
            ayt = ayt + ay

    return [axt, ayt]

#------------------------------------------------------------------------------#
def compute_destinations_par():
    """ Calculates v_des, f_alpha_des, and alpha_des values for all persons"""
    global alpha_des, f_alpha_des, v_des, gap, contact_p,contact_w,in_field,d_h

    d_h = tau*v_0
    gap = scidist.squareform(scidist.pdist(x.T)) #distance between i and j
    contact_p = np.zeros_like(gap)

    contact_p[gap<rsum] = gap[gap<rsum]

    dx,dy = np.subtract.outer(x[0],x[0]),np.subtract.outer(x[1],x[1])
    quad_C2 = 2*(dx**2 + dy**2 - rsum**2) #used in smallest_positive_quadroot
    b_dir = np.arctan2(dy,dx) #used to compute in_field below
    phi1 = putils.wrap_angle(alpha_current-H_min)
    phi2 = putils.wrap_angle(alpha_current+H_min)
    in_field=[]
    del_phi = np.abs(b_dir-alpha_current)
    for i in range(n):
        ifield = np.where((del_phi[:,i]<H_min) | ((2*np.pi-del_phi[:,i])<H_min))
        in_field.append(ifield[0])
    out = Parallel(n_jobs=num_cores)(delayed(compute_alpha_des_par)(i,[rsum[i,:],dx[i,:],dy[i,:],quad_C2[i,:],in_field[i]]) for i in range(n))

    for i in range(n):
        alpha_des[i],f_alpha_des[i],contact_w[i,:] = out[i]
    v_des = np.min(np.vstack([v_0,f_alpha_des/tau]),axis=0)
#------------------------------------------------------------------------------#
def compute_destinations(flag_periodic=False,L_periodic=0):
    """ Calculates v_des, f_alpha_des, and alpha_des values for all persons"""
    global alpha_des, f_alpha_des, v_des, gap, contact_p,contact_w,in_field
    global x,v,rsum,alpha_current,d_h
    d_h = tau*v_0
    if flag_periodic:
        x = np.hstack((x,x))
        x[0,n:] += L_periodic
        rsum = np.hstack((rsum,rsum))
        v = np.hstack((v,v))
        alpha_current = np.hstack((alpha_current,alpha_current))
        n_cad = 2*n

    else:
        n_cad = n

    gap = scidist.squareform(scidist.pdist(x.T)) #distance between i and j
    gap = gap[:n,:]
    contact_p = np.zeros_like(gap)
    contact_p[gap<rsum] = gap[gap<rsum]

    dx,dy = np.subtract.outer(x[0],x[0]),np.subtract.outer(x[1],x[1])
    if flag_periodic:
        dx,dy = dx[:,:n].T,dy[:,:n].T
        dxm,dym = -dx,-dy
    else:
        dxm,dym = dx,dy
    quad_C2 = 2*(dx**2 + dy**2 - rsum**2) #used in smallest_positive_quadroot
    b_dir = np.arctan2(dy,dx) #used to compute in_field below
    if not flag_periodic: b_dir=b_dir.T

    in_field=[]
    del_phi = np.abs(b_dir-alpha_current)
    for i in range(n):
        ifield = np.where((del_phi[i,:]<H_min) | ((2*np.pi-del_phi[i,:])<H_min))
        in_field.append(ifield[0])
        alpha_des[i],f_alpha_des[i],d_h_i = \
        pf.compute_alpha_des(n_cad,n_walls,i,rsum[i,:],dxm[i,:],dym[i,:],quad_C2[i,:],in_field[i], \
        x[:,i],v,o[:,i],gap[i,:],d_max,v_0[i], \
        r[i],alpha_0[i],ar,alpha_current[i],H[i])
        d_h[i] = min(d_h[i],d_h_i)
    if n_walls>0: contact_w = pf.pf_contact_w
    v_des = np.min(np.vstack([v_0,d_h/(1*tau)]),axis=0)
    if flag_periodic:
        x = x[:,:n]
        v = v[:,:n]
        rsum = rsum[:n,:n]
        alpha_current = alpha_current[:n]
#------------------------------------------------------------------------------#
def move_pedestrians():
    """ Moves all pedestrians forward in time by time_step based on calculated v_des and alpha_des values"""
    #Global values being saved
    global v, x

    #acceleration due to body collisions - needs to computed before moving the pedestrians to ensure both people colliding feel the force
    abcx = np.zeros(n)
    abcy = np.zeros(n)
    for i  in range(n):
        [abcx[i],abcy[i]] = compute_bodycollision_acceleration(i)

    #acceleration
    a = np.zeros_like(v)
    a[0,:] = (np.cos(alpha_des)*v_des-v[0,:])/tau + abcx
    a[1,:] = (np.sin(alpha_des)*v_des-v[1,:])/tau + abcy

    #update
    v = v + a * time_step
    x = x + v * time_step
#------------------------------------------------------------------------------#
def update_model():
    """ Once alpha_des, v_des have been calculated and pedestrians have moved forward in time to ready the model for the next iteration"""
    global alpha_0, alpha_current, x_full, v_full, t
    #update alpha_0 values
    alpha_0 = np.arctan2((o[1]-x[1]),(o[0]-x[0]))
    alpha_current = np.arctan2(v[1,:],v[0,:])
    ind = v[0,:]==0
    ind[v[1,:]!=0]=False
    alpha_current[ind]=alpha_0[ind]
    #save information about positions of each individual
    x_full = np.dstack((x_full,x))
    v_full = np.dstack((v_full,v))
    #increment time
    t = t + time_step
#------------------------------------------------------------------------------#
def advance_model():
    """Advances current model in time by time_step"""

    compute_destinations()
    move_pedestrians()
    update_model()
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#border0.5,k1000,interval50,nomean


reset_model()
#n_vals = [5,12,25,36,49,64,73,84,97]
#n_vals = [12,45,60,75,90,120,140]
n_vals = [12,18,24,30,36,45,60,75,90,105,120,135]            #setting the number of people
#n_vals = [45,60,75,90,105,115]
#n_vals = [5,15,30,60,100]
total_time = 90     
dis_interval = 10                                      #setting the simulation time
interval = 50                                              # how many intervals per second
direction = 'opposite'
#n_vals=[2]
border= 0.5
random = True
display = True
interp_index = 1 
if not random:
    np.random.seed(100)
t_iteration_vals = [None] * len(n_vals)
t_vals = [None] * len(n_vals)
occupancy_vals = [None] * len(n_vals)
avgspeed_vals = [None] * len(n_vals)
avgxspeed_vals = [None] * len(n_vals)

flag_par = False
num_cores = 2 #Only used if flag_par = True, adjust as needed
flag_periodic = True
L_periodic = 8


# IMPORT DATA
data = np.genfromtxt('D:\\model20-master\\old\\data\\data_pub_occupancy.csv', delimiter=',')
#-----------------------------storage for speed, X coordinate, Y coordinate, time-------------------------------------------------#

local_speed = np.zeros((total_time*interval+1,8*dis_interval+2,3*dis_interval+2))
Xrange = np.linspace(-4,4,8*dis_interval+2)
Yrange = np.linspace(-1.5,1.5,3*dis_interval+2)
Trange = np.linspace(0,total_time,total_time*interval+1)


T_vals = [3,5,7]
k_vals = [200.0,1000.0]
#k_vals = [10.0,100.0,1000.0,10000.0,100000.0]
xshift=2
T_legends=["T=3","T=5","T=7"]


correlation_vals = np.zeros((len(n_vals),len(T_vals)))

for index_k in range(len(k_vals)):
    for index in range(len(n_vals)):
        #Set number of pedestrians
        n = n_vals[index]
        all_positions = np.zeros( ( Trange.shape[0],2, n))

        #Parameters
        time_step = 1/interval
        d_max = 8
        H_min = np.pi/4
        H_max = np.pi/4
        v_0 = np.random.normal( 1.3, 0.2, n)  # comfortable speed v_0
        # v_0 = np.ones(n) * 1.3
        k = (k_vals[index_k])

        #Current position uniformly distributed
        x = np.zeros( ( 2, n))
        x[0,:] = np.linspace(-4,4,n+2)[1:-1]
        x[1][::] = np.random.uniform( -1.25, 1.25, n)  

        #Destination points, o, set to (50,0) for all

        o = np.zeros( ( 2, n))
        o[0,:] = 100.         #pedestrain target

        if direction == 'opposite':
            x[0,:] = np.linspace(-4,4,n+2)[1:-1]
            x[1,::2] = np.random.uniform( -0.6, 0.6, (x[1,::2]).shape[0])  +0.75
            x[1,1::2] = np.random.uniform( -0.6, 0.6, (x[1,1::2]).shape[0]  )  -0.75
            o[0,::2] = 100.
            o[0,1::2] = -100.
            o[1,::2] = 0.75
            o[1,1::2] = -0.75
        #Initialize the walls [a,b,c,startval,endval]
        n_walls = 2
        walls = np.zeros( (7, n_walls))
        # wall y = -1.5
        walls[:,0] = np.array([ 0, 1, 1.5, -4, 4, 0, 1])
        # wall y = 1.5
        walls[:,1] = np.array([ 0, 1, -1.5, -4, 4, 0, -1])

        check_model_ready()



        initialize_global_parameters()

        #------------------------------------------------------------------------------#
        #Increment the time
        start_time = time.time()
        j=0
        while (t<total_time-time_step*0.5):
            if t%1<=0.01: print("t=%f" %(t))
            #print ("t = %.2f" %(t))
            #compute alpha_des and v_des for each i
            j+=1
        # print(j)
            if flag_par:
                compute_destinations_par()
            else:
                compute_destinations(flag_periodic,L_periodic)

            move_pedestrians()
            for i in range(n):
                    if x[0,i] > 4:
                        x[0,i] = x[0,i] - 8
                    if x[0,i] < -4:
                        x[0,i] = x[0,i] + 8
                    if x[1,i] > 1.5:
                        x[1,i] = 1.4
                    if x[1,i] < -1.5:
                        x[1,i] = -1.4
            #Update alpha_0 and alpha_current
            update_model()
            if t == time_step:
                t_iteration_vals[index] = time.time() - start_time
                print( "Time for 1 iteration: %.3f" %(t_iteration_vals[index]) )
            if ((j-1) % 100) ==0: print("i,t=",j,t)
            for m0 in range(8*dis_interval+2):
                for n0 in range(3*dis_interval+2):
                # print(putils.local_speed_current(x,v,[Xrange[m],Yrange[n]]))
                    local_speed[j,m0,n0] = putils.local_speed_current(x,v,[Xrange[m0],Yrange[n0]])
            all_positions[j-1,:,:] =x
        end_time = time.time()
        t_vals[index] = end_time - start_time
        print( "Time Taken: %.3f" %(t_vals[index]) )



        #------------------------------------------------------------------------------#

        bounding_area = 3. * 8.
        occupancy_vals[index] = putils.occupancy_accurate(r,all_positions[-2,:,:],bounding_area)
        print("Occupancy =", occupancy_vals[index])
        avgspeed_vals[index] = np.abs(putils.average_speed(v_full,all_positions))
        avgxspeed_vals[index] = np.abs(putils.average_x_speed(v_full,all_positions))
        print("Average Speed = ", avgspeed_vals[index])
        print("Average x Speed = ", avgxspeed_vals[index])
        print("Comfortable Walking Speeds =", v_0)

        for k in range(len(T_vals)):

            data1 = np.average((local_speed[(20)*interval:  -(T_vals[k])*interval  ,int((border+2)*dis_interval):-int(border*dis_interval),int(border*dis_interval):int(-dis_interval*border)]),axis=2).flatten()
            data2 = np.average(local_speed[(20+T_vals[k])*interval:,int(border*dis_interval):int(-2*dis_interval-border*dis_interval),int(border*dis_interval):int(-dis_interval*border)],axis = 2).flatten()   

           # data1 = (local_speed[(20)*interval:  -(T_vals[k])*interval  ,int((border+xshift)*dis_interval):-int(border*dis_interval),int(border*dis_interval):int(-dis_interval*border)]).flatten()
           # data2 = (local_speed[(20+T_vals[k])*interval:,int(border*dis_interval):int(-xshift*dis_interval-border*dis_interval),int(border*dis_interval):int(-dis_interval*border)]).flatten()           
            cormatrix = np.corrcoef(data1,data2)
            #print((local_speed[(10+T_vals[k])*interval:,:-2*10-border*10,:]).shape)
            #print((local_speed[:-(10+T_vals[k])*interval,border*10+2*10:,:]).shape)
            correlation_vals[index,k] = cormatrix[0,1]

        fig, ax = plt.subplots()
        pl = ax.pcolormesh( Xrange,Trange, np.average(local_speed[:,:,:],axis=2), vmin=0.0, vmax=1., cmap=plt.cm.jet.reversed())
        plt.gca().invert_yaxis()
        fig.colorbar(pl)
        plt.title('Local Speed')
        plt.xlabel('X')
        plt.ylabel('T')

        plt.savefig('D:\\model20-master\\old\\results\\'+str(int(k_vals[index_k]))+'speed_heatmap'+str(n_vals[index])+'.png')
        plt.close()

        print(v_full.shape)
        fig = plt.figure()
        plot_singlespeed= plt.plot( Trange, np.linalg.norm(v_full[:,0,:],axis = 0), 'r', label = 'pedestrain')
        plt.legend(loc=1)
        plt.title('Single Person Speed')
        plt.ylim(0, 1.5)
        plt.xlabel('Time')
        plt.ylabel('Speed (m/s)')

        plt.savefig('D:\\model20-master\\old\\results\\'+str(int(k_vals[index_k]))+'speed_ped'+str(n_vals[index])+'.png')
        plt.close()

        fig = plt.figure()
        plot_singlespeed= plt.plot(all_positions[:-1,0,0],all_positions[:-1,1,0], 'r', label = 'pedestrain',linewidth = 0.5)
        plt.legend(loc=1)
        plt.title('Single Person Trajectory')
        plt.ylim(-1.5, 1.5)
        plt.xlim(-4,4)
        plt.xlabel('X coordinate')
        plt.ylabel('Y Coordinate')

        plt.savefig('D:\\model20-master\\old\\results\\'+str(int(k_vals[index_k]))+'trajectory_'+str(n_vals[index])+'.png')
        plt.close()



        if display:
            fig = plt.figure()
            plot_pub = plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
            plot_results = plt.plot(occupancy_vals,avgspeed_vals,'x-', label = 'My Results')
            plt.legend(loc=1)
            plt.title('Average Speed')
            plt.xlabel('Occupancy')
            plt.ylabel('Avg Speed (m/s)')
            # plt.show()
            plt.savefig('D:\\model20-master\\old\\results\\'+str(int(k_vals[index_k]))+'avg.png')
            plt.close()



            fig = plt.figure()
            plot_pub = plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
            plot_xresults = plt.plot(occupancy_vals,avgxspeed_vals,'x-', label = 'My Results')
            plt.legend(loc=1)
            plt.title('Average x Speed')
            plt.xlabel('Occupancy')
            plt.ylabel('Avg Speed (m/s)')
            plt.savefig('D:\\model20-master\\old\\results\\'+str(int(k_vals[index_k]))+'avgx.png')

            plot_results = plt.plot(occupancy_vals,avgxspeed_vals,'x-', label = 'My Results')
            plt.savefig('D:\\model20-master\\old\\results\\'+str(int(k_vals[index_k]))+'avgtotal.png')
            plt.close()


            fig = plt.figure()
            plot_results = plt.plot(occupancy_vals,correlation_vals,'x-') 
            plt.legend(plot_results, T_legends, loc=1)
            plt.title('Correlation Coefficient')
            plt.xlabel('Occupancy')
            plt.ylabel('Correlation Coefficient')
            plt.savefig('D:\\model20-master\\old\\results\\'+str(int(k_vals[index_k]))+'correlation.png')
            plt.close()

        #------------------------------------------------------------------------------#


        if False:

            camera = Camera(plt.figure())
            altpattern = np.linspace(0, 1, all_positions.shape[2])
            if direction == 'opposite':
                altpattern[::2] = 1
            else:
                altpattern[::2] = 0
            altpattern[1::2] = 0
            colors = cm.rainbow(altpattern)
            print(all_positions.shape)

            for m in range(all_positions.shape[0]-2):
                for interp in range(interp_index): 

                    diff = (all_positions[m+1] -all_positions[m])
                    xdiff = np.abs(diff[0]) 
                    diff[:,xdiff>2] = 0
                    plt.scatter(*(all_positions[m] + (interp+1)/(interp_index) *diff), c=colors, s=r*1200)
                    plt.ylim(-1.5, 1.5)
                    camera.snap()

            anim = camera.animate(blit=True,interval=12)
            anim.save(str(int(k_vals[index_k]))+'scatter'+str(all_positions.shape[2])+'.mp4')

            reset_model()
            plt.close()
#------------------------------------------------------------------------------#



print("\nn:\n", n_vals)
print("Occupancy:\n", occupancy_vals)
print("Average Speed:\n", avgspeed_vals)
print("Time Taken:\n", t_vals)
print("Iteration Times:\n", t_iteration_vals)
#------------------------------------------------------------------------------#
np.savetxt('D:\\model20-master\\old\\results\\results_occupancy.out', (n_vals, occupancy_vals, avgspeed_vals, t_vals, t_iteration_vals), delimiter = ',')
#------------------------------------------------------------------------------#



