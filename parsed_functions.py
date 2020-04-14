import numpy as np

from coordinateSystems import TangentPlaneCartesianSystem, GeographicSystem, MapProjection

def travel_time(X, X_ctr, c, t0=0.0, get_r=False):
    """ Units are meters, seconds.
        X is a (N,3) vector of source locations, and X_ctr is (N,3) receiver
        locations.

        t0 may be used to specify a fixed amount of additonal time to be
        added to the travel time.

        If get_r is True, return the travel time *and* calculated range.
        Otherwise
        just return the travel time.
    """
    ranges = (np.sum((X[:, np.newaxis] - X_ctr)**2, axis=2)**0.5).T
    time = t0+ranges/c
    if get_r == True:
        return time, ranges
    if get_r == False:
        return time

def min_power(threshold, r, wavelength=4.762, recv_gain=1.0):
    """ Calculate the free space loss. Defaults are for no receive gain and
        a 63 MHz wave.
        Units: watts, meters
    """
    four_pi_r = 4.0 * np.pi * r
    free_space_loss = (wavelength*wavelength) / (four_pi_r * four_pi_r)
    return np.ma.array([threshold[i] / (free_space_loss[i] * recv_gain) for i in range(len(threshold))])
    # return threshold[:,np.newaxis] / (free_space_loss * recv_gain)

def power_increase(r, wavelength=4.762, recv_gain=1.0):
    """ Calculate the free space loss. Defaults are for no receive gain and
        a 63 MHz wave.
        Units: watts, meters
    """
    four_pi_r = 4.0 * np.pi * r
    free_space_loss = (wavelength*wavelength) / (four_pi_r * four_pi_r)
#     return -10*np.log10((free_space_loss * recv_gain)/1e-3)
    return -10*np.log10((free_space_loss * recv_gain))

def quick_method(aves, sq, fde, xint=5000, altitude=7000,station_requirement=6,c0 = 3.0e8,
                 mindist = 300000):
    """ This function derives the minimum detectable source power and 
        corresponding source and flash detection efficiencies at points within
        300 km of the network with grid spacing xint at altitude m MSL

        aves is a (N-stations,(lat,lon,alt,threshold in dBm) array of station
        locations)

        sq is the quantile of the source powers in the power distribution

        fde is the quantile array of expected points per flash

        stations_requirement is the minimum number of stations to retreive a
        signal in order to find a solution
        
        c0 is th speed of light

        mindist is used to find the min/max x- and y- grid coordinates from 
        the min/max station locations. Ex: A value of 300000 (m) is at least 
        600 by 600 km in x and y.

        Also performs check of line of sight based on Earth curvature

        Returns: latitude of grid points, longitude of grid points, source
        detection efficiency (%), flash detection efficiency (%), minimum 
        detectable source power (dBW)       
    """
    center = (np.mean(aves[:,0]), np.mean(aves[:,1]), np.mean(aves[:,2]))
    geo  = GeographicSystem()
    mapp = MapProjection
    projl = MapProjection(projection='laea', lat_0=center[0], lon_0=center[1])

    lat, lon, alt  = aves[:,:3].T
    stations_ecef  = np.array(geo.toECEF(lon, lat, alt)).T

    center_ecef = np.array(geo.toECEF(center[1],center[0],center[2]))
    ordered_threshs = aves[:,-1]

    check = projl.fromECEF(stations_ecef[:,0],stations_ecef[:,1],stations_ecef[:,2])
    xmin = np.min(check[0]) - mindist
    xmax = np.max(check[0]) + mindist
    ymin = np.min(check[1]) - mindist
    ymax = np.max(check[1]) + mindist
    alts = np.array([altitude])
    initial_points = np.array(np.meshgrid(np.arange(xmin,xmax+xint,xint),
                                          np.arange(ymin,ymax+xint,xint), alts))

    x,y,z=initial_points.reshape((3,int(np.size(initial_points)/3)))
    points2 = np.array(projl.toECEF(x,y,z)).T

    xp,yp,zp = points2.T
    lonp,latp,zp = geo.fromECEF(xp,yp,zp)
    latp = latp.reshape(np.shape(initial_points)[1],np.shape(initial_points)[2])
    lonp = lonp.reshape(np.shape(initial_points)[1],np.shape(initial_points)[2])

    tanp_all = []
    for i in range(len(aves[:,0])): 
        tanp_all = tanp_all + [TangentPlaneCartesianSystem(aves[i,0],aves[i,1],aves[i,2])]

    masking2 = np.ma.empty((np.shape(stations_ecef)[0],np.shape(x)[0]))
    dt, ran  = travel_time(points2, stations_ecef, c=c0, get_r=True)
    selection = np.sum(ran<320000,axis=0) >0
    for i in range(len(stations_ecef[:,0])):
        masking2[i,selection] = tanp_all[i].toLocal(points2[selection].T)[2]>0
        
    ran = np.ma.masked_where(masking2==0, ran)
    ran = np.ma.masked_where(ran>=320000, ran)

    mins = min_power((10**(ordered_threshs/10.))*1e-3,ran)
    np.ma.set_fill_value(mins, 999)
    req_power = np.partition(mins.filled(), station_requirement-1,axis=0)[station_requirement-1]

    sde = np.zeros_like(req_power)
    for i in range(len(sq[0])-1):
        selects = (req_power >= sq[1,i]) & (req_power < sq[1,i+1])
        sde[selects] = 100-sq[0,i+1]
        
    selects = (req_power < sq[1,0]) 
    sde[selects] = 100
    sde = (sde.T.reshape(np.shape(initial_points[0,:,:,0])))

    fde_a = np.zeros_like(sde)
    xs  = 1000./np.arange(10,1000,1.) # Theoretical source detection efficiency that corresponds with fde

    selects = sde == 100. # Put into the next lowest or equivalent flash DE from given source DE
    fde_a[selects] = 100.
    for i in range(len(xs)-1):
        selects = (sde >= xs[1+i]) & (sde < xs[i])
        fde_a[selects] = fde[i]

    req_power = (req_power.T.reshape(np.shape(initial_points[0,:,:,0])))

    return latp,lonp,sde,fde_a,req_power