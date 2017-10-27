import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import leastsq
from coordinateSystems import GeographicSystem
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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


def received_power(power_emit, r, wavelength=4.762, recv_gain=1.0):
    """ Calculate the free space loss. Defaults are for no receive gain and
        a 63 MHz wave.
        Units: watts, meters
    """
    four_pi_r = 4.0 * np.pi * r
    free_space_loss = (wavelength*wavelength) / (four_pi_r * four_pi_r)
    return recv_gain * power_emit * free_space_loss


def precalc_station_terms(stations_ECEF):
    """ Given a (N_stn, 3) array of station locations,
        return dxvec and drsq which are (N, N, 3) and (N, N),
        respectively, where the first dimension is the solution
        for a different master station
    """
    dxvec = stations_ECEF-stations_ECEF[:, np.newaxis]
    r1    = (np.sum(stations_ECEF**2, axis=1))
    drsq  = r1-r1[:, np.newaxis]
    return dxvec, drsq


def linear_first_guess(t_i, dxvec, drsq, c=3.0e8):
    """ Given a vector of (N_stn) arrival times,
        calcluate a first-guess solution for the source location.
        Return f=(x, y, z, t), the source retrieval locations.
    """
    g = 0.5*(drsq - (c**2)*(t_i**2))
    K = np.vstack((dxvec.T, -c*t_i)).T
    f, residuals, rank, singular = lstsq(K, g)
    return f


def predict(p, stations_ECEF2, c=3.0e8):
    """ Predict arrival times for a particular fit of x,y,z,t source
        locations. p are the parameters to retrieve
    """
    return (p[3] + ((np.sum((p[:3] - stations_ECEF2) *
            (p[:3] - stations_ECEF2), axis=1))**0.5)) / c


def residuals(p, t_i, stations_ECEF2):
    return t_i - predict(p, stations_ECEF2)


def dfunc(p, t_i, stations_ECEF2, c=3.0e8):
    return -np.vstack(((p[:3] - stations_ECEF2).T / (c *
              (np.sum((p[:3] - stations_ECEF2) *
              (p[:3] - stations_ECEF2), axis=1))**0.5),
             np.array([1./c]*np.shape(stations_ECEF2)[0])))


def gen_retrieval_math(i, selection, t_all, t_mins, dxvec, drsq, center_ECEF,
                       stations_ECEF, dt_rms, min_stations=5,
                       max_z_guess=25.0e3):
    """ t_all is a N_stations x N_points masked array of arrival times at
        each station.
        t_min is an N-point array of the index of the first unmasked station 
        to receive a signal

        center_ECEF for the altitude check

        This streamlines the generator function, which emits a stream of 
        nonlinear least-squares solutions.
    """  
    m = t_mins[i]
    stations_ECEF2=stations_ECEF[selection]
    # Make a linear first guess
    p0 = linear_first_guess(np.array(t_all[:,i][selection]-t_all[m,i]),
                            dxvec[m][selection], 
                            drsq[m][selection])
    t_i =t_all[:,i][selection]-t_all[m,i]
    # Checking altitude in lat/lon/alt from local coordinates
    latlon = np.array(GeographicSystem().fromECEF(p0[0], p0[1],p0[2]))
    if (latlon[2]<0) | (latlon[2]>25000): 
        latlon[2] = 7000
        new = GeographicSystem().toECEF(latlon[0], latlon[1], latlon[2])
        p0[:3]=np.array(new)
    plsq = np.array([np.nan]*5)   
    plsq[:4], cov, infodict, mesg,ier = leastsq(residuals, p0, 
                            args=(t_i, stations_ECEF2), 
                            Dfun=dfunc,col_deriv=1,full_output=True) 
    plsq[4] = np.sum(infodict['fvec']*infodict['fvec'])/(
                     dt_rms*dt_rms*(float(np.shape(stations_ECEF2)[0]-4)))
    return plsq

def gen_retrieval(t_all, t_mins, dxvec, drsq, center_ECEF, stations_ECEF, 
                  dt_rms, min_stations=5, max_z_guess=25.0e3): 
    """ t_all is a N_stations x N_points masked array of arrival times at 
        each station.
        t_min is an N-point array of the index of the first unmasked station 
        to receive a signal
    
        center_ECEF for the altitude check

        This is a generator function, which emits a stream of nonlinear
        least-squares solutions.
    """    
    for i in range(t_all.shape[1]):
        selection=~np.ma.getmask(t_all[:,i])
        if np.all(selection == True):
            selection = np.array([True]*len(t_all[:,i]))
            yield gen_retrieval_math(i, selection, t_all, t_mins, dxvec, drsq,
                                     center_ECEF, stations_ECEF, dt_rms, 
                                     min_stations, max_z_guess=25.0e3)
        elif np.sum(selection)>=min_stations:
            yield gen_retrieval_math(i, selection, t_all, t_mins, dxvec, drsq,
                                     center_ECEF, stations_ECEF, dt_rms, 
                                     min_stations, max_z_guess=25.0e3)
        else: 
            yield np.array([np.nan]*5)

def gen_retrieval_full(t_all, t_mins, dxvec, drsq, center_ECEF, stations_ECEF, 
                       dt_rms, c0, min_stations=5, max_z_guess=25.0e3): 
    """ t_all is a N_stations x N_points masked array of arrival times at 
        each station.
        t_min is an N-point array of the index of the first unmasked station 
        to receive a signal
    
        center_ECEF for the altitude check

        This is a generator function, which emits a stream of nonlinear
        least-squares solutions.

        Timing comes out of least-squares function as t*c from the initial 
        station
    """    
    for i in range(t_all.shape[1]):
        selection=~np.ma.getmask(t_all[:,i])
        plsq = np.array([np.nan]*7)
        if np.all(selection == True):
            selection = np.array([True]*len(t_all[:,i]))
            plsq[:5] = gen_retrieval_math(i, selection, t_all, t_mins, dxvec,
                         drsq, center_ECEF, stations_ECEF, dt_rms, 
                         min_stations, max_z_guess=25.0e3)
            plsq[5] = plsq[3]/c0 + t_all[t_mins[i],i]
            plsq[6] = np.shape(stations_ECEF[selection])[0]
            yield plsq
        elif np.sum(selection)>=min_stations:
            plsq[:5] = gen_retrieval_math(i, selection, t_all, t_mins, dxvec,
                                     drsq, center_ECEF, stations_ECEF, dt_rms,
                                     min_stations, max_z_guess=25.0e3)
            plsq[5] = plsq[3]/c0 + t_all[t_mins[i],i]
            plsq[6] = np.shape(stations_ECEF[selection])[0]
            yield plsq
        else: 
            plsq[6] = np.shape(stations_ECEF[selection])[0]
            yield plsq

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def black_boxtesting(x,y,z,n,
              stations_local,ordered_threshs,stations_ecef,center_ecef,
              tanps,
              c0,dt_rms,tanp,projl,chi2_filter,min_stations=5,ntsd=3):
    """ This funtion incorporates most of the Monte Carlo functions and calls
        into one big block of code.

        x,y,z are the source location in the local tangent plane (m)
        n is the number of iterations

        stations_local is the the (N-stations, 3) array of station locations
        in the local tangent plane

        ordered_threshs is the N-station array of thresholds in the same order
        as the station arrays (in dBm).

        stations_ecef is (N,3) array in ECEF coordinates

        center_ecef is just the center location of the network, just easier
        to pass into the fuction separately to save some calculation

        c0 is th speed of light

        dt_rms is the standard deviation of the timing error (Gaussian, in s)

        tanp is the tangent plane object

        projl is the map projection object

        chi2_filter is the maximum allowed reduced chi2 filter for the
        calculation (use at most 5)

        min_stations is the minimum number of stations required to receive
        a source. This must be at least 5, can be higher to filter out more
        poor solutions

        Returned are the w,h,theta values of the covariance ellipses in one
        array and the standard deviation of the altitude solutions separately.
        Covariance ellipses are by default set at 3 standard deviations.
    """
    points = np.array([np.zeros(n)+x, np.zeros(n)+y, np.zeros(n)+z]).T
    powers = np.empty(n)
    # For the theoretical distribution:
    for i in range(len(powers)):
        powers[i] = np.max(1./np.random.uniform(0,1000,2000))
    
    # Calculate distance and power retrieved at each station and mask
    # the stations which have higher thresholds than the retrieved power
    points_f_ecef = (tanp.fromLocal(points.T)).T  
    dt, ran  = travel_time(points, stations_local, c0, get_r=True)
    pwr = received_power(powers, ran)
    masking = 10.*np.log10(pwr/1e-3) < ordered_threshs[:,np.newaxis]
    masking2 = np.empty_like(masking)
    for i in range(len(stations_ecef[:,0])):
        masking2[i] = tanps[i].toLocal(points_f_ecef.T)[2]<0

    masking = masking | masking2
    pwr = np.ma.masked_where(masking, pwr)
    dt  = np.ma.masked_where(masking, dt)
    ran = np.ma.masked_where(masking, ran)
    
    # Add error to the retreived times
    dt_e = dt + np.random.normal(scale=dt_rms, size=np.shape(dt))
    dt_mins = np.argmin(dt_e, axis=0)
    # Precalculate some terms in ecef (fastest calculation)
    points_f_ecef = (tanp.fromLocal(points.T)).T  
    full_dxvec, full_drsq = precalc_station_terms(stations_ecef)
    # Run the retrieved locations calculation
    # gen_retrieval returns a tuple of four positions, x,y,z,t.
    dtype=[('x', float), ('y', float), ('z', float), ('t', float), 
           ('chi2', float)]
    # Prime the generator function - pauses at the first yield statement.
    point_gen = gen_retrieval(dt_e, dt_mins, full_dxvec, full_drsq, 
                              center_ecef, stations_ecef, dt_rms, 
                              min_stations)
    # Suck up the values produced by the generator, produce named array.
    retrieved_locations = np.fromiter(point_gen, dtype=dtype)
    retrieved_locations = np.array([(a,b,c,e) for (a,b,c,d,e) in 
                                     retrieved_locations])
    chi2                = retrieved_locations[:,3]
    retrieved_locations = retrieved_locations[:,:3]
    retrieved_locations = np.ma.masked_invalid(retrieved_locations)
    #Convert back to local tangent plane
    soluts = tanp.toLocal(retrieved_locations.T)
    proj_soluts = projl.fromECEF(retrieved_locations[:,0], 
                                 retrieved_locations[:,1], 
                                 retrieved_locations[:,2])
    good = proj_soluts[2] > 0
    proj_soluts = (proj_soluts[0][good],proj_soluts[1][good],
                   proj_soluts[2][good])
    proj_soluts = np.ma.masked_invalid(proj_soluts)
    cov = np.cov(proj_soluts[0][chi2[good]<chi2_filter], proj_soluts[1][chi2[good]<chi2_filter])
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * ntsd * np.sqrt(vals)

    return np.array([w,h,theta]),np.std(proj_soluts[2][chi2[good]<chi2_filter]
             )