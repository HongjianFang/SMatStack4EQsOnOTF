#!/usr/bin/env python
# coding: utf-8

import numpy as np
from obspy.geodetics.base import gps2dist_azimuth

def SMatStack(stori,align_a,align_b,lshift=-20,rshift=20,reftrace=0):
    st = stori.copy()
    ntraces = len(st)
    shiftall = np.zeros(ntraces,)
    shift = np.arange(lshift,rshift)
    nshift = len(shift)
    rsample = st[0].stats.sampling_rate
    bg = int(align_a*rsample)
    ed = int(align_b*rsample)
    energy = st[reftrace].data[bg:ed]

    for jj in range(1,ntraces):  
        emetricp = np.zeros(nshift,)
        for ii in range(nshift):
            datasub = np.roll(st[jj].data[bg:ed],shift[ii])
            engyp = np.inner(energy,datasub)
            emetricp[ii] = engyp

        shiftv = np.argmax(emetricp)
        energy = energy + np.roll(st[jj].data[bg:ed],shift[shiftv])
        st[jj].data = np.roll(st[jj].data,shift[shiftv])
        st[jj].stats.shift = shift[shiftv]
        shiftall[jj] = shift[shiftv]

    return st,shiftall

def mcc(stori,align_before,align_after,maxiter=5,threshold=0.5):
    st = stori.copy()
    delta = st[0].stats.delta
    stnum = len(st)

    b = int(align_before/delta)
    e = int(align_after/delta)
    
    shift = np.zeros(stnum,)
    for iter in range(maxiter):
        stack = np.zeros(e-b,)
        for ii in range(stnum):
            tmpdata = st[ii].data[b:e]
            stack += tmpdata
        v2 = stack
        pol = np.ones(stnum,)
        coeff = np.zeros(stnum,)
        for ii in range(stnum):
            v1 = stori[ii].data[b:e]
            xcorr = np.correlate(v1,v2,'same')
            maxloc = np.where(xcorr==xcorr.max())[0][0]
            loc = len(v2)//2-maxloc
            shift[ii] = loc
            v1 = np.roll(v1,loc)
            coeffs = np.corrcoef(v2,v1)
            coeff[ii] = coeffs[0,1]
            if coeff[ii] < 0:
                pol[ii] = -1
            st[ii].data = np.roll(stori[ii].data,loc)
            

    return st,shift,pol


def distance_mesh(lat1,lon1,lat2,lon2):
    R = 6371
    dLat = lat2-lat1
    dLon = lon2-lon1
    a = np.sin(dLat/2) * np.sin(dLat/2) +  np.sin(dLon/2) * np.sin(dLon/2) * np.cos(lat1) * np.cos(lat2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

# randomly chosen cells
def clustersta(stlat,stlon,ncell=8000,seed=20):

    nsta = len(stlat)
    np.random.seed(seed)
    rlat = np.random.rand(ncell,)
    rlat = np.arccos(2*rlat-1)-np.pi/2
    np.random.seed(seed*2)
    rlon = np.random.rand(ncell,)
    rlon = 2*np.pi*rlon

    stlon[stlon<0]+=360
    stidx = np.zeros(nsta,dtype=int)
    aslat = np.zeros(nsta,)
    aslon = np.zeros(nsta,)

    for ii in range(nsta):
        dis = distance_mesh(np.radians(stlat[ii]),np.radians(stlon[ii]),rlat,rlon)
        idx = np.argmin(dis)
        stidx[ii] = idx
        aslat[ii] = rlat[idx]
        aslon[ii] = rlon[idx]
    return stidx,aslat,aslon

