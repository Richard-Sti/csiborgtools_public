import numpy as np
import math
import matplotlib.pyplot as plt
import fortranfile as ff
from os import listdir
import random

outnr1 = str(184).zfill(5)
outnr2 = str(2).zfill(5)
srcdir1 = '/mnt/extraspace/hdesmond/IC_test3/output_'+outnr1
srcdir2 = '/mnt/extraspace/hdesmond/IC_test3_inv/output_'+outnr2

for i in range(2):
    print("Starting ", i)
    
    if i==0:
        srcdir = srcdir1
        outnr = outnr1
    else:
        srcdir = srcdir2
        outnr = outnr2

    infofile = srcdir+'/info_'+outnr+'.txt'
    f = open(infofile, 'r')
    ncpuline = f.readline()
    line = ncpuline.split()

    ncpu = int(line[-1])
    print("ncpu:", ncpu)

    print("Reading in particles of output", int(srcdir[-5:]))

    srcdirlist = listdir(srcdir)

    if 'unbinding_'+srcdir[-5:]+'.out00001' not in srcdirlist:
        print("Couldn't find unbinding_"+srcdir[-5:]+".out00001 in", srcdir)
        print("use mergertreeplot.py -h or --help to print help message.")
        quit()


    #-----------------------
    # First read headers
    #-----------------------
    nparts = np.zeros(ncpu, dtype='int32')
    partfiles = [0]*ncpu

    for cpu in range(ncpu):
        srcfile = srcdir+'/part_'+srcdir[-5:]+'.out'+str(cpu+1).zfill(5)
        partfiles[cpu] = ff.FortranFile(srcfile)

        ncpu = partfiles[cpu].readInts()
        ndim = partfiles[cpu].readInts()
        nparts[cpu] = partfiles[cpu].readInts()
        localseed = partfiles[cpu].readInts()
        nstar_tot = partfiles[cpu].readInts()
        mstar_tot = partfiles[cpu].readReals('d')
        mstar_lost = partfiles[cpu].readReals('d')
        nsink = partfiles[cpu].readInts()

        del ndim, localseed, nstar_tot, mstar_tot, mstar_lost, nsink


    #-------------------
    # Allocate arrays
    #-------------------
    nparttot = nparts.sum()

    dum = np.zeros(nparttot, dtype='float16')

    if i==0:
        #x = np.zeros(nparttot, dtype='float16')
        #y = np.zeros(nparttot, dtype='float16')
        #z = np.zeros(nparttot, dtype='float16')

        mass = np.zeros(nparttot, dtype='float16')
        ID = np.zeros(nparttot, dtype='int32')
        
        level = np.zeros(nparttot, dtype='int32')
        
        clumpid = np.zeros(nparttot, dtype='int32')
    
    else:
        #x_inv = np.zeros(nparttot, dtype='float16')
        #y_inv = np.zeros(nparttot, dtype='float16')
        #z_inv = np.zeros(nparttot, dtype='float16')
        
        mass_inv = np.zeros(nparttot, dtype='float16')
        ID_inv = np.zeros(nparttot, dtype='int32')

        level_inv = np.zeros(nparttot, dtype='int32')

        clumpid_inv = np.zeros(nparttot, dtype='int32')


    #----------------------
    # Read particle data
    #----------------------

     #read(1)ncpu2          # What you would do in fortran
     #read(1)ndim2
     #read(1)npart2
     #read(1)
     #read(1)
     #read(1)
     #read(1)
     #read(1)
     #do i=1,ndim
        #read(1)m
        #x(1:npart2,i)=m
     #end do
     #! Skip velocity
     #do i=1,ndim
        #read(1)m
     #end do
     #! Read mass
     #read(1)m
     #if(nstar>0)then
        #read(1) ! Skip identity
        #read(1) ! Skip level
        #read(1)family
        #read(1)tag
        #read(1)age

    start_ind = np.zeros(ncpu, dtype='int')
    for cpu in range(ncpu-1):
        start_ind[cpu+1] = nparts[cpu] + start_ind[cpu]

    for cpu in range(ncpu):
        unbfile = srcdir+'/unbinding_'+srcdir[-5:]+'.out'+str(cpu+1).zfill(5)
        unbffile = ff.FortranFile(unbfile)
        
        if i==0:
            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')      # Think they're stored as double so must read as double
            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')      # Positions
            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')

            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')      # Velocities; this all just overwrites itself
            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')

            mass[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')     # Mass

            #vx[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
            #vy[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
            #vz[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
    
            #mass[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
            ID[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readInts()

            level[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readInts()

            clumpid[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = unbffile.readInts()

        else:
            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')

            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
            dum[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')

            mass_inv[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')

            #vx[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
            #vy[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
            #vz[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
    
            #mass[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readReals('d')
            ID_inv[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readInts()

            level_inv[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = partfiles[cpu].readInts()

            clumpid_inv[start_ind[cpu]:start_ind[cpu]+nparts[cpu]] = unbffile.readInts()

    del dum
    
    if i==0:
        print("Minimum clump ID:", np.min(clumpid))      # This is the clump a particle has been assigned to, so min should be 0 which means not in clump
        clumpid = np.absolute(clumpid)                  # Not sure why this is here...
    else:
        print("Minimum inv clump ID:", np.min(clumpid_inv))
        clumpid_inv = np.absolute(clumpid_inv)


#random.shuffle(ID); random.shuffle(ID_inv)          # If the IDs are randomised but not the clumpIDs then all of the below should be random

print(np.min(ID), np.median(ID), np.mean(ID), np.max(ID))
print(np.min(ID_inv), np.median(ID_inv), np.mean(ID_inv), np.max(ID_inv))

print(np.min(level), np.median(level), np.mean(level), np.max(level))
print(np.min(level_inv), np.median(level_inv), np.mean(level_inv), np.max(level_inv))

print(np.min(mass), np.median(mass), np.mean(mass), np.max(mass))
print(np.min(mass_inv), np.median(mass_inv), np.mean(mass_inv), np.max(mass_inv))

#plt.clf()
#plt.hist(mass)
#plt.show()


#index  lev  parent(2)  ncell    peak_x   peak_y(5)   peak_z  rho-       rho+(8)      rho_av    mass_cl   relevance(11)
clumparr = np.genfromtxt(srcdir1+"/clump_"+outnr1+".dat")
clumparr_inv = np.genfromtxt(srcdir2+"/clump_"+outnr2+".dat")

clumpID, parent, Mclump = clumparr[:,0].astype(int), clumparr[:,2].astype(int), clumparr[:,10]
clumpID_inv, parent_inv, Mclump_inv = clumparr_inv[:,0].astype(int), clumparr_inv[:,2].astype(int), clumparr_inv[:,10]

#clumpID_main = clumpID[clumpID==parent]      # IDs of main halos only from the clump file
#clumpID_main_inv = clumpID_inv[clumpID_inv==parent_inv]

#clumpID_big = clumpID[Mclump>np.median(Mclump)]         # IDs of halos more massive than the median
#clumpID_big_inv = clumpID_inv[Mclump_inv>np.median(Mclump_inv)]

clumpID_big = clumpID[Mclump>np.percentile(Mclump, 90)]
clumpID_big_inv = clumpID_inv[Mclump_inv>np.percentile(Mclump_inv, 90)]

#clumpID_small = clumpID[Mclump<=np.median(Mclump)]         # IDs of halos more massive than the median
#clumpID_small_inv = clumpID_inv[Mclump_inv<=np.median(Mclump_inv)]


#[np.where(clumpid==x) for x in clumpID_main]

print("CHECK:", len(ID), len(ID_inv), len(np.intersect1d(ID, ID_inv)), "(should all be the same)")

print("Total number of clumps in the two sims:", len(clumparr), len(clumparr_inv))

print("Total number of particles in the two sims:", len(ID), len(ID_inv), "(should be the same)")

print("Fraction of particles within halos in the two sims:", round(len(ID[clumpid!=0])/float(len(ID)), 6), round(len(ID_inv[clumpid_inv!=0])/float(len(ID)), 6))
print("Fraction of particles within halos in *both* sims:", round(len(np.intersect1d(ID[clumpid!=0], ID_inv[clumpid_inv!=0]))/float(len(np.intersect1d(ID, ID_inv))), 6), "(random value =", round(len(ID[clumpid!=0])/float(len(ID)) * len(ID_inv[clumpid_inv!=0])/float(len(ID_inv)), 6), "), ratio =", round(len(np.intersect1d(ID[clumpid!=0], ID_inv[clumpid_inv!=0]))/float(len(np.intersect1d(ID, ID_inv))) / (len(ID[clumpid!=0])/float(len(ID)) * len(ID_inv[clumpid_inv!=0])/float(len(ID_inv))), 3))

print("Fraction of particles in massive halos in the two sims:", round(np.sum(np.in1d(clumpid, clumpID_big))/float(len(ID)), 6), round(np.sum(np.in1d(clumpid_inv, clumpID_big_inv))/float(len(ID)), 6))
print("Fraction of particles in massive halos in *both* sims:", round(len(np.intersect1d(ID[np.in1d(clumpid, clumpID_big)], ID_inv[np.in1d(clumpid_inv, clumpID_big_inv)]))/float(len(np.intersect1d(ID, ID_inv))), 6), "(random value =", round(np.sum(np.in1d(clumpid, clumpID_big))/float(len(ID)) * np.sum(np.in1d(clumpid_inv, clumpID_big_inv))/float(len(ID)), 6), "), ratio =", round(len(np.intersect1d(ID[np.in1d(clumpid, clumpID_big)], ID_inv[np.in1d(clumpid_inv, clumpID_big_inv)]))/float(len(np.intersect1d(ID, ID_inv))) / (np.sum(np.in1d(clumpid, clumpID_big))/float(len(ID)) * np.sum(np.in1d(clumpid_inv, clumpID_big_inv))/float(len(ID_inv))), 3))