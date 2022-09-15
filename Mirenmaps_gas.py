
#--------------------
# imports
#--------------------

import os, sys, glob
import numpy as np
from utils import rotate_data, readsnapsgl, write_fits_image_totalmass,plot_cluster,mass_profile
from astropy.cosmology import FlatLambdaCDM 
import scipy.ndimage
from tqdm import tqdm

Code='GadgetX' # the simulation code
path="/home2/weiguang/The300/data/" 
#groupinfo=np.loadtxt("/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/G3X_Mass_snap_128-center-cluster.txt")
Simun = "simulation/"+Code+"/"
Outdir = './'

Use_Lra = True # use less rotations

# two arguments given to the function, the first and the last region

h = 0.678

#RAs = np.loadtxt('1k_rotations.txt',dtype=np.int32)
RAs = np.loadtxt('main_angles.txt',dtype=np.int32)

#RAs = [RAs[0]] # only one rotation first

selecth=np.load('/home2/weiguang/Project-300-Clusters/ML/Reselected_all_halos.npy')
#HID M200 Rid

lp = 198


#------------    
# Open data
#------------

clnum='0000'+str(lp)
clnum=clnum[-4:]
cname = "NewMDCLUSTER_"+clnum+"/"

# Check outputs
outcat = cname + "/"
if not os.path.exists(outcat):
    os.mkdir(outcat)

idr = np.where(np.int32(selecth[:,2]+0.1)==lp)[0]
if len(idr)<1:
    raise ValueError('No regions find in selected halo',lp)

Hids = np.int64(selecth[idr,0]+0.1)    #AHF halo IDs
sn = np.array([np.int32(str(i)[:3]) for i in Hids])
idshid=np.argsort(Hids)
Hids=Hids[idshid]; sn=sn[idshid]; idr=idr[idshid]
st=0

#for j, s, hid in zip(idr, sn , Hids): # loop over snaps

# Miren lsit
twin_list = np.loadtxt('Twins_Samples.txt.rtf')

idrs = np.int32(twin_list[:,0])
hids = np.int64(twin_list[:,1])
ss = np.int32(twin_list[:,2])

    



# load the data 
particles = [0]

for i in tqdm(range(len(idrs))):
    idr = idrs[i]
    lp = idr
    hid = hids[i]
    s = ss[i]
    
    clnum='0000'+str(lp)
    clnum=clnum[-4:]
    cname = "NewMDCLUSTER_"+clnum+"/"

    # Check outputs
    outcat = cname + "/"
    if not os.path.exists(outcat):
        os.mkdir(outcat)
    
    print(lp,s,hid)
    
    snapname = 'snap_'+str(s)
    #print(snapname)
    #ds = yt.load(path+Simun+cname+snapname, field_spec="my_def") # it can also be done using yt
    snapfile = path+Simun+cname+snapname

    head=readsnapsgl(path+Simun+cname+snapname,'HEAD')
    if head.Redshift<0:
        head.Redshift = 0.0000
    
    
    
    
    ra = 0

    for RA in RAs:
        Ms = 0
        imgs = []
        for particle in particles:

            print('PARTICLE TYPE = ', particle)
            pos = readsnapsgl(snapfile, 'POS ', ptype=particle)
            mass = readsnapsgl(snapfile, 'MASS', ptype=particle)
            print('path = ',snapfile)
            print('POSITION VECTOR = ', pos.shape)
            print('MASS VECTOR = ', mass.shape)

            # it looks like there might be a bug in Weiguang's code to read Gatgetx data
            if particle==1:
                if mass.shape==():
                    mass = np.ones(pos.shape[0])*0.12691148
            #------------    
            # Cuts
            #------------


            halo = np.load('/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GadgetX/G3X_Mass_snap_'+str(s)+'info.npy')
            idg = np.where((halo[:,0]==lp) & (halo[:,1]==hid))[0]
            if len(idg) == 1:
                cc = halo[idg[0],4:7]; rr = halo[idg[0],7]
            else:
                raise ValueError('Halo not find.... ',lp,hid)

            # apply mask (not needed here)

    #             mask2= np.where((pos[:,0]<=cc[0]+2*rr)&(pos[:,0]>=cc[0]-2*rr)&
    #                     (pos[:,1]<=cc[1]+2*rr)&(pos[:,1]>=cc[1]-2*rr)&
    #                     (pos[:,2]<=cc[2]+2*rr)&(pos[:,2]>=cc[2]-2*rr))

            pos_inside = pos.copy()
            mass_inside = mass.copy()
            pos_centered = pos_inside.copy()

            #center the data to be rotated
            pos_inside[:,0] = pos_inside[:,0]-cc[0]
            pos_inside[:,1] = pos_inside[:,1]-cc[1]
            pos_inside[:,2] = pos_inside[:,2]-cc[2]



            def indices_inside(pos,r,center = (0,0,0)): 

                x = pos[:,0]
                y = pos[:,1]
                z = pos[:,2]

                mask = np.where((x**2+y**2+z**2)<=r**2)

                return mask


            mask_sphere = indices_inside(pos_inside,rr)
            # write this to see the difference with respect to Mahf
            print('MASA 3D:',np.log10(mass_inside[mask_sphere].sum()*1e10))
            Ms = Ms + mass_inside[mask_sphere].sum()
            print(np.log10(halo[idg[0],3]))

            #------------    
            # Rotations
            #------------

            rot = rotate_data(pos_inside,RA)[0]

            # note that radius is comoving in kpc/h
            redshift = float(head.Redshift)

            new_rr = 2500*(1+redshift)*h
            mask = np.where((rot[:,0]<=new_rr)&(rot[:,0]>=-new_rr)&
                    (rot[:,1]<=new_rr)&(rot[:,1]>=-new_rr)&
                    (rot[:,2]<=2*new_rr)&(rot[:,2]>=-2*new_rr))

            rot = rot[mask]

            w = mass_inside[mask]
            #------------    
            # Create the 2D projection
            #------------
            N = 2048
            x = rot[:, 0] 
            y = rot[:,1] 
            img,xedges,yedges = np.histogram2d(x,y,bins=(N,N),weights=w)
            img = img.T
            #plot_cluster(img)
            M_profile = mass_profile(img)

            import matplotlib.pyplot as plt
            #plt.plot(np.arange(320),np.log10(M_profile))

            #print('MASA 2D = ',np.log10(img.sum()*1e10))
    #         print(np.log10(halo[idg[0],3]))
    #         print(hid)
    #         print(cc)

            del pos
            del mass

            # end particle loop

            imgs.append(img)
        #------------    
        # Save data
        #------------
        img_tot = np.zeros(img.shape)
        for i in range(len(particles)):
            img_tot = img_tot+imgs[i]


        fwhm2sig = 1./(2.*np.sqrt(2.*np.log(2.)))
        reso = 5000/2048 # in kpc
        img_tot = scipy.ndimage.filters.gaussian_filter(img_tot,(7./reso)*fwhm2sig)

    #     print('MASA AHF =', np.log10(halo[idg[0],3]))
    #     print('MASA 2D tot =',np.log10(img_tot.sum()*1e10))
    #     print('MASA 3D tot =', np.log10(Ms*1e10))
    #     M_profile = mass_profile(img_tot)
    #     print('MASA 2D CIL =',np.log10(M_profile[-1]))


        write_fits_image_totalmass(img_tot,
                                 outcat + snapname + "-Mgas-" + "cl-" + str(hid) + "-ra-{}-{}-{}-".format(RA[0],RA[1],RA[2])  +".fits",
                                 overwrite= True, 
                                 comments=("Simulation Region: " + clnum,
                                          "AHF Halo ID: "+str(hid), 
                                          "Simulation redshift: " + str(head.Redshift)[:7],
                                          "log M_200 = "+str(np.log10(halo[idg[0],3]))[:8]+" Msun/h",
                                          "R_200 = "+str(rr)[:8]+" kpc/h",
                                          )
                                          )


        ra+=1

        print("DOOOOOOOONNNNNEEEEEEEEE")



