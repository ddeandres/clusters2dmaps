import pymsz
import os, sys, glob
import numpy as np
from readsnapsgl import readsnapsgl
from astropy.cosmology import FlatLambdaCDM 
from astropy.io import fits

Code='GadgetX'
path="/home2/weiguang/The300/data/"
cpath="catalogues/AHF/"+Code+"/"
groupinfo=np.loadtxt("/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/G3X_Mass_snap_128-center-cluster.txt")
#regionID hID  Mvir(4) Xc(6)   Yc(7)   Zc(8)  Rvir(12) fMhires(38) cNFW (42) M500 R500 fgas f*
Simun = "simulation/"+Code+"/"


stn = np.int32(sys.argv[1])
edn = np.int32(sys.argv[2])

# Gadget X     - GadgetMUSIC
# 0.022 - 127  - 16 0.053
# 0.117- 123   - 15 0.111
# 0.194 - 120  - 14 0.176
# 0.305 - 116  - 12 0.333
# 0.43 - 111*  - 11
# 0.54 - 108*  - 10
# 0.67 - 105*  - 09
# 0.82 - 101*  - 08
# 0.901 - 99   - --
# 1.032 - 96   - 07 1.000

# all halos with pregens
# pregenIDs=np.loadtxt('/home2/weiguang/Project-300-Clusters/MAH/G3X-Halo-Progenitor-IDs-for-M200-gt-11.8.txt',dtype=np.int64)

# rotation angles 0-180, 30

RAs = np.loadtxt('29_rotations.txt')

selecth=np.load('/home2/weiguang/Project-300-Clusters/ML/Reselected_all_halos.npy')
#HID M200 Rid

def calc_z(npx,ar,z,rr): #return redshift by pixels and angular resolution
    #set up cosmology
    cosmo= FlatLambdaCDM(67.77,0.307115)
    zr=np.arange(1.0e-6,0.12,1.0e-6)[::-1]
    apkpz=cosmo.arcsec_per_kpc_proper(zr).value
    rrp=rr*2/0.6777/(1+z)
    return np.interp(npx, rrp/ar*apkpz, zr)

#for lp in np.arange(0,groupinfo.shape[0]):  # note region 10 is missing snap_100, Region 228 missing 101!!
#f=open('output_red_'+str(stn)+'.txt','w')
for lp in np.arange(stn,edn):  #all regions 

    clnum='0000'+str(lp)
    clnum=clnum[-4:]
    cname = "NewMDCLUSTER_"+clnum+"/"

    # Check outputs
    outcat = cname + "/"
    if not os.path.exists(outcat):
        os.mkdir(outcat)

    idr=np.where(np.int32(selecth[:,2]+0.1)==lp)[0]
    if len(idr)<1:
        raise ValueError('No regions find in selected halo',lp)
    
    Hids = np.int64(selecth[idr,0]+0.1)    #AHF halo IDs
    sn = np.array([np.int32(str(i)[:3]) for i in Hids])
    idshid=np.argsort(Hids)
    Hids=Hids[idshid]; sn=sn[idshid]; idr=idr[idshid]
    st=0
    for j, s, hid in zip(idr, sn, Hids):
        snapname = 'snap_'+str(s)
        #ff=fits.open(outcat + snapname + "-TT" + "-cl-" + str(hid) + "-ra-0-0-0.fits")
        #if not np.isnan(ff[0].data.max()): # if there is a non value, we redo the image
        #    ff.close()
        #    continue 
        #else:
        #    ff.close()

        #print('Updating ', lp, j, hid)
        if st!=s:
            st=s
            if ((lp == 10) and (snapname == 'snap_100')) or ((lp == 228) and (snapname == 'snap_110')):
                continue
            
            head=readsnapsgl(path+Simun+cname+snapname,'HEAD')
            if head.Redshift<0:
                head.Redshift = 0.0000

            halo = np.load('/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GadgetX/G3X_Mass_snap_'+str(s)+'info.npy')
            ##ReginIDs HIDs  HosthaloID Mvir(4) Xc(6)   Yc(7)   Zc(8)  Rvir(12) fMhires(38) cNFW (42) Mgas200 M*200 M500  R500 fgas500 f*500
        idg=np.where((halo[:,0]==lp) & (halo[:,1]==hid))[0]
        if len(idg) == 1:
            cc=halo[idg[0],4:7]; rr = halo[idg[0],7]
        else:
            raise ValueError('Halo not find.... ',lp,hid)
            
        # load simulation data
        simd = pymsz.load_data(path+Simun+cname+snapname, snapshot=True, center=cc, radius=rr*np.sqrt(2), restrict_r=True)
        cosmo= FlatLambdaCDM(simd.cosmology['h']*100,simd.cosmology['omega_matter'])


        #setup for outputs
#         if simd.cosmology['z'] <= 0.05:
#             outred = 0.1 + simd.cosmology['z']
#         else:
#             outred = simd.cosmology['z'] # use simulation redshift
        fixps = 640  # Maxi Number pixels per image refer to the massive clusters at z0.1
        angular = 5  # arcsec fixps/(head.Redshift+1) * cosmo.arcsec_per_kpc_proper(outred).value # need to change into physical
        #now we calculate the redshift to put this halo
        outred = calc_z(fixps, angular, simd.cosmology['z'], rr)
#        f.write(str(outred)+'\n')
        ra =0
        for pd in RAs:  ##["x","y","z"]:
            # print("projecting photons to %s" % proj_direc)
            #pj = pymsz.TT_model(simd, npixel=np.int32(2*rr*1.2*head.Time/head.Hubbleparam/10.+0.5), axis=proj_direc, redshift=0.1, AR=5.252134578)
            # increase to 2 times
            pj = pymsz.TT_model(simd, npixel=640, axis=pd, redshift=outred, AR=angular,Ncpu=8,sph_kernel='wendland4',zthick=rr)
            pj.write_fits_image(outcat + snapname + "-TT" + "-cl-" + str(hid) + "-ra-" + str(ra) + ".fits", 
                                overwrite=True, comments=("Simulation Region: " + clnum,
                                                          "AHF Halo ID: "+str(hid), 
                                                          "Simulation redshift: " + str(head.Redshift)[:6],
                                                          "log M_200 = "+str(np.log10(halo[idg[0],3]))[:6]+" Msun/h",
                                                          "R_200 = "+str(rr)[:6]+" kpc/h") )
            ra+=1
            
     