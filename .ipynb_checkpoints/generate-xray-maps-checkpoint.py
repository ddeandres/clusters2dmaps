import yt
import os, sys, glob
import numpy as np
from yt.utilities.cosmology import Cosmology
from yt.frontends.gadget.definitions import gadget_field_specs
#from readsnapsgl import readsnapsgl
import pyxsim
from astropy.cosmology import FlatLambdaCDM 
from utils import rotation_matrix

Code='GadgetX'
path="/home2/weiguang/The300/data/"
cpath="catalogues/AHF/"+Code+"/"
#groupinfo=np.loadtxt("/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/G3X_Mass_snap_128-center-cluster.txt")
#regionID hID  Mvir(4) Xc(6)   Yc(7)   Zc(8)  Rvir(12) fMhires(38) cNFW (42) M500 R500 fgas f*
Simun = "simulation/"+Code+"/"
#sname = ["snap_128","snap_127","snap_123","snap_120","snap_116","snap_111","snap_108","snap_105","snap_101","snap_099","snap_096"]
# sname = ["snap_128"]
Outdir = './'
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

Use_Lra = True  #use less rotations

#stn = np.int32(sys.argv[1])
#edn = np.int32(sys.argv[2])
# pregenIDs=np.loadtxt('/home2/weiguang/Project-300-Clusters/MAH/G3X-Halo-Progenitor-IDs-for-M200-gt-11.8.txt',dtype=np.int64)

# if Use_Lra:
#     RAs = np.loadtxt('/home2/weiguang/data/SZ/ML/Less_rotations.txt',dtype=np.int32)
# else:
#     RAs = np.loadtxt('/home2/weiguang/data/SZ/ML/More_rotations.txt',dtype=np.int32)

RAs = np.loadtxt('29_rotations.txt',dtype=np.float64)
stn = np.int32(sys.argv[1])
edn = np.int32(sys.argv[2])

selecth = np.load('/home2/weiguang/Project-300-Clusters/ML/Reselected_all_halos.npy')
#HID M200 Rid

print(yt.__version__, pyxsim.__version__)
Myxdef = ("Coordinates",
          "Velocities",
          "ParticleIDs",
          "Mass",
          ("InternalEnergy", "Gas"),
          ("Density", "Gas"),
          # Need to include this (NE block in simulation snapshot) for
          # corrected temperature field.  !!!Need to change mu for this simulation!!!
          ("ElectronAbundance", "Gas"),
          ("NeutralHydrogenAbundance", "Gas"),
          ("SmoothingLength", "Gas"),
          ("StarFomationRate", "Gas"),
          ("Age", ("Stars", "Bndry")),
          ("Z", ("Gas", "Stars")),
          ("Temperature", "Gas"),
          )
Mymdef = ("Coordinates",
          "Velocities",
          "ParticleIDs",
          "Mass",
          ("InternalEnergy", "Gas"),
          ("Density", "Gas"),
          # Need to include this (NE block in simulation snapshot) for
          # corrected temperature field.  !!!Need to change mu for this simulation!!!
          ("ElectronAbundance", "Gas"),
          ("NeutralHydrogenAbundance", "Gas"),
          ("SmoothingLength", "Gas"),
          ("StarFomationRate", "Gas"),
          ("Age", "Stars"),
          ("Z", ("Gas", "Stars")),
          # ("Temperature", "Gas"),
          )
gadget_field_specs["my_def"] = Myxdef

def calc_z(npx,ar,z,rr): #return redshift by pixels and angular resolution
    #set up cosmology
    cosmo= FlatLambdaCDM(67.77,0.307115)
    zr=np.arange(1.0e-6,0.12,1.0e-6)[::-1]
    apkpz=cosmo.arcsec_per_kpc_proper(zr).value
    rrp=rr*2/0.6777/(1+z)
    return np.interp(npx, rrp/ar*apkpz, zr)

# for lp in np.arange(46,groupinfo.shape[0]):

for lp in np.arange(stn,edn): # lp = 1
    clnum = '0000'+str(lp)
    clnum = clnum[-4:]
    cname = "NewMDCLUSTER_"+clnum+"/"
    print(lp, "NewMDCLUSTER_",clnum)
    
    # Check outputs
    outcat = cname + "/"
    if not os.path.exists(outcat):
        os.mkdir(outcat)

    idr = np.where(np.int32(selecth[:,2]+0.1)==lp)[0]
    if len(idr)<1:
        raise ValueError('No regions find in selected halo',lp)
    
    Hids = np.int64(selecth[idr,0]+0.1)    #AHF halo IDs
    sn = np.array([np.int32(str(i)[:3]) for i in Hids])
    idshid = np.argsort(Hids)
    Hids = Hids[idshid]; sn=sn[idshid]; idr=idr[idshid]
    st = 0
    
    for j, s, hid in zip(idr, sn , Hids):
        
        snapname = 'snap_'+str(s)
    #     if os.path.isfile(outcat + snapname + "-Athena-wfi-cl-" + str(hid) + "-ra-120-120-120.fits"):
    #         continue
        if st!=s:
            st=s
            if ((lp == 10) and (snapname == 'snap_100')) or ((lp == 228) and (snapname == 'snap_110')):
                continue

            # load simulation data
            ds = yt.load(path+Simun+cname+snapname, field_spec="my_def")  # read snapshots
            if ds.current_redshift < 1.0e-9:
                redshift = 0.000
            else:
                redshift = ds.current_redshift

            halo = np.load('/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GadgetX/G3X_Mass_snap_'+str(s)+'info.npy')
            ##ReginIDs HIDs  HosthaloID Mvir(4) Xc(6)   Yc(7)   Zc(8)  Rvir(12) fMhires(38) cNFW (42) Mgas200 M*200 M500  R500 fgas500 f*500
        idg=np.where((halo[:,0]==lp) & (halo[:,1]==hid))[0]
        if len(idg) == 1:
            cc=halo[idg[0],4:7]; rr = halo[idg[0],7]
        else:
            raise ValueError('Halo not find.... ',lp,hid)


        #sp = ds.box(cc-np.sqrt(2)*rr, cc+np.sqrt(2)*rr)  # increase 40% to match SZ images # select the cluster
        sp = ds.sphere(cc, (np.sqrt(2)*rr,'kpc/h'))
        ct = sp.cut_region(["obj[('Gas', 'StarFomationRate')] < 0.1"])  # Needed for G3X!!

        eff_A = 20000.                                    # CCD effective area


        cosmo = Cosmology(hubble_constant=ds.hubble_constant, omega_matter=ds.omega_matter, omega_lambda=ds.omega_lambda)
        angdd = cosmo.angular_diameter_distance(0, redshift).in_units("kpc")
        fixps = 640. # number of pixels per image
        angular = 5 # arcsec
        kpcparc = angdd.tolist()*np.pi/180/3600.
    #         ar_at = fixps/kpcparc; psf_at = 5
    #             pxln_xm = np.int32(rr*2.8*ds.scale_factor/(ar_xm*kpcparc))+1  #into pixel size of 10 kpc
        pxln_at = np.int32(rr/ds.hubble_constant/(1.+redshift)/(angular*kpcparc))+1  # must use physical units

        outred = calc_z(fixps, angular, redshift, rr)
        # print(outred)
        exp_tn = 100000 # at z=0.1 with total pixel size of 640
        # rescale the expostion time
        kpcparc1 = cosmo.angular_diameter_distance(0,0.1).in_units("kpc").tolist()*np.pi/180/3600.
        kpcparcx = cosmo.angular_diameter_distance(0, outred).in_units("kpc").tolist()*np.pi/180/3600.
        exp_tn *= (kpcparcx/kpcparc1)**2

        etname = np.str("%3.1f" % (exp_tn / 1000.)) + 'k'

        # resectied to XMM Newton telescope
        emin = 0.1 # 0.01     # The minimum energy of the spectrum in keV
        emax = 15   # 20.0    # The maximum energy of the spectrum in keV
        nchan = 10000   # The number of spectral channels
        apec_vers = "3.0.7"  # The version identifier string for the APEC files. Default: "2.0.2"
        apec_root = "/home2/weiguang/.local/spectral/modelData/"
        # The directory where the APEC model files are stored.
        # Optional, the native pyxsim files will be used if a location is not
        # provided.
        apec_model = pyxsim.TableApecModel(emin, emax, nchan, apec_root=apec_root,
                                           apec_vers=apec_vers, thermal_broad=False)

        T_apec = pyxsim.ThermalSourceModel(apec_model, Zmet=("Gas", "Z"), kT_min=emin,
                                           kT_max=emax, n_kT=nchan, method='invert_cdf')

        photons = pyxsim.PhotonList.from_data_source(ct, outred, eff_A, exp_tn, T_apec)

        # -----------------------------------------------------------
        # Description of the Photon field:
        # number_of_photons -> number of photons (subgrid) per particle, scalar   
        # energy -> energy per particle, scalar
        # x,y,z: coordinates of the particles, vector
        # vx,vy,vz : velocities of the particles
        # dx -> do not know what this is, but it is scalar 
        # -----------------------------------------------------------

        #------------------------------------------------------------
        # So we only need to rotate the vector fields and select the
        # particles inside the R_200 box. Then, overwrite the photon 
        # field to produce the desire projection
        #------------------------------------------------------------

        # original photon field in memory
        photons.__dict__['photons']['Energy'] = photons.__dict__['photons']['Energy'].to_ndarray() # the energy to numpy


        number_of_photons = photons['NumberOfPhotons']
        energy = np.array(photons['Energy'])
        x = photons['x']
        y = photons['y']
        z = photons['z']
        vx = photons['vx']
        vy = photons['vy']
        vz = photons['vz']
        dx = photons['dx']

        num_cells = x.shape[0] 
        pos = np.empty((num_cells,3))
        vel = np.empty((num_cells,3))

        pos[:,0] = x.to_ndarray()
        pos[:,1] = y.to_ndarray()
        pos[:,2] = z.to_ndarray()

        vel[:,0] = vx.to_ndarray()
        vel[:,1] = vy.to_ndarray()
        vel[:,2] = vz.to_ndarray()

        #RAs = [[0,0,0]]
        ra = 0
        for RA in RAs:

            print('rotation =', RA)
            R = rotation_matrix(RA)

            new_pos = np.dot(pos,R)
            new_vel =  np.dot(vel,R)


            #select the photons that are inside the R_200 box size
            h = 0.677
            rr2 =rr/h # in kpc
            cut_inside = np.where((new_pos[:,0]<=rr2)&(new_pos[:,0]>=-rr2)&
                        (new_pos[:,1]<=rr2)&(new_pos[:,1]>=-rr2)&
                        (new_pos[:,2]<=rr2)&(new_pos[:,2]>=-rr2))[0]


            # update photon field
            new_number_of_photons = number_of_photons[cut_inside]

    #         p_bins = np.empty(len(new_number_of_photons+1,))
    #         for index in range(0,len(p_bins)):
    #             p_bins[index] = np.sum(new_number_of_photons[:index])

            p_bins = [0]
            indx = 0

            for numberp in new_number_of_photons:
                p_bins.append(numberp+p_bins[indx])
                indx+=1

            p_bins = np.array(p_bins)

            photons.__dict__['p_bins'] = p_bins



            new_energy = energy[cut_inside]
            new_energy = np.concatenate(new_energy)
            new_energy = yt.units.yt_array.YTArray(new_energy, input_units='keV')

            new_pos = new_pos[cut_inside]
            new_x = yt.units.yt_array.YTArray(new_pos[:,0], input_units='kpc')
            new_y = yt.units.yt_array.YTArray(new_pos[:,1], input_units='kpc')
            new_z = yt.units.yt_array.YTArray(new_pos[:,2], input_units='kpc')

            new_num_cells = new_x.shape[0]

            new_vel = new_vel[cut_inside]
            new_vx = yt.units.yt_array.YTArray(new_vel[:,0], input_units='km/s')
            new_vy = yt.units.yt_array.YTArray(new_vel[:,1], input_units='km/s')
            new_vz = yt.units.yt_array.YTArray(new_vel[:,2], input_units='km/s')
            new_dx = dx[cut_inside]


            photons.__dict__['num_cells'] = new_num_cells

            photons.__dict__['photons']['NumberOfPhotons'] = new_number_of_photons 
            photons.__dict__['photons']['Energy'] = new_energy

            photons.__dict__['photons']['x'] = new_x
            photons.__dict__['photons']['y'] = new_y
            photons.__dict__['photons']['z'] = new_z

            photons.__dict__['photons']['vx'] = new_vx
            photons.__dict__['photons']['vy'] = new_vy
            photons.__dict__['photons']['vz'] = new_vz

            photons.__dict__['photons']['dx'] = new_dx

            N_H = 0.1  # galactic column density in units of 10^{22} cm^{-2}
            # forground galactic absorption model TB
            abs_TBabs = pyxsim.TBabsModel(N_H)
            # abs_wabs = pyxsim.WabsModel(N_H)

            # XMM-Newton angular resolution 6" FWHM; 15" HEW; PSF pn 16.6 MOS1 16.8 MOS2 17.0 (HEW)
            # Athena AR 5" HEW half energy width; 5" on-axis PSF 
            # ACIS AR 0.4920 arcsec; PSF FWHM (with detector) 0.5 arcsec
            #             ar_xm, psf_xm = 5.252134578, 16.8
            #             ar_at, psf_at = 5.252134578, 5
            #             ar_ac, psf_ac = 0.492, 0.5

            #             pxln_ac = np.int32(rr*2.8*ds.scale_factor/(ar_ac*kpcparc))+1
            #             print(pxln_xm,pxln_at,pxln_ac)
            Athena = pyxsim.InstrumentSimulator(angular/3600., fixps, angular/3600.,                                            "athena_wfi_1469_onaxis_w_filter_v20150326.arf",
                                                "athena_wfi_rmf_v20150326.rmf")
            #ACISS = pyxsim.InstrumentSimulator(ar_ac/3600., pxln_ac, psf_ac/3600.,
            #                                   "aciss_aimpt_cy19.arf", "aciss_aimpt_cy19.rmf")
            #             XMM_mos1 = pyxsim.InstrumentSimulator(ar_xm/3600., pxln_xm, psf_xm/3600.,
            #                                                   "/home2/weiguang/.local/Xray-Auxiliary/Xcop/mos1S001-a2029_reg1.arf", "/home2/weiguang/.local/Xray-Auxiliary/Xcop/mos1S001-a2029_reg1.rmf")
            # XMM_mos2 = pyxsim.InstrumentSimulator(ar_xm/3600., pxln_xm, psf_xm/3600.,
            #                                       "mos2S002-a2029_reg1.arf", "mos2S002-a2029_reg1.rmf")
            # XMM_pn = pyxsim.InstrumentSimulator(ar_xm/3600., pxln_xm, psf_xm/3600.,
            #                                     "pnS003-a2029_reg1.arf", "pnS003-a2029_reg1.rmf")
            print('projecting photons')

            events = photons.project_photons(normal='z') #, absorb_model=abs_TBabs, sky_center=[194.95, 27.98])
             #, absorb_model=abs_TBabs, sky_center=[194.95, 27.98]) #only for y-axis, because it is projection to z-x plane instead of x-z
            #         print("projecting photons to %s" % pd)

            print('Convolve Events with Instrumental Responses')
            athena_wfi = Athena(events)

            athena_wfi.write_fits_image(outcat + snapname + "-Athena-wfi-cl-" + str(hid) + "-ra-" + str(ra) +".fits", 
                                radius=rr/ds.hubble_constant/(1.+redshift), redshift=outred, 
                                comments=("Exposure time: " + etname,
                                      "Simulation Region: " + clnum,
                                      "AHF Halo ID: "+str(hid), 
                                      "Simulation redshift: " + str(redshift)[:6],        
                                      "log M_200 = "+str(np.log10(halo[idg[0],3]))[:6]+" Msun/h",
                                      "R_200 = "+str(rr)[:6]+" kpc/h"), overwrite=True)
            ra+=1