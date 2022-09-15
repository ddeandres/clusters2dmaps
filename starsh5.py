# todos:
# make this script to work in parallel, so that we do not have to wait 1h every time we use it. However, the code itself uses ~50 CPUs.

import numpy as np
from astropy.io import fits
#import cv2
import h5py
from tqdm import tqdm
import scipy.ndimage



path = "/data7/users/deandres/newML2/"

RAs = np.arange(0,29)


def read_total_mass(lp,hid,RA):
    region = 'total-mass/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-Mtotal-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    return data


def read_xr(lp,hid,RA):
    region = 'X-ray/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-Athena-wfi-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    return data

def read_sz(lp,hid,RA):
    region = 'SZ/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-TT-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    return data

def read_dm(lp,hid,RA):
    region = 'DM/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-DM-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    return data

def read_star(lp,hid,RA):
    region = 'star/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-Mstar-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    return data

def get_M2(lp,hid,RA):
    region = 'DM/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-DM-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    hdul = fits.open(path+region+file)
    M = np.float(hdul[0].header[-2][12:18])
    return M

def get_M3D(lp,hid,RA):
    region = 'total-mass/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-Mtotal-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    hdul = fits.open(path+region+file)
    header = hdul[0].header
    return float(header[-1][-8:])

def get_MCIL(lp,hid,RA):
    region = 'total-mass/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-Mtotal-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    hdul = fits.open(path+region+file)
    header = hdul[0].header
    return float(header[-2][-8:])

#-----------------------------------------------------------
# COMPUTE THE CHI PARAMETER
#-----------------------------------------------------------

def chi(delta,fs):
    return np.sqrt(2/((delta/0.1)**2+(fs/0.1)**2))


#-----------------------------------------------------------
# REBINING FUNCTION
#-----------------------------------------------------------

def rebin(a, shape):
    # it works only when a.shape = n*shape, here n is a positive integer.
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)


#-----------------------------------------------------------
# IMPORTANT PARAMETERS
#-----------------------------------------------------------

kernel_size = 0
new_resolution = int(320)


##### load dataset
params = np.loadtxt('/home2/weiguang/Project-300-Clusters/ML/DS_reselected_all_halos.txt')
colums = 'rID[0] Hid[1]  R200: DS[2], eta[3], detla[4], fm[5], fm2[6];   R500: DS[7], eta[8], detla[9], fm[10], fm2[11]'

##### set the important parameters for R=R_{200}
eta = params[:,3]
delta = params[:,4]
fs1 = params[:,5] ## the sum of all substructure mass fraction
fs2 = params[:,6] ## only for the most massive substructure
#### for R = R_500
eta500 = params[:,8]
delta500 = params[:,9]
fs500 = params[:,10]


chi_parameter = chi(delta,fs1)



images_xr = []
images_sz = []
images_dm = []
images_star = []
images_mass = []
M_200 = []
M_3D = []
M_CIL = []
region_list = []
hids_list = []
selecth = np.load('/home2/weiguang/Project-300-Clusters/ML/Reselected_all_halos.npy')



fwhm2sig = 1./(2.*np.sqrt(2.*np.log(2.)))


for lp in tqdm(range(1,325)):
#     print('reading data region =' , lp)
    idr = np.where(np.int32(selecth[:,2]+0.1)==lp)[0]
    if len(idr)<1:
        raise ValueError('No regions find in selected halo',lp)

    Hids = np.int64(selecth[idr,0]+0.1)    #AHF halo IDs
    sn = np.array([np.int32(str(i)[:3]) for i in Hids])
    idshid = np.argsort(Hids)
    Hids = Hids[idshid]; sn=sn[idshid]; idr=idr[idshid]
    st = 0
    
    for hid in Hids:
        
        
        for RA in RAs:
            #print(lp,hid,RA)
            img_star = read_star(lp,hid,RA)*1e10 # correct units Msun/h

            
            # these resize function should be modified by a rebining function for obvious reasons. We dont need the
            # mean values but rather, the total sum of the emmision over the beam.
            img_star = scipy.ndimage.filters.gaussian_filter(img_star,kernel_size*fwhm2sig)

            resized_star = rebin(img_star,(new_resolution,new_resolution))

            images_star.append(resized_star)
            
            #end RAs loop
        M_2 = get_M2(lp,hid,RA)   
        M_C = get_MCIL(lp,hid,RA)
        M_3 = get_M3D(lp,hid,RA)
        M_200.append(M_2) 
        M_3D.append(M_3)
        M_CIL.append(M_C)
        region_list.append(lp)
        hids_list.append(hid)
        #end Hids loop        
    #end region loop




images_star = np.array(images_star)


M_200 = np.array(M_200)
M_3D = np.array(M_3D)
M_CIL = np.array(M_CIL)
region_list = np.array(region_list)
hids_list = np.array(hids_list)

images_star = images_star.reshape((2580,29,new_resolution,new_resolution))


print('creating data set that consists of the following keys...')

print('shape M_200: ',M_200.shape)
print('shape region: ',region_list.shape)
print('shape hid: ',hids_list.shape)
print('integration factor ',(640/new_resolution)**2)


h5_path = path + "h5files/"
df = h5py.File(h5_path+'SmoothStars_reso{}_kernel{}.h5'.format(new_resolution,kernel_size), 'w')

df.create_dataset('star',data = images_star)
df.create_dataset('M_200',data = M_200)
df.create_dataset('region',data = region_list)
df.create_dataset('hid',data=hids_list)


df.close()