# redshift 0 
import numpy as np
import pandas as pd
from tqdm import tqdm

# redshift 0 
snaptoz = np.loadtxt('/data4/niftydata/TheThreeHundred/data/snapidzred.txt')
columns = ['ID(1)',	'hostHalo(2)',	'numSubStruct(3)','Mvir(4)', 'npart(5)', 'Xc(6)','Yc(7)', 'Zc(8)', 'VXc(9)',
           'VYc(10)','VZc(11)',	'Rvir(12)',	'Rmax(13)',	'r2(14)', 'mbp_offset(15)',	
           'com_offset(16)', 'Vmax(17)', 'v_esc(18)','sigV(19)'	'lambda(20)',
           'lambdaE(21)', 'Lx(22)',	'Ly(23)', 'Lz(24)',	'b(25)','c(26)','Eax(27)','Eay(28)',
            'Eaz(29)','Ebx(30)','Eby(31)',	'Ebz(32)','Ecx(33)','Ecy(34)','Ecz(35)','ovdens(36)','nbins(37)',
           'fMhires(38)','Ekin(39)','Epot(40)',	'SurfP(41)','Phi0(42)',
            'cNFW(43)', 'n_gas(44)','M_gas(45)','lambda_gas(46)', 'lambdaE_gas(47)',
            'Lx_gas(48)','Ly_gas(49)','Lz_gas(50)',	'b_gas(51)', 'c_gas(52)',
            'Eax_gas(53)','Eay_gas(54)','Eaz_gas(55)','Ebx_gas(56)','Eby_gas(57)',
            'Ebz_gas(58)','Ecx_gas(59)','Ecy_gas(60)','Ecz_gas(61)','Ekin_gas',
            '(62)',	'Epot_gas(63)',	'n_star(64)',	'M_star(65)',	'lambda_star(66)',	'lambdaE_star(67)'	,
           'Lx_star(68)',	'Ly_star(69)'	,'Lz_star(70)'	,'b_star(71)'	,
            'c_star(72)',	'Eax_star(73)',	'Eay_star(74)'	,'Eaz_star(75)',	'Ebx_star(76)'	,
            'Eby_star(77)'	,'Ebz_star(78)'	,'Ecx_star(79)',	'Ecy_star(80)'	,'Ecz_star(81)'	,
            'Ekin_star(82)'	,'Epot_star(83)',	'mean_z_gas(84)'	,'mean_z_star(85)'	,'n_star_excised(86)',
            'M_star_excised(87)',	'mean_z_star_excised(88)'	]


for snap in range(95,129):
    
    df = pd.DataFrame()
    print('SNAP:',snap)
    if snap==128:
        z = '0.000'
    else:
        z = str(round(snaptoz[snap,1],3))
        if len(z)==4:
            z = 'z'.ljust(5, '0')
    
    for i in tqdm(range(1,324)):
        cnum = ('%i'%i).zfill(4)
        file = '/data4/niftydata/TheThreeHundred/data/catalogues/AHF/GIZMO_R200c/NewMDCLUSTER_{}/GIZMO-NewMDCLUSTER_{}.snap_{}.z{}.AHF_halos'.format(cnum,cnum,str(snap).zfill(3),z)
        data = np.loadtxt(file)
        df = pd.concat((df,pd.DataFrame(data, columns = columns)))

    df['ID(1)'] = np.int64(df['ID(1)'])
    df['hostHalo(2)'] = np.int64(df['hostHalo(2)'])
    df['redshift'] = z    
    df.to_csv('/home2/deandres/clusters2dmaps/dataframes/try_{}.cvs'.format(snap))