# %%
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import xarray as xr
import pandas as pd
from mpl_toolkits.basemap import Basemap
import sys
# %%
def slice(latbounds,lonbounds,lat,lon,var):
    lat_slice0 = np.where(lat>latbounds[0],True,False)
    lat_slice = np.where(lat<latbounds[1],lat_slice0,False)

    lon_slice0 = np.where(lon>lonbounds[0],lat_slice,False)
    lon_slice = np.where(lon<lonbounds[1],lon_slice0,False)

    AOD_sliced = np.where(lon_slice == True,var,0.5)

    return AOD_sliced

def get_goes_data(yy,mm,dd,tt):

    j_day = (datetime(int(yy), int(mm), int(dd)) - datetime(int(yy),1,1)).days + 1
    new_path = glob.glob(f'/Users/jmceachern/data/noaa-goes18/ABI-L2-AODF/{yy}/{j_day}/OR_ABI-L2-AODF-M6_G18_s{yy}{j_day}{tt}*.nc')[0]
    ds = xr.open_dataset(new_path)
    coords = xr.open_dataset('goes18_abi_full_disk_lat_lon.nc')

    return ds, coords['latitude'], coords['longitude']

def plot_AOD(ds,AOD):
    H = ds['goes_imager_projection'].perspective_point_height + ds['goes_imager_projection'].semi_major_axis
    data_proj = crs.Geostationary(central_longitude=-137, satellite_height=H, false_easting=0, false_northing=0, globe=None)
    desired_proj = crs.PlateCarree(central_longitude=-110)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(1, 1, 1, projection=desired_proj)

    levels = np.linspace(0,1,10)
    norm = matplotlib.colors.BoundaryNorm(levels,len(levels))
    colors = list(plt.cm.Greys(np.linspace(0,1,len(levels)-1)))
    # colors[0] = "w"
    # colors[1] = "dodgerblue"
    # colors[2] = "mediumseagreen"
    # colors[3] = "seagreen"
    cmap = matplotlib.colors.ListedColormap(colors,"", len(colors))

    AOD_ones = np.clip(np.array(AOD),0,1)
    im = ax.imshow(AOD_ones, origin='upper', cmap=cmap,
            transform=data_proj) # the datas projection
    ax.coastlines('50m', linewidth=0.8)
    # ax.set_extent([lonbounds[0],lonbounds[1],latbounds[0],latbounds[1]])
    ax.add_feature(crs.cartopy.feature.BORDERS, linewidth=1)
    fig.colorbar(im,ticks=levels,shrink=0.15)
    return

def cfcompliant(path):
    """
    Converts Hysplit dataset to be cf compliant.
    Also, reformats julian datatime to standard datetime and creates LAT LONG arrays of the model domain.
    """
    ## open dataset
    ds = Dataset(path)
    PM25 = ds.variables['PM25'][:]
    tflag = ds.variables['TFLAG'][:]
    yyi, jdi, hhi = str(tflag[:,0,0][0])[0:4], str(tflag[:,0,0][0])[4:], str(tflag[:,0,1][0])[0:2]
    yyf, jdf, hhf = str(tflag[:,0,0][-1])[0:4], str(tflag[:,0,0][-1])[4:], str(tflag[:,0,1][-1])[0:2]
    if len(str(tflag[:,0,1][0])) == 5:
        hhi = f'0{str(tflag[:,0,1][0])[0:1]}'
    start_t = datetime.strptime(yyi+jdi+hhi,'%Y%j%H')
    end_t = datetime.strptime(yyf+jdf+hhf,'%Y%j%H')
    date_range = pd.date_range(start_t, end_t, freq="1H")

    ## get x coordinate dimensions and create an array
    ds_xr = xr.open_dataset(path)
    xnum = ds_xr.dims["COL"]
    dx = ds_xr.attrs["XCELL"]
    xorig = ds_xr.attrs["XORIG"]
    x = np.arange(0, xnum)

    ## get y coordinate dimensions and create an array
    ynum = ds_xr.dims["ROW"]
    dy = ds_xr.attrs["YCELL"]
    yorig = ds_xr.attrs["YORIG"]
    y = np.arange(0, ynum)

    ## create LAT and LONG 2D arrays based on the x and y coordinates
    X = np.arange(0, xnum) * dx + xorig
    Y = np.arange(0, ynum) * dy + yorig
    XX, YY = np.meshgrid(X, Y)
        
    ## get z coordinate dimensions and create an array
    Z = np.array(ds_xr.attrs["VGLVLS"][:-1])
    z = np.arange(0, len(Z))

    ## create new dataset a set up to be CF Compliant
    ds_cf = xr.Dataset(
                    data_vars=dict(
                                    pm25=(["time", "z", "y", "x"], PM25.astype("float32")),
                                    x=(["x"], x.astype("int32")),
                                    y=(["y"], y.astype("int32")),
                                    z=(["z"], z.astype("int32")),
                                    Times=(["time"], date_range.astype("S19")),
                                    ),
                    coords=dict(
                                LONG=(["y", "x"], XX.astype("float32")),
                                LAT=(["y", "x"], YY.astype("float32")),
                                HEIGHT=(["z"], Z.astype("float32")),
                                time=date_range,
                                ),
                    attrs=dict(description="BlueSky Canada PM25 Forecast"),
                    )

    ## add axis attributes from cf compliance
    ds_cf["time"].attrs["axis"] = "Time"
    ds_cf["x"].attrs["axis"] = "X"
    ds_cf["y"].attrs["axis"] = "Y"
    ds_cf["z"].attrs["axis"] = "Z"
        
    ## add units attributes from cf compliance
    ds_cf["pm25"].attrs["units"] = "um m^-3"
    ds_cf["LONG"].attrs["units"] = "degree_east"
    ds_cf["LAT"].attrs["units"] = "degree_north"
                        
    return ds_cf

# %% GOES18 obs. 
yy, mm, dd, tt = '2023', '07', '08', '19'
ds_obs, lat, lon = get_goes_data(yy,mm,dd,tt) 
AOD = ds_obs['AOD'] # direct AOD observations from the GOES18 satellite before being regridded 
plot_AOD(ds_obs,AOD)

# %% add the lats and lons from the NOAA file to a new file called GOES-18-coords
ds_new = xr.DataArray(AOD,
                    coords= [ds_obs['y'],ds_obs['x']],
                    dims=['y','x']
                    )
ds_new = ds_new.to_dataset(name='aod')
ds_new['lat'] = lat
ds_new['lon'] = lon
ds_new.to_netcdf("/Users/jmceachern/regrid/GOES-18-coords2.nc") 

# %% Pipeline output
path = "/Users/jmceachern/bsp-data/hysplit_conc.nc"
ds_xr = xr.open_dataset(path) # unmodified output from pipeline (useful because it has info that can be used for projection)

ds_hy = cfcompliant(path) # makes pipeline output cf compliant (adds lats and lons)
pm25 = ds_hy.variables['pm25'][-1,0,:,:] # pm2.5 data for last hour, time dim = 1st dim
lats = ds_hy.variables['LAT']
lons = ds_hy.variables['LONG']

# %% Regridded GOES data 
path = "/Users/jmceachern/regrid/final_remapnn.nc"
ds_obsf = Dataset(path)
aod = ds_obsf['aod']
aod0 = np.where(aod[:]==65535.0,0,aod)
aod1 = np.where(aod[:]==65533.0,0,aod0)
AOD_ones = np.clip(np.array(aod1),0,1)

# %%
path = "/Users/jmceachern/bsp-data/hysplit_conc.nc"
ds_xr = xr.open_dataset(path)
xcent = ds_xr.attrs["XCENT"] # central longitude
ycent = ds_xr.attrs["YCENT"] # central latitude
ymin = ds_xr.attrs['YORIG'] # minimum longitude??
ymax = np.max(lats)

# data_proj = crs.Stereographic(central_longitude=xcent, central_latitude=ycent, globe=None)
data_proj = crs.Mercator(central_longitude=xcent, min_latitude=ymin, max_latitude=ymax, globe=None, latitude_true_scale=None, false_easting=0.0, false_northing=0.0, scale_factor=None)
desired_proj = crs.PlateCarree(central_longitude=xcent, globe=None)

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1, 1, 1, projection=desired_proj)

levels = np.linspace(0,1,10)
norm = matplotlib.colors.BoundaryNorm(levels,len(levels))
colors = list(plt.cm.Greys(np.linspace(0,1,len(levels)-1)))
cmap = matplotlib.colors.ListedColormap(colors,"", len(colors))

im = ax.imshow(AOD_ones, origin='upper', cmap=cmap,
        transform=data_proj) # the datas projection
ax.coastlines('50m', linewidth=0.8)
# ax.set_extent([lonbounds[0],lonbounds[1],latbounds[0],latbounds[1]])
ax.add_feature(crs.cartopy.feature.BORDERS, linewidth=1)
fig.colorbar(im,ticks=levels,shrink=0.15)
# %%
