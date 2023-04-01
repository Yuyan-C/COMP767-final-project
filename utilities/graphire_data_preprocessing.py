import xarray as xr

def get_data_cube(longitude = -119.454, latitute=50.33, data_path = 'Data/seasfire.zarr'):
    lon_range = [longitude-0.5000001, longitude+0.50]
    lat_range = [latitute-0.50, latitute+0.5000001]

    ds = xr.open_dataset(data_path)
    
    dsc = ds.rio.write_crs(4326)
    med_ds = dsc.rio.clip_box(minx=lon_range[0] , miny= lat_range[0], maxx= lon_range[1],maxy= lat_range[1])
    features_to_keep = ['lst_day', 'ndvi', 'rel_hum', 'ssrd', 'sst', 't2m_min', 'tp', 'vpd', 'fcci_ba']
    med_ds = med_ds[features_to_keep]
    ds.close()

    return med_ds

