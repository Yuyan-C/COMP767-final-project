import xarray as xr
import numpy as np


def prepare_dataset(minx, miny, maxx, maxy, data_path):
    ds = xr.open_dataset(data_path)
    dsc = ds.rio.write_crs(4326)
    region = dsc.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    padded = dsc.rio.clip_box(minx=minx - 0.5, miny=miny - 0.5, maxx=maxx + 0.5, maxy=maxy + 0.5)
    features_to_keep = ['lst_day', 'ndvi', 'rel_hum', 'ssrd', 't2m_min', 'tp', 'vpd']
    padded = padded[features_to_keep]
    lat, lon, _ = region.indexes.values()
    lat_padded, lon_padded, _ = padded.indexes.values()
    y = region["fcci_ba"]
    X = [get_data_cube(padded, i, j) for i in lat for j in lon]
    X = np.array(X)
    ds.close()
    return X, y


def get_data_cube(ds, latitude, longitude):
    lon_range = [longitude - 0.5000001, longitude + 0.50]
    lat_range = [latitude - 0.50, latitude + 0.5000001]
    dsc = ds.rio.write_crs(4326)
    med_ds = dsc.rio.clip_box(minx=lon_range[0], miny=lat_range[0], maxx=lon_range[1], maxy=lat_range[1])
    med_ds = med_ds.to_array()
    return med_ds


#
# X, y = prepare_dataset(minx=16.125, miny=0.125, maxx=20.875, maxy=2.875, data_path='seasfire.zarr')
#
# np.save("forest_X.npy", X)
# np.save("forest_y.npy", y)


X = np.load("forest_X.npy")
y = np.load("forest_y.npy")
X = X.reshape((X.shape[0], X.shape[2], X.shape[1], -1))
y = y.reshape((y.shape[0], -1)).T


def reshape_dataset(X_new, y_new):
    X_new = X_new.reshape((X_new.shape[0] * X_new.shape[1], X_new.shape[2], X_new.shape[3], X_new.shape[4]))
    y_new = y_new.reshape((y_new.shape[0] * y_new.shape[1],))
    return X_new, y_new


def stack_months(X, y, months=3):
    X, y = prepare_dataset(minx=16.125, miny=0.125, maxx=20.875, maxy=2.875, data_path='seasfire.zarr')
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1], -1))
    y = y.reshape((y.shape[0], -1)).T
    X_new = np.array([np.stack([patch[i - months:i] for i in range(months, len(patch))]) for patch in X])
    y_new = y[:, 3:]

    X_train = X_new[:, :676, :, :]
    y_train = y_new[:, :676]

    X_val = X_new[:, 676:773, :, :]
    y_val = y_new[:, 676:773]

    X_test = X_new[:, 773:, :, :]
    y_test = y_new[:, 773:]

    X_train, y_train = reshape_dataset(X_train, y_train)
    X_val, y_val = reshape_dataset(X_val, y_val)
    X_test, y_test = reshape_dataset(X_test, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test




