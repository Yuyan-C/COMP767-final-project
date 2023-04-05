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
    y = region["fcci_ba"][:920, :, :] # get rid of nan

    # fill nan for X
    values = {"lst_day": 300, "ndvi": 0.7}
    padded = padded.fillna(value=values)

    # get neighborhood
    X = [get_data_cube(padded, i, j) for i in lat for j in lon]
    X = np.array(X)
    X = X[:, :, :920, :, :]
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
# np.save("newX.npy", X)
# np.save("newY.npy", y)


X = np.load("newX.npy")
y = np.load("newY.npy")


# print(y.shape)


def reshape_dataset(X_new, y_new):
    X_new = X_new.reshape((X_new.shape[0] * X_new.shape[1], X_new.shape[2], X_new.shape[3], X_new.shape[4]))
    y_new = y_new.reshape((y_new.shape[0] * y_new.shape[1],))
    return X_new, y_new



def stack_weeks(X, y, weeks=3):
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1], -1)) # (240, 920, 7, 25)
    y = y.reshape((y.shape[0], -1))
    y = y.T # shape: (240, 920)

    X_new = np.array([np.stack([patch[i - weeks:i] for i in range(weeks, len(patch))]) for patch in X])
    y_new = y[:, 3:]

    # shape (240, 917, 3, 7, 25) (240, 917)

    X_train = X_new[:, :644, :, :]
    y_train = y_new[:, :644]

    X_val = X_new[:, 644:736, :, :]
    y_val = y_new[:, 644:736]

    X_test = X_new[:, 736:, :, :]
    y_test = y_new[:, 736:]

    X_train, y_train = reshape_dataset(X_train, y_train)
    X_val, y_val = reshape_dataset(X_val, y_val)
    X_test, y_test = reshape_dataset(X_test, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

data = stack_weeks(X, y, weeks=3)
fns = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
for i in range(6):
    np.save(f"{fns[i]}.npy", data[i])

