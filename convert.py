import healpy as hp
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from pyproj import CRS, Transformer
from scipy.interpolate import griddata
import time

from download import Tile


NSIDE = 262144
DLAT = 15e-5
DLON = 24e-5

def add_lat_lon(chunk: xr.Dataset, dtype=np.float32):
    """
    Takes a xr.Dataset with coordinates in (x [m], y [m]) and adds coordinates in (lon [deg], lat[deg])
    :param chunk:
    :param dtype: datatype of the coordinates
    :return:
    """
    tfm = Transformer.from_crs(CRS.from_wkt(
        chunk["crs"].attrs["spatial_ref"]),
        CRS.from_epsg(4326), always_xy=True
    )

    #make 2D projected coords
    x = chunk["x"].values
    y = chunk["y"].values
    X, Y = np.meshgrid(x, y)  # shapes (Ny, Nx)

    lon, lat = tfm.transform(X, Y) #project -> lon/lat

    return chunk.assign_coords({ #add new cords to Dataset
        'lon': (("y", "x"), lon.astype(dtype)),
        'lat': (("y", "x"), lat.astype(dtype)),
    })


def regular_grid_to_healpix(
    chunk: xr.Dataset,
    *,
    nside: int = NSIDE,
    nest: bool = True,
    lon_name: str = "lon",
    lat_name: str = "lat",
    xy_dims: tuple[str, str] = ("y", "x"),
):
    """
    Re-grids a Dataset onto Healpix Grid
    :param chunk: Dataset to be regridded
    :param nside: Healpix Resolution
    :param nest: Whether to use nested indices
    :param lon_name: name of the dimension representing longitude
    :param lat_name: name of the dimension represting latitude
    :param xy_dims: name of the x and y dimensions
    :return: Regridded Dataset
    """
    lon = chunk[lon_name].values
    lat = chunk[lat_name].values

    theta = (np.pi / 2) - np.deg2rad(lat) # caluclate rad of all lat/lon coordinates
    phi   = np.deg2rad(lon) % (2 * np.pi)
    ipix2d = hp.ang2pix(nside, theta, phi, nest=nest).astype(np.uint64) # get the corresponding healpix pixel for each coordinate
    yx = xy_dims
    xy = (xy_dims[1], xy_dims[0])
    data_vars = [v for v in chunk.data_vars if tuple(chunk[v].dims) in (yx, xy)]

    m = np.isfinite(theta) & np.isfinite(phi)

    ip = ipix2d[m].ravel() #flatten all coordinate pixels into array
    order = np.argsort(ip) # get order of the pixel ids
    ip_s = ip[order] #sorts the healpix pixel ids

    change = np.empty(ip_s.shape, dtype=bool)
    change[0] = True
    change[1:] = ip_s[1:] != ip_s[:-1] #change[n] is True if ip_s[n] != ip_s[n-1]
    idx = np.nonzero(change)[0] #starting indices of each healpix pixel

    uip = ip_s[idx] #unique pixel ids
    count = np.diff(np.append(idx, ip_s.size)).astype(np.uint16) #number of grid pixels for each healpix pixel

    def to_pix_nanmean(vname: str) -> np.ndarray:
        a = chunk[vname].values
        if tuple(chunk[vname].dims) == xy:
            a = a.T
        a = a[m].ravel()[order] #process each variable / band the same way the coordinates were processed to align them with their coordinates
        ok = np.isfinite(a) #only process if not nan or +- infinite is the value of the band
        s = np.add.reduceat(np.where(ok, a.astype(np.float64), 0.0), idx) # sum of all grid pixel values inside healpix pixel
        c = np.add.reduceat(ok.astype(np.uint32), idx) #number of valid grid pixel values inside healpix pixel
        out = (s / np.maximum(c, 1)).astype(np.float32) #compute mean
        out[c == 0] = np.nan #set nan if there aren't any valid grid pixels inside healpix pixel
        return out

    pix_vars = {v: (("pix",), to_pix_nanmean(v)) for v in data_vars}

    rb = np.empty(uip.shape, dtype=bool)        # the next few lines build a running index meaning instead of storing the
    rb[0] = True                                # healpix pix_id for each pixel we store how many consecutive pixels come
    rb[1:] = uip[1:] != (uip[:-1] + 1)          # behind each other so instead of storing pix: [24, 25, 26, 27, 28, 51, 52, 53]
    r0 = np.nonzero(rb)[0]                      # we store start_pix: [24, 51] and run_len: [5, 3]
                                                # This is especially advantageous because we use NESTED healpix ordering
    start_ipix = uip[r0].astype(np.uint64)      # which increases the probability of many consecutive indices in our chunks
    run_len = np.diff(np.append(r0, uip.size)).astype(np.uint32)

    return xr.Dataset(
        data_vars=dict(
            start_ipix=(("runs",), start_ipix),
            run_len=(("runs",), run_len),
            count=(("pix",), count),
            **pix_vars,
        ),
        attrs=dict(nside=nside, nest=nest, ordering="NESTED" if nest else "RING", aggregator="nanmean"),
    )


def _circular_mean_deg(lon_deg: np.ndarray) -> float:
    """
    Circular mean in degrees for angles on a circle.
    """
    lon_rad = np.deg2rad(lon_deg)
    s = np.sin(lon_rad).mean()
    c = np.cos(lon_rad).mean()
    return (np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0


def _wrap_lon_around(lon_deg: np.ndarray, center_deg: float) -> np.ndarray:
    """
    Wrap lon_deg so values lie in [center_deg-180, center_deg+180).
    """
    return (lon_deg - center_deg + 180.0) % 360.0 - 180.0 + center_deg



def healpix_to_regular_grid(
    chunk: xr.Dataset,
    *,
    dlat: float = DLAT,
    dlon: float = DLON,
):
    """
    Re-grids Dataset from Healpix back to a regular grid
    :param chunk: Dataset to be re-gridded
    :param dlat: Target lat resolution (don't choose to high, limiting factor is the resolution of the healpix grid)
    :param dlon: Target lon resolution (don't choose to high, limiting factor is the resolution of the healpix grid)
    :return: Regridded dataset
    """
    chunk = chunk.load()
    nside = chunk.attrs["nside"]
    nest  = chunk.attrs["nest"]

    if 'ipix' in chunk.variables:
        ipix = chunk["ipix"].values
    else:
        ipix = np.concatenate([
            np.arange(s, s + l, dtype=np.int64)
            for s, l in zip(chunk["start_ipix"].values, chunk["run_len"].values)
        ])

    # healpix → lon/lat
    theta, phi = hp.pix2ang(nside, ipix, nest=nest)
    lat = np.rad2deg(0.5 * np.pi - theta)
    lon = np.rad2deg(phi)

    lon_center = _circular_mean_deg(lon)
    lon_w = _wrap_lon_around(lon, lon_center).astype(np.float32)

    # target regular grid
    lat_g = np.arange(lat.min(), lat.max(), dlat)
    lon_g = np.arange(lon_w.min(), lon_w.max(), dlon)
    lon2d, lat2d = np.meshgrid(lon_g, lat_g)

    data_vars = {}
    for v in chunk.data_vars:
        if chunk[v].dims in (("pix",), ("ipix",)) :
            grid = griddata(
                (lon_w, lat),
                chunk[v].values,
                (lon2d, lat2d),
                method="nearest",
            )
            data_vars[v] = (("lat", "lon"), grid)

    return xr.Dataset(
        data_vars=data_vars,
        coords=dict(lat=lat_g, lon=lon_g),
        attrs=dict(
            source="healpix",
            nside=nside,
            nest=nest,
            interpolation="linear",
        ),
    )

def merge_healpix_chunks(chunks: list[xr.Dataset], bbox: Tile):
    """
    Merges healpix chunks and crops the merged dataset to the given bbox
    :param chunks: list of Datasets to be merged
    :param bbox: bbox to crop to
    :return: merged and cropped dataset
    """
    ipix_all = []
    count_all = []
    data_vars = [v for v in chunks[0].data_vars if v not in ('count', 'run_len', 'start_ipix')]
    vars_all = {v: [] for v in data_vars}
    attrs = dict(chunks[0].attrs)
    attrs["bbox"] = str(bbox)

    nside = int(chunks[0].attrs["nside"])
    nest = bool(chunks[0].attrs.get("nest", True))

    for chunk in chunks:
        ipix = np.concatenate([
            np.arange(s, s + l, dtype=np.int64)
            for s, l in zip(chunk["start_ipix"].values, chunk["run_len"].values)
        ])

        ipix_all.append(ipix)
        count_all.append(chunk['count'])
        for v in data_vars:
            vars_all[v].append(chunk[v])

    ipix = np.concatenate(ipix_all)
    count = np.concatenate(count_all)
    _vars = {v: np.concatenate(vars_all[v]) for v in data_vars}

    u, inv = np.unique(ipix, return_inverse=True)

    count_sum = np.bincount(inv, weights=count)
    vars_sum = {v: np.bincount(inv, weights=_vars[v]) for v in data_vars}

    theta, phi = hp.pix2ang(nside, u.astype(np.int64), nest=nest)  # theta=colat, phi=lon(rad)
    lon = np.degrees(phi)
    lat = 90.0 - np.degrees(theta)

    m = (lon >= bbox.west) & (lon <= bbox.east) & (lat >= bbox.south) & (lat <= bbox.north)

    u = u[m]
    count_sum = count_sum[m]
    vars_sum = {v: vars_sum[v][m] for v in data_vars}

    return xr.Dataset(
        data_vars=dict(
            {
                "count": ("ipix", count_sum),
                **{
                    v: ("ipix", vars_sum[v] / count_sum)
                    for v in data_vars
                },
            }
        ),
        coords=dict(ipix=u),
        attrs=attrs
    )


def plot_radar_scatter_das(das: list[list[xr.DataArray]], titles: list[list[str]]):
    """
    Simple method to plot a grid of radar_scatter data arrays
    """
    DB_RANGE = [-30, -5]
    shape = (len(das), len(das[0]))
    fig, axes = plt.subplots(shape[0], shape[1])
    for row, row_das in enumerate(das):
        row_axes = axes if shape[0] == 1 else axes[row]
        for col, da in enumerate(row_das):
            col_ax = row_axes if shape[1] == 1 else row_axes[col]
            title = titles[row][col]
            da_db = 10 * np.log10(da.clip(min=1e-10))

            da_visual = np.clip((da_db - DB_RANGE[0]) / (DB_RANGE[1] - DB_RANGE[0]), 0, 1)

            da_visual = np.nan_to_num(da_visual, nan=0.0, posinf=1.0, neginf=0.0)
            image = col_ax.imshow(da_visual, cmap='gray', origin='lower')
            col_ax.set_title(title)

    cbar = fig.colorbar(image, ax=axes, orientation="vertical", fraction=0.03, pad=0.04)
    plt.show()

if __name__ == '__main__':
    t0 = time.time()
    ds = xr.load_dataset("results/tile_2.0-3.0=51.0-51.5_asof_2026-01-20.nc")
    ds = add_lat_lon(ds)
    print(f'loading takes {time.time() - t0}s')
    print(ds)
    t0 = time.time()
    ds_hp = regular_grid_to_healpix(ds)
    print(f'rg -> hp conversion takes {time.time() - t0}s')
    print(ds_hp)
    t0 = time.time()
    ds_rg = healpix_to_regular_grid(ds_hp, dlat=15e-5, dlon=24e-5)
    print(f'hp -> rg conversion takes {time.time() - t0}s')
    print(ds_rg)
    plot_radar_scatter_das(
        [[ds["VV"], ds["VH"]], [ds_rg["VV"], ds_rg["VH"]]],
        [["vv_og", "vh_og"], ["vv_processed", "vh_processed"]])


