import xarray as xr
import numpy as np

import convert
from download import Chunk

PIX_MAX = 8000000
RUNS_MAX = 6000

def init_store(
    zarr_path: str,
    *,
    pix_max: int = PIX_MAX,
    runs_max: int = RUNS_MAX,
    pix_dim: str = 'pix',
    runs_dim: str = 'runs',
    record_dim: str = 'record',
):
    ds_init = xr.Dataset(
        coords={
            record_dim: np.empty((0,), dtype=np.int64),
            pix_dim: np.arange(pix_max, dtype=np.int32),
            runs_dim: np.arange(runs_max, dtype=np.int32),
        }
    )

    # per-record metadata
    ds_init["time"] = xr.DataArray(np.empty((0,), dtype='datetime64[ns]'), dims=(record_dim,))
    ds_init["tile_lon0"] = xr.DataArray(np.empty((0,), dtype='float32'), dims=(record_dim,))
    ds_init["tile_lat0"] = xr.DataArray(np.empty((0,), dtype='float32'), dims=(record_dim,))
    ds_init["pix_len"] = xr.DataArray(np.empty((0,), dtype='int32'), dims=(record_dim,))
    ds_init["runs_len"] = xr.DataArray(np.empty((0,), dtype='int32'), dims=(record_dim,))

    # placeholder arrays (no data yet, just shape & dtype)
    ds_init["count"] = xr.DataArray(np.empty((0, pix_max), dtype='uint16'), dims=(record_dim, pix_dim))
    ds_init["VV"] = xr.DataArray(np.empty((0, pix_max), dtype='float32'), dims=(record_dim, pix_dim))
    ds_init["VH"] = xr.DataArray(np.empty((0, pix_max), dtype='float32'), dims=(record_dim, pix_dim))
    ds_init["start_ipix"] = xr.DataArray(np.empty((0, runs_max), dtype='uint64'), dims=(record_dim, runs_dim))
    ds_init["run_len"] = xr.DataArray(np.empty((0, runs_max), dtype='uint32'), dims=(record_dim, runs_dim))

    ds_init.to_zarr(zarr_path, mode='w', zarr_format=2)


def pad_and_add_record(
    ds_hp: xr.Dataset,
    *,
    time_value: np.datetime64,
    tile_lon0: float,
    tile_lat0: float,
    pix_max: int = PIX_MAX,
    runs_max: int = RUNS_MAX,
    record_dim: str = 'record',
    pix_dim: str = 'pix',
    runs_dim: str = 'runs',
) -> xr.Dataset:
    """
    Pads all (pix) and (runs) variables to fixed sizes and returns a 1-record dataset
    appendable via append_dim='record'.
    """
    pix_len  = int(ds_hp.sizes.get(pix_dim, 0))
    runs_len = int(ds_hp.sizes.get(runs_dim, 0))

    out = xr.Dataset(
        coords={
            record_dim: [0],
            pix_dim: np.arange(pix_max, dtype=np.int32),
            runs_dim: np.arange(runs_max, dtype=np.int32),
        }
    )

    out["time"] = xr.DataArray([time_value], dims=(record_dim,))
    out["tile_lon0"] = xr.DataArray([tile_lon0], dims=(record_dim,))
    out["tile_lat0"] = xr.DataArray([tile_lat0], dims=(record_dim,))
    out["pix_len"] = xr.DataArray([pix_len], dims=(record_dim,))
    out["runs_len"] = xr.DataArray([runs_len], dims=(record_dim,))

    def pad_1d(a: np.ndarray, maxlen: int, dtype) -> np.ndarray:
        is_int = np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.unsignedinteger)
        fill = 0 if is_int else np.nan
        outa = np.full((maxlen,), fill, dtype=dtype)
        n = min(a.size, maxlen)
        if n:
            outa[:n] = a[:n]
        return outa

    for v in ds_hp.data_vars:
        a = ds_hp[v].values
        if ds_hp[v].dims == (pix_dim,):
            out[v] = xr.DataArray(
                pad_1d(a, pix_max, ds_hp[v].dtype)[None, :],
                dims=(record_dim, pix_dim),
            )
        elif ds_hp[v].dims == (runs_dim,):
            out[v] = xr.DataArray(
                pad_1d(a, runs_max, ds_hp[v].dtype)[None, :],
                dims=(record_dim, runs_dim),
            )


    out.attrs.update(ds_hp.attrs)
    out.attrs.update(dict(padded=True, pix_max=pix_max, runs_max=runs_max))
    return out


def unpad_record(ds_r, *, pix_dim='pix', runs_dim='runs'):
    pix_len = int(ds_r["pix_len"].values)
    runs_len = int(ds_r["runs_len"].values)

    out = xr.Dataset(attrs=ds_r.attrs)

    for v in ds_r.data_vars:
        if ds_r[v].dims == ( pix_dim,):
            out[v] = ds_r[v].isel(**{pix_dim: slice(pix_len)})
        elif ds_r[v].dims == (runs_dim,):
            out[v] = ds_r[v].isel(**{runs_dim: slice(runs_len)})

    return out


def get_record_id(
    zarr_path: str,
    *,
    time_value: np.datetime64,
    tile_lon0: float,
    tile_lat0: float
) -> None | np.int64:
    lon0 = np.float32(tile_lon0)
    lat0 = np.float32(tile_lat0)

    ds = xr.open_zarr(zarr_path, zarr_format=2)
    mask = (
            (ds["time"] == time_value)
            & np.isclose(ds["tile_lon0"].values, lon0)
            & np.isclose(ds["tile_lat0"].values, lat0)
    )

    retrieved = np.nonzero(mask.values)[0]
    if len(retrieved) > 1:
        raise Exception(f'Duplicated Chunk found in {zarr_path}. This is strictly not allowed. Something went wrong')
    return None if len(retrieved) < 1 else retrieved[0]


def is_chunk_stored(zarr_path, *, chunk: Chunk) -> bool:
    _time = np.datetime64(chunk.date, 'ns')
    _lat0 = chunk.tile.south
    _lon0 = chunk.tile.west
    return False if get_record_id(zarr_path, time_value=_time, tile_lon0=_lon0, tile_lat0=_lat0) is None else True


def write_to_zarr(
    zarr_path: str,
    ds_hp,
    *,
    time_value: np.datetime64,
    tile_lon0: float,
    tile_lat0: float,
) -> bool:
    ds_store = pad_and_add_record(ds_hp, time_value=time_value, tile_lat0=tile_lat0, tile_lon0=tile_lon0)

    if get_record_id(zarr_path, time_value=time_value, tile_lat0=tile_lat0, tile_lon0=tile_lon0) is not None:
        return False

    ds_store.to_zarr(zarr_path, mode='a', append_dim='record')
    return True


def read_from_zarr(
    zarr_path: str,
    *,
    time_value: np.datetime64,
    tile_lon0: float,
    tile_lat0: float,
) -> bool | xr.Dataset:

    record = get_record_id(zarr_path, time_value=time_value, tile_lat0=tile_lat0, tile_lon0=tile_lon0)
    if record is None:
        return False
    ds = xr.open_zarr(zarr_path)
    ds_r = ds.isel(record=record)
    return unpad_record(ds_r)


def read_chunk_from_zarr(
    zarr_path: str,
    *,
    chunk: Chunk
) -> bool | xr.Dataset:
    _time = np.datetime64(chunk.date, 'ns')
    _lat0 = chunk.tile.south
    _lon0 = chunk.tile.west
    return read_from_zarr(zarr_path, time_value=_time, tile_lat0=_lat0, tile_lon0=_lon0)


if __name__ == '__main__':
    from pathlib import Path
    from convert import healpix_to_regular_grid

    lat0 = 51.0
    lon0 = 3.0
    time = np.datetime64("2026-01-20", 'ns')

    ret = read_from_zarr("sentinel1.zarr", time_value=time, tile_lat0=lat0, tile_lon0=lon0)
    ret_rg = healpix_to_regular_grid(ret, dlat=15e-5, dlon=24e-5)
    convert.plot_radar_scatter_das([[ret_rg['VV']]], [['']])
