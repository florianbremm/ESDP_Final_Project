# Sentinel 1 Data Archive

Student: Florian Bremm \
Mat.Nr.: 7440728

This Repository is my submission for the final project of the lecture "Earth System Data Processing". \
<span style="color:#CC4444">To use this script you must be signed up to CDSE Head to "Data Access" for more info</span>

## General Information
This project provides a solution for downloading, archiving and retrieving high resolution Geo-Data.
In its current form it's setup to download ESA Sentinel-1 radar data from the CDSE openEO endpoint.
That Data is regridded into high resolution HEALPix and stored in a Zarr archive. 
Any bounding box can be retrieved from the archive and be displayed.

### Chunking Strategy
The program uses chunking to reduce waiting time (download as well as retrieval) and space consumption.
Chunks are of rigid size (0.5° [S-N], 1° [E-W]) and position (originating from 0° N, 0° E).
In the download routine, the chunks that intersect the Area of Interest (AOI) are downloaded from the openEO.
Each chunk is separately converted to HEALPix.
The chunks also mirror the chunking strategy of the Zarr archive: One Zarr chunk contains one chunk and one timestamp.
In retrieval, all available chunks that intersect the AOI are selected and merged, however for this use-case the merged dataset is cropped before interpolating the regular grid to exactly match the query.

### Conversion and Storage
As mentioned, each Zarr chunk contains exactly one chunk of one timestamp.
NSIDE=262144 was used, resulting in ~25m per pixel.
This amounts to about 80MB per chunk (two variables with 30MB each + some overhead). 
Due to the healpix chunks having variable length, a padding is applied (Zarr needs chunks of equal size).
Each chunk needs a `ipix` array to keep track which value belongs to which HEALPix pixel.
To store this index data efficient a running index is used.
For example instead of storing `ipix=[7, 8, 9, 10, 11, 12, 34, 35, 36, 37]` we store `start_ipix=[7, 34]` and `run_len=[6, 4]`.
To boost the efficiency gain of this technique the HEALPix IDs are created Nested, not as Ring.
Visit https://healpix.sourceforge.io/pdf/intro.pdf for detailed information regarding HEALPix.

### Download
Requested Chunks get downloaded from the CDSE openEO endpoint.
The main bottleneck in download is the processing time on the openEO server, the download itself is relative fast, even though 300MB are downloaded per chunk and timestamp.
To speed up this process, two workers are used in parallel on the server (more would be possible without problems, but are not supported by the endpoint, at least not with a basic account).
The downloading routine monitors their progress and starts the next job as soon as one of the two jobs completes.
The actual download and process happens afterwards synchronous while the next chunk is already processed.

For the CDSE openEO endpoint documentation see: https://documentation.dataspace.copernicus.eu/APIs/openEO/openEO.html \
For ussage instructions for the python openEO library see: https://open-eo.github.io/openeo-python-client/ \
An online control panel where you can manage and inspect your jobs exists at: https://openeofed.dataspace.copernicus.eu \
To check your credits see: https://marketplace-portal.dataspace.copernicus.eu/billing

## Data Access
To access the data, authentication is required. However, access is granted to every person and registration requires less than 5 Minutes. Visit https://dataspace.copernicus.eu/ for registration. After registration is completed online, this client must be registered interactively during notebook execution. This process provides minimal friction, compared to a system with a token that must be manually created and provided to the client. See `load_sentinel1.ipynb` for details.

## Requirements 
To run this program, you need to install some packages besides the python core libraries.
If you want to install those automatically run `pip install -r requirements.txt`
Required are :
h5netcdf, healpy, matplotlib, matplotlib-inline, netCDF4, numpy, openeo, scipy, xarray, zarr, pyproj

## Usage
Run the program by entering `python main.py`.
The program will first ask you if you want to download or retrieve.
After that it will query all required parameters for the selected operation from you.
If you want to change parameters other than the AOI and the timestamp you have to do this in Code.
The configuration constants for the different processes are stored in their respective .py file in the top.

Here's a list of constants by the file in which they are located: \
main.py:
- `ZARR_PATH`: path of the archive

download.py:
- `MAX_INFLIGHT`: max number of chunk-jobs running in parallel on the server
- `MAX_PENDING`: number of chunk-jobs that will be made kept ready to be run 
- `POLL_WAIT_TIME`: time in [s] to wait between status queries 
- `MAX_TRYS`: number of tries for each chunk-job before it is marked as failed 
- `CHUNK_SIZE`: [float, float] size of the chunks, must be at least [0.1, 0.1]
- `COLLECTION`: CollectionID on the endpoint of the collection to be downloaded
- `BANDS`: Bands of the collection to be downloaded

convert.py:
- `NSIDE`: NSIDE of the resulting HEALPix grid. #healpix_dots = 12 * NSIDE ^ 2 on complete sphere
- `DLAT`: Lateral distance between dots in regular grid [deg]
- `DLON`: Longitudinal distance between dots in regular grid [deg]

store.py:
- `PIX_MAX`: Size of pix dimension in Zarr archive. **Must** be bigger than the number of pixels in each of your chunks. Current Value 8e6 is tuned for areas around 50° N with chunk size `[1, 0.5]` and a relatively large safety margin. For Areas closer to the Equator choose 1e7 or 11e6
- `RUNS_MAX`: Size of Runs dimensions. Same as with `PIX_MAX` current value 6000 *should* be sufficient for all Areas but maybe increase to 7000 or 8000 when working close to the equator. 