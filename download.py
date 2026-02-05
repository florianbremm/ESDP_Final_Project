from typing import Iterable, Callable
from copy import copy
import openeo
from openeo.rest.result import SaveResult
from openeo import Connection
import os, time, json, hashlib
from datetime import datetime, timedelta
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
from dataclasses import dataclass

#required are also netcdf4, h5netcdf


MAX_INFLIGHT = 2
MAX_PENDING = 2
POLL_WAIT_TIME = 15
MAX_TRYS = 3
CHUNK_SIZE = [1, 0.5]
COLLECTION = "SENTINEL1_GRD"
BANDS = ("VV", "VH")

_connection = None

@dataclass
class Tile:
    west: float
    south: float
    east: float
    north: float

    def get_dict(self):
        return {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north
        }

@dataclass
class Chunk:
    tile: Tile
    date: datetime

    def __str__(self):
        return f'tile_{self.tile.west}-{self.tile.east}={self.tile.south}-{self.tile.north}_asof_{self.date.strftime('%Y-%m-%d')}'


@dataclass
class Recipe:
    result: SaveResult
    chunk: Chunk

    def __str__(self):
        return str(self.chunk)


class Job(Recipe):
    def __init__(self, recipe: Recipe):
        self.result = recipe.result
        self.chunk = recipe.chunk
        self.trys = 0

        connection = get_connection()
        self.job = connection.create_job(
            self.result,
            title=str(recipe),
            job_options={"soft-errors": 0.6}
        )

    def status(self):
        return self.job.status()

    def start(self):
        self.job.start()
        self.trys += 1

    def get_results(self):
        return self.job.get_results()

  
def get_connection() -> Connection:
    """
    Singleton implementation for connection
    :return: Connection Object either from cache or newly created
    """
    global _connection
    if _connection and _connection is not None and _connection.__class__ == Connection:
        return _connection
    else:
        _connection = openeo.connect('openeofed.dataspace.copernicus.eu')
        _connection.authenticate_oidc()
        return _connection


def aoi_to_extend(area_of_interest: Tile) -> Tile:
    """
    expands AOI to completely cover all intersecting chunks
    :param area_of_interest:
    :return: expanded tile
    """
    extend = Tile(0, 0, 0, 0)

    while extend.west >= area_of_interest.west:
        extend.west -= CHUNK_SIZE[0]

    while extend.west <= area_of_interest.west - CHUNK_SIZE[0]:
        extend.west += CHUNK_SIZE[0]

    extend.east = extend.west
    while extend.east < area_of_interest.east:
        extend.east += CHUNK_SIZE[0]

    while extend.south >= area_of_interest.south:
        extend.south -= CHUNK_SIZE[1]

    while extend.south <= area_of_interest.south - CHUNK_SIZE[1]:
        extend.south += CHUNK_SIZE[1]

    extend.north = extend.south
    while extend.north < area_of_interest.north:
        extend.north += CHUNK_SIZE[1]

    return extend


def generate_chunks(area_of_interest: Tile, time_of_interest: tuple[str, str]) -> list[Chunk]:
    """
    generates chunks (in spatial x temporal grid) for a specified area and time of interest
    :param area_of_interest:
    :param time_of_interest:
    :return: list of all chunks necessary to cover the interest
    """
    extend = aoi_to_extend(area_of_interest)
    tiles: list[Tile] = []
    x = 0
    while extend.west + CHUNK_SIZE[0] * x < extend.east:
        y = 0
        while extend.south + CHUNK_SIZE[1] * y < extend.north:
            tiles.append(Tile(
                west=extend.west + CHUNK_SIZE[0] * x,
                south=extend.south + CHUNK_SIZE[1] * y,
                east=min(extend.west + CHUNK_SIZE[0] * (x + 1), extend.east),
                north=min(extend.south + CHUNK_SIZE[1] * (y + 1), extend.north),
            ))
            y += 1
        x += 1

    first = datetime.strptime(time_of_interest[0], '%Y-%m-%d')
    last = datetime.strptime(time_of_interest[1], '%Y-%m-%d')

    dates = []
    while first <= last:
        dates.append(copy(first))
        first += timedelta(days=1)

    chunks = []

    for tile in tiles:
        for date in dates:
            chunks.append(Chunk(tile, date))

    return chunks


def build_job_recipes(chunks: list[Chunk], collection=COLLECTION, bands: Iterable[str]=BANDS) -> list[Recipe]:
    """
    Builds a recipe to download the chunks doesn't control for chunk existence
    :param chunks: ``[{"date":'%Y-%m-%d',"tile":{"west":float,"south":float,"east":float,"north":float}}]`` The chunks to create a recipe for
    :param collection: ID of the collection you wish to download
    :param bands: ``[str]`` Bands inside the collection
    :return: ``["result":openeo.SaveResult,"title":str,"chunk":chunk]`` List of recipes. "result" is the openeo.SaveResult that can be used to create a job
    """
    job_recipes = []
    connection = get_connection()

    for chunk in chunks:

        datacube = connection.load_collection(  # config for the download
            collection,
            temporal_extent=[
                (chunk.date - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                (chunk.date + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
            ],
            spatial_extent=chunk.tile.get_dict(),
            bands=bands,
        )

        dataslice = datacube.reduce_temporal('last')
        result = dataslice.save_result(
            format="netCDF"
        )

        job_recipe = Recipe(result=result, chunk=chunk)
        job_recipes.append(job_recipe)
    return job_recipes


def download_chunks(
        chunks: list[Chunk],
        process_completed: Callable[[Job], bool],
        check_recipe: Callable[[Recipe], bool] = lambda _: True,
        collection: str = COLLECTION,
        bands: Iterable[str]=BANDS,
        verbose: int = 2
) -> tuple[list, list]:
    """

    :param check_recipe: boolean function to check whether it's necessary to execute a recipe
    :param chunks: ``[{"date":'%Y-%m-%d',"tile":{"west":float,"south":float,"east":float,"north":float}}]`` The chunks to download
    :param process_completed: function to execute on a completed Job
    :param collection: ID of the collection you wish to download
    :param bands: ``[str]`` Bands inside the collection
    :param verbose: 0 -> no logs, 1 -> failures, completions, 2 -> status changes
    :return:
    """
    job_recipes = build_job_recipes(chunks, collection, bands)

    pending_recipes = job_recipes
    pending: list[Job] = []
    running: list[Job] = []
    finished: list[Job] = []
    processed: list[Job] = []
    failed: list[Job] = []

    while pending_recipes or pending or running:
        still_running: list[Job] = []

        for running_job in running:
            status = running_job.status()
            if status == 'finished':
                if verbose == 2:
                    print(f'Job "{running_job}" finished. Will start processing ASAP')
                finished.append(running_job)
            elif status in ('error', 'canceled'):
                if running_job.trys < MAX_TRYS:
                    if verbose > 0:
                        print(f'Job "{running_job}" failed in try {running_job.trys}. Retrying')
                    pending.append(running_job)
                else:
                    if verbose > 0:
                        print(f'Job "{running_job}" failed {MAX_TRYS} times. This is the retry limit. Will not retry')
                    failed.append(running_job)
            else:
                still_running.append(running_job)
        running = still_running

        while pending_recipes and len(pending) < MAX_PENDING:
            next_recipe = pending_recipes.pop(0)
            if check_recipe(next_recipe):
                next_job = Job(next_recipe)
                pending.append(next_job)
                if verbose == 2:
                    print(f'Job "{next_job}" created')
            else:
                if verbose == 2:
                    print(f'Recipe "{next_recipe}" skipped')
        while pending and len(running) < MAX_INFLIGHT:
            next_job = pending.pop(0)
            next_job.start()
            running.append(next_job)
            if verbose == 2:
                print(f'Job "{next_job}" started')

        while finished:
            finished_job = finished.pop(0)
            if verbose == 2:
                print(f'Processing Job {finished_job}')
            process_completed(finished_job)
            processed.append(finished_job)
            if verbose > 0:
                print(f'Job {finished_job} completed')
        if running:
            time.sleep(POLL_WAIT_TIME)

    return processed, failed