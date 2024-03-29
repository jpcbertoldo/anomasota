""" Generate data.json from src/ """

from collections import OrderedDict
from copy import deepcopy
import json5
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
import time
import re
import datetime
import warnings
import hashlib

import click
from pprint import pprint

from anomasota.data import common
from anomasota.data.parse_models_single_paper import parse_models_single_paper


NOW = time.time()


def fmt_timestamp(timestamp: float) -> str:
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d-%H-%M-%S")


def _get_datadir_subpaths(datadir: Path) -> Tuple[Path, Path, Path, Path, Path]:
    return (
        # data.json and data.checksum
        datadir / "data.json",
        datadir / "data.checksum",
        datadir / "data-no-metadata.checksum",
        # src/
        (srcdir := datadir / "src"),
        srcdir / "000-manual.json",
        # bkp/
        srcdir / "bkp",
    )


def _validate_datadir(datadir: Path) -> None:
    assert datadir.is_dir(), f"Data directory is not a directory, {datadir=}"
    datajson, _, _, srcdir, manualjson, bkpdir = _get_datadir_subpaths(datadir)
    assert srcdir.is_dir(), f"Source directory is not a directory, {srcdir=}"
    assert manualjson.is_file(), f"Manual JSON file is not a file, {manualjson=}"
    bkpdir.mkdir(exist_ok=True)
    assert bkpdir.is_dir(), f"Backup directory is not a directory, {bkpdir=}"
    
    
def _parse_manual(jsonfpath: Path) -> Dict[str, Any]:
    return json5.loads(jsonfpath.read_text())


def _get_checksum(data: Union[Path, dict]) -> str:
    
    if isinstance(data, Path):
        return hashlib.md5(data.read_bytes()).hexdigest()

    elif isinstance(data, dict):
        return hashlib.md5(json5.dumps(data).encode()).hexdigest()
    
    raise TypeError(f"Expected Path or dict, got {type(data)}")


def _bkp(datadir) -> None:
    
    datajson, data_checksum, data_no_metadata_checksum, _, _, bkpdir = _get_datadir_subpaths(datadir)
    
    if not datajson.exists():
        warnings.warn("data.json does not exist, skipping backup")
        return 
    
    bkp_subdirs = [d for d in bkpdir.iterdir() if d.is_dir() and d.name.startswith("bkp-")] 
    bkp_count = len(bkp_subdirs)
    
    newbkpdir = bkpdir / f"bkp-{bkp_count:05d}-{fmt_timestamp(NOW)}"
    newbkpdir.mkdir()
    
    files_to_copy = [datajson, data_checksum, data_no_metadata_checksum]
    for f in files_to_copy:
        (newbkpdir / f.name).write_bytes(f.read_bytes())


# TODO move this to a config file
_SOURCES_PARSER_FUNCTIONS: Dict[str, callable] = {
    "000-manual": _parse_manual,
    "001-padim": parse_models_single_paper,
    "002-patchcore": parse_models_single_paper,
    "003-spade": parse_models_single_paper,
    "004-semi-orthogonal": parse_models_single_paper,
    "005-gaussian-ad": parse_models_single_paper,
    # "005-gaussian-ad-table-i": parse_models_single_paper,
    # "005-gaussian-ad-table-v-table-viii": parse_models_single_paper,
    # "006-": parse_models_single_paper,
    # "007-": parse_models_single_paper,
    # "008-": parse_models_single_paper,
}
    
    
def _parse_rootname(srcjson: Path) -> str:
    """name before the first dot in the stem if any, else the whole stem"""
    return srcjson.stem.split(".", maxsplit=1)[0]


@click.command()
@click.option('--datadir', default=common.DEFAULT_DATADIR_PATH, type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True, allow_dash=False, path_type=Path))
@click.option("--dryrun", is_flag=True)
@click.option("--bkp", is_flag=True)
def main(datadir: Path, dryrun: bool, bkp: bool) -> None:
    
    _validate_datadir(datadir)
    datajson_fpath, data_checksum_fpath, data_no_metadata_checksum_fpath, srcdir, manualjson, _ = _get_datadir_subpaths(datadir)
    
    # find source files in srcdir
    regex_source_file_json = re.compile(r"^\d{3}-.*\.json$")
    srcjsons = sorted(
        p 
        for p in srcdir.iterdir() 
        if p.is_file() 
        and regex_source_file_json.match(p.name)
    )
    
    print(f"sources found (n: {len(srcjsons)}):")
    for p in srcjsons:
        print(f"\t{p.name}")
    
    print("grouping sources by root name (name before the first dot in the stem if any)")
    srcjson_by_rootname: Dict[str, List[Path]] = OrderedDict()
    for srcjson in srcjsons:
        srcjson_by_rootname.setdefault(_parse_rootname(srcjson), []).append(srcjson)
    
    print("grouped sources:")
    for rootname, srcjsons in srcjson_by_rootname.items():
        print(f"{rootname} (n: {len(srcjsons)}):")
        for p in srcjsons:
            print(f"\t{p.name}")
    
    merged_data = {}
    
    def count_objects(data: Dict[str, Any]) -> Dict[str, int]:
        return {k: len(v) for k, v in data.items() if isinstance(v, list) or isinstance(v, dict)}
    
    for rootname, rootname_srcjsons in srcjson_by_rootname.items():

        print(f"parsing `{rootname=}`")
        
        try:
            parser = _SOURCES_PARSER_FUNCTIONS[rootname]
        
        except KeyError as ex: 
            warnings.warn(f"no parser function found for `{rootname=}`, skipping")
            continue

        print(f"`{rootname=}` using parser `{parser.__name__}`")  # IMPORTANT CALL!!!!!

        rootname_data_persrc = OrderedDict()
        
        for srcjson in rootname_srcjsons:
            
            print(f"\tparsing `{srcjson.name}`")
            rootname_data_persrc[srcjson.name] = parser(srcjson)

        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------

        if len(rootname_data_persrc) == 1:
            print(f"`{rootname=}` only one source file, no need to merge")
            data = rootname_data_persrc.popitem()[1]        
       
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        # merge data from different sources with the same rootname

        elif parser == parse_models_single_paper:
            
            print(f"`{rootname=}` papers ({len(rootname_data_persrc)}) have to be merged")
            # this is done to avoid having the same paper multiple times in the data.json

            data = {}
            
            for srcidx, (srcjson, srcdata) in enumerate(rootname_data_persrc.items()):

                if srcidx == 0:
                    # this is done to avoid having different papers while they should be the same
                    first_srcdata = deepcopy(srcdata)
                    first_srcjson = srcjson
                    
                else:
                    srcdata = deepcopy(srcdata)
                    srcpaper = srcdata.pop(common.DK_PAPERS)
                    assert srcpaper == first_srcdata[common.DK_PAPERS], f"the paper `{srcjson=}` is different from the first one (`{first_srcjson=}`) with the same `{rootname=}`"
                
                # TODO: have a function that merges stuff and reuse it here
                for key, objs in srcdata.items():
                    if key == common.DK_PERFORMANCES:
                        data.setdefault(key, []).extend(objs)    
                    else:
                        data.setdefault(key, {}).update(objs)
                
            # avoid mistakes with the variables names
            del srcidx, srcjson, srcdata, srcpaper, first_srcdata, first_srcjson
            
        del rootname_data_persrc
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
            
        counts = count_objects(data)
        print(f"parsed data count: {counts}")
        
        total_parsed_objs = sum(counts.values())
        if total_parsed_objs == 0:
            warnings.warn(f"no objects parsed, {rootname=} {rootname_srcjsons=}")
            continue
        
        # keys: papers, datasets, metrics, models, performances
        for key, objects in data.items():
            
            objects_list = objects.values() if isinstance(objects, dict) else objects
            
            for obj in objects_list:
                obj["metadata"] = {
                    "src-rootname": rootname,
                    "src-rootname-files": [srcjson.name for srcjson in rootname_srcjsons],
                    "src-parser": parser.__name__,
                }

            # check that there is no intersection between the new objects and the existing ones, then merge
            merged_objects = merged_data.setdefault(key, [] if key == "performances" else {})
            
            # all except performances are dictionaries (umique keys)
            if key == "performances":
                
                merged_unique_perfobjs = set(map(common.performance_unicity_tuple, merged_objects))
                unique_objects = set(map(common.performance_unicity_tuple, objects_list))
                intersec = merged_unique_perfobjs.intersection(unique_objects)
                assert len(intersec) == 0, f"Duplicate performances, {rootname=} {rootname_srcjsons=}, {sorted(intersec)=}"
                
                merged_objects.extend(objects)
                
            else:
                
                interct = set(merged_objects.keys()).intersection(set(objects.keys()))
                assert len(interct) == 0, f"Duplicate object ids found. {rootname=} {rootname_srcjsons=}, {key=}, {sorted(interct)=}"
                
                merged_objects.update(objects)    
            
    print("validating merged data")
    common.validate_data(merged_data)            
    
    print(f"merged data count: {count_objects(merged_data)}")
    
    checksum_no_metadata = _get_checksum(merged_data)
    
    print("writing data.json")
    merged_data["metadata"] = metadata = {
        "checksum-no-metadata": checksum_no_metadata,
        "datetime": fmt_timestamp(NOW),
        "source-files": [
            {
                "filename": p.name,
                "checksum": _get_checksum(p),
            } 
            for p in srcjsons
        ],
    }

    if dryrun:
        warnings.warn("dryrun, not writing data.json")
    
    elif data_no_metadata_checksum_fpath.exists() and data_no_metadata_checksum_fpath.read_text() == checksum_no_metadata:
        warnings.warn("no changes detected, not writing data.json")
    
    else:
        
        if bkp:
            print("backing up data.json")
            _bkp(datadir)        
                
        datajson_fpath.write_text(json5.dumps(merged_data, indent=4, sort_keys=False))
        data_no_metadata_checksum_fpath.write_text(checksum_no_metadata)
        data_checksum_fpath.write_text(_get_checksum(datajson_fpath))
    

if __name__ == "__main__":
    main()
