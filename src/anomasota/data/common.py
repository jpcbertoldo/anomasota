
from typing import Dict, List, Tuple, Any
import time
import datetime
import warnings
from collections import Counter
from functools import wraps, partial

import jmespath

# ==================================================================================================
# data (root level)
# ==================================================================================================

# DK stands for Data Key

# dict
DK_DATASETS = "datasets"
DK_METRICS = "metrics"
DK_MODELS = "models"
DK_PAPERS = "papers"
DATA_KEYS_DICTIONARIES = (DK_DATASETS, DK_METRICS, DK_MODELS, DK_PAPERS, )

# list
DK_PERFORMANCES = "performances"

DATA_KEYS = (DK_DATASETS, DK_METRICS, DK_MODELS, DK_PAPERS, DK_PERFORMANCES,)

# ==================================================================================================
# Dataset
# ==================================================================================================

# DSK stands for DataSet Key
DSK_ID = "id"
DSK_DESCRIPTION = "description"
DSK_PAPERS = "papers"

DATASET_KEYS = (DSK_ID, DSK_DESCRIPTION, DSK_PAPERS)

# DS_PK stands for DataSet Paper Key
DS_PK_SOURCE = "source"
DATASET_PAPERS_KEYS = (DS_PK_SOURCE,)

# ==================================================================================================
# Metrics
# ==================================================================================================

# MEK stands for MEtric Key
MEK_ID = "id"
MEK_DESCRIPTION = "description"

METRIC_KEYS = (MEK_ID, MEK_DESCRIPTION,)

# ==================================================================================================
# Models
# ==================================================================================================

# MOK stands for MOdel Key
MOK_ID = "id"
MOK_NAME = "name"
MOK_VERSION = "version"
MOK_TAGS = "tags"
MOK_PAPERS = "papers"

MODEL_ID_KEYS = (MOK_NAME, MOK_VERSION,)
MODEL_KEYS = (MOK_ID, MOK_NAME, MOK_VERSION, MOK_TAGS, MOK_PAPERS,)

# MO_PK stands for MOdel --> Papers Key
MO_PK_SOURCE = "source"
MODEL_PAPERS_KEYS = (MO_PK_SOURCE,)

# ==================================================================================================
# Papers
# ==================================================================================================

# PAK stands for PAper Key
PAK_ID = "id"
PAK_NAME = "name"
PAK_AUTHOR = "author"
PAK_YEAR = "year"

PAPER_ID_KEYS = (PAK_NAME, PAK_AUTHOR, PAK_YEAR,)
PAPER_KEYS = (PAK_ID, PAK_NAME, PAK_AUTHOR, PAK_YEAR,)

# ==================================================================================================
# Performance
# ==================================================================================================

# PEK stands for Performance Key

# these define the unicity of a performance
PEK_DATASET = "dataset"
PEK_METRIC = "metric"
PEK_MODEL = "model"
PEK_SOURCE_TYPE = "source-type"
PEK_SOURCE = "source"

PERFORMANCE_UNICITY_KEYS = (PEK_DATASET, PEK_METRIC, PEK_MODEL, PEK_SOURCE_TYPE, PEK_SOURCE,)

# these are also mandatory keys
PEK_EXTRA = "extra"
PEK_TAGS = "tags"
PEK_VALUE = "value"

PERFORMANCE_KEYS = PERFORMANCE_UNICITY_KEYS + (PEK_EXTRA, PEK_TAGS, PEK_VALUE,)

# -----------------------------------------
# Source type
# -----------------------------------------

# PEST stands for PErformance Source Type
PEST_LITTERATURE = "litterature"

PEST_TYPES = (PEST_LITTERATURE,)

# ==================================================================================================
# ==================================================================================================


def _get_src_file(obj: Dict[str, Any]) -> str:
    """This metadata is injected in the script generate.py"""
    return obj["metadata"]["src-file"]


def _add_src_file_to_assertion_msg(wrapped: callable) -> callable:
    """The object is assumed to be the first argument of the wrapped function"""
    
    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        try:
            return wrapped(*args, **kwargs)
        except AssertionError as ex:
            msg = ex.args[0]
            obj = args[0]
            msg += f", {_get_src_file(obj)=}"
            raise AssertionError(msg)

    return wrapper


def build_paperid(name: str, author: str, year: str) -> str:
    return f"{name}.{author}.{year}"


def build_modelid(name: str, version: str) -> str:
    return f"{name}/{version}"


def performance_unicity_tuple(perfobj: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return tuple(perfobj[key] for key in PERFORMANCE_UNICITY_KEYS)
    

@_add_src_file_to_assertion_msg
def validate_paperobj_id(paperobj: dict) -> None:
    
    assert PAK_ID in paperobj, f"Paper object does not have an ID, {paperobj.keys()=}"
    assert set(PAPER_ID_KEYS).issubset(set(paperobj.keys())), f"Paper object should have keys {PAPER_ID_KEYS}, {paperobj.keys()=}"
    
    id_, name, author, year = paperobj["id"], paperobj["name"], paperobj["author"], paperobj["year"]
    rebuilt_id = build_paperid(name, author, year)
    assert id_ == rebuilt_id, f"Paper ID is not valid, {id_=} != {rebuilt_id}"


@_add_src_file_to_assertion_msg
def validate_modelobj_id(modelobj: dict) -> None:
    
    assert MOK_ID in modelobj, f"Model object does not have an ID, {modelobj.keys()=}"
    assert set(MODEL_ID_KEYS).issubset(set(modelobj.keys())), f"Model object should have keys {MODEL_ID_KEYS}, {modelobj.keys()=}"
    
    id_, name, version = modelobj["id"], modelobj["name"], modelobj["version"]
    rebuilt_id = build_modelid(name, version)
    assert id_ == rebuilt_id, f"Model ID is not valid, {id_=} != {rebuilt_id}"
    

def validate_key_id_are_same(objects_dict: Dict[str, Any]) -> None:
    for key, obj in objects_dict.items():
        assert isinstance(key, str), f"An object's key is not a string, {key=}"
        assert "id" in obj, f"An object does not have an ID, {obj.keys()=}"
        objid = obj["id"]
        assert key == objid, f"An object's ID does not match its key, {key=}, {objid=}, {_get_src_file(obj)=}"


@_add_src_file_to_assertion_msg
def validate_datasetobj(datasetobj: Dict[str, Any]) -> None:
    """ Validate a dataset object """
    missing_keys = set(DATASET_KEYS).difference(datasetobj.keys())
    assert not missing_keys, f"Dataset object is missing keys, {missing_keys=}"
    
    papers = datasetobj[DSK_PAPERS]
    
    papers_missing_keys = set(DATASET_PAPERS_KEYS).difference(papers.keys())
    assert not papers_missing_keys, f"Dataset object's papers is missing keys, {papers_missing_keys=}"   


@_add_src_file_to_assertion_msg
def validate_metricobj(metricobj: Dict[str, Any]) -> None:
    """ Validate a metric object """
    missing_keys = set(METRIC_KEYS).difference(metricobj.keys())
    assert not missing_keys, f"Metric object is missing keys, {missing_keys=}"


@_add_src_file_to_assertion_msg
def validate_modelobj(modelobj: Dict[str, Any]) -> None:
    """ Validate a model object """
    missing_keys = set(MODEL_KEYS).difference(modelobj.keys())
    assert not missing_keys, f"Model object is missing keys, {missing_keys=}"

    papers = modelobj[MOK_PAPERS]
    
    missing_papers_keys = set(MODEL_PAPERS_KEYS).difference(papers.keys())
    assert not missing_papers_keys, f"Model object's papers is missing keys, {missing_papers_keys=}"
    

@_add_src_file_to_assertion_msg
def validate_paperobj(paperobj: Dict[str, Any]) -> None:
    """ Validate a paper object """
    missing_keys = set(PAPER_KEYS).difference(paperobj.keys())
    assert not missing_keys, f"Paper object is missing keys, {missing_keys=}"

    validate_paperobj_id(paperobj)


@_add_src_file_to_assertion_msg
def validate_performanceobj(performanceobj: Dict[str, Any]) -> None: 
    """ Validate a performance object """
    missing_keys = set(PERFORMANCE_KEYS).difference(performanceobj.keys())
    assert not missing_keys, f"Performance object is missing keys, {missing_keys=}"    

    source_type = performanceobj[PEK_SOURCE_TYPE]
    assert source_type in PEST_TYPES, f"Performance object source type is not valid, {source_type=}, {PEST_TYPES=}"
    
    
def validate_data(data) -> None:
    
    assert isinstance(data, dict), f"Data is not a dictionary, {type(data)=}"
    
    sorted_keys = set(data.keys())
    expected_sorted_keys = set(DATA_KEYS)
    wrong_keys = sorted_keys ^ expected_sorted_keys
    assert not wrong_keys, f"Data keys are not valid, {wrong_keys=}"
        
    # validate all key-id
    for datakey in DATA_KEYS_DICTIONARIES:
        try: 
            assert isinstance(data[datakey], dict), f"{datakey} is not a dictionary, {type(data[datakey])=}"
            validate_key_id_are_same(data[datakey])
        except AssertionError as ex: 
            raise AssertionError(f"Key-ID validation failed for {datakey=}. {ex}") from ex
    
    datasets = data[DK_DATASETS]
    metrics = data[DK_METRICS]
    models = data[DK_MODELS]
    papers = data[DK_PAPERS]
    
    performances = data[DK_PERFORMANCES]
    
    # validate performances unicity 
    performances_unicity_tuples = list(map(performance_unicity_tuple, performances))
    counts = Counter(performances_unicity_tuples)
    not_unique = [tup for tup, count in counts.items() if count > 1]
    assert len(not_unique) == 0, f"Not all performances are unique: {not_unique=}"
    
    # validate all dataset objects
    for datasetid, datasetobj in datasets.items(): 
        try: 
            validate_datasetobj(datasetobj)
        except AssertionError as ex: 
            raise AssertionError(f"Dataset object validation failed for {datasetid=}. {ex}") from ex
            
    # validate all metric objects
    for metricid, metricobj in metrics.items(): 
        try: 
            validate_metricobj(metricobj)
        except AssertionError as ex: 
            raise AssertionError(f"Metric object validation failed for {metricid=}. {ex}") from ex
    
    # validate all model objects
    for modelid, modelobj in models.items():
        try: 
            validate_modelobj(modelobj)
        except AssertionError as ex:
            raise AssertionError(f"Model object validation failed for {modelid=}. {ex}") from ex
        
    # validate all paper objects
    for paperid, paperobj in papers.items():
        try:
            validate_paperobj_id(paperobj)
        except AssertionError as ex:
            raise AssertionError(f"Paper object validation failed for {paperid=}. {ex}") from ex

    # validate all performance objects
    for performanceobj in performances:
        try:
            validate_performanceobj(performanceobj)
        except AssertionError as ex:
            raise AssertionError(f"Performance object validation failed for {performanceid=}. {ex}") from ex
    
    # "q" stands for "query"
    qdatasets = partial(jmespath.search, data=datasets)
    qmetrics = partial(jmespath.search, data=metrics)
    qmodels = partial(jmespath.search, data=models)
    qpapers = partial(jmespath.search, data=papers)
    qperformances = partial(jmespath.search, data=performances)
    
    datasetids = set(datasets.keys())
    metricids = set(metrics.keys())
    modelids = set(models.keys())
    paperids = set(papers.keys())
    
    datasetids_in_performances = set(qperformances(f"[*].{PEK_DATASET}"))
    metricids_in_performances = set(qperformances(f"[*].{PEK_METRIC}"))
    modelids_in_performances = set(qperformances(f"[*].{PEK_MODEL}"))

    paperids_in_models = set(qmodels(f'*.{MOK_PAPERS}.{MO_PK_SOURCE}'))
    paperids_in_datasets = set(qdatasets(f'*.{DSK_PAPERS}.{DS_PK_SOURCE}'))

    paperids_in_performances = set(qperformances(f'[?"{PEK_SOURCE_TYPE}" == \'{PEST_LITTERATURE}\'].{PEK_SOURCE}'))

    referenced_paperids = paperids_in_performances | paperids_in_models | paperids_in_datasets    
    
    # for any id \in {dataset, metric, model}
    # 1) all id values in the collection dictionary are used in the performances (at least once)
    # 2) all id values found in performances are from the collection dictionary
    # recal: set1 ^ set2 is the symmetric difference of set1 and set2
    
    # datasets
    mutually_exclusive = datasetids ^ datasetids_in_performances
    assert not mutually_exclusive, f"There are mutually exclusive DATASET ids between data.{DK_DATASETS} and data.{DK_PERFORMANCES}, {mutually_exclusive=}"
    
    # metrics
    mutually_exclusive = metricids ^ metricids_in_performances
    assert not mutually_exclusive, f"There are mutually exclusive METRIC ids between data.{DK_METRICS} and data.{DK_PERFORMANCES}, {mutually_exclusive=}"
    
    # models
    mutually_exclusive = modelids ^ modelids_in_performances
    assert not mutually_exclusive, f"There are mutually exclusive MODEL ids between data.{DK_MODELS} and data.{DK_PERFORMANCES}, {mutually_exclusive=}"
    
    # papers
    mutually_exclusive = paperids ^ referenced_paperids
    assert not mutually_exclusive, f"There are mutually exclusive PAPER ids between data.{DK_PAPERS} and (data.{DK_PERFORMANCES} U data.{DK_DATASETS} U data.{DK_MODELS}), {mutually_exclusive=}"
    
    # =================
    # warnings
    # =================
    
    perfobj_debug_projection_str = '[*].{dataset: dataset, metric: metric, model: model, "src-file": metadata."src-file", tagkeys: keys(tags)}'
    perfobjs_without_task_tag_querystr = f' {perfobj_debug_projection_str} | [?!contains(tagkeys, `task`)]'
    
    try:
        perfobjs_without_task_tag = qperformances(perfobjs_without_task_tag_querystr)
        
    except jmespath.exceptions.JMESPathError as ex:
        warnings.warn(f"JMESPath query failed: {perfobjs_without_task_tag_querystr=}. {ex}")

    if perfobjs_without_task_tag:
        tabchar = '\n\t'
        fmt_perfobj = lambda obj: f"dataset={obj['dataset']}, metric={obj['metric']}, model={obj['model']}, src-file={obj['src-file']}"
        warnings.warn(f"There are performance objects without a 'task' tag: \n{tabchar.join(map(fmt_perfobj, perfobjs_without_task_tag))}")