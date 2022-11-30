""" Parse a .json that cotains a single paper, several models, and their respective performances.

It is assumed that all the models and performances are from the same paper, which should be unique in the json file.

ATTENTION: the validations and assumptions here are not generic, they are specific to json files supposed to be parsed with this parser.


MVTecAD categories order:

"""

import json5
from pathlib import Path
from copy import deepcopy

from typing import Dict, List, Tuple, Any
import time
import datetime
import warnings

from anomasota.data import common
from anomasota.data.common import (
    DK_DATASETS,
    DK_METRICS,
    DK_MODELS,
    DK_PAPERS,
    DK_PERFORMANCES,
    MOK_ID,
    MOK_NAME,
    MOK_VERSION,
    MOK_TAGS,
    MOK_PAPERS,
    MO_PK_SOURCE,
    MODEL_ID_KEYS,
    PAK_ID,
    PAK_NAME,
    PAK_AUTHOR,
    PAK_YEAR,
    PAPER_ID_KEYS,
    PEK_DATASET,
    PEK_METRIC,
    PEK_MODEL,
    PEK_SOURCE_TYPE,
    PEK_SOURCE,
    PERFORMANCE_UNICITY_KEYS,
    PEK_EXTRA,
    PEK_TAGS,
    PEK_VALUE,
    PERFORMANCE_KEYS,
    PERFORMANCE_SOURCE_TYPE_LITTERATURE,
)


# these are redifined here because files parsed with this parser follow a different structure
_DATA_MANDATORY_KEYS = (DK_MODELS, DK_PAPERS, DK_PERFORMANCES, )

_DK_ADHOC_MODELS_COMMON_TAGS = "models-common-tags"
_DK_ADHOC_PERFORMANCES_COMMON_TAGS = "performances-common-tags"
_DATA_OPTIONAL_KEYS = (_DK_ADHOC_MODELS_COMMON_TAGS, _DK_ADHOC_PERFORMANCES_COMMON_TAGS, )

_ADHOC_DATA_KEYS = _DATA_MANDATORY_KEYS + _DATA_OPTIONAL_KEYS

# modelobj
_MODELOBJ_FORBIDDEN_KEYS = (MOK_ID,) + MODEL_ID_KEYS

# paperobj
_PAPEROBJ_FORBIDDEN_KEYS = (PAK_ID, ) + PAPER_ID_KEYS

# performanceobj
_PERFOBJ_MANDATORY_KEYS = ("model", "metric", "tags", "extra", )
_PERFOBJ_FORBIDEN_KEYS = ("source", "source-type")
_PERFOBJ_ADHOC_KEY_DATASETS = "datasets"  # datasetS in the plural!!!

        
def _inject_modelobj_fields(modelobj: Dict[str, Any], modelid: str, source_paperid: str, common_model_tags: Dict[str, str]) -> Dict[str, Any]:

    # the MODEL should not have source paper because they will be added later
    assert MO_PK_SOURCE not in modelobj.get(MOK_PAPERS, {}), f"modelobj.papers.source should be empty for this parser (it is automatically set), {modelid=}, found {modelobj[MOK_PAPERS].keys()=}"
    
    # the MODEL should not have the id field and its parsed fields because they will be injected
    model_keys = set(modelobj.keys())
    forbidden_keys_in_modelobj = set(_MODELOBJ_FORBIDDEN_KEYS) & model_keys

    assert not forbidden_keys_in_modelobj, f"modelobj should not have the following keys: {sorted(_MODELOBJ_FORBIDDEN_KEYS)}; found {sorted(forbidden_keys_in_modelobj)}"

    model_name, model_version = common.parse_modelid(modelid)

    model_tags = modelobj.get(MOK_TAGS, {})
    model_tagkeys = set(model_tags.keys())
    conflicting_tagkeys = model_tagkeys.intersection(set(common_model_tags.keys()))

    assert len(conflicting_tagkeys) == 0, f"Conflicting model tags, {conflicting_tagkeys=}"
    
    modelobj[MOK_ID] = modelid        
    modelobj[MOK_NAME] = model_name
    modelobj[MOK_VERSION] = model_version
    modelobj.setdefault(MOK_PAPERS, {})[MO_PK_SOURCE] = source_paperid
    modelobj[MOK_TAGS] = {**common_model_tags, **model_tags}
    
    return modelobj


def _inject_paperobj_fields(paperobj: Dict[str, Any], paperid: str) -> Dict[str, Any]:
    
    # the PAPER should not have the id field and its parsed fields because they will be injected
    paper_keys = set(paperobj.keys())
    forbidden_keys_in_paperobj = set(_PAPEROBJ_FORBIDDEN_KEYS) & paper_keys
    
    assert not forbidden_keys_in_paperobj, f"paperobj should not have the following keys:  {sorted(_PAPEROBJ_FORBIDDEN_KEYS)}; found {sorted(forbidden_keys_in_paperobj)}"
    
    paper_name, paper_author, paper_year = common.parse_paperid(paperid)
    
    paperobj[PAK_ID] = paperid
    paperobj[PAK_NAME] = paper_name
    paperobj[PAK_AUTHOR] = paper_author
    paperobj[PAK_YEAR] = paper_year
    
    return paperobj
    
    
def _get_perfobjs_from_prototype(protoperfobj: Dict[str, Any], performances_perdataset: Dict[str, float], paperid: str, common_perf_tags: Dict[str, str]) -> List[Dict[str, Any]]:
    
    protoperfobj[PEK_SOURCE_TYPE] = PERFORMANCE_SOURCE_TYPE_LITTERATURE
    protoperfobj[PEK_SOURCE] = paperid
    
    protoperfobj_tags = protoperfobj[PEK_TAGS]
    
    conflicting_tagkeys = set(protoperfobj_tags.keys()).intersection(set(common_perf_tags.keys()))
    assert not conflicting_tagkeys, f"Conflicting performance tags, {conflicting_tagkeys=}"

    # merge common tags
    protoperfobj[PEK_TAGS] = {**common_perf_tags, **protoperfobj_tags}
    
    return_perfobjs = []
    
    for datasetid, value in performances_perdataset.items():
    
        perfobj = deepcopy(protoperfobj)
        
        perfobj[PEK_DATASET] = datasetid
        perfobj[PEK_VALUE] = value
        
        return_perfobjs.append(perfobj)
    
    return return_perfobjs

    
def parse_models_single_paper(jsonfpath: Path) -> Dict[str, Any]:
    
    data = json5.loads(jsonfpath.read_text())

    data_keys = set(data.keys())
    assert data_keys.issubset(_ADHOC_DATA_KEYS), f"Data keys are not valid, {data_keys=}, {_ADHOC_DATA_KEYS=}"
    assert data_keys.issuperset(_DATA_MANDATORY_KEYS), f"Data keys are not valid, {data_keys=}, {_DATA_MANDATORY_KEYS=}"
    
    papers = data.pop(DK_PAPERS)
    models = data.pop(DK_MODELS)
    performances = data.pop(DK_PERFORMANCES)
    
    models_common_tags = data.pop("models-common-tags", {})
    performances_comon_tags = data.pop("performances-common-tags", {})

    # don't validate because they will be deduced from the key
    # common.validate_key_id_are_same(papers)

    assert len(papers) == 1, f"Only one paper is allowed, {papers=}"
    assert len(models) > 0, f"At least one model is required, {models=}"
    assert len(performances) >= len(models), f"At least one performance is required for each model, {performances=}, {models=}"
    
    paperid = list(papers.keys())[0]
    paperobj = papers[paperid]
    
    try:
        paperobj = _inject_paperobj_fields(deepcopy(paperobj), paperid)
        common.validate_paperobj(paperobj)  # double check...
        
    except AssertionError as ex:
        raise common.ParsingError(f"Error parsing paper {paperid=} {jsonfpath.name=} {ex}") from ex

    papers_parsed = {paperid: paperobj}
    
    models_parsed = {}
    for modelid, modelobj in models.items():
        
        try:
            modelobj = _inject_modelobj_fields(deepcopy(modelobj), modelid, paperid, models_common_tags)
            common.validate_modelobj(modelobj)  # double check
            
        except AssertionError as ex:
            raise common.ParsingError(f"Error parsing model {modelid=} {jsonfpath.name=} {ex}") from ex
        
        models_parsed[modelid] = modelobj
    
    performances_parsed = []
    
    # "proto" stands for "prototype" because the performances are not yet parsed and they have multiple datasets
    for protoperfobj in performances:
        
        perfobj_keys = set(protoperfobj.keys())
        missing_mandatory_keys = set(_PERFOBJ_MANDATORY_KEYS) - perfobj_keys
        forbidden_keys_in_obj = set(_PERFOBJ_FORBIDEN_KEYS).intersection(perfobj_keys)

        assert not missing_mandatory_keys, f"Missing mandatory keys in performanceobj, {missing_mandatory_keys=} {jsonfpath.name=}"
        
        assert not forbidden_keys_in_obj, f"Forbidden keys found in performanceobj, {forbidden_keys_in_obj=} {jsonfpath.name=}"
        
        assert _PERFOBJ_ADHOC_KEY_DATASETS in protoperfobj, f"Missing mandatory key {_PERFOBJ_ADHOC_KEY_DATASETS=} {jsonfpath.name=}"
    
        modelid = protoperfobj[PEK_MODEL]
        perfs_per_dataset: Dict[str, float] = protoperfobj.pop(_PERFOBJ_ADHOC_KEY_DATASETS) 
        
        assert modelid in models_parsed, f"Model ID is not valid, it must be one of the models declared in the same file; found {modelid=} but available models are {models_parsed.keys()=} in {jsonfpath.name=}"
        
        assert len(perfs_per_dataset) > 0, f"At least one dataset is required, {perfs_per_dataset=} {jsonfpath.name=}"
        
        try:
            perfobjs = _get_perfobjs_from_prototype(deepcopy(protoperfobj), perfs_per_dataset, paperid, performances_comon_tags)
            
            for perfobj in perfobjs:
                common.validate_performanceobj(perfobj)
            
        except AssertionError as ex:
            raise common.ParsingError(f"Error parsing performanceobj prototype {jsonfpath.name=} {ex}") from ex
        
        performances_parsed.extend(perfobjs)
        
    data[DK_PAPERS] = papers_parsed
    data[DK_MODELS] = models_parsed
    data[DK_PERFORMANCES] = performances_parsed
    
    return data