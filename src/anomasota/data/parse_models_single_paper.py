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


# these are redifined here because files parsed with this parser follow a different structure
DATA_MANDATORY_KEYS = ("models", "papers", "performances", )
DATA_OPTIONAL_KEYS = ("models-common-tags", "performances-common-tags", )
DATA_KEYS = DATA_MANDATORY_KEYS + DATA_OPTIONAL_KEYS

# same for the performanceobj
PERFOBJ_MANDATORY_KEYS = ("model", "metric", "tags", "extra", "datasets" )  # datasetS (in the plural!!!)
PERFOBJ_FORBIDEN_KEYS = ("source", "source-type")
 

def parse_models_single_paper(jsonfpath: Path) -> Dict[str, Any]:
    
    data = json5.loads(jsonfpath.read_text())
    data_keys = set(data.keys())

    assert data_keys.issubset(DATA_KEYS), f"Data keys are not valid, {data_keys=}, {DATA_KEYS=}"
    assert data_keys.issuperset(DATA_MANDATORY_KEYS), f"Data keys are not valid, {data_keys=}, {DATA_MANDATORY_KEYS=}"
    
    papers = data.pop('papers')
    models = data.pop('models')
    performances = data.pop('performances')

    common.validate_key_id_are_same(papers)
    common.validate_key_id_are_same(papers)

    assert len(papers) == 1, f"Only one paper is allowed, {papers=}"
    assert len(models) > 0, f"At least one model is required, {models=}"
    assert len(performances) >= len(models), f"At least one performance is required for each model, {performances=}, {models=}"
    
    paper = list(papers.values())[0]
    common.validate_paperobj_id(paper)
    
    models_common_tags = data.pop("models-common-tags", {})
    models_common_tagkeys = set(models_common_tags.keys())
    
    performances_comon_tags = data.pop("performances-common-tags", {})
    performances_comon_tagkeys = set(performances_comon_tags.keys())
    
    models_parsed = {}
    for modelid, modelobj in models.items():
        
        common.validate_modelobj_id(modelobj)

        # the models should not have source paper because they will be added later
        assert modelobj.get("papers", {}).get("source", None) is None, f"modelobj.papers.source should be empty for this parser (it is automatically set), {modelid=}"
        
        new_modelobj = deepcopy(modelobj)
        
        # inject paperid
        new_modelobj.setdefault("papers", {})["source"] = paper["id"]
        
        model_tags = modelobj.get("tags", {})
        model_tagkeys = set(model_tags.keys())
        
        conflicting_tagkeys = model_tagkeys.intersection(models_common_tagkeys)
        assert len(conflicting_tagkeys) == 0, f"Conflicting model tags, {conflicting_tagkeys=}"
        
        # inject common tags
        new_modelobj["tags"] = {**models_common_tags, **model_tags}
        
        models_parsed[modelid] = new_modelobj
    
    performances_parsed = []
    for perfobj in performances:
        
        perfobj_keys = set(perfobj.keys())
        
        assert set(PERFOBJ_MANDATORY_KEYS).issubset(perfobj_keys), f"Performance object keys are not valid, {perfobj.keys()=}, {PERFOBJ_MANDATORY_KEYS=}"
        
        forbidden_keys_in_obj = set(PERFOBJ_FORBIDEN_KEYS).intersection(perfobj_keys)
        assert len(forbidden_keys_in_obj) == 0, f"Performance object keys are not valid, {perfobj.keys()=}, {PERFOBJ_FORBIDEN_KEYS=}"
        
        modelid = perfobj["model"]
        assert modelid in models_parsed, f"Model ID is not valid, it must be one of the models declared in the same file, {modelid=}, {models_parsed.keys()=}"
        
        perfs_per_dataset: Dict["str", float] = perfobj.pop("datasets")  # datasetS (in the plural!!!)
        
        assert len(perfs_per_dataset) > 0, f"At least one dataset is required, {perfs_per_dataset=}"
        
        prototype_perfobj = deepcopy(perfobj)
                
        prototype_perfobj["source-type"] = "litterature"
        prototype_perfobj["source"] = paper["id"]
        
        perfobj_tags = perfobj["tags"]
        perfobj_tagkeys = set(perfobj_tags.keys())
        
        conflicting_tagkeys = perfobj_tagkeys.intersection(performances_comon_tagkeys)
        assert len(conflicting_tagkeys) == 0, f"Conflicting performance tags, {conflicting_tagkeys=}"
        
        # merge common tags
        prototype_perfobj["tags"] = {**performances_comon_tags, **perfobj_tags}
        
        for datasetid, value in perfs_per_dataset.items():
            new_perfobj = deepcopy(prototype_perfobj)
            new_perfobj["dataset"] = datasetid
            new_perfobj["value"] = value
            performances_parsed.append(new_perfobj)
    
    data["papers"] = papers
    data['models'] = models_parsed
    data["performances"] = performances_parsed
    
    return data