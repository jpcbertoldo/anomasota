"""Queries on data.json"""

from copy import deepcopy
import json5
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
import time
import re
import datetime
import textwrap
import warnings
import hashlib

import click
import jmespath
import pandas as pd
from pandas import DataFrame

from anomasota import data as datamodule

# from scipy.stats import wilcoxon

DataFrameWithOptionalTagsUniqueValues = Union[DataFrame, Tuple[DataFrame, Dict[str, List[str]]]]


@click.group()
@click.option(
    '--datadir', 
    default=datamodule.DEFAULT_DATADIR_PATH, 
    type=click.Path(
        exists=True, 
        file_okay=False, 
        dir_okay=True, 
        readable=True, 
        writable=True, 
        allow_dash=False, 
        path_type=Path,
    )
)
@click.pass_context
def cli(context, datadir):
    pass
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called by means other than the `if` block below)
    context.ensure_object(dict)
    context.obj['datadir'] = datadir
    

@cli.command("models")
@click.pass_context
def command_models(context):
    context.obj["return_tag_unique_values"] = False
    return models(**context.obj)


def models(datadir: Path = datamodule.DEFAULT_DATADIR_PATH, return_tag_unique_values: bool = True) -> DataFrameWithOptionalTagsUniqueValues:
    """Get all models from data.json"""
   
    data, datasets, metrics, models, papers, performances, model_tagkeys, performance_tagkeys = datamodule.load(datadir)
        
    MODEL_FIELDS_TOCOPY = (datamodule.common.MOK_ID, datamodule.common.MOK_NAME, datamodule.common.MOK_VERSION, )
    
    records = [
        {
            **{objkey: modelobj[objkey] for objkey in MODEL_FIELDS_TOCOPY},
            **{tagkey: modelobj[datamodule.common.MOK_TAGS].get(tagkey, None) for tagkey in model_tagkeys}
        }
        for modelobj in models.values()
    ]
    
    modelsdf = pd.DataFrame.from_records(records).set_index(datamodule.common.MOK_ID)
    
    tags_unique_values = {
        tagkey: sorted(tagval for tagval in modelsdf[tagkey].unique() if tagval is not None)
        for tagkey in model_tagkeys
    }
    
    if return_tag_unique_values:
        return modelsdf, tags_unique_values
    
    return modelsdf


@cli.command("performances")
@click.pass_context
def command_performances(context):
    context.obj["return_tag_unique_values"] = False
    return performances(**context.obj)


def performances(datadir: Path = datamodule.DEFAULT_DATADIR_PATH, return_tag_unique_values: bool = True) -> DataFrameWithOptionalTagsUniqueValues:

    data, datasets, metrics, models, papers, performances, model_tagkeys, performance_tagkeys = datamodule.load(datadir)
    
    PERFORMANCE_FIELDS_TOCOPY = (
        datamodule.common.PEK_DATASET,
        datamodule.common.PEK_METRIC,
        datamodule.common.PEK_MODEL,
        datamodule.common.PEK_EXTRA,
        datamodule.common.PEK_VALUE,   
    )
    
    def get_tags(perfobj_):
        return {
            tagkey: perfobj_[datamodule.common.PEK_TAGS].get(tagkey, None) 
            for tagkey in performance_tagkeys
        } 
    
    records = [
        {
            datamodule.common.PEK_SOURCE: f"{perfobj[datamodule.common.PEK_SOURCE_TYPE]}/{perfobj[datamodule.common.PEK_SOURCE]}",
            **{objkey: perfobj[objkey] for objkey in PERFORMANCE_FIELDS_TOCOPY},
            **get_tags(perfobj),
        }
        for perfobj in performances
    ]
    
    performancesdf = pd.DataFrame.from_records(records)
    
    # ensure unicity of (dataset, metric, model, source)
    UNICITY_FIELDS = ["dataset", "metric", "model", "source"]
    COLUMNS_TO_DROP = performance_tagkeys + [datamodule.common.PEK_EXTRA]
    duplicate_counts = performancesdf.drop(columns=COLUMNS_TO_DROP).groupby(UNICITY_FIELDS).size().unique().tolist()
    assert len(duplicate_counts) == 1 and duplicate_counts[0] == 1, f"Duplicate counts: {duplicate_counts}"
    
    # TODO MAKE ALERT FOR NON-UNICITY OF SOURCES
    
    tags_unique_values = {
        tagkey: sorted(tagval for tagval in performancesdf[tagkey].unique() if tagval is not None)
        for tagkey in performance_tagkeys
    }
        
    if return_tag_unique_values:
        return performancesdf, tags_unique_values
    
    return performancesdf

if __name__ == "__main__":
    cli()
