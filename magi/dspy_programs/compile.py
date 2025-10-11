

from __future__ import annotations 

from typing import Any ,Callable ,Iterable 

import dspy 
from dspy .teleprompt import BootstrapFewShot ,MIPROv2 

def compile_program (
program :dspy .Module ,
trainset :Iterable [Any ],
metric_fn :Callable [[Any ,Any ],float ],
*,
heavy :bool =False ,
)->dspy .Module :
    optimizer =MIPROv2 (metric =metric_fn ,auto ="heavy"if heavy else "medium")
    return optimizer .compile (program ,trainset =trainset )

def bootstrap_program (
program :dspy .Module ,
trainset :Iterable [Any ],
metric_fn :Callable [[Any ,Any ],float ],
*,
shots :int =6 ,
)->dspy .Module :
    optimizer =BootstrapFewShot (metric =metric_fn ,max_bootstrapped_demos =shots )
    return optimizer .compile (program ,trainset =trainset )
