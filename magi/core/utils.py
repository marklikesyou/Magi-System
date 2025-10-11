from __future__ import annotations 

import asyncio 
import hashlib 
import json 
import logging 
import re 
import time 
from collections import OrderedDict 
from concurrent .futures import ThreadPoolExecutor ,as_completed 
from functools import wraps 
from typing import Any ,Callable ,Dict ,Optional ,TypeVar 

try :
    import tiktoken 
    TIKTOKEN_AVAILABLE =True 
except ImportError :
    TIKTOKEN_AVAILABLE =False 

T =TypeVar ("T")

logger =logging .getLogger (__name__ )


class LRUCache :
    def __init__ (self ,max_size :int =100 ):
        self .cache =OrderedDict ()
        self .max_size =max_size 

    def get (self ,key :str )->Optional [Any ]:
        if key in self .cache :
            self .cache .move_to_end (key )
            return self .cache [key ]
        return None 

    def put (self ,key :str ,value :Any ):
        if key in self .cache :
            self .cache .move_to_end (key )
        self .cache [key ]=value 
        if len (self .cache )>self .max_size :
            self .cache .popitem (last =False )

    def clear (self ):
        self .cache .clear ()


def hash_query (query :str ,constraints :str ="")->str :
    content =f"{query }||{constraints }"
    return hashlib .sha256 (content .encode ()).hexdigest ()


def retry_with_backoff (
max_retries :int =3 ,
initial_delay :float =1.0 ,
backoff_factor :float =2.0 ,
exceptions :tuple =(Exception ,),
):
    def decorator (func :Callable [...,T ])->Callable [...,T ]:
        @wraps (func )
        def wrapper (*args ,**kwargs )->T :
            delay =initial_delay 
            last_exception =None 

            for attempt in range (max_retries ):
                try :
                    return func (*args ,**kwargs )
                except exceptions as e :
                    last_exception =e 
                    if attempt <max_retries -1 :
                        logger .warning (
                        f"Attempt {attempt +1 } failed for {func .__name__ }: {e }. "
                        f"Retrying in {delay } seconds..."
                        )
                        time .sleep (delay )
                        delay *=backoff_factor 
                    else :
                        logger .error (
                        f"All {max_retries } attempts failed for {func .__name__ }: {e }"
                        )

            raise last_exception 

        return wrapper 

    return decorator 


def sanitize_input (text :str ,max_length :int =10000 )->str :
    if not text or not isinstance (text ,str ):
        return ""

    text =text [:max_length ]

    text =re .sub (r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]","",text )

    text =re .sub (r"<script[^>]*>.*?</script>","",text ,flags =re .IGNORECASE |re .DOTALL )
    text =re .sub (r"javascript:","",text ,flags =re .IGNORECASE )

    return text .strip ()


def validate_json_response (response :str ,required_keys :list [str ])->Dict [str ,Any ]:
    if not response :
        return {}

    response =response .strip ()
    if response .startswith ("```json"):
        response =response [7 :]
    if response .startswith ("```"):
        response =response [3 :]
    if response .endswith ("```"):
        response =response [:-3 ]

    try :
        data =json .loads (response )
        if not isinstance (data ,dict ):
            return {}

        for key in required_keys :
            if key not in data :
                data [key ]=""if key !="confidence"else 0.5 

        return data 
    except json .JSONDecodeError :
        logger .warning (f"Failed to parse JSON response: {response [:100 ]}...")
        return {}


def count_tokens (text :str ,model :str ="gpt-4")->int :
    if TIKTOKEN_AVAILABLE :
        try :
            if "gpt"in model .lower ():
                encoding =tiktoken .encoding_for_model (model )
            else :
                encoding =tiktoken .get_encoding ("cl100k_base")
            return len (encoding .encode (text ))
        except Exception :
            return len (text )//4 
    else :
        return len (text )//4 


def truncate_to_token_limit (text :str ,max_tokens :int ,model :str ="gpt-4")->str :
    tokens =count_tokens (text ,model )
    if tokens <=max_tokens :
        return text 

    if TIKTOKEN_AVAILABLE :
        try :
            if "gpt"in model .lower ():
                encoding =tiktoken .encoding_for_model (model )
            else :
                encoding =tiktoken .get_encoding ("cl100k_base")

            encoded =encoding .encode (text )
            truncated =encoded [:max_tokens ]
            return encoding .decode (truncated )
        except Exception :
            char_limit =(max_tokens *4 )-100 
            return text [:char_limit ]
    else :
        char_limit =(max_tokens *4 )-100 
        return text [:char_limit ]


def parallel_execute (functions :list [Callable ],max_workers :int =3 )->list [Any ]:
    results =[None ]*len (functions )

    with ThreadPoolExecutor (max_workers =max_workers )as executor :
        future_to_index ={
        executor .submit (func ):i 
        for i ,func in enumerate (functions )
        }

        for future in as_completed (future_to_index ):
            index =future_to_index [future ]
            try :
                results [index ]=future .result ()
            except Exception as e :
                logger .error (f"Parallel execution failed for task {index }: {e }")
                results [index ]=None 

    return results 


async def async_parallel_execute (functions :list [Callable ])->list [Any ]:
    tasks =[asyncio .create_task (asyncio .to_thread (func ))for func in functions ]
    return await asyncio .gather (*tasks ,return_exceptions =True )


class TokenTracker :
    def __init__ (self ):
        self .total_input_tokens =0 
        self .total_output_tokens =0 
        self .total_cost =0.0 
        self .model_costs ={
        "gpt-4o":{"input":0.03 ,"output":0.06 },
        "gpt-4o-mini":{"input":0.00015 ,"output":0.0006 },
        "gpt-3.5-turbo":{"input":0.0015 ,"output":0.002 },
        "gemini-2.0-flash-exp":{"input":0.0 ,"output":0.0 },
        "gemini-2.5-flash-lite":{"input":0.0 ,"output":0.0 },
        }

    def track (self ,input_text :str ,output_text :str ,model :str ):
        input_tokens =count_tokens (input_text ,model )
        output_tokens =count_tokens (output_text ,model )

        self .total_input_tokens +=input_tokens 
        self .total_output_tokens +=output_tokens 

        if model in self .model_costs :
            costs =self .model_costs [model ]
            self .total_cost +=(input_tokens *costs ["input"]/1000 )+(
            output_tokens *costs ["output"]/1000 
            )

    def get_stats (self )->Dict [str ,Any ]:
        return {
        "total_input_tokens":self .total_input_tokens ,
        "total_output_tokens":self .total_output_tokens ,
        "total_tokens":self .total_input_tokens +self .total_output_tokens ,
        "estimated_cost_usd":round (self .total_cost ,4 ),
        }

    def reset (self ):
        self .total_input_tokens =0 
        self .total_output_tokens =0 
        self .total_cost =0.0 


class CircuitBreaker :
    def __init__ (
    self ,
    failure_threshold :int =5 ,
    recovery_timeout :float =60.0 ,
    expected_exception :type =Exception ,
    ):
        self .failure_threshold =failure_threshold 
        self .recovery_timeout =recovery_timeout 
        self .expected_exception =expected_exception 
        self .failure_count =0 
        self .last_failure_time =None 
        self .state ="closed"

    def call (self ,func :Callable [...,T ],*args ,**kwargs )->T :
        if self .state =="open":
            if (
            self .last_failure_time 
            and time .time ()-self .last_failure_time >self .recovery_timeout 
            ):
                self .state ="half-open"
            else :
                raise Exception (f"Circuit breaker is open for {func .__name__ }")

        try :
            result =func (*args ,**kwargs )
            if self .state =="half-open":
                self .state ="closed"
                self .failure_count =0 
            return result 
        except self .expected_exception as e :
            self .failure_count +=1 
            self .last_failure_time =time .time ()

            if self .failure_count >=self .failure_threshold :
                self .state ="open"
                logger .error (f"Circuit breaker opened for {func .__name__ }")

            raise e 

    def reset (self ):
        self .failure_count =0 
        self .last_failure_time =None 
        self .state ="closed"