import numpy as np
import pandas as pd
from py2neo import GraphService
from py2neo.matching import NodeMatcher, RelationshipMatcher
from py2neo.data import Node, Relationship
import toml
from typing import Literal, Union, Optional, Dict
import functools, signal, errno
from typing import NewType
import os


_SECRETS = toml.load("./.streamlit/secrets.toml")
SERVICE = GraphService(
    _SECRETS["database"]["neo4j_endpoint"],
    auth=(_SECRETS["database"]["neo4j_username"], 
          _SECRETS["database"]["neo4j_password"]))

class Match:
    def __init__(self, graphname: str):
        self.graph = SERVICE[graphname]
        self.nodematcher = NodeMatcher(self.graph)
        self.relationmatcher = RelationshipMatcher(self.graph)
    
    def _match_node(self, 
                    nodetype: str, 
                    properties: Dict, 
                    **kwargs):
        return self.nodematcher.match(nodetype, **properties, **kwargs)

    def _count_node(self,
                   nodetype: str, 
                   properties: Dict, 
                   **kwargs):
        return self._match_node(nodetype, properties, **kwargs).count()
    
    @classmethod
    def count_node(cls, graphname: str,
                   nodetype: str, 
                   properties: Dict,
                   **kwargs):
        return cls(graphname)._count_node(nodetype, properties, **kwargs)




class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator
