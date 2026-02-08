import numpy as np
from enum import Enum, auto


## Initial label types to store, to be clarified and then will need to be versioned (probably)
LABEL_DTYPE_EXP = np.dtype([
    ("cc",        np.bool_),
    ("topology",  np.int8),
    ("mode",      np.int8),
    ("nneutron",  np.int8),
    ("nantineut", np.int8),
    ("nproton",   np.int8),    
    ("nantiprot", np.int8),    
    ("npipm",     np.int8),
    ("npi0",      np.int8),
    ("nkapm",     np.int8),
    ("nka0",      np.int8),
    ("nem",       np.int8),
    ("nmuon",     np.int8),    
    ("nstrange",  np.int8),
    ("ncharm",    np.int8),    
    ("enu",       np.float32),
    ("q0",        np.float32),
])

class Topology(Enum):

    ## Default
    NONE = -1

    ## CC topologies
    CC0pi = auto()
    CC1pi0 = auto()
    CC1pipm = auto()
    CC2pi = auto()
    CCNpi = auto()
    CCOther = auto()

    ## NC topologies
    NC0pi = auto()
    NC1pipm = auto()
    NC1pi0 = auto()
    NC2pi = auto()
    NCNpi = auto()
    NCOther = auto()
    
    ## A method to dump the list
    @classmethod
    def print_members(cls):
        for member in cls:
            print(f"{member.name}: {member.value}")

    @classmethod
    def name_from_index(cls, index):
        for member in cls:
            if member.value == index:
                return member.name
        return f"Unknown label for index {index}"

class Mode(Enum):

    ## Default
    NONE = -1

    ## CC modes
    CCQE = auto()
    CC2p2h = auto()
    CCRES = auto()
    CCDIS = auto()
    CCCOH = auto()

    ## NC modes
    NCQE = auto()
    NC2p2h = auto()
    NCRES = auto()
    NCDIS = auto()
    NCCOH = auto()

    ## Other
    IMD = auto()
    NUEE = auto()
    
    ## A method to dump the list
    @classmethod
    def print_members(cls):
        for member in cls:
            print(f"{member.name}: {member.value}")

    @classmethod
    def name_from_index(cls, index):
        for member in cls:
            if member.value == index:
                return member.name
        return f"Unknown label for index {index}"
