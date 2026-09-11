import numpy as np
from enum import Enum, auto


## The particle stack labels (for both truth and visible
PARTICLE_STACK_DTYPE = np.dtype([
    ("nneutron",  np.int8),
    ("nantineut", np.int8),
    ("nproton",   np.int8),    
    ("nantiprot", np.int8),
    ("npipm",     np.int8),
    ("npip",      np.int8),
    ("npim",      np.int8),
    ("npi0",      np.int8),
    ("nkapm",     np.int8),
    ("nkap",      np.int8),
    ("nkam",      np.int8),
    ("nka0",      np.int8),
    ("nem",       np.int8),
    ("nepm",      np.int8),
    ("ngamma",    np.int8),    
    ("nmuon",     np.int8),    
    ("nlambda0",  np.int8),
    ("ndeuteron", np.int8),
    ("ntritium",  np.int8),
    ("nalpha",    np.int8),
    ("nhelium3",  np.int8),
    ("nnuclfrag", np.int8),
    ("ncharged",  np.int8),
    ("ncluster",  np.int8),    
])

## These are event summary variables
EVENT_LABEL_DTYPE = np.dtype([

    ## Interaction truth
    ("cc",        np.bool_),
    ("enu",       np.float32),
    ("q0",        np.float32),
    ("mode",      np.int8),

    ## Summaries from the two particle stacks
    ("topology_truth",    np.int8),
    ("topology_visible",  np.int8),
    ("cctopology_truth",  np.int8),
    ("cctopology_visible",np.int8),

    ## Global deposition features
    ("edep_5mm",      np.float32),
    ("edep_10mm",     np.float32),
    ("edep_20mm",     np.float32),
    ("edep_50mm",     np.float32),
])

class Topology(Enum):

    ## Default
    NONE = -1

    ## CC topologies
    CC0pi = auto() ## 0
    CC1pi0 = auto() ## 1
    CC1pipm = auto() ## 2
    CC2pi = auto() ## 3
    CCNpi = auto() ## 4
    CCOther = auto() ## 5

    ## NC topologies
    NC0pi = auto() ## 6
    NC1pipm = auto() ## 7
    NC1pi0 = auto() ## 8
    NC2pi = auto() ## 9
    NCNpi = auto() ## 10
    NCOther = auto() ## 11
    
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

class CCTopology(Enum):

    ## Default
    NONE = -1
    CC0pi0p = auto()                    # 0
    CC0pi1p = auto()                    # 1
    CC0piNp = auto()                    # 2

    CC1pipm0pi0_0p = auto()             # 3
    CC1pipm0pi0_1p = auto()             # 4
    CC1pipm0pi0_Np = auto()             # 5

    CCNpipm0pi0_0p = auto()             # 6
    CCNpipm0pi0_1p = auto()             # 7
    CCNpipm0pi0_Np = auto()             # 8

    CC0pipm1pi0_0p = auto()             # 9
    CC0pipm1pi0_1p = auto()             # 10
    CC0pipm1pi0_Np = auto()             # 11

    CC0pipmNpi0_0p = auto()             # 12
    CC0pipmNpi0_1p = auto()             # 13
    CC0pipmNpi0_Np = auto()             # 14

    CCmixedpi_0p = auto()               # 15
    CCmixedpi_1p = auto()               # 16
    CCmixedpi_Np = auto()               # 17

    CCOther = auto()                    # 18
    NC = auto()                         # 19
    
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
    
