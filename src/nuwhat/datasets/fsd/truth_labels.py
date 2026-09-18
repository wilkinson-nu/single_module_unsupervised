from enum import Enum, auto

class Label(Enum):

    ## Default
    DATA = -1
    
    ## e+/- or photon induced
    EM = auto()

    ## Neutron induced
    NEUTRON = auto()

    ## Proton induced
    PROTON = auto()

    ## Charged pion induced
    PION = auto()
    
    ## Multiple muons deposit energy into the active volume
    MULTIMUON = auto()

    ## The muon interacted outside the active volume
    EXTERNAL = auto()

    ## Stopping muon which is captured by a nucleus and then decays not through a Michel
    STOPPINGCAPTURE = auto()

    ## Muon which decays into a Michel inside the volume
    STOPPINGMICHEL = auto()

    ## This is a catch-all category for stopping events that aren't the other two...
    STOPPINGOTHER = auto()
    
    ## This is meant to be for through-going muons without much colinear activity
    THROUGHCLEAN = auto()

    ## This is to try and get at through-going muons with colinear showers
    THROUGHMESSY = auto()

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
        
