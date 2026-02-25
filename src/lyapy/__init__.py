# lyappy/__init__.py

from .maps import (
    ChaoticMap, LogisticMap, UlamMap, GaussMap,
    BernoulliMap, TentMap, AsymetricMap, ChebyshevMap, GeneralizedBernoulliMap
)

def available_maps():
    """Lista dinamicamente todas as subclasses de ChaoticMap."""
    # Buscamos as subclasses diretamente da classe base importada
    maps = [m.__name__ for m in ChaoticMap.__subclasses__()]
    print("Available maps on Lyapy:")
    for m in maps:
        print(f" - {m}")
    return maps

# Define o que será exportado ao usar 'from lyappy import *'
__all__ = [
    'available_maps', 'LogisticMap', 'UlamMap', 'GaussMap',
    'BernoulliMap', 'TentMap', 'AsymetricMap', 'ChebyshevMap','GeneralizedBernoulliMap'
]
