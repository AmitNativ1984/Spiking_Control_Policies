"""Spiking (SNN) networks: population encoding, spiking actor, spike decoding."""

from .decoder import SpikeDecoder
from .encoder import DEFAULT_TYPE_BOUNDS, PopulationSpikeEncoder, bounds_from_layout
from .pop_spiking_actor import PopulationSpikingActorNetwork
from .popsan import POPSANNetwork, POPSANNetworkBuilder

__all__ = [
    "SpikeDecoder",
    "PopulationSpikeEncoder",
    "DEFAULT_TYPE_BOUNDS",
    "bounds_from_layout",
    "PopulationSpikingActorNetwork",
    "POPSANNetwork",
    "POPSANNetworkBuilder",
]
