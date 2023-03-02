from .veri import VeRi
from .vehicleid import VehicleID
from .duke import duke


__imgreid_factory = {
    'veri': VeRi,
    'vehicleID': VehicleID,
    'duke': duke,
}
def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)
