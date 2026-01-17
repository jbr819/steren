from abc import ABC, abstractmethod
import numpy as np


class Network(ABC):
    @abstractmethod
    def load_weights(self):
        pass

    @abstractmethod
    def inference(self, image : np.ndarray):
        pass

    @abstractmethod
    def set_device(self, device : str):
        pass


class Detector(Network, ABC):
    pass 
    

class Segmenter(Network,ABC):
    pass
   

