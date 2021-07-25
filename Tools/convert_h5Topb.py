import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

modelloc=str(sys.argv[1])
pbmodelloc=str(sys.argv[2])
model = keras.models.load_model(modelloc)

import cmsml
cmsml.tensorflow.save_graph(pbmodelloc, model, variables_to_constants=True)
