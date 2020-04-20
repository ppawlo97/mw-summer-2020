from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense


class MLP(tf.keras.Model):
    def __init__(self,
                 num_classes: int,
                 hidden_units: Tuple[int] = (256, 256),
                 name: str = "mlp",
                 **kwargs):
        super(MLP, self).__init__(name=name, **kwargs)

        if not hidden_units:
            raise Exception("MLP must have at least one hidden layer!")
        
        self.num_hidden = len(hidden_units)

        for ix, units in enumerate(hidden_units, start=1):
            setattr(self, f"dense_{ix}", Dense(units,
                                               activation="relu",
                                               name=f"dense_{ix}"))

        self.classifier = Dense(num_classes, name="classifier")

    
    def call(self, inputs, training=False):
        x = self.dense_1(inputs)
        for ix in range(2, self.num_hidden + 1):
            layer = self.get_layer(f"dense_{ix}")
            x = layer(x)
        output = self.classifier(x)
        return output