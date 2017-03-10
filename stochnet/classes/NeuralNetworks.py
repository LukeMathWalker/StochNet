from keras.models import Model


class StochNeuralNetwork:

    def __init__(self, input_tensor, NN_body, TopLayer_obj):
        # TODO: fix eliminating input_tensor argument embedding it in NN_body somehow
        self.input_tensor = input_tensor
        self.body = NN_body
        self.TopLayer_obj = TopLayer_obj
        output_layer = self.output_layer_obj.get_layer()
        output = output_layer(self.body)
        self.model = Model(input=self.input_tensor, output=output)
        self.model.compile(optimizer='adam', loss=self.output_layer_obj.loss_function)

    def fit(self, X_data, y_data, batch_size=32, nb_epoch=10, verbose=1, callbacks=None, validation_split=0.0):
        self.model.fit(X_data,
                       y_data,
                       batch_size=batch_size,
                       nb_epoch=nb_epoch,
                       callbacks=callbacks,
                       validation_split=validation_split)

    def predict(self, X_data, batch_size=32, verbose=0):
        return self.model.predict(X_data, batch_size=batch_size, verbose=verbose)
