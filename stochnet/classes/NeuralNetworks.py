from keras.models import Model
import tensorflow as tf


class StochNeuralNetwork:

    def __init__(self, input_tensor, NN_body, TopLayer_obj, mode='min', optimizer='adam'):
        self.input_tensor = input_tensor
        self.body = NN_body
        self.TopLayer_obj = TopLayer_obj
        output_layer = self.TopLayer_obj.add_layer_on_top(self.body)
        self.model = Model(inputs=self.input_tensor, outputs=output_layer)
        self.model.compile(optimizer=optimizer,
                           loss=self.TopLayer_obj.loss_function)

    def fit(self, X_data, y_data, batch_size=32, epochs=10, verbose=1,
            callbacks=None, validation_split=0.0, validation_data=None):
        return self.model.fit(X_data,
                       y_data,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=callbacks,
                       validation_split=validation_split,
                       validation_data=validation_data)

    def fit_generator(self, training_generator, samples_per_epoch, epochs=5, verbose=1, callbacks=None, validation_generator=None, nb_val_samples=100):
        return self.model.fit_generator(generator=training_generator,
                                 samples_per_epoch=samples_per_epoch,
                                 epochs=epochs,
                                 verbose=verbose,
                                 callbacks=callbacks,
                                 validation_data=validation_generator,
                                 nb_val_samples=nb_val_samples)

    def predict(self, X_data, batch_size=32, verbose=0):
        return self.model.predict(X_data, batch_size=batch_size, verbose=verbose)

    def evaluate(self, X_data, y_data, batch_size=32, verbose=1, sample_weight=None):
        return self.model.evaluate(X_data, y_data, batch_size=batch_size,
                                   verbose=verbose, sample_weight=sample_weight)

    def visualize_performance_by_sampling(self, X_data, y_data, NN_prediction,
                                          feature_labels=None,
                                          fitted_scaler=None, max_display=10):
        to_be_used_for_sampling = self.get_first_M_predictions(NN_prediction, max_display)
        samples = self.sample(to_be_used_for_sampling, max_number_of_samples=max_display)
        descriptions = self.TopLayer_obj.get_description(to_be_used_for_sampling)
        number_of_samples = len(samples)
        for j in range(number_of_samples):
            print("\n\n\nInput data:\n")
            print(X_data[j, ...])
            print("\nDescription of the model provided as output by the neural network:\n")
            print(descriptions[j])
            if feature_labels is not None:
                print('\nLabels:\n')
                print(feature_labels)
            print("\nCorrect output (rescaled):\n")
            print(y_data[j, ...])
            print("\nOutput obtained by sampling (rescaled):\n")
            print(samples[j, ...])
            if fitted_scaler is not None:
                print("\nCorrect output:\n")
                print(fitted_scaler.inverse_transform(y_data[j, ...]))
                print("\nOutput obtained by sampling:\n")
                print(fitted_scaler.inverse_transform(samples[j, ...]))
            # TODO: improve formatting

    def sample(self, NN_prediction, max_number_of_samples=10):
        # TODO: add inverse scaling and labels
        to_be_used_for_sampling = self.get_first_M_predictions(NN_prediction, max_number_of_samples)
        samples = self.TopLayer_obj.sample(to_be_used_for_sampling)
        return samples

    def get_first_M_predictions(self, NN_prediction, M):
        batch_size = int(NN_prediction.shape[0])
        slicing_size = min(batch_size, M)
        first_M_predictions = tf.slice(NN_prediction, [0, 0], [slicing_size, -1])
        return first_M_predictions

    def save(self, filepath):
        self.model.save(filepath)
        return

    # TO-DO: save the whole class instance, instead of simply saving the Keras/TF model
