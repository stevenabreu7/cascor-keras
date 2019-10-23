from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import SGD, RMSprop
import numpy as np
import time


def safe_concatenate(x, y):
    """Wrapper function for the concatenate function from
    keras.layers.

    Params:
        x   single layer
        y   list of layers (possibly empty)

    Returns:
        concatenated layer of x and y (if y is empty, that is
        only x)
    """
    if not y:
        return x
    else:
        return concatenate([x] + y)


def correlation_loss(y_true, y_pred):
    """This returns the error correlation, as proposed by Fahlman et al.
    The error correlation should be maximized, hence the sign change.
    """
    # y_true, y_pred: (N_patterns, output_dim=10)
    error = K.abs(y_pred - y_true)
    # l2 normalize predictions and error
    norm_ypred = K.l2_normalize(y_pred, axis=-1)
    norm_error = K.l2_normalize(error, axis=-1)
    # element-wise multiplication, then sum over patterns (axis 0)
    sum_patterns = K.sum(norm_ypred * norm_error, axis=0)
    # absolute value, then sum over output
    sum_patterns = K.abs(sum_patterns)
    return -1. * K.sum(sum_patterns)


class Cascor:
    def __init__(self, data, loss='categorical_crossentropy', hid_act='tanh',
                 out_act='softmax', optimizer='rms'):
        """Builds a cascade-correlation base network.

        The network is represented by:
        - one input layer (no computations)
        - cascading hidden layers, each consisting of
            - a fully connected (dense) layer with an activation function
                input: last concatenation layer (or input layer)
                output: one dimensional value
            - a concatenation layer
                input: input layer and all previous fully connected layers
                output: concatenation of the input and all hidden layers
        - one output layer
            input: the last concatenation layer (inputs and all hidden outs)
            output: final output of the network

        Parameters:
            data        data tuple: (x_train, y_train), (x_test, y_test)
        Optional:
            loss        string: what loss function to use
                        - 'corr' (error correlation loss)
                        - default: 'categorical_crossentropy'
            hid_act     hidden activation function
            out_act     output activation function
            optimizer   optimizer ('sgd' or 'rms')
        """

        # load the data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        input_dim = self.x_train.shape[1]
        output_dim = self.y_train.shape[1]

        # determine loss function
        if loss == 'corr':
            self.loss = correlation_loss
        elif loss == 'categorical_crossentropy':
            self.loss = 'categorical_crossentropy'
        else:
            print('Ambiguous loss function. Using cat cross entropy.')
            self.loss = 'categorical_crossentropy'

        # save parameters for reuse
        self.hid_act = hid_act
        self.out_act = out_act
        self.input_dim = input_dim
        self.output_dim = output_dim

        # input layer
        self.in_layer = Input(shape=(input_dim,))

        # single hidden layer
        self.hid_layers = []

        # output layer
        outl = Dense(output_dim, activation=out_act, name='output')
        self.out_layer = outl(
            self.in_layer
        )

        # setup the model
        self.model = Model(inputs=self.in_layer, outputs=self.out_layer)

        # setup optimizer
        if optimizer == 'rms':
            self.optimizer = RMSprop()
        elif optimizer == 'sgd':
            self.optimizer = SGD()
        else:
            self.optimizer = RMSprop()

        # compile model
        self.compile_model()

    def reset_model(self, n_hidden=1):
        """Reset the computational graph in order to clean up memory.
        All model weights are saved and restored.

        Params:
            n_hidden    number of units in hidden layers
        """

        weights = self.model.get_weights().copy()

        K.clear_session()

        self.in_layer = Input(shape=(self.input_dim,))

        for i in range(len(self.hid_layers)):
            hidl = Dense(n_hidden, activation=self.hid_act)
            self.hid_layers[i] = hidl(
                safe_concatenate(self.in_layer, self.hid_layers[:i])
            )

        outl = Dense(self.output_dim, activation=self.out_act, name='output')
        self.out_layer = outl(
            concatenate([self.in_layer] + self.hid_layers)
        )

        self.model = Model(inputs=self.in_layer, outputs=self.out_layer)

        if isinstance(self.optimizer, RMSprop):
            self.optimizer = RMSprop()
        else:
            assert False, 'Unknown optimizer'

        self.model.set_weights(weights)

        self.compile_model()

    def add_unit(self, n=1, hid_weights=None, out_weights=None,
                 prev_out_weights=None):
        """Adds a new cascading unit into the network.

        Params:
            hid_weights         weights for the hidden unit
            out_weights         weights for the output unit
            prev_out_weights    weights for previous output unit

        Optional:
            n                   how many units in this layer
                                default = 1

        Returns:
            hid_weights         weights for the hidden unit
            out_weights         weights for the output unit
        """

        # create new hidden layer on the concatenated layer
        # that was previously fed to the output layer
        hidl = Dense(n, activation=self.hid_act)
        self.hid_layers.append(
            hidl(
                safe_concatenate(self.in_layer, self.hid_layers)
            )
        )

        # output layer
        outl = Dense(self.output_dim, activation=self.out_act, name='output')
        self.out_layer = outl(
            concatenate([self.in_layer] + self.hid_layers)
        )

        # re-setup model
        self.model = Model(inputs=self.in_layer, outputs=self.out_layer)

        # set weights, if they are given
        start = time.time()
        if hid_weights:
            hidl.set_weights(hid_weights)
        if out_weights:
            outl.set_weights(out_weights)
        if prev_out_weights:
            out_weights = outl.get_weights()
            out_weights[1] = prev_out_weights[1]
            stop = prev_out_weights[0].shape[0]
            out_weights[0][:stop] = prev_out_weights[0]
            outl.set_weights(out_weights)
        print('Setting weights: {:.4f}s'.format(time.time() - start))

        # re-compile model
        start = time.time()
        self.compile_model()
        print('Compiling model: {:.4f}s'.format(time.time() - start))

        if hid_weights:
            assert np.array_equal(hid_weights[0], hidl.get_weights()[0])
            assert np.array_equal(hid_weights[1], hidl.get_weights()[1])
        if out_weights:
            assert np.array_equal(out_weights[0], outl.get_weights()[0])
            assert np.array_equal(out_weights[1], outl.get_weights()[1])

        return hidl.get_weights(), outl.get_weights()

    def remove_last_unit(self):
        """Remove the unit that was last inserted (everything is frozen
        except for the output layer)
        """
        # remove last hidden layer
        self.hid_layers.pop(-1)

        # update output layer
        outl = Dense(self.output_dim, activation=self.out_act, name='output')
        self.out_layer = outl(
            safe_concatenate(self.in_layer, self.hid_layers)
        )

        # update model
        self.model = Model(inputs=self.in_layer, outputs=self.out_layer)

        # re-compile model
        self.compile_model()

    def compile_model(self):
        """Compile the current model (or re-compile).
        """
        metrics = ['accuracy']
        self.model.compile(self.optimizer, loss=self.loss, metrics=metrics)

    def freeze_weights(self):
        """Freeze all non-output dense layers' weights.
        """
        # iterate through all but the last layer (output layer)
        for i in range(len(self.model.layers) - 1):
            # freeze all non-output dense layers
            if isinstance(self.model.layers[i], Dense):
                self.model.layers[i].trainable = False
        self.compile_model()

    def train(self, n_units=1, pool_size=8, reuse_pool_n=1,
              reuse_comp=0.0, unit_epochs=1, max_units=10):
        """Train the Cascor network.

        Parameters:
            n_units     number of units in each new layer
            pool_size   size of the candidate pool for new units
            batch_size  size of batches
            reuse_comp  compensation for reusing weights

        Returns:
            hist        history object for training losses
        """

        batch_size = 128

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=2)

        histories = []
        for epoch in range(max_units):

            self.freeze_weights()

            epoch_start = time.time()

            # get current output weight vector
            prev_w = self.model.get_layer('output').get_weights()

            # iterate through candidate pool and train each one
            candidate_weights = []
            candidate_scores = []
            for j in range(pool_size):

                print('Unit {:02}. Candidate {:02}/{:02}'.format(
                    epoch+1, j+1, pool_size
                ))

                # first unit should reuse previous weights
                if j < reuse_pool_n:
                    unit_weights = self.add_unit(n=n_units,
                                                 prev_out_weights=prev_w)
                else:
                    unit_weights = self.add_unit(n=n_units)

                trainable_layers = len(list(filter(
                    lambda x: x.trainable and isinstance(x, Dense),
                    self.model.layers
                )))
                assert trainable_layers == 2

                candidate_weights.append(unit_weights)
                del unit_weights

                # train the network with the new unit
                ht = self.model.fit(self.x_train,
                                    self.y_train,
                                    epochs=unit_epochs,
                                    shuffle=False,
                                    validation_data=(self.x_test, self.y_test),
                                    batch_size=batch_size)
                score = ht.history['val_accuracy'][-1]

                # save the "unit's" score
                candidate_scores.append(score)

                # remove the unit again
                self.remove_last_unit()

            # select the best candidate
            best_all_idx = candidate_scores.index(max(candidate_scores))
            best_all_score = candidate_scores[best_all_idx]
            best_new_idx = candidate_scores.index(max(candidate_scores[1:]))
            best_new_score = candidate_scores[best_new_idx]

            if reuse_pool_n > 0:
                if best_all_idx == best_new_idx:
                    # best score is not reusing output weights -> use
                    print('New weights (superior). Candidate {}.'.format(
                        best_new_idx
                    ))
                    best_idx = best_new_idx
                elif best_all_score < (best_new_score + reuse_comp):
                    # best new score is slightly worse than reusing score, use
                    print('New weights (compensated). Candidate {}.'.format(
                        best_new_idx
                    ))
                    best_idx = best_new_idx
                else:
                    # reuse the output weights
                    print('Previous weights. Candidate 0.')
                    best_idx = 0
            else:
                print('Chose candidate {}.'.format(best_all_idx))
                best_idx = best_all_idx

            # add the best unit
            hid_weights, out_weights = candidate_weights[best_idx]
            self.add_unit(n=n_units, hid_weights=hid_weights,
                          out_weights=out_weights)

            # train the network
            histry = self.model.fit(self.x_train, self.y_train, shuffle=False,
                                    epochs=100, batch_size=batch_size,
                                    validation_data=(self.x_test, self.y_test),
                                    callbacks=[early_stopping])

            epoch_duration = time.time() - epoch_start

            histry = histry.history
            histry['candidates'] = [','.join(map(str, candidate_scores))]
            histry['candidates'] += [''] * (len(histry['accuracy']) - 1)
            histry['epoch_time'] = [epoch_duration] * len(histry['accuracy'])
            histories.append(histry)

            self.reset_model(n_hidden=n_units)

        # combine histories into single dictionary
        history = {}
        for key in histories[0].keys():
            history[key] = []
            history['inserted'] = []
            for i in range(len(histories)):
                history[key] += histories[i][key]
                inserted = [True] + [False] * (len(histories[i][key])-1)
                history['inserted'] += inserted

        return history
