
### UNFINISHED. JUST FOR TESTING ###

import numpy as np
import tensorflow as tf
from tensorflow.keras import models as km
from tensorflow.keras import metrics as kmet

from .tools.kde import nll


CHAN_LIST = (
    'PAS1', 'PBS1', 'PCS1', 'PDS1', 'PES1', 'PFS1',
    'PAS2', 'PBS2', 'PCS2', 'PDS2', 'PES2', 'PFS2'
)


class MLSimRescaler(km.Model):

    def __init__(
        self,
        model,
        rpred_model,
        sim_df,
        data_df,
        bandwidth: float = 0.003,
        radius_bandwidth: float = 1.0,
        samples_data: int = 1000,
        samples_sim: int = 1000,
        meta_steps: int = 10,
        inner_lr: float = 1e-4,
        radius_weight: float = 1.0
    ):

        super().__init__()

        self.sim_df = sim_df.copy()
        self.data_df = data_df.copy()

        self.bandwidth = bandwidth
        self.radius_bandwidth = radius_bandwidth

        self.samples_data = samples_data
        self.samples_sim = samples_sim

        self.meta_steps = meta_steps
        self.inner_lr = inner_lr
        self.radius_weight = radius_weight

        self.model = model
        self.rpred_model = rpred_model

        self.loss_tracker = kmet.Mean(name='loss')
        self.loss_tracker_rpred = kmet.Mean(name='rloss')
        self.loss_tracker_supervised = kmet.Mean(name='supervised_rloss')
        self.loss_tracker_meta = kmet.Mean(name='meta_rloss')

        self.process()

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.loss_tracker_rpred,
            self.loss_tracker_supervised,
            self.loss_tracker_meta
        ]

    # --------------------------------------------------
    # Data processing
    # --------------------------------------------------

    def process(self):

        sim_df = self.sim_df.copy()
        data_df = self.data_df.copy()

        amp_cols = [c + "OFamps" for c in CHAN_LIST]

        # Calculate PSum if necessary.
        if 'PSum' not in sim_df.columns:
            sim_df['PSum'] = sim_df[amp_cols].sum(axis=1)

        if 'PSum' not in data_df.columns:
            data_df['PSum'] = data_df[amp_cols].sum(axis=1)

        # Calculate radius if necessary.
        if 'R' not in sim_df.columns:
            sim_df['R'] = np.sqrt(
                sim_df['X']**2 + sim_df['Y']**2
            )

        # Normalize radius to units of 50 mm.
        if sim_df['R'].max() > 1:
            sim_df['R'] /= 50.0

        # Shuffle rows before extracting arrays.
        sim_df = sim_df.sample(frac=1, ignore_index=True)
        data_df = data_df.sample(frac=1, ignore_index=True)

        self.sim_psum = sim_df['PSum'].to_numpy(dtype=np.float32)
        self.data_psum = data_df['PSum'].to_numpy(dtype=np.float32)

        sim_amps = sim_df[amp_cols].to_numpy(dtype=np.float32)
        data_amps = data_df[amp_cols].to_numpy(dtype=np.float32)

        if np.any(self.sim_psum == 0) or np.any(self.data_psum == 0):
            raise ValueError("Encountered events with PSum = 0.")

        self.norm_sim = sim_amps / self.sim_psum[:, None]
        self.norm_data = data_amps / self.data_psum[:, None]

        self.sim_R = sim_df['R'].to_numpy(dtype=np.float32)

    # --------------------------------------------------
    # Tensor preparation
    # --------------------------------------------------

    def get_batches(self):

        data = tf.convert_to_tensor(
            self.norm_data[:self.samples_data],
            dtype=tf.float32
        )

        sim = tf.convert_to_tensor(
            self.norm_sim[:self.samples_sim],
            dtype=tf.float32
        )

        radius = tf.convert_to_tensor(
            self.sim_R[:self.samples_sim],
            dtype=tf.float32
        )

        return data, sim, radius

    # --------------------------------------------------
    # Loss functions
    # --------------------------------------------------

    def get_amplitude_loss(self, data, rescaled_sim):

        # Maintain constant total normalized amplitude.
        norm_psum = tf.reduce_sum(rescaled_sim, axis=1)

        energy_loss = (
            tf.reduce_mean(tf.square(norm_psum - 1.0)) * 1e4
        )

        distribution_loss = nll(
            data,
            rescaled_sim,
            self.bandwidth
        )

        return distribution_loss + energy_loss

    def get_loss(self, data, sim):

        rescaled_sim = self.model(sim, training=True)

        return self.get_amplitude_loss(data, rescaled_sim)

    def get_radius_loss(self, data_prediction, sim_radius):

        # Both inputs are in units of 50 mm.
        data_prediction = tf.reshape(data_prediction * 50.0, [-1])
        sim_radius = tf.reshape(sim_radius * 50.0, [-1])

        return nll(
            data_prediction,
            sim_radius,
            self.radius_bandwidth
        )

    # --------------------------------------------------
    # Gradient application
    # --------------------------------------------------

    @staticmethod
    def apply_gradients(optimizer, gradients, variables):

        pairs = [
            (g, v)
            for g, v in zip(gradients, variables)
            if g is not None
        ]

        if pairs:
            optimizer.apply_gradients(pairs)

    # --------------------------------------------------
    # Differentiable predictor training
    # --------------------------------------------------

    def differentiable_predictor_training(
        self,
        rescaled_sim,
        sim_radius
    ):
        """
        Virtually train the radius predictor for meta_steps.

        The predictor's actual weights are NOT modified.

        Instead, we create tensor-valued copies of its weights
        and perform differentiable SGD updates on those copies.

        The outer gradient can then propagate through these
        updates and back into the rescaling model.
        """

        if not hasattr(self.rpred_model, 'stateless_call'):
            raise RuntimeError(
                "Differentiable predictor training requires "
                "Keras 3 Model.stateless_call()."
            )

        # Copy the predictor's current parameters.
        fast_weights = [
            tf.identity(v)
            for v in self.rpred_model.trainable_variables
        ]

        # Copy non-trainable state (e.g. BatchNorm statistics).
        fast_non_trainable = [
            tf.identity(v)
            for v in self.rpred_model.non_trainable_variables
        ]

        sim_radius = tf.reshape(sim_radius, [-1])

        # Unroll the predictor's training process.
        for _ in range(self.meta_steps):

            with tf.GradientTape(
                watch_accessed_variables=False
            ) as inner_tape:

                inner_tape.watch(fast_weights)

                prediction, updated_non_trainable = (
                    self.rpred_model.stateless_call(
                        fast_weights,
                        fast_non_trainable,
                        rescaled_sim,
                        training=True
                    )
                )

                prediction = tf.reshape(prediction, [-1])

                inner_loss = tf.reduce_mean(
                    tf.square(prediction - sim_radius)
                )

            # Differentiate the supervised loss with respect
            # to the virtual predictor weights.
            gradients = inner_tape.gradient(
                inner_loss,
                fast_weights
            )

            # Differentiable SGD update.
            fast_weights = [
                w - self.inner_lr * g
                for w, g in zip(fast_weights, gradients)
            ]

            fast_non_trainable = updated_non_trainable

        return fast_weights, fast_non_trainable

    # --------------------------------------------------
    # Meta-objective
    # --------------------------------------------------

    def get_meta_loss(self, data, rescaled_sim, sim_radius):

        # Virtually train the predictor on RESCALED SIMULATION.
        fast_weights, fast_non_trainable = (
            self.differentiable_predictor_training(
                rescaled_sim,
                sim_radius
            )
        )

        # Evaluate the virtually trained predictor on ORIGINAL DATA.
        data_prediction, _ = self.rpred_model.stateless_call(
            fast_weights,
            fast_non_trainable,
            data,
            training=False
        )

        # Compare the predicted data-radius distribution against
        # the true simulation-radius distribution.
        return self.get_radius_loss(
            data_prediction,
            sim_radius
        )

    # --------------------------------------------------
    # Phase 1: Amplitude pretraining
    # --------------------------------------------------

    def train_rescaler_step(self, _=None):

        data, sim, _ = self.get_batches()

        with tf.GradientTape() as tape:

            loss = self.get_loss(data, sim)

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)

        self.apply_gradients(
            self.model.optimizer,
            gradients,
            variables
        )

        self.loss_tracker.update_state(loss)

        return {
            'loss': self.loss_tracker.result()
        }

    # --------------------------------------------------
    # Phase 2: Supervised predictor training
    # --------------------------------------------------

    def train_step_R(self, _=None):

        _, sim, radius = self.get_batches()

        # Rescale simulation using the CURRENT rescaler.
        # Freeze the rescaler during this training step.
        rescaled_sim = tf.stop_gradient(
            self.model(sim, training=False)
        )

        with tf.GradientTape() as tape:

            prediction = self.rpred_model(
                rescaled_sim,
                training=True
            )

            prediction = tf.reshape(prediction, [-1])

            loss = tf.reduce_mean(
                tf.square(prediction - radius)
            )

        variables = self.rpred_model.trainable_variables
        gradients = tape.gradient(loss, variables)

        self.apply_gradients(
            self.rpred_model.optimizer,
            gradients,
            variables
        )

        self.loss_tracker_supervised.update_state(loss)

        return {
            'supervised_rloss': self.loss_tracker_supervised.result()
        }

    # --------------------------------------------------
    # Phase 3: Meta-training
    # --------------------------------------------------

    def meta_train_step(self, _=None):

        data, sim, radius = self.get_batches()

        with tf.GradientTape() as outer_tape:

            # Rescale SIMULATION only.
            rescaled_sim = self.model(
                sim,
                training=True
            )

            # Preserve the amplitude-matching objective.
            amplitude_loss = self.get_amplitude_loss(
                data,
                rescaled_sim
            )

            # Differentiate through the predictor's virtual
            # supervised training process.
            meta_loss = self.get_meta_loss(
                data,
                rescaled_sim,
                radius
            )

            total_loss = (
                amplitude_loss
                + self.radius_weight * meta_loss
            )

        # The outer objective updates ONLY the rescaler.
        variables = self.model.trainable_variables

        gradients = outer_tape.gradient(
            total_loss,
            variables
        )

        self.apply_gradients(
            self.model.optimizer,
            gradients,
            variables
        )

        self.loss_tracker.update_state(amplitude_loss)
        self.loss_tracker_meta.update_state(meta_loss)

        return {
            'loss': self.loss_tracker.result(),
            'meta_rloss': self.loss_tracker_meta.result(),
            'total_loss': total_loss
        }

    # --------------------------------------------------
    # Optional train_step interface
    # --------------------------------------------------

    def train_step(self, _, toggle_r_loss=0):

        if toggle_r_loss == 0:
            return self.train_rescaler_step()

        elif toggle_r_loss == 1:
            return self.meta_train_step()

        else:
            raise ValueError("toggle_r_loss must be 0 or 1.")

    # --------------------------------------------------
    # Testing
    # --------------------------------------------------

    def test_step(self, _=None):

        data, sim, _ = self.get_batches()

        loss = self.get_loss(data, sim)

        self.loss_tracker.update_state(loss)

        return {
            'loss': self.loss_tracker.result()
        }

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    def call(self, inputs, training=False):

        return self.model(inputs, training=training)

    def apply(self, sim_df=None):

        if sim_df is None:
            sim_df = self.sim_df.copy()
        else:
            sim_df = sim_df.copy()

        amp_cols = [c + "OFamps" for c in CHAN_LIST]

        if 'PSum' not in sim_df.columns:
            sim_df['PSum'] = sim_df[amp_cols].sum(axis=1)

        psum = sim_df['PSum'].to_numpy(dtype=np.float32)

        if np.any(psum == 0):
            raise ValueError("Encountered events with PSum = 0.")

        norm_sim = (
            sim_df[amp_cols].to_numpy(dtype=np.float32)
            / psum[:, None]
        )

        pred = self.model(
            tf.convert_to_tensor(norm_sim, dtype=tf.float32),
            training=False
        ).numpy()

        unnorm = pred * psum[:, None]

        for ind, c in enumerate(CHAN_LIST):
            sim_df[c + "OFamps"] = unnorm[:, ind]

        sim_df['PSum'] = sim_df[amp_cols].sum(axis=1)

        return sim_df

    def predict_R(self, data=None):

        # Real data is passed DIRECTLY to the radius predictor.
        if data is None:
            data = self.norm_data

        data = tf.convert_to_tensor(data, dtype=tf.float32)

        return self.rpred_model(
            data,
            training=False
        ) * 50.0