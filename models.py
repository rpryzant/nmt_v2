import abc

import tensorflow as tf
from tensorflow.python.layers import core as layers_core


from model_base import BaseModel



class VanillaModel(BaseModel):

    def _build_encoder(self):
        source_embedded = tf.nn.embedding_lookup(
            self.encoder_embeddings, self.iterator.source)

        if self.config.encoder_type == 'uni':
            outputs, encoder_state = self.build_unidirectional_encoder(source_embedded)
        elif self.config.encoder_type == 'bi':
            outputs, state = self.build_bidirectional_encoder(source_embedded)
            # alternate between fw/bw states 
            encoder_state = []
            for layer in range(self.config.num_layers / 2):
                encoder_state.append(state[0][layer])
                encoder_state.append(state[1][layer])
            encoder_state = tuple(encoder_state)

        return outputs, encoder_state


    def _build_decoder_cell(self, encoder_outputs, encoder_state):
        cell = self._build_rnn_cell()

        if self.mode == "test" and self.config.beam_width > 0:
            initial_state = tf.contrib.seq2seq.tile_batch(
                encoder_state, multiplier=self.config.beam_width)
        else:
            initial_state = encoder_state

        return cell, initial_state
