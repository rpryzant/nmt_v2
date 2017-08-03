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


class DotAttentionModel(VanillaModel):
    # TODO - get attentional scores back, diagnostic plots, etc etc
    # TODO -- FIX BUG!!

    def _build_decoder_cell(self, encoder_outputs, encoder_state):
        source_sequence_length = self.iterator.source_sequence_length

        if self.mode == 'inference' and self.config.beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(
                encoder_outputs, multiplier=self.config.beam_width)
            source_sequence_length = tf.contrib.seq2seq.tile_batch(
                source_sequence_length, multiplier=self.config.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(
                 encoder_state, multiplier=self.config.beam_width)
            batch_size = self.batch_size * self.config.beam_width
        else:
            memory = encoder_outputs
            batch_size = self.batch_size
    
        # dot product attention
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            self.config.num_units, 
            memory, 
            memory_sequence_length=source_sequence_length)

        cell = self._build_rnn_cell()

        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=self.config.num_units,
            name='attention')

        initial_state = cell.zero_state(batch_size, tf.float32).clone(
            cell_state=encoder_state)

        return cell, initial_state


