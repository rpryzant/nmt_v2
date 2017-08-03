import tensorflow as tf
import os
import numpy as np
import math
import time

import model_base
import models
import input_pipeline
import utils
import inference
import evaluation

# shut up tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def run_sample_decode(test_model, test_sess, out_dir, config, 
                      summary_writer, src_file, tgt_file):
    """ sample decoding on the test set """
    with test_model.graph.as_default():
        loaded_test_model, global_step = model_base.create_or_load_model(
            test_model.model, out_dir, test_sess, "test")

    decode_ids = np.random.choice(len(src_file), config.sample_decodings)
    decoding_src = [src_file[id] for id in decode_ids]
    decoding_tgt = [tgt_file[id] for id in decode_ids]

    test_sess.run(test_model.iterator.initializer,
        feed_dict={
            test_model.src_placeholder: decoding_src,
            test_model.batch_size_placeholder: config.batch_size
        })

    _, nmt_outputs = loaded_test_model.test(test_sess)
    for i, (src, tgt, pred) in enumerate(zip(decoding_src, decoding_tgt, nmt_outputs)):
        print 'src: ', src
        print 'tgt: ', tgt
        print 'nmt: ', inference.format_decoding(pred, target_beam=0, eos=config.eos)
        print
        if i >= config.sample_decodings:
            break

def run_eval(eval_model, eval_sess, out_dir, config, summary_writer):
    # TODO -- WRITE SUMMARIES FROM THIS
    with eval_model.graph.as_default():
        loaded_eval_model, global_step = model_base.create_or_load_model(
            eval_model.model, out_dir, eval_sess, "eval")
    # TODO -- MAKE SURE GLOBAL STEP IS RIGHT!!
    eval_sess.run(eval_model.iterator.initializer)

    total_loss = 0
    total_batches = 0
    while True:
        try:
            loss, word_count = loaded_eval_model.eval(eval_sess)
            total_loss += loss
            total_batches += 1
        except tf.errors.OutOfRangeError:
            break
    avg_loss = total_loss / total_batches
    return total_loss / total_batches





def train(config):
    out_dir = config.out_dir

    if not config.attention:
        model_creator = models.VanillaModel
    else:
        model_creator = models.DotAttentionModel

    train_model = model_base.build_model_graph(model_creator, config, "train")
    eval_model = model_base.build_model_graph(model_creator, config, "eval")
    test_model = inference.build_inference_graph(model_creator, config)

    test_src = inference.load_data(test_model.src_file)
    test_tgt = inference.load_data(test_model.tgt_file)

    train_sess = tf.Session(graph=train_model.graph)
    eval_sess = tf.Session(graph=eval_model.graph)
    test_sess = tf.Session(graph=test_model.graph)

    with train_model.graph.as_default():
        loaded_train_model, global_step = model_base.create_or_load_model(
            train_model.model, out_dir, train_sess, "train")

    summary_writer = tf.summary.FileWriter(
        os.path.join(out_dir, "train_log"), train_model.graph)

    train_sess.run(train_model.iterator.initializer)

    total_time, total_loss, total_word_count = 0, 0, 0
    
    while global_step < config.num_train_steps:
        start_time = time.time()
        try:
            step_result = loaded_train_model.train(train_sess, debug=True)
            _, loss, word_count, global_step, \
            summary, src, src_len,  tgt, tgt_len, preds = step_result
            utils.add_summary(summary_writer, global_step, "train-loss", loss)
        except tf.errors.OutOfRangeError:
            # epoch is done 
            train_sess.run(train_model.iterator.initializer)

        summary_writer.add_summary(summary, global_step)
        total_time += (time.time() - start_time)
        total_loss += loss
        total_word_count += word_count

        if global_step % config.steps_per_stats == 0:
            # TODO -- when steps_per_stats > train file size, 
            #         we do this twice
            avg_loss = total_loss / config.steps_per_stats
            avg_step_time = total_time / config.steps_per_stats
            # time.time() is in milliseconds
            words_per_second = total_word_count / total_time
            print 'INFO: ' + \
                  ' step %d lr %g step-time %.2fs wps %.2f loss %.2f' % \
                  (global_step,
                    loaded_train_model.learning_rate.eval(session=train_sess),
                    avg_step_time, words_per_second, avg_loss)
            total_time, total_loss, total_word_count = 0, 0, 0


        if global_step % config.steps_per_eval == 0:
            # record test predictions
            inference.translate_file(
                test_model=test_model,
                test_sess=test_sess,
                out_file=os.path.join(out_dir, 'translations-%d' % global_step),
                eos=config.eos,
                src_file=test_src)

            # save and evaluate
            loaded_train_model.saver.save(
                train_sess,
                os.path.join(out_dir, "model.ckpt"),
                global_step=global_step)

            eval_loss = run_eval(
                eval_model, eval_sess, out_dir, config, summary_writer)
            utils.add_summary(summary_writer, global_step, "eval-loss", eval_loss)

            print 'EVAL: loss %.2f' % eval_loss


        if global_step % config.steps_per_sample == 0:
            # do inference on a sample
            run_sample_decode(
                 test_model, test_sess, out_dir, config, summary_writer, test_src, test_tgt)















