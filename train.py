import tensorflow as tf
import os
import numpy as np
import math

import model_base
import models
import input_pipeline
import vocab_utils as vocab_utils
import inference



def run_sample_decode(test_model, test_sess, out_dir, config, 
                      summary_writer, src_file, tgt_file):
    """ sample decoding on the test set """
    with test_model.graph.as_default():
        loaded_test_model, global_step = model_base.create_or_load_model(
            test_model.model, out_dir, test_sess, "test")



    test_sess.run(test_model.iterator.initializer,
        feed_dict={
            test_model.src_placeholder: src_file,
            test_model.batch_size_placeholder: 4
        })

    _, nmt_outputs = loaded_test_model.test(test_sess)

    for x in  tgt_file:
        print x
    
    for output in nmt_outputs:
        print output.shape
        print ' '.join(x for x in output[:,0])
    quit()



def run_eval(eval_model, eval_sess, out_dir, config, summary_writer):
    # TODO -- WRITE SUMMARIES FROM THIS
    with eval_model.graph.as_default():
        loaded_eval_model, global_step = model_base.create_or_load_model(
            eval_model.model, out_dir, eval_sess, "eval")
    # TODO -- MAKE SURE GLOBAL STEP IS RIGHT!!
    eval_sess.run(eval_model.iterator.initializer)

    total_loss = 0
    total_predictions = 0
    total_batches = 0
    while True:
        try:
            loss, predict_count = loaded_eval_model.eval(eval_sess)
            total_loss += loss
            total_predictions += predict_count
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
        model_creator = models.AttentionModel

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

    while global_step < config.num_train_steps:
        try:
            step_result = loaded_train_model.train(train_sess, debug=True)
            _, loss, predict_count, global_step, summary, src, src_len,  tgt, tgt_len, preds = step_result
            print loss
            print src
            print src_len
            print tgt
            print tgt_len
            print preds
            print
        except tf.errors.OutOfRangeError:
            # epoch is done 
            train_sess.run(train_model.iterator.initializer)






        summary_writer.add_summary(summary, global_step)
        print global_step, config.steps_per_eval
        if global_step % config.steps_per_eval == 0:
            # save and evaluate
            loaded_train_model.saver.save(
                train_sess,
                os.path.join(out_dir, "model.ckpt"),
                global_step=global_step)

            # with train_model.graph.as_default():
            #     variables_names = [v.name for v in tf.trainable_variables()]
            #     values = train_sess.run(variables_names)
            #     for k, v in zip(variables_names, values):
            #         print k
            # print

            run_sample_decode(
                test_model, test_sess, out_dir, config, summary_writer, test_src, test_tgt)

            eval_loss = run_eval(
                eval_model, eval_sess, out_dir, config, summary_writer)
            print 'EVAL LOSS ', eval_loss















