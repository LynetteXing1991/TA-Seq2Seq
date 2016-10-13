from __future__ import print_function

import logging
import time
import numpy
import os
import cPickle
import string

from collections import Counter
from theano import tensor, function,shared
from toolz import merge
from progressbar import ProgressBar

from blocks.algorithms import (GradientDescent, StepClipping,
                               AdaDelta, AdaGrad, Scale, CompositeRule)
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from search_decoder_with_extra_class import BeamSearch
from blocks.select import Selector

from checkpoint import CheckpointNMT, LoadNMT
from model import BidirectionalEncoder, Decoder, topicalq_transformer
from sampling import BleuValidator, Sampler, SamplingBase, pplValidation
from stream import (get_tr_stream, get_dev_stream, get_tr_stream_with_topic_target,get_dev_stream_with_topicalq,
    get_tr_stream_unsorted, _ensure_special_tokens)
from SimplePrinting import SimplePrinting
from learning_rate_halver import (LearningRateHalver, 
                                  LearningRateDoubler, 
                                  OldModelRemover)
from afterprocess import afterprocesser

try:
    from blocks.extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)


def main(mode, config, use_bokeh=False):

    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
    topical_transformer=topicalq_transformer(config['source_topic_vocab_size'],config['topical_embedding_dim'], config['enc_nhids'],config['topical_word_num'],config['batch_size']);
    decoder = Decoder(vocab_size=config['trg_vocab_size'],
                      topicWord_size=config['trg_topic_vocab_size'],
                      embedding_dim=config['dec_embed'],
                      topical_dim=config['topical_embedding_dim'],
                      state_dim=config['dec_nhids'],
                      representation_dim=config['enc_nhids'] * 2,
                      match_function=config['match_function'],
                      use_doubly_stochastic=config['use_doubly_stochastic'],
                      lambda_ds=config['lambda_ds'],
                      use_local_attention=config['use_local_attention'],
                      window_size=config['window_size'],
                      use_step_decay_cost=config['use_step_decay_cost'],
                      use_concentration_cost=config['use_concentration_cost'],
                      lambda_ct=config['lambda_ct'],
                      use_stablilizer=config['use_stablilizer'],
                      lambda_st=config['lambda_st'])
    # here attended dim (representation_dim) of decoder is 2*enc_nhinds
    # because the context given by the encoder is a bidirectional context

    if mode == "train":

        # Create Theano variables
        logger.info('Creating theano variables')
        source_sentence = tensor.lmatrix('source')
        source_sentence_mask = tensor.matrix('source_mask')
        target_sentence = tensor.lmatrix('target')
        target_sentence_mask = tensor.matrix('target_mask')
        target_topic_sentence=tensor.lmatrix('target_topic');
        target_topic_binary_sentence=tensor.lmatrix('target_binary_topic');
        #target_topic_sentence_mask=tensor.lmatrix('target_topic_mask');
        sampling_input = tensor.lmatrix('input')
        source_topical_word=tensor.lmatrix('source_topical')
        source_topical_mask=tensor.matrix('source_topical_mask')

        topic_embedding=topical_transformer.apply(source_topical_word);


        # Get training and development set streams
        tr_stream = get_tr_stream_with_topic_target(**config)
        #dev_stream = get_dev_tr_stream_with_topic_target(**config)

        # Get cost of the model
        representations = encoder.apply(source_sentence, source_sentence_mask)
        tw_representation=topical_transformer.look_up.apply(source_topical_word.T);
        content_embedding=representations[0,:,(representations.shape[2]/2):];
        cost = decoder.cost(representations,
                            source_sentence_mask,
                            tw_representation,
                            source_topical_mask,
                            target_sentence,
                            target_sentence_mask,
                            target_topic_sentence,
                            target_topic_binary_sentence,
                            topic_embedding,content_embedding)

        logger.info('Creating computational graph')
        perplexity = tensor.exp(cost)
        perplexity.name = 'perplexity'

        cg = ComputationGraph(cost)
        costs_computer = function([target_sentence,
                                   target_sentence_mask,
                                   source_sentence,
                                   source_sentence_mask,source_topical_word,target_topic_sentence,target_topic_binary_sentence], (perplexity),on_unused_input='ignore')

        # Initialize model
        logger.info('Initializing model')
        encoder.weights_init = decoder.weights_init = IsotropicGaussian(
            config['weight_scale'])
        encoder.biases_init = decoder.biases_init = Constant(0)
        encoder.push_initialization_config()
        decoder.push_initialization_config()
        encoder.bidir.prototype.weights_init = Orthogonal()
        decoder.transition.weights_init = Orthogonal()
        encoder.initialize()
        decoder.initialize()

        topical_transformer.weights_init=IsotropicGaussian(
            config['weight_scale']);
        topical_transformer.biases_init=Constant(0);
        topical_transformer.push_allocation_config();#don't know whether the initialize is for
        topical_transformer.look_up.weights_init=Orthogonal();
        topical_transformer.transformer.weights_init=Orthogonal();
        topical_transformer.initialize();
        word_topical_embedding=cPickle.load(open(config['topical_embeddings'], 'rb'));
        np_word_topical_embedding=numpy.array(word_topical_embedding,dtype='float32');
        topical_transformer.look_up.W.set_value(np_word_topical_embedding);
        topical_transformer.look_up.W.tag.role=[];


        # apply dropout for regularization
        if config['dropout'] < 1.0:
            # dropout is applied to the output of maxout in ghog
            logger.info('Applying dropout')
            dropout_inputs = [x for x in cg.intermediary_variables
                              if x.name == 'maxout_apply_output']
            cg = apply_dropout(cg, dropout_inputs, config['dropout'])

        # Apply weight noise for regularization
        if config['weight_noise_ff'] > 0.0:
            logger.info('Applying weight noise to ff layers')
            enc_params = Selector(encoder.lookup).get_params().values()
            enc_params += Selector(encoder.fwd_fork).get_params().values()
            enc_params += Selector(encoder.back_fork).get_params().values()
            dec_params = Selector(
                decoder.sequence_generator.readout).get_params().values()
            dec_params += Selector(
                decoder.sequence_generator.fork).get_params().values()
            dec_params += Selector(decoder.state_init).get_params().values()
            cg = apply_noise(
                cg, enc_params+dec_params, config['weight_noise_ff'])


        # Print shapes
        shapes = [param.get_value().shape for param in cg.parameters]
        logger.info("Parameter shapes: ")
        for shape, count in Counter(shapes).most_common():
            logger.info('    {:15}: {}'.format(shape, count))
        logger.info("Total number of parameters: {}".format(len(shapes)))

        # Print parameter names
        enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                                   Selector(decoder).get_parameters())
        logger.info("Parameter names: ")
        for name, value in enc_dec_param_dict.items():
            logger.info('    {:15}: {}'.format(value.get_value().shape, name))
        logger.info("Total number of parameters: {}"
                    .format(len(enc_dec_param_dict)))


        # Set up training model
        logger.info("Building model")
        training_model = Model(cost)

        # Set extensions
        logger.info("Initializing extensions")
        extensions = [
            FinishAfter(after_n_batches=config['finish_after']),
            TrainingDataMonitoring([perplexity], after_batch=True),
            CheckpointNMT(config['saveto'],
                          config['model_name'],
                          every_n_batches=config['save_freq'])
        ]

        # # Set up beam search and sampling computation graphs if necessary
        # if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
        #     logger.info("Building sampling model")
        #     sampling_representation = encoder.apply(
        #         sampling_input, tensor.ones(sampling_input.shape))
        #     generated = decoder.generate(
        #         sampling_input, sampling_representation)
        #     search_model = Model(generated)
        #     _, samples = VariableFilter(
        #         bricks=[decoder.sequence_generator], name="outputs")(
        #             ComputationGraph(generated[1]))
        #
        # # Add sampling
        # if config['hook_samples'] >= 1:
        #     logger.info("Building sampler")
        #     extensions.append(
        #         Sampler(model=search_model, data_stream=tr_stream,
        #                 model_name=config['model_name'],
        #                 hook_samples=config['hook_samples'],
        #                 every_n_batches=config['sampling_freq'],
        #                 src_vocab_size=config['src_vocab_size']))
        #
        # # Add early stopping based on bleu
        # if False:
        #     logger.info("Building bleu validator")
        #     extensions.append(
        #         BleuValidator(sampling_input, samples=samples, config=config,
        #                       model=search_model, data_stream=dev_stream,
        #                       normalize=config['normalized_bleu'],
        #                       every_n_batches=config['bleu_val_freq'],
        #                       n_best=3,
        #                       track_n_models=6))
        #
        # logger.info("Building perplexity validator")
        # extensions.append(
        #         pplValidation( config=config,
        #                 model=costs_computer, data_stream=dev_stream,
        #                 model_name=config['model_name'],
        #                 every_n_batches=config['sampling_freq']))


        # Plot cost in bokeh if necessary
        if use_bokeh and BOKEH_AVAILABLE:
            extensions.append(
                Plot('Cs-En', channels=[['decoder_cost_cost']],
                     after_batch=True))

        # Reload model if necessary
        if config['reload']:
            extensions.append(LoadNMT(config['saveto']))

        initial_learning_rate = config['initial_learning_rate']
        log_path = os.path.join(config['saveto'], 'log')
        if config['reload'] and os.path.exists(log_path):
            with open(log_path, 'rb') as source:
                log = cPickle.load(source)
                last = max(log.keys()) - 1
                if 'learning_rate' in log[last]:
                    initial_learning_rate = log[last]['learning_rate']

        # Set up training algorithm
        logger.info("Initializing training algorithm")
        algorithm = GradientDescent(
            cost=cost, parameters=cg.parameters,
            step_rule=CompositeRule([Scale(initial_learning_rate),
                                     StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()]),
            on_unused_sources='ignore')

        _learning_rate = algorithm.step_rule.components[0].learning_rate
        if config['learning_rate_decay']:
            extensions.append(
                LearningRateHalver(record_name='validation_cost',
                                   comparator=lambda x, y: x > y,
                                   learning_rate=_learning_rate,
                                   patience_default=3))
        else:
            extensions.append(OldModelRemover(saveto=config['saveto']))

        if config['learning_rate_grow']:
            extensions.append(
                LearningRateDoubler(record_name='validation_cost',
                                    comparator=lambda x, y: x < y,
                                    learning_rate=_learning_rate,
                                    patience_default=3))

        extensions.append(
            SimplePrinting(config['model_name'], after_batch=True))

        # Initialize main loop
        logger.info("Initializing main loop")
        main_loop = MainLoop(
            model=training_model,
            algorithm=algorithm,
            data_stream=tr_stream,
            extensions=extensions
        )

        # Train!
        main_loop.run()

    elif mode == 'translate':

        logger.info('Creating theano variables')
        sampling_input = tensor.lmatrix('source')
        source_topical_word=tensor.lmatrix('source_topical')
        tw_vocab_overlap=tensor.lmatrix('tw_vocab_overlap')
        tw_vocab_overlap_matrix=cPickle.load(open(config['tw_vocab_overlap'], 'rb'));
        tw_vocab_overlap_matrix=numpy.array(tw_vocab_overlap_matrix,dtype='int32');
        #tw_vocab_overlap=shared(tw_vocab_overlap_matrix);

        topic_embedding=topical_transformer.apply(source_topical_word);

        sutils = SamplingBase()
        unk_idx = config['unk_id']
        src_eos_idx = config['src_vocab_size'] - 1
        trg_eos_idx = config['trg_vocab_size'] - 1
        trg_vocab = _ensure_special_tokens(
            cPickle.load(open(config['trg_vocab'], 'rb')), bos_idx=0,
            eos_idx=trg_eos_idx, unk_idx=unk_idx)
        trg_ivocab = {v: k for k, v in trg_vocab.items()}

        logger.info("Building sampling model")
        sampling_representation = encoder.apply(
            sampling_input, tensor.ones(sampling_input.shape))
        topic_embedding=topical_transformer.apply(source_topical_word);
        tw_representation=topical_transformer.look_up.apply(source_topical_word.T);
        content_embedding=sampling_representation[0,:,(sampling_representation.shape[2]/2):];
        generated = decoder.generate(sampling_input,sampling_representation, tw_representation,topical_embedding=topic_embedding,content_embedding=content_embedding);

        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs
        beam_search = BeamSearch(samples=samples)

        logger.info("Loading the model..")
        model = Model(generated)
        #loader = LoadNMT(config['saveto'])
        loader = LoadNMT(config['validation_load']);
        loader.set_model_parameters(model, loader.load_parameters_default())

        logger.info("Started translation: ")
        test_stream = get_dev_stream_with_topicalq(**config)
        ts = test_stream.get_epoch_iterator()
        rts = open(config['val_set_source']).readlines()
        ftrans_original = open(config['val_output_orig'], 'w')
        saved_weights = []
        total_cost = 0.0

        pbar = ProgressBar(max_value=len(rts)).start()
        for i, (line, line_raw) in enumerate(zip(ts, rts)):
            trans_in = line_raw.split()
            seq = sutils._oov_to_unk(
                line[0], config['src_vocab_size'], unk_idx)
            seq1=line[1];
            input_topical=numpy.tile(seq1,(config['beam_size'],1))
            input_ = numpy.tile(seq, (config['beam_size'], 1))

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs, attendeds, weights = \
                beam_search.search(
                    input_values={sampling_input: input_,source_topical_word:input_topical,tw_vocab_overlap:tw_vocab_overlap_matrix},
                    tw_vocab_overlap=tw_vocab_overlap_matrix,
                    max_length=3*len(seq), eol_symbol=trg_eos_idx,
                    ignore_first_eol=True)

            # normalize costs according to the sequence lengths
            if config['normalized_bleu']:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            best = numpy.argsort(costs)[0]
            try:
                total_cost += costs[best]
                trans_out = trans[best]
                weight = weights[best][:, :len(trans_in)]
                trans_out = sutils._idx_to_word(trans_out, trg_ivocab)
            except ValueError:
                logger.info(
                    "Can NOT find a translation for line: {}".format(i+1))
                trans_out = '<UNK>'

            saved_weights.append(weight)
            print(' '.join(trans_out), file=ftrans_original)
            pbar.update(i + 1)

        pbar.finish()
        logger.info("Total cost of the test: {}".format(total_cost))
        cPickle.dump(saved_weights, open(config['attention_weights'], 'wb'))
        ftrans_original.close()
        # ap = afterprocesser(config)
        # ap.main()

    elif mode == 'score':
        logger.info('Creating theano variables')
        source_sentence = tensor.lmatrix('source')
        source_sentence_mask = tensor.matrix('source_mask')
        target_sentence = tensor.lmatrix('target')
        target_sentence_mask = tensor.matrix('target_mask')
        target_topic_sentence=tensor.lmatrix('target_topic');
        target_topic_binary_sentence=tensor.lmatrix('target_binary_topic');
        source_topical_word=tensor.lmatrix('source_topical')

        topic_embedding=topical_transformer.apply(source_topical_word);
        # Get cost of the model
        representations = encoder.apply(source_sentence, source_sentence_mask)
        costs = decoder.cost(representations,
                            source_sentence_mask,
                            target_sentence,
                            target_sentence_mask,
                            target_topic_sentence,
                            target_topic_binary_sentence,
                            topic_embedding)

        config['batch_size'] = 1
        config['sort_k_batches'] = 1
        # Get test set stream
        test_stream = get_tr_stream_with_topic_target(**config)

        logger.info("Building sampling model")


        logger.info("Loading the model..")
        model = Model(costs)
        loader = LoadNMT(config['validation_load'])
        loader.set_model_parameters(model, loader.load_parameters_default())

        costs_computer = function([target_sentence,
                                   target_sentence_mask,
                                   source_sentence,
                                   source_sentence_mask,source_topical_word,target_topic_sentence,target_topic_binary_sentence], (costs),on_unused_input='ignore')

        iterator = test_stream.get_epoch_iterator()

        scores = []
        att_weights = []
        for i, (src, src_mask, trg, trg_mask,te,te_mask,tt,tt_mask,tb,tb_mask) in enumerate(iterator):
            costs  = costs_computer(*[trg, trg_mask, src, src_mask,te,tt,tb])
            cost = costs.sum()
            print(i, cost)
            scores.append(cost)

        print(sum(scores)/10007);

