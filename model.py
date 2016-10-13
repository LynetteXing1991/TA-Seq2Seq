from theano import tensor
from toolz import merge

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP)
# from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter)
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans

from picklable_itertools.extras import equizip
from attention import SequenceContentAttention
#from SequenceGenerator import SequenceGenerator
from SequenceGenerator_forPickTopicWord import SequenceGenerator
from GRU import GRU
from match_functions import (
    SumMatchFunction, CatMatchFunction,
    DotMatchFunction, GeneralMatchFunction)


# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass


class LookupFeedbackWMT15(LookupFeedback):
    """Zero-out initial readout feedback by checking its value."""

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0

        shp = [outputs.shape[i] for i in range(outputs.ndim)]
        outputs_flat = outputs.flatten()
        outputs_flat_zeros = tensor.switch(outputs_flat < 0, 0,
                                           outputs_flat)

        lookup_flat = tensor.switch(
            outputs_flat[:, None] < 0,
            tensor.alloc(0., outputs_flat.shape[0], self.feedback_dim),
            self.lookup.apply(outputs_flat_zeros))
        lookup = lookup_flat.reshape(shp+[self.feedback_dim])
        return lookup


class BidirectionalWMT15(Bidirectional):
    """Wrap two Gated Recurrents each having separate parameters."""

    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]


class BidirectionalEncoder(Initializable):
    """Encoder of RNNsearch model."""

    def __init__(self, vocab_size, embedding_dim, state_dim, **kwargs):
        super(BidirectionalEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim

        self.lookup = LookupTable(name='embeddings')
        self.bidir = BidirectionalWMT15(
            GatedRecurrent(activation=Tanh(), dim=state_dim))
        self.fwd_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='fwd_fork')
        self.back_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='back_fork')

        self.children = [self.lookup, self.bidir,
                         self.fwd_fork, self.back_fork]

    def _push_allocation_config(self):
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim

        self.fwd_fork.input_dim = self.embedding_dim
        self.fwd_fork.output_dims = [self.bidir.children[0].get_dim(name)
                                     for name in self.fwd_fork.output_names]
        self.back_fork.input_dim = self.embedding_dim
        self.back_fork.output_dims = [self.bidir.children[1].get_dim(name)
                                      for name in self.back_fork.output_names]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation'])
    def apply(self, source_sentence, source_sentence_mask):
        # Time as first dimension
        source_sentence = source_sentence.T
        source_sentence_mask = source_sentence_mask.T

        embeddings = self.lookup.apply(source_sentence)

        representation = self.bidir.apply(
            merge(self.fwd_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask}),
            merge(self.back_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask})
        )
        return representation

class topicalq_transformer(Initializable):

    def __init__(self, vocab_size, topical_embedding_dim, state_dim,word_num,batch_size,
                 **kwargs):
        super(topicalq_transformer, self).__init__(**kwargs)
        self.vocab_size = vocab_size;
        self.word_embedding_dim = topical_embedding_dim;
        self.state_dim = state_dim;
        self.word_num=word_num;
        self.batch_size=batch_size;
        self.look_up=LookupTable(name='topical_embeddings');
        self.transformer=MLP(activations=[Tanh()],
                                dims=[self.word_embedding_dim*self.word_num, self.state_dim],
                                name='topical_transformer');
        self.children = [self.look_up,self.transformer];

    def _push_allocation_config(self):
        self.look_up.length = self.vocab_size
        self.look_up.dim = self.word_embedding_dim


    # do we have to push_config? remain unsure
    @application(inputs=['source_topical_word_sequence'],
                 outputs=['topical_embedding'])
    def apply(self, source_topical_word_sequence):
        # Time as first dimension
        source_topical_word_sequence=source_topical_word_sequence.T;
        word_topical_embeddings = self.look_up.apply(source_topical_word_sequence);
        word_topical_embeddings=word_topical_embeddings.swapaxes(0,1);
        #requires testing
        concatenated_topical_embeddings=tensor.reshape(word_topical_embeddings,[word_topical_embeddings.shape[0],word_topical_embeddings.shape[1]*word_topical_embeddings.shape[2]]);
        topical_embedding=self.transformer.apply(concatenated_topical_embeddings);
        return topical_embedding

class Decoder(Initializable):
    """Decoder of RNNsearch model."""

    def __init__(self, vocab_size, topicWord_size, embedding_dim, state_dim,topical_dim,
                 representation_dim, match_function='SumMacthFunction',
                 use_doubly_stochastic=False, lambda_ds=0.001,
                 use_local_attention=False, window_size=10,
                 use_step_decay_cost=False,
                 use_concentration_cost=False, lambda_ct=10,
                 use_stablilizer=False, lambda_st=50,
                 theano_seed=None, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.topicWord_size= topicWord_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.theano_seed = theano_seed

        # Initialize gru with special initial state
        self.transition = GRU(
            attended_dim=state_dim, dim=state_dim,
            activation=Tanh(), name='decoder')

        self.energy_computer = globals()[match_function](name='energy_comp')

        # Initialize the attention mechanism
        self.attention = SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=representation_dim,
            match_dim=state_dim,
            energy_computer=self.energy_computer,
            use_local_attention=use_local_attention,
            window_size=window_size,
            name="attention")

        self.topical_attention=SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=topical_dim,
            match_dim=state_dim,
            energy_computer=self.energy_computer,
            use_local_attention=use_local_attention,
            window_size=window_size,
            name="topical_attention")#not sure whether the match dim would be correct.


        # Initialize the readout, note that SoftmaxEmitter emits -1 for
        # initial outputs which is used by LookupFeedBackWMT15
        readout = Readout(
            source_names=['states', 'feedback',
                          self.attention.take_glimpses.outputs[0]],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=-1, theano_seed=theano_seed),
            feedback_brick=LookupFeedbackWMT15(vocab_size, embedding_dim),
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=state_dim, name='maxout_bias').apply,
                 Maxout(num_pieces=2, name='maxout').apply,
                 Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
                        use_bias=False, name='softmax0').apply,
                 Linear(input_dim=embedding_dim, name='softmax1').apply]),
            merged_dim=state_dim,
            name='readout')

        # calculate the readout of topic word,
        # no specific feedback brick, use the trival feedback break
        # no post_merge and merge, use Bias and Linear
        topicWordReadout = Readout(
            source_names=['states', 'feedback',
                          self.attention.take_glimpses.outputs[0]],
            readout_dim=self.topicWord_size,
            emitter=SoftmaxEmitter(initial_output=-1, theano_seed=theano_seed),
            name='twReadout')


        # Build sequence generator accordingly
        self.sequence_generator = SequenceGenerator(
            readout=readout,
            topicWordReadout=topicWordReadout,
            topic_vector_names=['topicSumVector'],
            transition=self.transition,
            attention=self.attention,
            topical_attention=self.topical_attention,
            q_dim=self.state_dim,
            #q_name='topic_embedding',
            topical_name='topic_embedding',
            content_name='content_embedding',
            use_step_decay_cost=use_step_decay_cost,
            use_doubly_stochastic=use_doubly_stochastic, lambda_ds=lambda_ds,
            use_concentration_cost=use_concentration_cost, lambda_ct=lambda_ct,
            use_stablilizer=use_stablilizer, lambda_st=lambda_st,
            fork=Fork([name for name in self.transition.apply.sequences
                       if name != 'mask'], prototype=Linear())
        )

        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'tw_representation','tw_mask','target_sentence','target_sentence_mask',
                         'target_topic_sentence','target_topic_binary_sentence','topic_embedding','content_embedding'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,tw_representation,tw_mask,
             target_sentence, target_sentence_mask, target_topic_sentence,target_topic_binary_sentence,topic_embedding,content_embedding):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T
        target_topic_sentence=target_topic_sentence.T
        target_topic_binary_sentence=target_topic_binary_sentence.T
        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended': representation,
            'attended_mask': source_sentence_mask,
            'topical_attended':tw_representation,
            'topical_attended_mask':tw_mask,
            'topic_embedding':topic_embedding,
            'content_embedding':content_embedding,
            'tw_outputs': target_topic_sentence,
            'tw_binary': target_topic_binary_sentence}
        )

        '''
        return (cost * target_sentence_mask).sum() / \
            target_sentence_mask.shape[1]
        '''
        #return tensor.exp((cost * target_sentence_mask).sum()/tensor.sum(target_sentence_mask))
        return (cost * target_sentence_mask).sum()/tensor.sum(target_sentence_mask)

    @application
    def generate(self, source_sentence, representation, tw_vocab_overlap,topic_embedding,**kwargs):
        return self.sequence_generator.generate(
            tw_vocab_overlap=tw_vocab_overlap,
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            topic_embedding=topic_embedding,
            **kwargs)
