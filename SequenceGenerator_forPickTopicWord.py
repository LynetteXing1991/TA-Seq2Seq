from theano import tensor

from blocks.bricks import Initializable, Random, Bias, NDimensionalSoftmax
from blocks.bricks.base import application, Brick, lazy
from blocks.bricks.parallel import Fork, Merge
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import recurrent
from blocks.roles import add_role, COST
from blocks.utils import dict_union, dict_subset
from blocks.bricks.sequence_generators import (
    BaseSequenceGenerator, FakeAttentionRecurrent)
from attention_with_topicalq import AttentionRecurrent

from blocks.bricks.wrappers import WithExtraDims

class PickTargetProb(Brick):
    """A softmax brick.

    Works with 2-dimensional inputs only. If you need more,
    see :class:`NDimensionalSoftmax`.
    """

    @application(inputs=['y', 'x'], outputs=['cost'])
    def apply(self, application_call, y, x):
        if y.ndim == x.ndim - 1:
            print("y.ndim == x.ndim - 1");
            indices = tensor.arange(y.shape[0]) * x.shape[1] + y
            cost = x.flatten()[indices]
        else:
            raise TypeError('rank mismatch between x and y')
        return cost


class NDPickTargetProb(PickTargetProb):
    decorators = [WithExtraDims()]

class SelectTarget(Random):

    @application
    def emit(self, probs):
        batch_size = probs.shape[0]
        pvals_flat = probs.reshape((batch_size, -1))
        generated = self.theano_rng.multinomial(pvals=pvals_flat)
        return generated.reshape(probs.shape).argmax(axis=-1)

    @application
    def cost(self, y,x):
        indices = tensor.arange(y.shape[0]) * x.shape[1] + y
        cost = x.flatten()[indices]

        return cost


class SequenceGenerator(BaseSequenceGenerator):
    r"""A more user-friendly interface for :class:`BaseSequenceGenerator`.

    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component for the sequence generator.
    transition : instance of :class:`.BaseRecurrent`
        The recurrent transition to be used in the sequence generator.
        Will be combined with `attention`, if that one is given.
    attention : object, optional
        The attention mechanism to be added to ``transition``,
        an instance of
        :class:`~blocks.bricks.attention.AbstractAttention`.
    add_contexts : bool
        If ``True``, the
        :class:`.AttentionRecurrent` wrapping the
        `transition` will add additional contexts for the attended and its
        mask.
    \*\*kwargs : dict
        All keywords arguments are passed to the base class. If `fork`
        keyword argument is not provided, :class:`.Fork` is created
        that forks all transition sequential inputs without a "mask"
        substring in them.

    """
    def __init__(self, readout,topicWordReadout,topic_vector_names, transition,topical_name,content_name,q_dim,q_name, attention=None,topical_attention=None,
                 use_step_decay_cost=False,
                 use_doubly_stochastic=False, lambda_ds=0.001,
                 use_concentration_cost=False, lambda_ct=10,
                 use_stablilizer=False, lambda_st=50,
                 add_contexts=True, **kwargs):
        self.use_doubly_stochastic = use_doubly_stochastic
        self.use_step_decay_cost = use_step_decay_cost
        self.use_concentration_cost = use_concentration_cost
        self.use_stablilizer = use_stablilizer
        self.lambda_ds = lambda_ds
        self.lambda_ct = lambda_ct
        self.lambda_st = lambda_st
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        if attention:
            transition = AttentionRecurrent(
                transition, attention,topical_attention,topical_attended_name='topical_attended',topical_attended_mask_name='topical_attended_mask',content_name=content_name,topical_name=topical_name,
                add_contexts=add_contexts, name="att_trans")
        else:
            transition = FakeAttentionRecurrent(transition,
                                                name="with_fake_attention")

        self.topicWordReadout=topicWordReadout;
        self._topic_vector_names=topic_vector_names;
        self.probPick=NDPickTargetProb();
        self.sampleTarget=SelectTarget();
        #self._q_names=[q_name];
        self.topical_name=topical_name;
        self.content_name=content_name;
        self._topical_context_names=['topical_attended','topical_attended_mask'];
        super(SequenceGenerator, self).__init__(
            readout, transition, **kwargs)
        self.children+=[self.topicWordReadout,self.probPick,self.sampleTarget];

    def _push_allocation_config(self):

        super(SequenceGenerator, self)._push_allocation_config();
        transition_sources = (self._state_names + self._context_names +
                              self._glimpse_names)
        self.topicWordReadout.source_dims = [self.transition.get_dim(name)
                                    if name in transition_sources
                                    else self.readout.get_dim(name)
                                    for name in self.readout.source_names]
        self.topicWordReadout.push_allocation_config()

    @application
    def cost_matrix(self, application_call, outputs,tw_outputs, tw_binary,mask=None, **kwargs):
        """Returns generation costs for output sequences.

        See Also
        --------
        :meth:`cost` : Scalar cost.

        """
        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        topical_word_contexts=dict_subset(kwargs, self._topical_context_names)
        topical_embeddings=dict_subset(kwargs,[self.topical_name]);
        content_embeddings=dict_subset(kwargs,[self.content_name]);
        #q=dict_subset(kwargs, self._q_names, must_have=True,pop=True);
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts,topical_word_contexts,topical_embeddings,content_embeddings))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}
        glimpses_modified={'weighted_averages':glimpses['weighted_averages'],'weigths':glimpses['weights']}

        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(batch_size)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses_modified, contexts))
        #costs = self.readout.cost(readouts, outputs)

        #topicSumVec = dict_subset(kwargs, self._topic_vector_names, must_have=True);
        twReadouts=self.topicWordReadout.readout(feedback=feedback,**dict_union(states,glimpses_modified, contexts));
        twExp=tensor.exp(twReadouts);
        rwExp=tensor.exp(readouts);
        Z=twExp.sum(keepdims=True,axis=2)+rwExp.sum(keepdims=True,axis=2);#remains uncertain,keepdims, and the # of axis
        twExp/=Z;
        rwExp/=Z;
        twCost=self.probPick.apply(tw_outputs,twExp,extra_ndim=twExp.ndim - 2);
        rwCost=self.probPick.apply(outputs,rwExp,extra_ndim=rwExp.ndim - 2);
        totalCost=twCost*tw_binary+rwCost;
        costs=-tensor.log(totalCost);

        if self.use_doubly_stochastic:
            # Doubly stochastic cost
            # \lambda\sum_{i}(1-\sum_{t}w_{t, i})^2
            # the first dimensions of weights returned by transition
            # is batch, time
            weights = glimpses['weights']
            weights_sum_time = tensor.sum(weights, 0)
            penalties = tensor.ones_like(weights_sum_time) - weights_sum_time
            penalties_squared = tensor.pow(penalties, 2)
            ds_costs = tensor.sum(penalties_squared, 1)
            costs += (self.lambda_ds * ds_costs)[None, :]

        def step_decay_cost(states):
            # shape is time, batch, features
            eta = 0.0001
            xi = 100
            states_norm = states.norm(2, axis=2)
            zz = tensor.zeros([1, states.shape[1]])
            padded_norm = tensor.join(0, zz, states_norm)[:-1, :]
            diffs = states_norm - padded_norm
            costs = eta * (xi ** diffs)
            return costs

        if self.use_step_decay_cost:
            costs += step_decay_cost(states['states'])

        def stablilizer_cost(states):
            states_norm = states.norm(2, axis=2)
            zz = tensor.zeros([1, states.shape[1]])
            padded_norm = tensor.join(0, zz, states_norm)[:-1, :]
            diffs = states_norm - padded_norm
            costs = tensor.pow(diffs, 2)
            return costs

        if self.use_stablilizer:
            costs += self.lambda_st * stablilizer_cost(states['states'])

        if self.use_concentration_cost:
            # weights has shape [batch, time, source sentence len]
            weights = glimpses['weights']
            maxis = tensor.max(weights, axis=2)
            lacks = tensor.ones_like(maxis) - maxis
            costs += self.lambda_ct * lacks

        if mask is not None:
            costs *= mask

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)

        # This variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        for name in self._state_names:
            application_call.add_auxiliary_variable(
                results[name][-1].copy(), name=name+"_final_value")

        return costs


    @recurrent
    def generate(self, outputs,tw_vocab_overlap, **kwargs):
        """A sequence generation step.

        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The outputs from the previous step.

        Notes
        -----
        The contexts, previous states and glimpses are expected as keyword
        arguments.

        """
        states = dict_subset(kwargs, self._state_names)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        topical_word_contexts=dict_subset(kwargs, self._topical_context_names)
        topical_embeddings=dict_subset(kwargs,[self.topical_name]);
        content_embeddings=dict_subset(kwargs,[self.content_name]);
        glimpses = dict_subset(kwargs, self._glimpse_names)
        next_glimpses = self.transition.take_glimpses(
            as_dict=True,
            **dict_union(
                states, glimpses,topical_embeddings,content_embeddings,contexts,topical_word_contexts));
        glimpses_modified={'weighted_averages':next_glimpses['weighted_averages'],'weigths':next_glimpses['weights']}
        next_readouts = self.readout.readout(
            feedback=self.readout.feedback(outputs),
            **dict_union(states, glimpses_modified, contexts))
        next_tw_readouts=self.topicWordReadout.readout(
            feedback=self.readout.feedback(outputs),
            **dict_union(states, glimpses_modified, contexts))
        twExp=tensor.exp(next_tw_readouts);
        rwExp=tensor.exp(next_readouts);
        Z=twExp.sum(keepdims=True,axis=1)+rwExp.sum(keepdims=True,axis=1);#remains uncertain,keepdims, and the # of axis
        twExp/=Z;
        rwExp/=Z;
        probs=tensor.dot(twExp,tw_vocab_overlap)+rwExp;
        next_outputs = self.sampleTarget.emit(probs);
        next_costs = self.sampleTarget.cost(next_outputs, probs)
        next_costs=-tensor.log(next_costs);
        next_feedback = self.readout.feedback(next_outputs)
        next_inputs = (self.fork.apply(next_feedback, as_dict=True)
                       if self.fork else {'feedback': next_feedback})
        next_states = self.transition.compute_states(
            as_list=True,
            **dict_union(next_inputs, states, next_glimpses, contexts,topical_word_contexts))
        return (next_states + [next_outputs] +
                list(next_glimpses.values()) + [next_costs])

    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('states')
    def generate_states(self):
        return self._state_names + ['outputs'] + self._glimpse_names

    @generate.property('outputs')
    def generate_outputs(self):
        return (self._state_names + ['outputs'] +
                self._glimpse_names + ['costs'])

    @generate.property('contexts')
    def generate_contexts(self):
        return (self.transition.apply.contexts++self._topical_context_names+[self.content_name]+[self.topical_name]
        +['tw_vocab_overlap'])

    def get_dim(self, name):
        if name in (self._state_names + self._context_names +
                    self._glimpse_names):
            return self.transition.get_dim(name)
        elif name == 'outputs':
            return self.readout.get_dim(name)
        return super(BaseSequenceGenerator, self).get_dim(name)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        # TODO: support dict of outputs for application methods
        # to simplify this code.
        state_dict = dict(
            self.transition.initial_states(
                batch_size, as_dict=True, *args, **kwargs),
            outputs=self.readout.initial_outputs(batch_size))
        return [state_dict[state_name]
                for state_name in self.generate.states]

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.generate.states
