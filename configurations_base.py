def default():
    config = {}
    config['saveto'] = 'models/default'
    config['model_name'] = 'default'

    # model hyperparameters
    config['seq_len'] = 200
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # traning options
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['initial_learning_rate'] = 1.0
    config['learning_rate_decay'] = False
    config['learning_rate_grow'] = False
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # data basics
    config['stream'] = 'stream'
    datadir = open('datadir_wsj', 'rb').readline().rstrip()
    config['datadir'] = datadir
    # dictionary options
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # dictionaries
    config['src_data'] = datadir + 'train/' + 'source'
    config['trg_data'] = datadir + 'train/' + 'target'
    config['src_vocab'] = datadir + 'train/' + 'source.vocab.pkl'
    config['trg_vocab'] = datadir + 'train/' + 'target.vocab.pkl'
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30000
    # validation options
    config['val_set_source'] = datadir + 'test/' + 'source1'
    config['val_set_target'] = datadir + 'test/' + 'target1'
    config['bleu_script'] = datadir + 'multi-bleu.perl'
    config['bleu_script_1'] = datadir + 'CompBleu.exe'
    config['normalized_bleu'] = True
    # testing options
    config['beam_size'] = 12
    # model saving related
    config['finish_after'] = 1000000
    config['reload'] = True
    config['save_freq'] = 500
    config['sampling_freq'] = 20
    config['hook_samples'] = 2
    config['bleu_val_freq'] = 1000
    config['val_burn_in'] = 20000

    # afterprocess config
    config['translation_table'] = datadir + 'translation_table'
    config['replacement_table'] = datadir + 'test/' + 'replacement_table'

    config['use_doubly_stochastic'] = False
    config['lambda_ds'] = 0.001

    config['use_local_attention'] = False
    config['window_size'] = 10

    config['use_step_decay_cost'] = False

    config['use_concentration_cost'] = False
    config['lambda_ct'] = 10

    # arxiv.org/abs/1511.08400
    config['use_stablilizer'] = False
    config['lambda_st'] = 1

    return config

def aaai2017Douban():
    config = {}

    # model hyperparameters
    config['seq_len'] = 200
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # traning options
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['initial_learning_rate'] = 1.0
    config['learning_rate_decay'] = False
    config['learning_rate_grow'] = False
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # data basics
    config['stream'] = 'stream'
    datadir = 'D:\users\chxing\\aaai2017Exp\\filterDouban\data.r50.q1min30max.tokenized.2-20.1-20\\'
    config['datadir'] = datadir
    config['saveto'] = datadir+'models\\'
    config['validation_load']=datadir+'model_for_test\\'
    config['model_name'] = 's2saModel'
    # dictionary options
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # dictionaries
    config['src_data'] = datadir + 'query'
    config['trg_data'] = datadir + 'response'
    config['src_vocab'] = datadir + 'query.vocab'
    config['trg_vocab'] = datadir + 'response.vocab'
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30000
    # # validation options
    config['val_set_source'] = datadir + 'query.dev'
    config['val_set_target'] = datadir + 'response.dev'
    # config['bleu_script'] = datadir + 'multi-bleu.perl'
    # config['bleu_script_1'] = datadir + 'CompBleu.exe'
    # config['normalized_bleu'] = True
    # testing options
    config['normalized_bleu'] = True
    config['beam_size'] = 12
    # model saving related
    config['finish_after'] = 1000000
    config['reload'] = True
    config['save_freq'] = 500
    config['sampling_freq'] = 20
    config['hook_samples'] = 2
    config['bleu_val_freq'] = 1000
    config['val_burn_in'] = 20000

    # # afterprocess config
    # config['translation_table'] = datadir + 'translation_table'
    # config['replacement_table'] = datadir + 'test/' + 'replacement_table'

    config['use_doubly_stochastic'] = False
    config['lambda_ds'] = 0.001

    config['use_local_attention'] = False
    config['window_size'] = 10

    config['use_step_decay_cost'] = False

    config['use_concentration_cost'] = False
    config['lambda_ct'] = 10

    # arxiv.org/abs/1511.08400
    config['use_stablilizer'] = False
    config['lambda_st'] = 1

    config['match_function'] = 'SumMatchFunction'
    config['attention_images'] = config['saveto'] + '/attention_images/'
    config['attention_weights'] = config['saveto'] + '/attention_weights'
    config['val_output_orig'] = config['saveto'] + '/test_output_orig'
    config['val_output_repl'] = config['saveto'] + '/test_output_repl'

    return config

def aaai2017Douban_targettopic_test():
    config = {}

    # model hyperparameters
    config['seq_len'] = 200
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # traning options
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['initial_learning_rate'] = 1.0
    config['learning_rate_decay'] = False
    config['learning_rate_grow'] = False
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # data basics
    config['stream'] = 'stream'
    datadir='D:\users\chxing\data\\topical\chinese\\former_douban\\';
    config['datadir'] = datadir
    config['saveto'] = datadir+'models\\'
    config['validation_load']=datadir+'models\\'
    config['model_name'] = 's2saTargetTopic'
    # dictionary options
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # dictionaries
    config['src_vocab'] = datadir + 'tokensized_chinese_query_vocab.pkl'
    config['trg_vocab'] = datadir + 'tokensized_chinese_response_vocab.pkl'
    config['src_data'] = datadir + 'tokenized_chinese_query.txt'
    config['trg_data'] = datadir + 'tokenized_chinese_response.txt'
    config['src_vocab_size'] = 20000
    config['trg_vocab_size'] = 20000
    config['val_set_source'] = datadir + 'tokenized_chinese_query.txt'
    config['val_set_target'] = datadir + 'response.dev'
    config['normalized_bleu'] = True
    config['beam_size'] = 12
    # model saving related
    config['finish_after'] = 1000000
    config['reload'] = True
    config['save_freq'] = 500
    config['sampling_freq'] = 20
    config['hook_samples'] = 2
    config['bleu_val_freq'] = 1000
    config['val_burn_in'] = 20000

    # # afterprocess config
    # config['translation_table'] = datadir + 'translation_table'
    # config['replacement_table'] = datadir + 'test/' + 'replacement_table'

    config['use_doubly_stochastic'] = False
    config['lambda_ds'] = 0.001

    config['use_local_attention'] = False
    config['window_size'] = 10

    config['use_step_decay_cost'] = False

    config['use_concentration_cost'] = False
    config['lambda_ct'] = 10

    # arxiv.org/abs/1511.08400
    config['use_stablilizer'] = False
    config['lambda_st'] = 1

    config['match_function'] = 'SumMatchFunction'
    config['attention_images'] = config['saveto'] + '/attention_images/'
    config['attention_weights'] = config['saveto'] + '/attention_weights'
    config['val_output_orig'] = config['saveto'] + '/test_output_orig'
    config['val_output_repl'] = config['saveto'] + '/test_output_repl'

    config['topical_vocab_size']=1737#adding the unk
    config['topical_vocab']='D:\users\chxing\data\\topical\chinese\\former_douban\\aaai2017topicTest\\tokensized_chinese_response_vocab.pkl';
    config['tw_vocab_overlap']='D:\users\chxing\data\\topical\chinese\\former_douban\\aaai2017topicTest\\W.pkl'
    config['trg_vocab_size'] = 20000
    config['trg_vocab'] = 'D:\users\chxing\data\\topical\chinese\\former_douban\\aaai2017topicTest\\tokensized_chinese_topic_vocab.pkl'
    return config

def aaai2017Douban_with_extra_class():
    config = {}

    # model hyperparameters
    config['seq_len'] = 200
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # traning options
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['initial_learning_rate'] = 1.0
    config['learning_rate_decay'] = False
    config['learning_rate_grow'] = False
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # data basics
    config['stream'] = 'stream'
    datadir = 'D:\users\chxing\\aaai2017Exp\\filterDouban\data.r50.q1min30max.tokenized.withBaikeWord.2-20.1-20\\'
    config['datadir'] = datadir
    config['saveto'] = datadir+'models\\'
    config['validation_load']=datadir+'models\\models_for_test\\'
    config['model_name'] = 's2sa_decoder_with_extra_class'
    # dictionary options
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # dictionaries
    config['src_data'] = datadir + 'query'
    config['trg_data'] = datadir + 'response'
    config['src_vocab'] = datadir + 'query.pkl'
    config['trg_vocab'] = datadir + 'response_withTw.pkl'
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30112
    # # validation options
    config['val_set_source'] = datadir + 'query.dev'
    config['val_set_target'] = datadir + 'response.dev'
    # config['bleu_script'] = datadir + 'multi-bleu.perl'
    # config['bleu_script_1'] = datadir + 'CompBleu.exe'
    # config['normalized_bleu'] = True
    # testing options
    config['normalized_bleu'] = True
    config['beam_size'] = 12
    # model saving related
    config['finish_after'] = 1000000
    config['reload'] = True
    config['save_freq'] = 500
    config['sampling_freq'] = 20
    config['hook_samples'] = 2
    config['bleu_val_freq'] = 1000
    config['val_burn_in'] = 20000

    # # afterprocess config
    # config['translation_table'] = datadir + 'translation_table'
    # config['replacement_table'] = datadir + 'test/' + 'replacement_table'

    config['use_doubly_stochastic'] = False
    config['lambda_ds'] = 0.001

    config['use_local_attention'] = False
    config['window_size'] = 10

    config['use_step_decay_cost'] = False

    config['use_concentration_cost'] = False
    config['lambda_ct'] = 10

    # arxiv.org/abs/1511.08400
    config['use_stablilizer'] = False
    config['lambda_st'] = 1

    config['match_function'] = 'SumMatchFunction'
    config['attention_images'] = config['saveto'] + '/attention_images/'
    config['attention_weights'] = config['saveto'] + '/attention_weights'
    config['val_output_orig'] = config['validation_load'] + '/test_output_orig'
    config['val_output_repl'] = config['saveto'] + '/test_output_repl'

    config['topical_vocab_size']=2851#adding the unk and EOS
    config['topical_vocab']=datadir + 'topic_withSpecialTok.pkl';
    config['tw_vocab_overlap']=datadir+'W.pkl'

    return config

def aaai2017Douban_with_extra_class_nonPopularTopicWord():
    config = {}

    # model hyperparameters
    config['seq_len'] = 200
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # traning options
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['initial_learning_rate'] = 1.0
    config['learning_rate_decay'] = False
    config['learning_rate_grow'] = False
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # data basics
    config['stream'] = 'stream'
    datadir = 'D:\users\chxing\\aaai2017Exp\\filterDouban\data.r50.q1min30max.tokenized.withBaikeWord.2-20.1-20\\'
    config['datadir'] = datadir
    config['saveto'] = datadir+'extraClass_nonPopularTopicWord_models\\'
    config['validation_load']=datadir+'extraClass_nonPopularTopicWord_models\\models_for_test\\'
    config['model_name'] = 's2sa_decoder_with_extra_class_nonPopularTopicWord'
    # dictionary options
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # dictionaries
    config['src_data'] = datadir + 'query'
    config['trg_data'] = datadir + 'response'
    config['src_vocab'] = datadir + 'query.pkl'
    config['trg_vocab'] = datadir + 'response_withTw.pkl'
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30112
    # # validation options
    config['val_set_source'] = datadir + 'query.dev'
    config['val_set_target'] = datadir + 'response.dev'
    # config['bleu_script'] = datadir + 'multi-bleu.perl'
    # config['bleu_script_1'] = datadir + 'CompBleu.exe'
    # config['normalized_bleu'] = True
    # testing options
    config['normalized_bleu'] = True
    config['beam_size'] = 12
    # model saving related
    config['finish_after'] = 1000000
    config['reload'] = True
    config['save_freq'] = 500
    config['sampling_freq'] = 20
    config['hook_samples'] = 2
    config['bleu_val_freq'] = 1000
    config['val_burn_in'] = 20000

    # # afterprocess config
    # config['translation_table'] = datadir + 'translation_table'
    # config['replacement_table'] = datadir + 'test/' + 'replacement_table'

    config['use_doubly_stochastic'] = False
    config['lambda_ds'] = 0.001

    config['use_local_attention'] = False
    config['window_size'] = 10

    config['use_step_decay_cost'] = False

    config['use_concentration_cost'] = False
    config['lambda_ct'] = 10

    # arxiv.org/abs/1511.08400
    config['use_stablilizer'] = False
    config['lambda_st'] = 1

    config['match_function'] = 'SumMatchFunction'
    config['attention_images'] = config['saveto'] + '/attention_images/'
    config['attention_weights'] = config['saveto'] + '/attention_weights'
    config['val_output_orig'] = config['validation_load'] + '/test_output_orig'
    config['val_output_repl'] = config['saveto'] + '/test_output_repl'

    config['topical_vocab_size']=2848#adding the unk and EOS
    config['topical_vocab']=datadir + 'topic_nonPopularWord\\topic_withSpecialTok_withoutPopularWord.pkl';
    config['tw_vocab_overlap']=datadir+'topic_nonPopularWord\\W.pkl'

    return config

def aaai2017Douban_with_extra_class_topictt():
    config = {}

    # model hyperparameters
    config['seq_len'] = 200
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # traning options
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['initial_learning_rate'] = 1.0
    config['learning_rate_decay'] = False
    config['learning_rate_grow'] = False
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # data basics
    config['stream'] = 'stream'
    datadir = 'D:\users\chxing\\aaai2017Exp\\filterDouban\data.r50.q1min30max.tokenized.withBaikeWord.2-20.1-20\\'
    config['datadir'] = datadir
    config['saveto'] = datadir+'extraClass_topictt_models\\'
    config['validation_load']=datadir+'extraClass_topictt_models\\models_for_test\\'
    config['model_name'] = 's2sa_decoder_extraClass_topictt_models'
    # dictionary options
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # dictionaries
    config['src_data'] = datadir + 'query'
    config['trg_data'] = datadir + 'response'
    config['src_vocab'] = datadir + 'query.pkl'
    config['trg_vocab'] = datadir + 'topic_tt\\response_withTw.pkl'
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30213
    # # validation options
    config['val_set_source'] = datadir + 'query.dev'
    config['val_set_target'] = datadir + 'response.dev'
    # config['bleu_script'] = datadir + 'multi-bleu.perl'
    # config['bleu_script_1'] = datadir + 'CompBleu.exe'
    # config['normalized_bleu'] = True
    # testing options
    config['normalized_bleu'] = True
    config['beam_size'] = 12
    # model saving related
    config['finish_after'] = 1000000
    config['reload'] = True
    config['save_freq'] = 500
    config['sampling_freq'] = 20
    config['hook_samples'] = 2
    config['bleu_val_freq'] = 1000
    config['val_burn_in'] = 20000

    # # afterprocess config
    # config['translation_table'] = datadir + 'translation_table'
    # config['replacement_table'] = datadir + 'test/' + 'replacement_table'

    config['use_doubly_stochastic'] = False
    config['lambda_ds'] = 0.001

    config['use_local_attention'] = False
    config['window_size'] = 10

    config['use_step_decay_cost'] = False

    config['use_concentration_cost'] = False
    config['lambda_ct'] = 10

    # arxiv.org/abs/1511.08400
    config['use_stablilizer'] = False
    config['lambda_st'] = 1

    config['match_function'] = 'SumMatchFunction'
    config['attention_images'] = config['saveto'] + '/attention_images/'
    config['attention_weights'] = config['saveto'] + '/attention_weights'
    config['val_output_orig'] = config['validation_load'] + '/test_output_orig'
    config['val_output_repl'] = config['saveto'] + '/test_output_repl'

    config['topical_vocab_size']=1736#adding the unk and EOS
    config['topical_vocab']=datadir + 'topic_tt\\topic_withSpecialTok_withoutPopularWord.pkl';
    config['tw_vocab_overlap']=datadir+'topic_tt\\W.pkl'

    return config

def aaai2017Douban_with_extra_class_nonPopularTopicWord_latterpart():
    config = {}

    # model hyperparameters
    config['seq_len'] = 200
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # traning options
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['initial_learning_rate'] = 1.0
    config['learning_rate_decay'] = False
    config['learning_rate_grow'] = False
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # data basics
    config['stream'] = 'stream'
    datadir = 'D:\users\chxing\\aaai2017Exp\\filterDouban\data.r50.q1min30max.tokenized.withBaikeWord.2-20.1-20\\'
    config['datadir'] = datadir
    config['saveto'] = datadir+'extraClass_nonPopularTopicWord_models\\'
    config['validation_load']=datadir+'extraClass_nonPopularTopicWord_models\\models_for_test\\'
    config['model_name'] = 's2sa_decoder_with_extra_class_nonPopularTopicWord'
    # dictionary options
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # dictionaries
    config['src_data'] = datadir + 'query'
    config['trg_data'] = datadir + 'response'
    config['src_vocab'] = datadir + 'query.pkl'
    config['trg_vocab'] = datadir + 'response_withTw.pkl'
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30112
    # # validation options
    config['val_set_source'] = datadir + 'query.dev.latterpart'
    config['val_set_target'] = datadir + 'response.dev'
    # config['bleu_script'] = datadir + 'multi-bleu.perl'
    # config['bleu_script_1'] = datadir + 'CompBleu.exe'
    # config['normalized_bleu'] = True
    # testing options
    config['normalized_bleu'] = True
    config['beam_size'] = 12
    # model saving related
    config['finish_after'] = 1000000
    config['reload'] = True
    config['save_freq'] = 500
    config['sampling_freq'] = 20
    config['hook_samples'] = 2
    config['bleu_val_freq'] = 1000
    config['val_burn_in'] = 20000

    # # afterprocess config
    # config['translation_table'] = datadir + 'translation_table'
    # config['replacement_table'] = datadir + 'test/' + 'replacement_table'

    config['use_doubly_stochastic'] = False
    config['lambda_ds'] = 0.001

    config['use_local_attention'] = False
    config['window_size'] = 10

    config['use_step_decay_cost'] = False

    config['use_concentration_cost'] = False
    config['lambda_ct'] = 10

    # arxiv.org/abs/1511.08400
    config['use_stablilizer'] = False
    config['lambda_st'] = 1

    config['match_function'] = 'SumMatchFunction'
    config['attention_images'] = config['saveto'] + '/attention_images/'
    config['attention_weights'] = config['saveto'] + '/attention_weights'
    config['val_output_orig'] = config['validation_load'] + '/test_output_orig.latterpart'
    config['val_output_repl'] = config['saveto'] + '/test_output_repl'

    config['topical_vocab_size']=2848#adding the unk and EOS
    config['topical_vocab']=datadir + 'topic_nonPopularWord\\topic_withSpecialTok_withoutPopularWord.pkl';
    config['tw_vocab_overlap']=datadir+'topic_nonPopularWord\\W.pkl'

    return config

def aaai2017Douban_with_extra_class_topicEncoder():
    config = {}

    # model hyperparameters
    config['seq_len'] = 200
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # traning options
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['initial_learning_rate'] = 1.0
    config['learning_rate_decay'] = False
    config['learning_rate_grow'] = False
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # data basics
    config['stream'] = 'stream'
    datadir = 'D:\users\chxing\\aaai2017Exp\\filterDouban\data.r50.q1min30max.tokenized.withBaikeWord.2-20.1-20\\'
    config['datadir'] = datadir
    config['saveto'] = datadir+'models_decoder_with_extra_class_topicEncoder\\'
    config['validation_load']=datadir+'models_decoder_with_extra_class_topicEncoder\\models_for_test\\'
    config['model_name'] = 's2sa_decoder_with_extra_class_topicEncoder'
    # dictionary options
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # dictionaries
    config['src_data'] = datadir + 'query'
    config['trg_data'] = datadir + 'response'
    config['src_vocab'] = datadir + 'query.pkl'
    config['trg_vocab'] = datadir + '\\topic_highFreqRemoved\\response_withTw.pkl'
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30248
    # # validation options
    config['val_set_source'] = datadir + 'query.dev.latterpart'
    config['val_set_target'] = datadir + 'response.dev'
    config['normalized_bleu'] = True
    config['beam_size'] = 12
    # model saving related
    config['finish_after'] = 1000000
    config['reload'] = True
    config['save_freq'] = 500
    config['sampling_freq'] = 20
    config['hook_samples'] = 2
    config['bleu_val_freq'] = 1000
    config['val_burn_in'] = 20000

    config['use_doubly_stochastic'] = False
    config['lambda_ds'] = 0.001

    config['use_local_attention'] = False
    config['window_size'] = 10

    config['use_step_decay_cost'] = False

    config['use_concentration_cost'] = False
    config['lambda_ct'] = 10

    # arxiv.org/abs/1511.08400
    config['use_stablilizer'] = False
    config['lambda_st'] = 1

    config['match_function'] = 'SumMatchFunction'
    config['attention_images'] = config['saveto'] + '/attention_images/'
    config['attention_weights'] = config['saveto'] + '/attention_weights'
    config['val_output_orig'] = config['validation_load'] + '//test_output_orig.latterpart'
    config['val_output_repl'] = config['saveto'] + '/test_output_repl'

    config['trg_topic_vocab_size']=2626#adding the unk and EOS
    config['topic_vocab_output']=datadir + '\\topic_highFreqRemoved\\topic_withSpecialTok_highFreqRemoved.pkl';
    config['tw_vocab_overlap']=datadir+'\\topic_highFreqRemoved\\W.pkl'

    #topical related features--------------------------------------------------------
    config['source_topic_vocab_size']=1735
    config['topical_embedding_dim']=200
    config['topical_word_num']=10
    config['topical_embeddings']='D:\users\chxing\\aaai2017Exp\\filterDouban\\topicSetting\\tLDAtopic\word_topic_normalize.tt10.pkl'
    config['topic_vocab_input']='D:\users\chxing\\aaai2017Exp\\filterDouban\\topicSetting\\tLDAtopic\LDADic.tt10.pkl';
    config['topical_data']= datadir + 'response.topic.input'
    config['topical_test_set']= datadir+'query.dev.topic.input.latterpart'

    return config

def aaai2017Douban_with_extra_class_topicEncoder_perplexity():
    config = {}

    # model hyperparameters
    config['seq_len'] = 200
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # traning options
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['initial_learning_rate'] = 1.0
    config['learning_rate_decay'] = False
    config['learning_rate_grow'] = False
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # data basics
    config['stream'] = 'stream'
    datadir = 'D:\users\chxing\\aaai2017Exp\\filterDouban\data.r50.q1min30max.tokenized.withBaikeWord.2-20.1-20\\'
    config['datadir'] = datadir
    config['saveto'] = datadir+'models_decoder_with_extra_class_topicEncoder\\'
    config['validation_load']=datadir+'models_decoder_with_extra_class_topicEncoder\\models_for_test\\'
    config['model_name'] = 's2sa_decoder_with_extra_class_topicEncoder'
    # dictionary options
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # dictionaries
    config['src_data'] = datadir + 'query'
    config['trg_data'] = datadir + 'response.test'
    config['src_vocab'] = datadir + 'query.pkl'
    config['trg_vocab'] = datadir + '\\topic_highFreqRemoved\\response_withTw.pkl'
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30248
    # # validation options
    config['val_set_source'] = datadir + 'query.dev.latterpart'
    config['val_set_target'] = datadir + 'response.dev'
    config['normalized_bleu'] = True
    config['beam_size'] = 12
    # model saving related
    config['finish_after'] = 1000000
    config['reload'] = True
    config['save_freq'] = 500
    config['sampling_freq'] = 20
    config['hook_samples'] = 2
    config['bleu_val_freq'] = 1000
    config['val_burn_in'] = 20000

    config['use_doubly_stochastic'] = False
    config['lambda_ds'] = 0.001

    config['use_local_attention'] = False
    config['window_size'] = 10

    config['use_step_decay_cost'] = False

    config['use_concentration_cost'] = False
    config['lambda_ct'] = 10

    # arxiv.org/abs/1511.08400
    config['use_stablilizer'] = False
    config['lambda_st'] = 1

    config['match_function'] = 'SumMatchFunction'
    config['attention_images'] = config['saveto'] + '/attention_images/'
    config['attention_weights'] = config['saveto'] + '/attention_weights'
    config['val_output_orig'] = config['validation_load'] + '//test_output_orig.latterpart'
    config['val_output_repl'] = config['saveto'] + '/test_output_repl'

    config['trg_topic_vocab_size']=2626#adding the unk and EOS
    config['topic_vocab_output']=datadir + '\\topic_highFreqRemoved\\topic_withSpecialTok_highFreqRemoved.pkl';
    config['tw_vocab_overlap']=datadir+'\\topic_highFreqRemoved\\W.pkl'

    #topical related features--------------------------------------------------------
    config['source_topic_vocab_size']=1735
    config['topical_embedding_dim']=200
    config['topical_word_num']=10
    config['topical_embeddings']='D:\users\chxing\\aaai2017Exp\\filterDouban\\topicSetting\\tLDAtopic\word_topic_normalize.tt10.pkl'
    config['topic_vocab_input']='D:\users\chxing\\aaai2017Exp\\filterDouban\\topicSetting\\tLDAtopic\LDADic.tt10.pkl';
    config['topical_data']= datadir + 'query.topic.input'
    config['topical_test_set']= datadir+'query.dev.topic.input.latterpart'

    return config

def topicAwareJPData():
    config = {}

    # model hyperparameters
    config['seq_len'] = 200
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # traning options
    config['batch_size'] = 128
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['initial_learning_rate'] = 1.0
    config['learning_rate_decay'] = False
    config['learning_rate_grow'] = False
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # data basics
    config['stream'] = 'stream'
    datadir = 'D:\users\chxing\\aaai2017Exp\JPdata\\r50maxcount.q1min10max.2-20.1-20\\'
    config['datadir'] = datadir
    config['saveto'] = datadir+'s2sa_topicAwareGeneration\\'
    config['validation_load']=datadir+'s2sa_topicAwareGeneration\\models_for_test\\'
    config['model_name'] = 's2sa_decoder_with_extra_class'
    # dictionary options
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # dictionaries
    config['src_data'] = datadir + 'train.query'
    config['trg_data'] = datadir + 'train.response'
    config['src_vocab'] = datadir + 'query.vocab'
    config['trg_vocab'] = datadir + 'topicSetting\outputTopicWord\\response.vocab.withTw'
    config['src_vocab_size'] = 40000
    config['trg_vocab_size'] = 40010
    # # validation options
    config['val_set_source'] = datadir + 'dev.query'
    config['val_set_target'] = datadir + 'dev.response'
    # config['bleu_script'] = datadir + 'multi-bleu.perl'
    # config['bleu_script_1'] = datadir + 'CompBleu.exe'
    # config['normalized_bleu'] = True
    # testing options
    config['normalized_bleu'] = True
    config['beam_size'] = 12
    # model saving related
    config['finish_after'] = 1000000
    config['reload'] = True
    config['save_freq'] = 500
    config['sampling_freq'] = 20
    config['hook_samples'] = 2
    config['bleu_val_freq'] = 1000
    config['val_burn_in'] = 20000

    # # afterprocess config
    # config['translation_table'] = datadir + 'translation_table'
    # config['replacement_table'] = datadir + 'test/' + 'replacement_table'

    config['use_doubly_stochastic'] = False
    config['lambda_ds'] = 0.001

    config['use_local_attention'] = False
    config['window_size'] = 10

    config['use_step_decay_cost'] = False

    config['use_concentration_cost'] = False
    config['lambda_ct'] = 10

    # arxiv.org/abs/1511.08400
    config['use_stablilizer'] = False
    config['lambda_st'] = 1

    config['match_function'] = 'SumMatchFunction'
    config['attention_images'] = config['saveto'] + '/attention_images/'
    config['attention_weights'] = config['saveto'] + '/attention_weights'
    config['val_output_orig'] = config['validation_load'] + '/test_output_orig'
    config['val_output_repl'] = config['saveto'] + '/test_output_repl'

    config['trg_topic_vocab_size']=3064#adding the unk and EOS
    config['topic_vocab_output']=datadir + 'topicSetting\outputTopicWord\\tVocab.withSpecialTok.highFreqRemoved.pkl';
    config['tw_vocab_overlap']=datadir+'topicSetting\outputTopicWord\\decoderW.pkl'

    #topical related features--------------------------------------------------------
    config['source_topic_vocab_size']=2990
    config['topical_embedding_dim']=200
    config['topical_word_num']=20
    config['topical_embeddings']=datadir + 'topicSetting\inputTopicWord\\inputW.pkl';
    config['topic_vocab_input']=datadir + 'topicSetting\inputTopicWord\\inputTopicVocab.pkl'
    config['topical_data']= datadir + 'train.query.topic.input'
    config['topical_dev_set']= datadir+'dev.query.topic.input'
    #config['topical_test_set']= datadir+'test.query.topic.input'



    return config
