from configurations_base import default

def wsj():
    config = default()
    config['model_name'] = 'wsj'
    config['saveto'] = 'models/wsj'
    config['step_rule'] = 'AdaGrad'
    config['match_function'] = 'SumMatchFunction'
    config['attention_images'] = config['saveto'] + '/attention_images/'
    config['attention_weights'] = config['saveto'] + '/attention_weights'
    config['val_output_orig'] = config['saveto'] + '/test_output_orig'
    config['val_output_repl'] = config['saveto'] + '/test_output_repl'
    return config
