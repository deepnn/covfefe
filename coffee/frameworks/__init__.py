from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
from .common import epsilon
from .common import floatx
from .common import set_epsilon
from .common import set_floatx
from .common import get_uid
from .common import cast_to_floatx
from .common import image_dim_ordering
from .common import set_image_dim_ordering
from .common import is_coffee_tensor

frameworks = {'theano', 'tensorflow', 'caffe', 'torch'}
frameworks_dim_ordering = {'ch_first', 'ch_last'}
float_types = {'float16', 'float32', 'float64'}

_coffee_base_dir = os.path.expanduser('~')
if not os.access(_coffee_base_dir, os.W_OK):
    _coffee_base_dir = '/tmp'

_coffee_dir = os.path.join(_coffee_base_dir, '.coffee')
if not os.path.exists(_coffee_dir):
    os.makedirs(_coffee_dir)


_FRAMEWORK = 'torch'

_config_path = os.path.expanduser(os.path.join(_coffee_dir, 'coffee.json'))
if os.path.exists(_config_path):
    _config = json.load(open(_config_path))
    _floatx = _config.get('floatx', floatx())
    assert _floatx in float_types
    _epsilon = _config.get('epsilon', epsilon())
    assert type(_epsilon) == float
    _framework = _config.get('framework', _FRAMEWORK)
    assert _framework in frameworks
    _image_dim_ordering = _config.get('image_dim_ordering', image_dim_ordering())
    assert _image_dim_ordering in frameworks_dim_ordering

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_image_dim_ordering(_image_dim_ordering)
    _FRAMEWORK = _framework

# save config file
if not os.path.exists(_config_path):
    _config = {'floatx': floatx(),
               'epsilon': epsilon(),
               'framework': _FRAMEWORK,
               'image_dim_ordering': image_dim_ordering()}
    with open(_config_path, 'w') as f:
        f.write(json.dumps(_config, indent=4))

if 'COFFEE_FRAMEWORK' in os.environ:
    _framework = os.environ['COFFEE_FRAMEWORK']
    assert _framework in frameworks
    _FRAMEWORK = _framework

# import framework
if _FRAMEWORK == 'theano':
    sys.stderr.write('Using Theano framework.\n')
    from .theano import *
elif _FRAMEWORK == 'tensorflow':
    sys.stderr.write('Using TensorFlow framework.\n')
    from .tf import *
elif _FRAMEWORK == 'caffe':
    sys.stderr.write('Using Caffe framework.\n')
    from .caffe import *
elif _FRAMEWORK == 'torch':
    sys.stderr.write('Using torch framework.\n')
    from .torch import *
else:
    raise Exception('Unknown framework: ' + str(_FRAMEWORK))


def framework():
    '''Publicly accessible method
    for determining the current framework.
    '''
    return _FRAMEWORK
