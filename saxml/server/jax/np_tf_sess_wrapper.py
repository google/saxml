"""np_tf_sess_wrapper with more efficient PyTree."""

from jax import tree_util
from saxml.server.tf import np_tf_sess_wrapper

# nested -> sequence.
np_tf_sess_wrapper.tree_flatten = tree_util.tree_flatten
# tree, sequence -> nested.
np_tf_sess_wrapper.tree_unflatten = tree_util.tree_unflatten
# function, tree -> tree.
np_tf_sess_wrapper.tree_map = tree_util.tree_map

wrap_tf_session = np_tf_sess_wrapper.wrap_tf_session
wrap_tf_session_class_member = np_tf_sess_wrapper.wrap_tf_session_class_member
