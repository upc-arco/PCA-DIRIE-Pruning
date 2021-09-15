# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers are the fundamental building blocks for NLP models.

They can be used to assemble new `tf.keras` layers or models.
"""
# pylint: disable=wildcard-import
from nlp.modeling.layers.attention import *
from nlp.modeling.layers.cls_head import *
from nlp.modeling.layers.dense_einsum import DenseEinsum
from nlp.modeling.layers.gated_feedforward import GatedFeedforward
from nlp.modeling.layers.gaussian_process import RandomFeatureGaussianProcess
from nlp.modeling.layers.masked_lm import MaskedLM
from nlp.modeling.layers.masked_softmax import MaskedSoftmax
from nlp.modeling.layers.mat_mul_with_margin import MatMulWithMargin
from nlp.modeling.layers.mobile_bert_layers import MobileBertEmbedding
from nlp.modeling.layers.mobile_bert_layers import MobileBertMaskedLM
from nlp.modeling.layers.mobile_bert_layers import MobileBertTransformer
from nlp.modeling.layers.multi_channel_attention import *
from nlp.modeling.layers.on_device_embedding import OnDeviceEmbedding
from nlp.modeling.layers.position_embedding import RelativePositionBias
from nlp.modeling.layers.position_embedding import RelativePositionEmbedding
from nlp.modeling.layers.relative_attention import MultiHeadRelativeAttention
from nlp.modeling.layers.relative_attention import TwoStreamRelativeAttention
from nlp.modeling.layers.rezero_transformer import ReZeroTransformer
from nlp.modeling.layers.self_attention_mask import SelfAttentionMask
from nlp.modeling.layers.spectral_normalization import *
from nlp.modeling.layers.talking_heads_attention import TalkingHeadsAttention
from nlp.modeling.layers.text_layers import BertPackInputs
from nlp.modeling.layers.text_layers import BertTokenizer
from nlp.modeling.layers.text_layers import SentencepieceTokenizer
from nlp.modeling.layers.tn_transformer_expand_condense import TNTransformerExpandCondense
from nlp.modeling.layers.transformer import *
from nlp.modeling.layers.transformer_scaffold import TransformerScaffold
from nlp.modeling.layers.transformer_xl import TransformerXL
from nlp.modeling.layers.transformer_xl import TransformerXLBlock
