# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modified by Jiyun Zhou
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import sys
sys.path.append("..")
from models.modeling_utils import ProteinConfig
from models.modeling_utils import ProteinModel
from models.modeling_utils import prune_linear_layer
from models.modeling_utils import get_activation_fn
from models.modeling_utils import LayerNorm
from models.modeling_utils import MLMHead
from models.modeling_utils import ValuePredictionHead
from models.modeling_utils import MultiValuePredictionHead
from models.modeling_utils import MethylValuePredictionHead
from models.modeling_utils import SequenceValuePredictionHead
from models.modeling_utils import SimpleMLP
import numpy as np
import torch.nn.functional as F
from registry import registry

logger = logging.getLogger(__name__)

URL_PREFIX = "https://s3.amazonaws.com/proteindata/pytorch-models/"
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base': URL_PREFIX + "bert-base-pytorch_model.bin",
}
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base': URL_PREFIX + "bert-base-config.json"
}


class ProteinBertConfig(ProteinConfig):
    r"""
        :class:`~pytorch_transformers.ProteinBertConfig` is the configuration class to store the
        configuration of a `ProteinBertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `ProteinBertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the ProteinBert encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the ProteinBert encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the ProteinBert encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `ProteinBertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size: int = 30,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 8096,
                 type_vocab_size: int = 2,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class ProteinBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        #self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.word_embeddings = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be
        # able to load any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, methylation_data, token_type_ids=None, position_ids=None):
        seq_length = methylation_data.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=methylation_data.device)
            position_ids = position_ids.unsqueeze(0).expand_as(methylation_data[:,:,0])
            #position_ids = position_ids.unsqueeze(0).expand_as(methylation_data)
        
        #words_embeddings = self.word_embeddings(methylation_data)
        words_embeddings = methylation_data
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ProteinBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in
        # ProteinBertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original ProteinBert paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) \
            if self.output_attentions else (context_layer,)
        return outputs


class ProteinBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ProteinBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = ProteinBertSelfAttention(config)
        self.output = ProteinBertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask):
        self_outputs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ProteinBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation_fn(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ProteinBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ProteinBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ProteinBertAttention(config)
        self.intermediate = ProteinBertIntermediate(config)
        self.output = ProteinBertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class ProteinBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [ProteinBertLayer(config) for _ in range(config.num_hidden_layers)])

    def run_function(self, start, chunk_size):
        def custom_forward(hidden_states, attention_mask):
            all_hidden_states = ()
            all_attentions = ()
            chunk_slice = slice(start, start + chunk_size)
            for layer in self.layer[chunk_slice]:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_outputs = layer(hidden_states, attention_mask)
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = (hidden_states,)
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs

        return custom_forward

    def forward(self, hidden_states, attention_mask, chunks=None):
        all_hidden_states = ()
        all_attentions = ()

        if chunks is not None:
            assert isinstance(chunks, int)
            chunk_size = (len(self.layer) + chunks - 1) // chunks
            for start in range(0, len(self.layer), chunk_size):
                outputs = checkpoint(self.run_function(start, chunk_size),
                                     hidden_states, attention_mask)
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + outputs[1]
                if self.output_attentions:
                    all_attentions = all_attentions + outputs[-1]
                hidden_states = outputs[0]
        else:
            for i, layer_module in enumerate(self.layer):
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(hidden_states, attention_mask)
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            # Add last layer
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = (hidden_states,)
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class ProteinBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProteinBertAbstractModel(ProteinModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = ProteinBertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

#CNN model for DNA sequence
class DNA_CNN(nn.Module):
    def __init__(self, num_kernel):
        super(DNA_CNN,self).__init__()
        self.num_kernel = num_kernel
        self.one_hot_embedding = nn.Embedding(5,4,padding_idx=0)
        self.one_hot_embedding.weight.data = torch.from_numpy(np.array([[0.,0.,0.,0.],
                                                                        [1.,0.,0.,0.],
                                                                        [0.,1.,0.,0.],
                                                                        [0.,0.,1.,0.],
                                                                        [0.,0.,0.,1.]])).type(torch.FloatTensor)
        self.one_hot_embedding.weight.requires_grad = False
        self.conv1 = nn.Conv1d(4,self.num_kernel,10,padding=4)
        self.conv2 = nn.Conv1d(self.num_kernel,self.num_kernel,10,padding=4)
        self.conv3 = nn.Conv1d(self.num_kernel,self.num_kernel,10,padding=5)
        self.batch = nn.BatchNorm1d(self.num_kernel)
        self.layer_batch1 = nn.LayerNorm((self.num_kernel, 2000))
        self.layer_batch2 = nn.LayerNorm((self.num_kernel, 1999))
        self.layer_batch3 = nn.LayerNorm((self.num_kernel, 2000))
        self.dropout = nn.Dropout(p=0.2)
        self.pool = nn.MaxPool1d(20, 20)
        self.maxpool = nn.MaxPool1d(2000, 2000)

    def forward(self,sequence):
        sequence = self.one_hot_embedding(sequence)
        sequence = torch.transpose(sequence,1,2)
        sequence = F.relu(self.conv1(sequence))
        sequence = self.layer_batch1(sequence)
        sequence = F.relu(self.conv2(sequence))
        sequence = self.layer_batch2(sequence)
        sequence = F.relu(self.conv3(sequence))
        sequence = self.layer_batch3(sequence)
        
        feature = self.maxpool(sequence)
        feature = feature.view(-1,self.num_kernel)
        feature = self.dropout(feature)       
 
        sequence = self.pool(sequence)
        sequence = self.batch(sequence)
        sequence = self.dropout(sequence)
        return sequence, feature


#CNN model for CpG sequence
class METH_CNN(nn.Module):
    def __init__(self, num_kernel):
        super(METH_CNN,self).__init__()
        self.num_kernel = num_kernel
        self.conv1 = nn.Conv2d(1, self.num_kernel, (11, 1), stride=(1, 1), padding=(5, 0))
        self.conv2 = nn.Conv2d(self.num_kernel, self.num_kernel, (11, 1), stride=(1, 1), padding=(5, 0))
        self.conv3 = nn.Conv2d(self.num_kernel, self.num_kernel, (11, 1), stride=(1, 1), padding=(5, 0))
        self.batch = nn.BatchNorm2d(self.num_kernel)
        self.dropout = nn.Dropout(p=0.2)
        self.pool1 = nn.MaxPool2d((2, 1), stride=(2, 1))
        self.pool2 = nn.MaxPool2d((2, 1), stride=(2, 1))
        self.maxpool = nn.MaxPool2d((50, 1), stride=(50, 1))#5 5 12

    def forward(self,sequence):
        task_size = sequence.size(dim=2)
        sequence = torch.unsqueeze(sequence, -1)
        sequence = torch.transpose(sequence, 2, 3)
        sequence = torch.transpose(sequence, 1, 2)
        sequence = F.relu(self.conv1(sequence))
        sequence = self.pool1(sequence)
        sequence = F.relu(self.conv2(sequence))
        sequence = self.pool2(sequence)
        sequence = F.relu(self.conv3(sequence))

        sequence = self.batch(sequence)
        feature = self.maxpool(sequence)
        feature = feature.contiguous().view(-1, task_size, self.num_kernel)
        feature = self.dropout(feature)

        sequence = torch.squeeze(sequence, -1)
        sequence = torch.transpose(sequence, 1, 2)
        sequence = torch.transpose(sequence, 2, 3)
        sequence = self.dropout(sequence)
        return sequence, feature


class Methyl_CNN(nn.Module):
    def __init__(self, num_kernel, num_features):
        super(Methyl_CNN,self).__init__()
        self.num_kernel = num_kernel
        self.conv1 = nn.Conv1d(num_features, self.num_kernel, 10, padding=4)
        self.conv2 = nn.Conv1d(self.num_kernel, self.num_kernel, 10, padding=4)
        self.conv3 = nn.Conv1d(self.num_kernel, self.num_kernel, 10, padding=5)
        self.batch = nn.BatchNorm1d(self.num_kernel)
        self.layer_batch1 = nn.LayerNorm((self.num_kernel, 1999))
        self.layer_batch2 = nn.LayerNorm((self.num_kernel, 1998))
        self.layer_batch3 = nn.LayerNorm((self.num_kernel, 1999))
        self.dropout = nn.Dropout(p=0.2)
        self.maxpool = nn.MaxPool1d(1999, 1999)

    def forward(self,sequence):
        sequence = torch.transpose(sequence,1,2)
        sequence = F.relu(self.conv1(sequence))
        sequence = self.layer_batch1(sequence)
        sequence = F.relu(self.conv2(sequence))
        sequence = self.layer_batch2(sequence)
        sequence = F.relu(self.conv3(sequence))
        sequence = self.layer_batch3(sequence)

        feature = self.maxpool(sequence)
        feature = feature.view(-1,self.num_kernel)
        feature = self.dropout(feature)
        return feature


@registry.register_task_model('embed', 'transformer')
class ProteinBertModel(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)
        
        #self.embeddings = nn.Linear(config.input_size, config.hidden_size, bias=False)
        self.embeddings = ProteinBertEmbeddings(config)
        self.encoder = ProteinBertEncoder(config)
        self.pooler = ProteinBertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class ProteinModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self,
                methylation_data,
                input_mask=None):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)

        # Since input_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(methylation_data)
        #embedding_output = input_ids #.unsqueeze(-1)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       chunks=None)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


"""Builds a model that incorporates both the DNA and CpG modules for prediction"""
@registry.register_task_model('single_cell_prediction', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.feature_cnn = METH_CNN(config.hidden_size)
        self.feature_bert = ProteinBertModel(config)

        self.bert = ProteinBertModel(config)
        self.cnn = DNA_CNN(config.hidden_size)

        self.mlm = MultiValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()

    def forward(self,DNA_data,
                feature_data=None,
                input_mask=None,
                targets=None,
                high_ids=None,
                low_ids=None):

        sequence_outputs, feature_outputs = self.feature_cnn(feature_data)
        feature_outputs = torch.mean(sequence_outputs, dim=2)
        input_mask = torch.from_numpy(np.ones((feature_outputs.size()[0],feature_outputs.size()[1]))).cuda()
        outputs = self.feature_bert(feature_outputs, input_mask=input_mask)
        feature_output, pooled_feature = outputs[:2]

        DNA_output, DNA_feature = self.cnn(DNA_data)
        DNA_output = torch.transpose(DNA_output,1,2)
        input_mask = torch.from_numpy(np.ones((DNA_output.size()[0],DNA_output.size()[1]))).cuda()
        outputs = self.bert(DNA_output, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]

        # add hidden states and attention if they are here
        outputs = self.mlm(pooled_output, feature_output=pooled_feature) + outputs[2:]
        return outputs


"""create and train a model that contains both the DNA module and CpG module"""
@registry.register_task_model('single_cell_regression', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        #self.feature_cnn = nn.Conv1d(config.num_features, config.hidden_size, 1) #13 21 64
        self.feature_cnn = METH_CNN(config.hidden_size)
        self.feature_bert = ProteinBertModel(config)

        self.bert = ProteinBertModel(config)
        self.cnn = DNA_CNN(config.hidden_size)

        self.mlm = MultiValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)
        self.init_weights()

    def forward(self,DNA_data,
                methyl_data=None,
                input_mask=None,
                targets=None,
                high_ids=None,
                low_ids=None,
                weights=None):

        #feature_data = torch.transpose(feature_data, 1, 2)
        sequence_outputs, feature_outputs = self.feature_cnn(methyl_data)
        feature_outputs = torch.mean(sequence_outputs, dim=2)
        #pooled_feature = torch.mean(feature_outputs, dim=1)
        #feature_outputs = torch.transpose(feature_outputs, 1, 2)
        input_mask = torch.from_numpy(np.ones((feature_outputs.size()[0], feature_outputs.size()[1]))).cuda()
        outputs = self.feature_bert(feature_outputs, input_mask=input_mask)
        feature_output, pooled_feature = outputs[:2]

        DNA_output, DNA_feature = self.cnn(DNA_data)
        DNA_output = torch.transpose(DNA_output,1,2)
        input_mask = torch.from_numpy(np.ones((DNA_output.size()[0],DNA_output.size()[1]))).cuda()
        outputs = self.bert(DNA_output, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]

        # add hidden states and attention if they are here
        outputs = self.mlm(pooled_output,feature_output=pooled_feature,high_ids=high_ids,low_ids=low_ids) + outputs[2:]
        return outputs


"""creates and trains a model that incorporates both the DNA and CpG modules for mouse embryo"""
@registry.register_task_model('single_mouse_regression', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)
        self.feature_cnn = METH_CNN(config.hidden_size)
        self.feature_bert = ProteinBertModel(config)

        self.bert = ProteinBertModel(config)
        self.cnn = DNA_CNN(config.hidden_size)

        self.mlm = MultiValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)
        self.init_weights()

    def forward(self,DNA_data,
                methyl_data=None,
                input_mask=None,
                targets=None,
                high_ids=None,
                low_ids=None):

        sequence_outputs, feature_outputs = self.feature_cnn(methyl_data)
        feature_outputs = torch.mean(sequence_outputs, dim=2)
        input_mask = torch.from_numpy(np.ones((feature_outputs.size()[0], feature_outputs.size()[1]))).cuda()
        outputs = self.feature_bert(feature_outputs, input_mask=input_mask)
        feature_output, pooled_feature = outputs[:2]

        DNA_output, DNA_feature = self.cnn(DNA_data)
        DNA_output = torch.transpose(DNA_output,1,2)
        input_mask = torch.from_numpy(np.ones((DNA_output.size()[0],DNA_output.size()[1]))).cuda()
        outputs = self.bert(DNA_output, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]

        # add hidden states and attention if they are here
        # feature_output=pooled_feature
        outputs = self.mlm(pooled_output,feature_output=pooled_feature,high_ids=high_ids,low_ids=low_ids) + outputs[2:]
        return outputs


"""Creates a model that incorporates both the DNA and CpG modules for mouse embryo prediction"""
@registry.register_task_model('single_mouse_prediction', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.feature_cnn = METH_CNN(config.hidden_size)
        self.feature_bert = ProteinBertModel(config)

        self.bert = ProteinBertModel(config)
        self.cnn = DNA_CNN(config.hidden_size)

        self.mlm = MultiValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()

    def forward(self,DNA_data,
                feature_data=None,
                input_mask=None,
                targets=None,
                high_ids=None,
                low_ids=None):

        sequence_outputs, feature_outputs = self.feature_cnn(feature_data)
        feature_outputs = torch.mean(sequence_outputs, dim=2)
        input_mask = torch.from_numpy(np.ones((feature_outputs.size()[0],feature_outputs.size()[1]))).cuda()
        outputs = self.feature_bert(feature_outputs, input_mask=input_mask)
        feature_output, pooled_feature = outputs[:2]

        DNA_output, DNA_feature = self.cnn(DNA_data)
        DNA_output = torch.transpose(DNA_output,1,2)
        input_mask = torch.from_numpy(np.ones((DNA_output.size()[0],DNA_output.size()[1]))).cuda()
        outputs = self.bert(DNA_output, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]

        # add hidden states and attention if they are here
        #feature_output=pooled_feature
        outputs = self.mlm(pooled_output, feature_output=pooled_feature) + outputs[2:]
        return outputs


"""create and train a model using only the CpG module"""
@registry.register_task_model('single_methylation_regression', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.feature_cnn = METH_CNN(config.hidden_size)
        self.feature_bert = ProteinBertModel(config)

        self.mlm = MethylValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)
        self.init_weights()

    def forward(self,methyl_data=None,
                input_mask=None,
                targets=None,
                high_ids=None,
                low_ids=None,
                weights=None):

        sequence_outputs, feature_outputs = self.feature_cnn(methyl_data)
        feature_outputs = torch.mean(sequence_outputs, dim=2)
        input_mask = torch.from_numpy(np.ones((feature_outputs.size()[0], feature_outputs.size()[1]))).cuda()
        outputs = self.feature_bert(feature_outputs, input_mask=input_mask)
        feature_output, pooled_feature = outputs[:2]

        # add hidden states and attention if they are here
        outputs = self.mlm(pooled_feature, high_ids=high_ids, low_ids=low_ids) + outputs[2:]
        return outputs


"""create a model using only the CpG module for prediction"""
@registry.register_task_model('single_methylation_prediction', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.feature_cnn = METH_CNN(config.hidden_size)
        self.feature_bert = ProteinBertModel(config)

        self.mlm = MethylValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()

    def forward(self,feature_data=None,
                input_mask=None,
                targets=None,
                high_ids=None,
                low_ids=None):

        sequence_outputs, feature_outputs = self.feature_cnn(feature_data)
        feature_outputs = torch.mean(sequence_outputs, dim=2)
        input_mask = torch.from_numpy(np.ones((feature_outputs.size()[0],feature_outputs.size()[1]))).cuda()
        outputs = self.feature_bert(feature_outputs, input_mask=input_mask)
        feature_output, pooled_feature = outputs[:2]

        # add hidden states and attention if they are here
        outputs = self.mlm(pooled_feature) + outputs[2:]
        return outputs


"""create and train a model using only the DNA module"""
@registry.register_task_model('single_sequence_regression', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)
        self.cnn = DNA_CNN(config.hidden_size)
        self.mlm = SequenceValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()

    def forward(self,DNA_data,
                feature_data=None,
                input_mask=None,
                targets=None,
                high_ids=None,
                low_ids=None):

        DNA_output, DNA_feature = self.cnn(DNA_data)
        DNA_output = torch.transpose(DNA_output,1,2)
        input_mask = torch.from_numpy(np.ones((DNA_output.size()[0],DNA_output.size()[1]))).cuda()
        outputs = self.bert(DNA_output, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]

        # add hidden states and attention if they are here
        outputs = self.mlm(pooled_output, high_ids=high_ids, low_ids=low_ids) + outputs[2:] #feature_output=pooled_feature
        return outputs


"""create a model using only the DNA module for prediction"""
@registry.register_task_model('single_sequence_prediction', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)
        self.cnn = DNA_CNN(config.hidden_size)
        self.mlm = SequenceValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()

    def forward(self,DNA_data,
                feature_data=None,
                input_mask=None,
                targets=None):

        DNA_output, DNA_feature = self.cnn(DNA_data)
        DNA_output = torch.transpose(DNA_output,1,2)
        input_mask = torch.from_numpy(np.ones((DNA_output.size()[0],DNA_output.size()[1]))).cuda()
        outputs = self.bert(DNA_output, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]

        # add hidden states and attention if they are here
        outputs = self.mlm(pooled_output) + outputs[2:] #feature_output=pooled_feature
        return outputs


"""create and train a CNN model that incorporates both the DNA and CpG modules"""
@registry.register_task_model('single_cnn_regression', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.feature_cnn = nn.Conv1d(config.num_features, config.hidden_size, 9, padding=4) #13 21 64
        self.feature_pool = nn.MaxPool1d(100, 100)

        self.cnn = DNA_CNN(config.hidden_size)
        self.mlm = MultiValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()

    def forward(self,DNA_data,
                feature_data=None,
                input_mask=None,
                targets=None,
                high_ids=None,
                low_ids=None):

        feature_data = torch.transpose(feature_data, 1, 2)
        feature_outputs = self.feature_cnn(feature_data)
        feature_output = self.feature_pool(feature_outputs)
        feature_output = feature_output.view(-1, 512)

        DNA_output, DNA_feature = self.cnn(DNA_data)
        # add hidden states and attention if they are here
        outputs = self.mlm(DNA_feature, feature_output=feature_output, high_ids=high_ids, low_ids=low_ids)#
        return outputs


"""create a CNN model that incorporates both the DNA and CpG modules for prediction"""
@registry.register_task_model('single_cnn_prediction', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.feature_cnn = nn.Conv1d(config.num_features, config.hidden_size, 9, padding=4) #13 21 64
        self.feature_pool = nn.MaxPool1d(100, 100)

        self.cnn = DNA_CNN(config.hidden_size)
        self.mlm = MultiValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()

    def forward(self,DNA_data,
                feature_data=None,
                input_mask=None,
                targets=None):

        feature_data = torch.transpose(feature_data, 1, 2)
        feature_outputs = self.feature_cnn(feature_data)
        feature_output = self.feature_pool(feature_outputs)
        feature_output = feature_output.view(-1, 512)

        DNA_output, DNA_feature = self.cnn(DNA_data)
        # add hidden states and attention if they are here
        outputs = self.mlm(DNA_feature, feature_output=feature_output)#
        return outputs


@registry.register_task_model('cell_variant_prediction', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)
        self.cnn = DNA_CNN(config.hidden_size)
        self.mlm = SequenceValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()

    def forward(self,DNA_data,
                input_mask=None,
                targets=None):
        DNA_output, DNA_feature = self.cnn(DNA_data)
        DNA_output = torch.transpose(DNA_output,1,2)
        input_mask = torch.from_numpy(np.ones((DNA_output.size()[0], DNA_output.size()[1]))).cuda()
        outputs = self.bert(DNA_output, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]

        # add hidden states and attention if they are here
        outputs = self.mlm(pooled_output) + outputs[2:]
        return outputs


@registry.register_task_model('DNA_motif_discovery', 'transformer')
class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)
        self.cnn = DNA_CNN()
        self.mlm = ValuePredictionHead(
            config.hidden_size, config.task_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()

    def forward(self,DNA_data,
                input_mask=None,
                targets=None):
        DNA_output, DNA_motif = self.cnn(DNA_data)
        return DNA_motif

