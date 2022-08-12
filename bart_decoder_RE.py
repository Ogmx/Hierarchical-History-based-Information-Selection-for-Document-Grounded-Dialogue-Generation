import os
import math
import random
import warnings
import torch
from torch.autograd import Variable
import pdb
from typing import Dict, List, Optional, Tuple

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import BartTokenizer, BartForSequenceClassification
from transformers.modeling_utils import PreTrainedModel, Conv1D
from transformers import BartConfig 
from transformers.activations import ACT2FN
from transformers.modeling_bart import (
    Attention,
    LayerNorm,
    invert_mask,
    BartEncoder,
    PretrainedBartModel,
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    BartForConditionalGeneration,
    BartModel,
    shift_tokens_right,
    BART_INPUTS_DOCSTRING,
    BART_GENERATION_EXAMPLE,
    _prepare_bart_decoder_inputs, _reorder_buffer, _make_linear_from_emb, BartClassificationHead
)
from transformers.modeling_bert import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)

_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"


class MHAAttention(Attention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
        attention_type=''
    ):
        super().__init__(embed_dim, num_heads)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if self.encoder_decoder_attention:
            if attention_type == 'doc':
                self.cache_key = "encoder_decoder_doc"
            elif attention_type == 'z':
                self.cache_key = "encoder_decoder_z"
            else:
                self.cache_key = "encoder_decoder"
        else:
            self.cache_key = "self"

        #self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

class PSA_Attention(Attention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__(embed_dim, num_heads)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def fill_with_neg_inf(self, t):
        """FP16-compatible function that fills a input_ids with -inf."""
        return t.float().fill_(float("-inf")).type_as(t)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state and static_kv:
                # previous time steps are cached - no need to recompute key and value if they are static
                key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k_z = k = v_z = v = None
            else:
                k = self.k_proj(key[0])
                v = self.v_proj(key[0])
                k_z = self.k_proj(key[1])
                v_z = self.v_proj(key[1])
        else:
            k = self.k_proj(key[0])
            v = self.v_proj(key[0])
            k_z = self.k_proj(key[1])
            v_z = self.v_proj(key[1])

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
            k_z = self._shape(k_z, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)
            v_z = self._shape(v_z, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }
        # PSA z
        k = torch.cat([k_z, k], dim=1)
        v = torch.cat([v_z, v], dim=1)

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = torch.triu(self.fill_with_neg_inf(torch.zeros(tgt_len, src_len)), 1).to(
                dtype=torch.float32, device=k.device
            )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None:
            key_padding_mask = shift_tokens_right(torch.cat([key_padding_mask, torch.ones(bsz, 1).cuda()], dim=-1), 1).bool()
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (
            bsz,
            src_len,
        )
        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = key_padding_mask
        return k, v, new_key_padding_mask

class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = gelu

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = attention_mask * -10000.0
            scores = scores + attention_mask

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze(1)

        return representations, scores

class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout)
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, output_attentions=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask, output_attentions=output_attentions
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn_weights

class VAE_BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
                self.padding_idx,
                config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

        # average SA to get Z
        self.averageSelfAttention = AverageSelfAttention(embed_dim)

    def forward(
        self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False
    ):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            BaseModelOutput or Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (tuple(torch.FloatTensor)): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *output_hidden_states:* is True.
                - **all_attentions** (tuple(torch.FloatTensor)): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # get z by average SA
        representations, _ = self.averageSelfAttention(x, attention_mask.squeeze(1).squeeze(1))

        outputs = representations

        return outputs  # mean, logvar, last hidden state, (presents), (all hidden_states), (attentions)

class MHADecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.encoder_attn_doc = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
            attention_type='doc'
        )
        self.encoder_attn_doc_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None
    ):
        residual = x
        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Cross attention with doc
        residual = x
        assert self.encoder_attn_doc.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_doc_layer_norm(x)
        x, _ = self.encoder_attn_doc(
            query=x,
            key=encoder_hidden_states_doc,
            key_padding_mask=encoder_attn_mask_doc,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_doc_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding

class VAE_DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig, have_doc=True):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MHAAttention(
            self.embed_dim, 
            config.decoder_attention_heads, 
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        if have_doc:
            self.encoder_attn_doc = MHAAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                encoder_decoder_attention=True,
                attention_type='doc'
            )
            self.encoder_attn_doc_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None
    ):
        residual = x
        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Cross attention with doc
        if encoder_hidden_states_doc is not None:
            residual = x
            assert self.encoder_attn_doc.cache_key != self.self_attn.cache_key
            if self.normalize_before:
                x = self.encoder_attn_doc_layer_norm(x)
            x, _ = self.encoder_attn_doc(
                query=x,
                key=encoder_hidden_states_doc,
                key_padding_mask=encoder_attn_mask_doc,
                layer_state=layer_state,  # mutates layer state
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_doc_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding

class VAE_PSA_DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig, have_doc=True):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = PSA_Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        if have_doc:
            self.encoder_attn_doc = MHAAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                encoder_decoder_attention=True,
                attention_type='doc'
            )
            self.encoder_attn_doc_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None,
        encoder_hidden_states_z=None,
        encoder_attn_mask_z=None
    ):
        residual = x
        z = encoder_hidden_states_z
        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # PSA Self Attention
        x, self_attn_weights = self.self_attn(
            query=x,
            key=(x, z),
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Cross attention with doc
        if encoder_hidden_states_doc is not None:
            residual = x
            assert self.encoder_attn_doc.cache_key != self.self_attn.cache_key
            if self.normalize_before:
                x = self.encoder_attn_doc_layer_norm(x)
            x, _ = self.encoder_attn_doc(
                query=x,
                key=encoder_hidden_states_doc,
                key_padding_mask=encoder_attn_mask_doc,
                layer_state=layer_state,  # mutates layer state
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_doc_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding

class MHABartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        config.extra_pos_embeddings = 2
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [MHADecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_past_key_values (dict or None): dictionary used for storing state during generation
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `decoder_past_key_values` instead.",
                FutureWarning,
            )
            decoder_past_key_values = unused.pop("decoder_cached_states")

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # check attention mask doc and invert
        if encoder_attn_mask_doc is not None:
            encoder_attn_mask_doc = invert_mask(encoder_attn_mask_doc)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        encoder_hidden_states_doc = encoder_hidden_states_doc.transpose(0, 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_state = decoder_past_key_values[idx] if decoder_past_key_values is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
                encoder_hidden_states_doc=encoder_hidden_states_doc,
                encoder_attn_mask_doc=encoder_attn_mask_doc
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None
        '''
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        '''
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )

class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding

class E_VAEDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.encoder_attn_z = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
            attention_type='z'
        )
        self.encoder_attn_doc = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
            attention_type='doc'
        )
        self.encoder_attn_doc_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn_z_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None,
        encoder_hidden_states_z=None,
        encoder_attn_mask_z=None
    ):
        residual = x
        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Cross attention with doc
        residual = x
        assert self.encoder_attn_doc.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_doc_layer_norm(x)
        x, _ = self.encoder_attn_doc(
            query=x,
            key=encoder_hidden_states_doc,
            key_padding_mask=encoder_attn_mask_doc,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_doc_layer_norm(x)

        # Cross attention with z
        residual = x
        assert self.encoder_attn_z.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_z_layer_norm(x)
        x, _ = self.encoder_attn_z(
            query=x,
            key=encoder_hidden_states_z,
            key_padding_mask=encoder_attn_mask_z,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_z_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding

class E_VAEBartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens

        config.extra_pos_embeddings = 2
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [E_VAEDecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None,
        encoder_hidden_states_z=None,
        encoder_attn_mask_z=None,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_past_key_values (dict or None): dictionary used for storing state during generation
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `decoder_past_key_values` instead.",
                FutureWarning,
            )
            decoder_past_key_values = unused.pop("decoder_cached_states")

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # check attention mask doc and invert
        if encoder_attn_mask_doc is not None:
            encoder_attn_mask_doc = invert_mask(encoder_attn_mask_doc)

        # check attention mask doc and invert
        if encoder_attn_mask_z is not None:
            encoder_attn_mask_z = invert_mask(encoder_attn_mask_z)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        encoder_hidden_states_doc = encoder_hidden_states_doc.transpose(0, 1)
        encoder_hidden_states_z = encoder_hidden_states_z.transpose(0, 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_state = decoder_past_key_values[idx] if decoder_past_key_values is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
                encoder_hidden_states_doc=encoder_hidden_states_doc,
                encoder_attn_mask_doc=encoder_attn_mask_doc,
                encoder_hidden_states_z=encoder_hidden_states_z,
                encoder_attn_mask_z=encoder_attn_mask_z
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None
        '''
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        '''
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )

class VAE_BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding, add_z=False, psa_z=False, have_doc=True):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.add_z = add_z
        self.psa_z = psa_z
        config.extra_pos_embeddings = 2
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        if add_z:
            self.input_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        if psa_z:
            self.attn_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.layers = nn.ModuleList(
                [VAE_PSA_DecoderLayer(config, have_doc) for _ in range(config.decoder_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [VAE_DecoderLayer(config, have_doc) for _ in range(config.decoder_layers)]
            )
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None,
        encoder_hidden_states_z=None,
        encoder_attn_mask_z=None,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_past_key_values (dict or None): dictionary used for storing state during generation
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `decoder_past_key_values` instead.",
                FutureWarning,
            )
            decoder_past_key_values = unused.pop("decoder_cached_states")

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # check attention mask doc and invert
        if encoder_attn_mask_doc is not None:
            encoder_attn_mask_doc = invert_mask(encoder_attn_mask_doc)

        # check attention mask doc and invert
        if encoder_attn_mask_z is not None:
            encoder_attn_mask_z = invert_mask(encoder_attn_mask_z)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        z = encoder_hidden_states_z
        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        if self.add_z:
            assert (z is not None)
            input_proj = self.input_proj(z)
            x += input_proj

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        if encoder_hidden_states_doc is not None:
            encoder_hidden_states_doc = encoder_hidden_states_doc.transpose(0, 1)
        encoder_hidden_states_z = encoder_hidden_states_z.transpose(0, 1)

        # decoder layers
        if self.psa_z:
            assert (z is not None)
            z = self.attn_proj(encoder_hidden_states_z)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_state = decoder_past_key_values[idx] if decoder_past_key_values is not None else None

            if self.psa_z:
                x, layer_self_attn, layer_past = decoder_layer(
                    x,
                    encoder_hidden_states,
                    encoder_attn_mask=encoder_padding_mask,
                    decoder_padding_mask=decoder_padding_mask,
                    layer_state=layer_state,
                    causal_mask=decoder_causal_mask,
                    output_attentions=output_attentions,
                    encoder_hidden_states_doc=encoder_hidden_states_doc,
                    encoder_attn_mask_doc=encoder_attn_mask_doc,
                    encoder_hidden_states_z=z,
                    encoder_attn_mask_z=encoder_attn_mask_z
                )
            else:
                x, layer_self_attn, layer_past = decoder_layer(
                    x,
                    encoder_hidden_states,
                    encoder_attn_mask=encoder_padding_mask,
                    decoder_padding_mask=decoder_padding_mask,
                    layer_state=layer_state,
                    causal_mask=decoder_causal_mask,
                    output_attentions=output_attentions,
                    encoder_hidden_states_doc=encoder_hidden_states_doc,
                    encoder_attn_mask_doc=encoder_attn_mask_doc
                )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None
        '''
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        '''
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )

class E_add_BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.mlp = nn.Linear(768*2, 768)
        config.extra_pos_embeddings = 2
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [MHADecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None,
        encoder_hidden_states_z=None,
        encoder_attn_mask_z=None,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_past_key_values (dict or None): dictionary used for storing state during generation
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `decoder_past_key_values` instead.",
                FutureWarning,
            )
            decoder_past_key_values = unused.pop("decoder_cached_states")

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # check attention mask doc and invert
        if encoder_attn_mask_doc is not None:
            encoder_attn_mask_doc = invert_mask(encoder_attn_mask_doc)

        # check attention mask doc and invert
        if encoder_attn_mask_z is not None:
            encoder_attn_mask_z = invert_mask(encoder_attn_mask_z)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        z = encoder_hidden_states_z
        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x += z
        # if z.shape[1] == 1:
        #     x_len = x.shape[1]
        #     zz = z.repeat(1, x_len, 1)
        #     x = torch.cat([x, zz], dim=-1)
        #     x = self.mlp(x)

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        encoder_hidden_states_doc = encoder_hidden_states_doc.transpose(0, 1)
        encoder_hidden_states_z = encoder_hidden_states_z.transpose(0, 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_state = decoder_past_key_values[idx] if decoder_past_key_values is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
                encoder_hidden_states_doc=encoder_hidden_states_doc,
                encoder_attn_mask_doc=encoder_attn_mask_doc
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None
        '''
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        '''
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )

class R_add_HD_BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.mlp = nn.Linear(768*2, 768)
        config.extra_pos_embeddings = 2
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [MHADecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None,
        encoder_hidden_states_z=None,
        encoder_attn_mask_z=None,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_past_key_values (dict or None): dictionary used for storing state during generation
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `decoder_past_key_values` instead.",
                FutureWarning,
            )
            decoder_past_key_values = unused.pop("decoder_cached_states")

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # check attention mask doc and invert
        if encoder_attn_mask_doc is not None:
            encoder_attn_mask_doc = invert_mask(encoder_attn_mask_doc)

        # check attention mask doc and invert
        if encoder_attn_mask_z is not None:
            encoder_attn_mask_z = invert_mask(encoder_attn_mask_z)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        z = encoder_hidden_states_z
        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x += z

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        encoder_hidden_states_doc = encoder_hidden_states_doc.transpose(0, 1)
        encoder_hidden_states_z = encoder_hidden_states_z.transpose(0, 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_state = decoder_past_key_values[idx] if decoder_past_key_values is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
                encoder_hidden_states_doc=encoder_hidden_states_doc,
                encoder_attn_mask_doc=encoder_attn_mask_doc
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None
        '''
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        '''
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )

class R_add_BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.mlp = nn.Linear(768 * 2, 768)
        config.extra_pos_embeddings = 2
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [BartDecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        encoder_hidden_states_z=None,
        encoder_attn_mask_z=None,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_past_key_values (dict or None): dictionary used for storing state during generation
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `decoder_past_key_values` instead.",
                FutureWarning,
            )
            decoder_past_key_values = unused.pop("decoder_cached_states")

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # check attention mask doc and invert
        if encoder_attn_mask_z is not None:
            encoder_attn_mask_z = invert_mask(encoder_attn_mask_z)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()
        z = encoder_hidden_states_z
        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x += z

        # if z.shape[1] == 1:
        #     x_len = x.shape[1]
        #     zz = z.repeat(1, x_len, 1)
        #     x = torch.cat([x, zz], dim=-1)
        #     x = self.mlp(x)

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        encoder_hidden_states_doc = encoder_hidden_states_z.transpose(0, 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_state = decoder_past_key_values[idx] if decoder_past_key_values is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None
        '''
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        '''
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )

class E_mlp_BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.mlp = nn.Linear(768*2, 768)
        config.extra_pos_embeddings = 2
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [MHADecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None,
        encoder_hidden_states_z=None,
        encoder_attn_mask_z=None,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_past_key_values (dict or None): dictionary used for storing state during generation
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `decoder_past_key_values` instead.",
                FutureWarning,
            )
            decoder_past_key_values = unused.pop("decoder_cached_states")

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # check attention mask doc and invert
        if encoder_attn_mask_doc is not None:
            encoder_attn_mask_doc = invert_mask(encoder_attn_mask_doc)

        # check attention mask doc and invert
        if encoder_attn_mask_z is not None:
            encoder_attn_mask_z = invert_mask(encoder_attn_mask_z)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        z = encoder_hidden_states_z
        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions

        if z.shape[1] == 1:
            x_len = x.shape[1]
            zz = z.repeat(1, x_len, 1)
            x = torch.cat([x, zz], dim=-1)
            x = self.mlp(x)

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        encoder_hidden_states_doc = encoder_hidden_states_doc.transpose(0, 1)
        encoder_hidden_states_z = encoder_hidden_states_z.transpose(0, 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_state = decoder_past_key_values[idx] if decoder_past_key_values is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
                encoder_hidden_states_doc=encoder_hidden_states_doc,
                encoder_attn_mask_doc=encoder_attn_mask_doc
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None
        '''
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        '''
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )

class R_mlp_BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.mlp = nn.Linear(768 * 2, 768)
        config.extra_pos_embeddings = 2
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [BartDecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        encoder_hidden_states_z=None,
        encoder_attn_mask_z=None,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_past_key_values (dict or None): dictionary used for storing state during generation
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `decoder_past_key_values` instead.",
                FutureWarning,
            )
            decoder_past_key_values = unused.pop("decoder_cached_states")

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # check attention mask doc and invert
        if encoder_attn_mask_z is not None:
            encoder_attn_mask_z = invert_mask(encoder_attn_mask_z)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()
        z = encoder_hidden_states_z
        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions

        if z.shape[1] == 1:
            x_len = x.shape[1]
            zz = z.repeat(1, x_len, 1)
            x = torch.cat([x, zz], dim=-1)
            x = self.mlp(x)

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        encoder_hidden_states_doc = encoder_hidden_states_z.transpose(0, 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_state = decoder_past_key_values[idx] if decoder_past_key_values is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None
        '''
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        '''
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )

class E_MlpZ_CLS_BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        dim = 768
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.mlp = nn.Linear(dim*2, dim)
        config.extra_pos_embeddings = 2
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [MHADecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.classify_layers = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.Linear(dim * 2, dim),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def get_cos_loss(self, R_reps, E_reps, his_reps, nxt_reps):
        loss_R = F.cosine_embedding_loss(R_reps, his_reps, Tensor([1]))
        loss_E = F.cosine_embedding_loss(E_reps, nxt_reps, Tensor([1]))
        return loss_R + loss_E

    def get_sep_mask(self, tgt_reps, his_reps, nxt_reps):
        tgt_reps = tgt_reps.detach()
        his_reps = his_reps.detach()
        nxt_reps = nxt_reps.detach()
        inputs = torch.cat([his_reps, tgt_reps, nxt_reps], dim=-1)
        outputs = self.classify_layers(inputs)
        sep_mask = (outputs > 0.5).int()
        R_reps = torch.masked_fill(inputs, sep_mask, 0)
        E_reps = torch.masked_fill(inputs, ~sep_mask, 0)
        loss = self.get_cos_loss(R_reps, E_reps, his_reps, nxt_reps)
        return sep_mask, loss

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None,
        encoder_hidden_states_z=None,
        encoder_attn_mask_z=None,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_past_key_values (dict or None): dictionary used for storing state during generation
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `decoder_past_key_values` instead.",
                FutureWarning,
            )
            decoder_past_key_values = unused.pop("decoder_cached_states")

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # check attention mask doc and invert
        if encoder_attn_mask_doc is not None:
            encoder_attn_mask_doc = invert_mask(encoder_attn_mask_doc)

        # check attention mask doc and invert
        if encoder_attn_mask_z is not None:
            encoder_attn_mask_z = invert_mask(encoder_attn_mask_z)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        z = encoder_hidden_states_z
        if z.shape[1] == 1:
            x_len = x.shape[1]
            zz = z.repeat(1, x_len, 1)
            x = torch.cat([x, zz], dim=-1)
            x = self.mlp(x)

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        encoder_hidden_states_doc = encoder_hidden_states_doc.transpose(0, 1)
        encoder_hidden_states_z = encoder_hidden_states_z.transpose(0, 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_state = decoder_past_key_values[idx] if decoder_past_key_values is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
                encoder_hidden_states_doc=encoder_hidden_states_doc,
                encoder_attn_mask_doc=encoder_attn_mask_doc
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None
        '''
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        '''
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )

class E_CatZ_CLS_BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        dim = 768
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens

        config.extra_pos_embeddings = 2
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [E_VAEDecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]

        self.classify_layers = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.Linear(dim * 2, dim),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def get_cos_loss(self, R_reps, E_reps, his_reps, nxt_reps):
        loss_R = F.cosine_embedding_loss(R_reps, his_reps, Tensor([1]).cuda())
        loss_E = F.cosine_embedding_loss(E_reps, nxt_reps, Tensor([1]).cuda())
        return loss_R + loss_E

    def get_sep_mask(self, tgt_reps, his_reps, nxt_reps):
        tgt_reps = tgt_reps.detach()
        his_reps = his_reps.detach()[:, :tgt_reps.shape[1]]
        nxt_reps = nxt_reps.detach()[:, :tgt_reps.shape[1]]
        inputs = torch.cat([his_reps, tgt_reps, nxt_reps], dim=-1)
        outputs = self.classify_layers(inputs)

        sep_mask = (((outputs - 0.5).sgn() + 1) * 0.5)
        sep_mask = sep_mask - outputs.detach() + outputs
        R_reps = (tgt_reps.clone() * sep_mask)
        E_reps = (tgt_reps.clone() * ((sep_mask*-1)+1))
        loss = self.get_cos_loss(R_reps, E_reps, his_reps, nxt_reps)
        return sep_mask.int(), loss

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        encoder_hidden_states_doc=None,
        encoder_attn_mask_doc=None,
        encoder_hidden_states_z=None,
        encoder_attn_mask_z=None,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_past_key_values (dict or None): dictionary used for storing state during generation
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `decoder_past_key_values` instead.",
                FutureWarning,
            )
            decoder_past_key_values = unused.pop("decoder_cached_states")

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # check attention mask doc and invert
        if encoder_attn_mask_doc is not None:
            encoder_attn_mask_doc = invert_mask(encoder_attn_mask_doc)

        # check attention mask doc and invert
        if encoder_attn_mask_z is not None:
            encoder_attn_mask_z = invert_mask(encoder_attn_mask_z)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        sep_mask, sep_loss = self.get_sep_mask(x, encoder_hidden_states, encoder_hidden_states_z)
        sep_positions = self.embed_positions(sep_mask, use_cache=use_cache)
        x += sep_positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        encoder_hidden_states_doc = encoder_hidden_states_doc.transpose(0, 1)
        encoder_hidden_states_z = encoder_hidden_states_z.transpose(0, 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_state = decoder_past_key_values[idx] if decoder_past_key_values is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
                encoder_hidden_states_doc=encoder_hidden_states_doc,
                encoder_attn_mask_doc=encoder_attn_mask_doc,
                encoder_hidden_states_z=encoder_hidden_states_z,
                encoder_attn_mask_z=encoder_attn_mask_z
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None
        '''
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        '''
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns, sep_loss] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )

##################################################################

class MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = MHABartDecoder(config, self.shared)       

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class REMultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.H_encoder = BartEncoder(config, self.shared)
        self.D_encoder = BartEncoder(config, self.shared)
        self.N_encoder = BartEncoder(config, self.shared)
        self.R_decoder = MHABartDecoder(config, self.shared)
        self.E_decoder = MHABartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            R_decoder_input_ids, R_decoder_padding_mask, R_causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids[0],
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
            E_decoder_input_ids, E_decoder_padding_mask, E_causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids[1],
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            E_decoder_padding_mask, E_causal_mask, R_decoder_padding_mask, R_causal_mask = None, None, None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs_R = self.R_decoder(
            R_decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            R_decoder_padding_mask,
            decoder_causal_mask=R_causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1]
        )

        decoder_outputs_E = self.E_decoder(
            E_decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            E_decoder_padding_mask,
            decoder_causal_mask=E_causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1]
        )

        if not return_dict:
            return decoder_outputs_R + decoder_outputs_E + encoder_outputs
        output_E = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs_E.last_hidden_state,
            past_key_values=decoder_outputs_E.past_key_values,
            decoder_hidden_states=decoder_outputs_E.hidden_states,
            decoder_attentions=decoder_outputs_E.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        output_R = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs_R.last_hidden_state,
            past_key_values=decoder_outputs_R.past_key_values,
            decoder_hidden_states=decoder_outputs_R.hidden_states,
            decoder_attentions=decoder_outputs_R.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        output = [output_R, output_E]
        return output

class RMultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.H_encoder = BartEncoder(config, self.shared)
        # self.D_encoder = BartEncoder(config, self.shared)
        self.R_decoder = MHABartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.R_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class EMultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.N_encoder = BartEncoder(config, self.shared)
        self.E_decoder = MHABartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.E_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class E_VAEMultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.N_encoder = BartEncoder(config, self.shared)
        self.H_encoder = BartEncoder(config, self.shared)
        self.E_decoder = E_VAEBartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.E_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1],
            encoder_hidden_states_z=encoder_outputs[2],
            encoder_attn_mask_z=attention_mask[2]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class E_mlp_VAE_MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.N_encoder = BartEncoder(config, self.shared)
        self.H_encoder = BartEncoder(config, self.shared)
        self.E_decoder = E_mlp_BartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.E_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1],
            encoder_hidden_states_z=encoder_outputs[2],
            encoder_attn_mask_z=attention_mask[2]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class R_mlp_VAE_MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.N_encoder = BartEncoder(config, self.shared)
        # self.H_encoder = BartEncoder(config, self.shared)
        self.R_decoder = R_mlp_BartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.R_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_z=encoder_outputs[1],
            encoder_attn_mask_z=attention_mask[1]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class E_add_VAE_MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.N_encoder = BartEncoder(config, self.shared)
        self.H_encoder = BartEncoder(config, self.shared)
        self.E_decoder = E_add_BartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.E_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1],
            encoder_hidden_states_z=encoder_outputs[2],
            encoder_attn_mask_z=attention_mask[2]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class R_add_VAE_MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.N_encoder = BartEncoder(config, self.shared)
        # self.H_encoder = BartEncoder(config, self.shared)
        self.R_decoder = R_add_BartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.R_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_z=encoder_outputs[1],
            encoder_attn_mask_z=attention_mask[1]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class R_add_HD_VAE_MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.N_encoder = BartEncoder(config, self.shared)
        # self.H_encoder = BartEncoder(config, self.shared)
        self.R_decoder = R_add_BartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.R_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1],
            encoder_hidden_states_z=encoder_outputs[2],
            encoder_attn_mask_z=attention_mask[2]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class tgt_PSA_VAE_MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig, add_z=False, psa_z=False):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.N_encoder = BartEncoder(config, self.shared)
        self.H_encoder = BartEncoder(config, self.shared)
        self.E_decoder = VAE_BartDecoder(config, self.shared, add_z, psa_z)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.E_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1],
            encoder_hidden_states_z=encoder_outputs[2],
            encoder_attn_mask_z=attention_mask[2]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class nxt_PSA_HD_VAE_MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig, add_z=False, psa_z=False):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.N_encoder = BartEncoder(config, self.shared)
        # self.H_encoder = BartEncoder(config, self.shared)
        self.E_decoder = VAE_BartDecoder(config, self.shared, add_z, psa_z)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.E_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1],
            encoder_hidden_states_z=encoder_outputs[2],
            encoder_attn_mask_z=attention_mask[2]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class E_MlpZ_CLS_MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.N_encoder = BartEncoder(config, self.shared)
        self.H_encoder = BartEncoder(config, self.shared)
        self.E_decoder = E_mlp_BartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.E_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1],
            encoder_hidden_states_z=encoder_outputs[2],
            encoder_attn_mask_z=attention_mask[2]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class E_CatZ_CLS_MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.N_encoder = BartEncoder(config, self.shared)
        self.H_encoder = BartEncoder(config, self.shared)
        self.E_decoder = E_CatZ_CLS_BartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.E_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1],
            encoder_hidden_states_z=encoder_outputs[2],
            encoder_attn_mask_z=attention_mask[2]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()

class MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = MultiHeadBartModel(config)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
   
    @classmethod
    def from_pretrained_multi(
        cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path))
            ckpt = torch.load(model_file_path)
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading default pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if n[6:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                else:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)

            print(model)
            del bart_model

            return model

class REMultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        RE_model = REMultiHeadBartModel(config)
        self.model = RE_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path))
            ckpt = torch.load(model_file_path)
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading RE pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[8:]].data)

            print(model)
            del bart_model

            return model

class RMultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        R_model = RMultiHeadBartModel(config)
        self.model = R_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.H_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path))
            ckpt = torch.load(model_file_path)
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading R pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[8:]].data)

            print(model)
            del bart_model

            return model

class EMultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        E_model = EMultiHeadBartModel(config)
        self.model = E_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.N_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path))
            ckpt = torch.load(model_file_path)
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading E pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[8:]].data)

            print(model)
            del bart_model

            return model

        # rec_z, rec_mu, rec_logvar = self.encoder2z.cxy2z(enc_hidden_cxy)
        # prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        # real_z = rec_z
        # kl_loss = self.encoder2z.kl_loss(rec_mu, rec_logvar, prior_mu, prior_logvar)

class E_VAEMultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        E_model = E_VAEMultiHeadBartModel(config)
        self.model = E_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.N_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path))
            ckpt = torch.load(model_file_path)
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading E_VAE pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    if "encoder_attn_doc" in n:
                        name_split = n.split('encoder_attn_doc')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    if "encoder_attn_z" in n:
                        name_split = n.split('encoder_attn_z')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    else:
                        continue

            print(model)
            del bart_model

            return model

        # rec_z, rec_mu, rec_logvar = self.encoder2z.cxy2z(enc_hidden_cxy)
        # prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        # real_z = rec_z
        # kl_loss = self.encoder2z.kl_loss(rec_mu, rec_logvar, prior_mu, prior_logvar)

class E_mlp_VAE_MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        E_model = E_mlp_VAE_MultiHeadBartModel(config)
        self.model = E_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.N_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'model.pt')
            ckpt = torch.load(model_file_path+'model.pt')
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading E_mlp pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    if "encoder_attn_doc" in n:
                        name_split = n.split('encoder_attn_doc')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    if "encoder_attn_z" in n:
                        name_split = n.split('encoder_attn_z')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    else:
                        continue

            print(model)
            del bart_model

            return model

        # rec_z, rec_mu, rec_logvar = self.encoder2z.cxy2z(enc_hidden_cxy)
        # prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        # real_z = rec_z
        # kl_loss = self.encoder2z.kl_loss(rec_mu, rec_logvar, prior_mu, prior_logvar)

class R_mlp_VAE_MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        model = R_mlp_VAE_MultiHeadBartModel(config)
        self.model = model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.N_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'model.pt')
            ckpt = torch.load(model_file_path+'model.pt')
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading R_mlp pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    if "encoder_attn_doc" in n:
                        name_split = n.split('encoder_attn_doc')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    if "encoder_attn_z" in n:
                        name_split = n.split('encoder_attn_z')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    else:
                        continue

            print(model)
            del bart_model

            return model

        # rec_z, rec_mu, rec_logvar = self.encoder2z.cxy2z(enc_hidden_cxy)
        # prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        # real_z = rec_z
        # kl_loss = self.encoder2z.kl_loss(rec_mu, rec_logvar, prior_mu, prior_logvar)

class E_add_VAE_MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        E_model = E_add_VAE_MultiHeadBartModel(config)
        self.model = E_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.N_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'model.pt')
            ckpt = torch.load(model_file_path+'model.pt')
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading E_add_HD pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    if "encoder_attn_doc" in n:
                        name_split = n.split('encoder_attn_doc')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    if "encoder_attn_z" in n:
                        name_split = n.split('encoder_attn_z')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    else:
                        continue

            print(model)
            del bart_model

            return model

        # rec_z, rec_mu, rec_logvar = self.encoder2z.cxy2z(enc_hidden_cxy)
        # prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        # real_z = rec_z
        # kl_loss = self.encoder2z.kl_loss(rec_mu, rec_logvar, prior_mu, prior_logvar)

class R_add_VAE_MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        model = R_add_VAE_MultiHeadBartModel(config)
        self.model = model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.N_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'model.pt')
            ckpt = torch.load(model_file_path+'model.pt')
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading R_add_H pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    if "encoder_attn_doc" in n:
                        name_split = n.split('encoder_attn_doc')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    if "encoder_attn_z" in n:
                        name_split = n.split('encoder_attn_z')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    else:
                        continue

            print(model)
            del bart_model

            return model

        # rec_z, rec_mu, rec_logvar = self.encoder2z.cxy2z(enc_hidden_cxy)
        # prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        # real_z = rec_z
        # kl_loss = self.encoder2z.kl_loss(rec_mu, rec_logvar, prior_mu, prior_logvar)

class R_add_HD_VAE_MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        model = R_add_HD_VAE_MultiHeadBartModel(config)
        self.model = model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.N_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'model.pt')
            ckpt = torch.load(model_file_path+'model.pt')
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading R_add_HD pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    if "encoder_attn_doc" in n:
                        name_split = n.split('encoder_attn_doc')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    elif "encoder_attn_z" in n:
                        name_split = n.split('encoder_attn_z')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    else:
                        continue

            print(model)
            del bart_model

            return model

        # rec_z, rec_mu, rec_logvar = self.encoder2z.cxy2z(enc_hidden_cxy)
        # prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        # real_z = rec_z
        # kl_loss = self.encoder2z.kl_loss(rec_mu, rec_logvar, prior_mu, prior_logvar)

class tgt_PSA_VAE_MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig, add_z, psa_z):
        super().__init__(config)
        model = tgt_PSA_VAE_MultiHeadBartModel(config, add_z, psa_z)
        self.model = model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.H_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, add_z=False, psa_z=False
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config, add_z, psa_z)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'tgt_model.pt')
            ckpt = torch.load(model_file_path+'tgt_model.pt')
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading tgt_PSA pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    if "encoder_attn_doc" in n:
                        name_split = n.split('encoder_attn_doc')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    elif "encoder_attn_z" in n:
                        name_split = n.split('encoder_attn_z')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    else:
                        continue

            print(model)
            del bart_model

            return model

        # rec_z, rec_mu, rec_logvar = self.encoder2z.cxy2z(enc_hidden_cxy)
        # prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        # real_z = rec_z
        # kl_loss = self.encoder2z.kl_loss(rec_mu, rec_logvar, prior_mu, prior_logvar)

class nxt_PSA_HD_VAE_MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig, add_z, psa_z):
        super().__init__(config)
        model = nxt_PSA_HD_VAE_MultiHeadBartModel(config, add_z, psa_z)
        self.model = model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.N_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, add_z=False, psa_z=False
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config, add_z, psa_z)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'nxt_model.pt')
            ckpt = torch.load(model_file_path+'nxt_model.pt')
            model.load_state_dict(ckpt['model'])
            return model
        else:
            # initialize scratch
            print('Loading nxt_PSA_HD pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    if "encoder_attn_doc" in n:
                        name_split = n.split('encoder_attn_doc')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    if "encoder_attn_z" in n:
                        name_split = n.split('encoder_attn_z')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    else:
                        continue

            print(model)
            del bart_model

            return model

class E_MlpZ_CLS_MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        E_model = E_mlp_VAE_MultiHeadBartModel(config)
        self.model = E_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.N_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'model.pt')
            ckpt = torch.load(model_file_path+'model.pt')
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading E_mlp pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    if "encoder_attn_doc" in n:
                        name_split = n.split('encoder_attn_doc')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    if "encoder_attn_z" in n:
                        name_split = n.split('encoder_attn_z')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    else:
                        continue

            print(model)
            del bart_model

            return model

        # rec_z, rec_mu, rec_logvar = self.encoder2z.cxy2z(enc_hidden_cxy)
        # prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        # real_z = rec_z
        # kl_loss = self.encoder2z.kl_loss(rec_mu, rec_logvar, prior_mu, prior_logvar)

class E_CatZ_CLS_MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        E_model = E_CatZ_CLS_MultiHeadBartModel(config)
        self.model = E_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def get_encoder(self):
        return self.model.N_encoder

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'model.pt')
            ckpt = torch.load(model_file_path+'model.pt')
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading E_CatZ_CLS pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if "shared" in n:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[8:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[8:]].data)
                else:
                    if "encoder_attn_doc" in n:
                        name_split = n.split('encoder_attn_doc')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    if "encoder_attn_z" in n:
                        name_split = n.split('encoder_attn_z')
                        name = name_split[0] + 'encoder_attn' + name_split[1]
                        p.data.copy_(bart_state_dict[name[8:]].data)
                    else:
                        continue

            print(model)
            del bart_model

            return model

        # rec_z, rec_mu, rec_logvar = self.encoder2z.cxy2z(enc_hidden_cxy)
        # prior_z, prior_mu, prior_logvar = self.encoder2z.c2z(enc_hidden_ctx)
        # real_z = rec_z
        # kl_loss = self.encoder2z.kl_loss(rec_mu, rec_logvar, prior_mu, prior_logvar)

class RE_VAE_MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig, add_z=False, psa_z=False):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.N_encoder = BartEncoder(config, self.shared)
        self.tgt_encoder = BartEncoder(config, self.shared)
        self.nxt_encoder = BartEncoder(config, self.shared)
        self.vae_encoder = VAE_encoder(config, self.shared)
        self.tgt_decoder = VAE_BartDecoder(config, self.shared, add_z, psa_z)
        self.nxt_decoder = VAE_BartDecoder(config, self.shared, add_z, psa_z)
        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def decode_tgt(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.tgt_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1],
            encoder_hidden_states_z=encoder_outputs[2],
            encoder_attn_mask_z=attention_mask[2]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def decode_nxt(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.nxt_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1],
            encoder_hidden_states_z=encoder_outputs[2],
            encoder_attn_mask_z=attention_mask[2]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class RE_VAE_MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig, add_z, psa_z):
        super().__init__(config)
        base_model = RE_VAE_MultiHeadBartModel(config, add_z, psa_z)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, add_z=False, psa_z=False
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config, add_z, psa_z)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path))
            ckpt = torch.load(model_file_path + 'model.pt')
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading default pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if n[6:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[10:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[10:]].data)
                elif n[18:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[18:]].data)
                elif 'encoder_attn_doc' in n:
                    name = n.replace('encoder_attn_doc', 'encoder_attn')
                    if name in bart_keys:
                        p.data.copy_(bart_state_dict[name].data)
                elif 'encoder_attn_z' in n:
                    name = n.replace('encoder_attn_z', 'encoder_attn')
                    if name in bart_keys:
                        p.data.copy_(bart_state_dict[name].data)
                else:
                    print(n)
                    continue
                # else:
                #     name_split = n.split('encoder_attn_doc')
                #     name = name_split[0] + 'encoder_attn' + name_split[1]
                #     p.data.copy_(bart_state_dict[name[6:]].data)


            print(model)
            del bart_model

            return model

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoder_typ='tgt',
        **unused,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the masked language modeling loss.
                Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
                with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

                # Mask filling only works for bart-large
                from transformers import BartTokenizer, BartForConditionalGeneration
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
                TXT = "My friends are <mask> but they eat too many carbs."

                model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
                input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
                logits = model(input_ids).logits

                masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
                probs = logits[0, masked_index].softmax(dim=0)
                values, predictions = probs.topk(5)

                tokenizer.decode(predictions).split()
                # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        if decoder_typ == 'tgt':
            outputs = self.model.decode_tgt(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if decoder_typ == 'nxt':
            outputs = self.model.decode_nxt(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def get_encoder(self):
        return self.model.tgt_encoder

class Bart_VAE_MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig, add_z=False, psa_z=False, have_doc=True):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.N_encoder = BartEncoder(config, self.shared)
        self.tgt_encoder = BartEncoder(config, self.shared)
        self.nxt_encoder = BartEncoder(config, self.shared)
        self.vae_encoder = VAE_encoder(config, self.shared)
        self.tgt_decoder = VAE_BartDecoder(config, self.shared, add_z, psa_z, have_doc)
        self.nxt_decoder = VAE_BartDecoder(config, self.shared, add_z, psa_z, have_doc)
        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def decode_tgt(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        decoder_outputs = self.tgt_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1] if len(encoder_outputs) == 3 else None,
            encoder_attn_mask_doc=attention_mask[1] if len(encoder_outputs) == 3 else None,
            encoder_hidden_states_z=encoder_outputs[2] if len(encoder_outputs) == 3 else encoder_outputs[1],
            encoder_attn_mask_z=attention_mask[2] if len(encoder_outputs) == 3 else attention_mask[1]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def decode_nxt(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.nxt_decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1] if len(encoder_outputs) == 3 else None,
            encoder_attn_mask_doc=attention_mask[1] if len(encoder_outputs) == 3 else None,
            encoder_hidden_states_z=encoder_outputs[2] if len(encoder_outputs) == 3 else encoder_outputs[1],
            encoder_attn_mask_z=attention_mask[2] if len(encoder_outputs) == 3 else attention_mask[1]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class Bart_VAE_MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig, add_z, psa_z, have_doc):
        super().__init__(config)
        base_model = Bart_VAE_MultiHeadBartModel(config, add_z, psa_z, have_doc)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, add_z=False, psa_z=False, have_doc=True
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config, add_z, psa_z, have_doc)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path))
            ckpt = torch.load(model_file_path + 'model.pt')
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading default pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if n[6:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif n[10:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[10:]].data)
                elif n[18:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[18:]].data)
                elif 'encoder_attn_doc' in n:
                    name = n.replace('encoder_attn_doc', 'encoder_attn')
                    if name in bart_keys:
                        p.data.copy_(bart_state_dict[name].data)
                elif 'encoder_attn_z' in n:
                    name = n.replace('encoder_attn_z', 'encoder_attn')
                    if name in bart_keys:
                        p.data.copy_(bart_state_dict[name].data)
                else:
                    print(n)
                    continue
                # else:
                #     name_split = n.split('encoder_attn_doc')
                #     name = name_split[0] + 'encoder_attn' + name_split[1]
                #     p.data.copy_(bart_state_dict[name[6:]].data)


            print(model)
            del bart_model

            return model

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoder_typ='tgt',
        **unused,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the masked language modeling loss.
                Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
                with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

                # Mask filling only works for bart-large
                from transformers import BartTokenizer, BartForConditionalGeneration
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
                TXT = "My friends are <mask> but they eat too many carbs."

                model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
                input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
                logits = model(input_ids).logits

                masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
                probs = logits[0, masked_index].softmax(dim=0)
                values, predictions = probs.topk(5)

                tokenizer.decode(predictions).split()
                # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        if decoder_typ == 'tgt':
            outputs = self.model.decode_tgt(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if decoder_typ == 'nxt':
            outputs = self.model.decode_nxt(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def get_encoder(self):
        return self.model.tgt_encoder

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        enc_size = 768
        z_dim = 768
        # Rec Net
        self.cxy2z_layer = nn.Sequential(
            nn.Linear(enc_size * 2, z_dim * 2),
            nn.ReLU()
        )
        self.cxy3z_layer = nn.Sequential(
            nn.Linear(enc_size * 3, z_dim * 2),
            nn.ReLU()
        )
        self.cxy2z_mu = nn.Linear(z_dim * 2, z_dim)
        self.cxy2z_logvar = nn.Linear(z_dim * 2, z_dim)
        # Prior Net
        self.c2z_layer = nn.Sequential(
            nn.Linear(enc_size, z_dim * 2),
            nn.ReLU()
        )
        self.c2z_mu = nn.Linear(z_dim * 2, z_dim)
        self.c2z_logvar = nn.Linear(z_dim * 2, z_dim)
        self.mlp = nn.Linear(z_dim, enc_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.rand(std.size()).cuda())
        # eps = torch.rand(std.size())
        # eps = torch.rand(std.size()).cuda()
        return mu + std * eps

    def kl_loss(self, mu1, logvar1, mu2, logvar2):
        # mu1, logvar1 -> RecognitionNet
        # mu2, logvar2 -> PriorNet
        kld = -0.5 * torch.sum(
            1 + logvar1 - logvar2 - torch.exp(logvar1) / torch.exp(logvar2) - torch.pow(mu1 - mu2, 2) / torch.exp(
                logvar2), -1)
        return kld

    def kl_loss_gaussian(self, mu1, logvar1):
        # mu2=0, logvar2=0
        kld = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), -1)
        return kld

    def cxy2z(self, hidden):
        hidden_ = self.cxy2z_layer(hidden)
        mu = self.cxy2z_mu(hidden_)
        logvar = self.cxy2z_logvar(hidden_)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def cxy3z(self, hidden):
        hidden_ = self.cxy3z_layer(hidden)
        mu = self.cxy2z_mu(hidden_)
        logvar = self.cxy2z_logvar(hidden_)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def c2z(self, hidden):
        hidden_ = self.c2z_layer(hidden)
        mu = self.c2z_mu(hidden_)
        logvar = self.c2z_logvar(hidden_)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class VAE_model(VAE):
    def __init__(self):
        super().__init__()
        VAE_model = VAE()
        self.model = VAE_model

    @classmethod
    def from_pretrained_multi(
            self, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""
        # this is actually a complete checkpoint
        model = VAE()
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'VAE_model.pt')
            ckpt = torch.load(model_file_path+'VAE_model.pt')
            model.load_state_dict(ckpt['model'])
            return model
        else:
            print("Using new VAE model")
            return model

class VAE_encoder(nn.Module):

    def __init__(self, config: BartConfig, shared):
        super().__init__()
        enc_size = 768
        z_dim = 768

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = shared
        self.encoder = VAE_BartEncoder(config, self.shared)

        nx = config.d_model
        nz = config.d_model
        self.post_mean = Conv1D(nz, nx)
        self.post_logvar = Conv1D(nz, nx)
        self.prior_mean = Conv1D(nz, nx)
        self.prior_logvar = Conv1D(nz, nx)


    def reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = torch.randn(std.size(), device=mean.device, dtype=mean.dtype)
        return z.mul(std) + mean

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def kl_loss_gaussian(self, mu1, logvar1):
        # mu2=0, logvar2=0
        kld = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), -1)
        return kld

    def cxy2z(self, inputs_ids, inputs_mask):
        representations = self.encoder(input_ids=inputs_ids, attention_mask=inputs_mask)
        posterior_mean = self.post_mean(representations)  # (bs,d)
        posterior_logvar = self.post_logvar(representations)  # (bs,d)
        latent_mean, latent_logvar = posterior_mean, posterior_logvar
        z = self.reparameterize(latent_mean, latent_logvar)
        assert not torch.isnan(z).any(), 'training get nan z'
        return z, latent_mean, latent_logvar

    def c2z(self, inputs_ids, inputs_mask):
        representations = self.encoder(input_ids=inputs_ids, attention_mask=inputs_mask)
        prior_mean = self.prior_mean(representations)  # (bs,d)
        prior_logvar = self.prior_logvar(representations)  # (bs,d)
        latent_mean, latent_logvar = prior_mean, prior_logvar
        z = self.reparameterize(latent_mean, latent_logvar)
        assert not torch.isnan(z).any(), 'training get nan z'
        return z, latent_mean, latent_logvar

class VAE_encoder_model(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        model = VAE_encoder(config)
        self.model = model

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""
        # this is actually a complete checkpoint
        config = BartConfig.from_pretrained(args.model_name)
        model = cls(config)
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'VAE_model.pt')
            ckpt = torch.load(model_file_path+'VAE_model.pt')
            model.load_state_dict(ckpt['model'])
            return model
        else:
            # initialize scratch
            print('Loading default VAE-encoder weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if n[6:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                elif 'encoder_prior' in n:
                    name = n.replace('encoder_prior', 'encoder')
                    if name in bart_keys:
                        p.data.copy_(bart_state_dict[name].data)
                else:
                    # print(n)
                    continue
            print(model)
            del bart_model

            return model

class mlp_decoder(nn.Module):
    def __init__(self, config: BartConfig):
        super(mlp_decoder, self).__init__()

        enc_size = 768
        z_dim = 768
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.z2nxt = nn.Linear(z_dim, enc_size)
        self.his2nxt = nn.Linear(enc_size, enc_size)
        self.his_z2nxt = nn.Linear(enc_size * 2, enc_size)
        self.tgt_nxt2tgt = nn.Linear(enc_size * 2, enc_size)

    def forward(self, z, his, nxt_ids):
        z = self.z2nxt(z).squeeze() # squeeze(0) ??
        his = self.his2nxt(his.mean(1))
        nxt_emb = self.embedding(nxt_ids).mean(1)
        z = z.reshape(nxt_emb.shape)
        his = his.reshape(nxt_emb.shape)
        nxt = self.his_z2nxt(torch.cat([his, z], dim=-1))

        loss = F.cosine_embedding_loss(nxt, nxt_emb, Tensor([1]).cuda())
        return (loss, nxt)

    def get_loss_fixed_emb(self, z, his, nxt_ids):
        z = self.z2nxt(z).squeeze() # squeeze(0) ??
        his = self.his2nxt(his.mean(1))
        nxt_emb = self.embedding(nxt_ids).mean(1)
        z = z.reshape(nxt_emb.shape)
        his = his.reshape(nxt_emb.shape)
        nxt = self.his_z2nxt(torch.cat([his, z], dim=-1))
        loss = F.cosine_embedding_loss(nxt, nxt_emb.detach(), Tensor([1]).cuda())
        return (loss, nxt)

class mlp_decoder_model(mlp_decoder):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        model = mlp_decoder(config)
        self.model = model

    @classmethod
    def from_pretrained_multi(
            self, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""
        # this is actually a complete checkpoint
        config = BartConfig.from_pretrained(args.model_name)
        model = mlp_decoder(config)
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path) + 'mlp_dec2_model.pt')
            ckpt = torch.load(model_file_path+'mlp_dec2_model.pt')
            model.load_state_dict(ckpt['model'])
            return model
        else:
            print("Using new mlp dec2 model")
            return model

class CLSBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.encoder = BartEncoder(config, self.shared)
        self.decoder = MHABartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask[0],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states_doc=encoder_outputs[1],
            encoder_attn_mask_doc=attention_mask[1]
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly

class CLSBartForSequenceClassification(BartForSequenceClassification):
    base_model_prefix = "model"
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = CLSBartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classif_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @classmethod
    def from_pretrained_multi(
        cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path))
            ckpt = torch.load(model_file_path)
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading CLS pre-trained BART weights')
            bart_model = BartForSequenceClassification.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if n in bart_keys:
                    p.data.copy_(bart_state_dict[n].data)
                elif n[6:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[6:]].data)
                else:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name].data)
            print(model)
            del bart_model

            return model

class CLS(BartForSequenceClassification):

    def __init__(self, config: BartConfig):
        super().__init__(config)
        E_model = CLSBartForSequenceClassification(config)
        self.model = E_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
        cls, args, model_file_path
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(model_file_path):
            print('Loading exisitng model at ' + str(model_file_path))
            ckpt = torch.load(model_file_path)
            model.load_state_dict(ckpt['model'])
            return model

        else:
            # initialize scratch
            print('Loading CLS pre-trained BART weights')
            bart_model = BartModel.from_pretrained(args.model_name)
            bart_state_dict = bart_model.state_dict()
            bart_keys = []
            for key, value in bart_state_dict.items():
                bart_keys.append(key)

            for n, p in model.named_parameters():
                if n[6:] in bart_keys:
                    p.data.copy_(bart_state_dict[n[6:]].data)
            print(model)
            del bart_model

            return model

