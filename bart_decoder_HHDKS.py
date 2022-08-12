import os
import math
import random
import warnings
import torch
import pdb
from typing import Dict, List, Optional, Tuple

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

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
    _prepare_bart_decoder_inputs
)

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

class Doc_His_Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.his_num = 3
        self.utt_len = 30
        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.mlp = nn.Linear(embed_dim * 2, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

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
        his_word = key[0]
        his_utt = key[1]
        one_his_len = int(his_word.shape[0] / 3)
        doc = query
        doc_len = doc.size(0)
        his_utt_len = his_utt.size(0)
        his_word_len, bsz, embed_dim = his_word.size()
        assert embed_dim == self.embed_dim
        assert list(his_word.size()) == [his_word_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key_word" in saved_state and static_kv:
                # previous time steps are cached - no need to recompute key and value if they are static
                key = None
        else:
            saved_state = None
            layer_state = {}

        q_doc = self.q_proj(doc) * self.scaling
        if static_kv:
            if key is None:
                k_his_word = k_his_utt = v_doc = v_his_word = None
            else:
                k_his_word = self.k_proj(his_word)
                k_his_utt = self.k2_proj(his_utt)
                v_his_word = self.v_proj(his_word)
                v_doc = self.v2_proj(doc)
        else:
            k_his_word = self.k_proj(his_word)
            k_his_utt = self.v_proj(his_utt)
            v_his_word = self.v_proj(his_word)
            v_doc = self.v2_proj(doc)

        q_doc = self._shape(q_doc, doc_len, bsz)
        if k_his_word is not None:
            k_his_word = self._shape(k_his_word, -1, bsz)
        if k_his_utt is not None:
            k_his_utt = self._shape(k_his_utt, -1, bsz)
        if v_his_word is not None:
            v_his_word = self._shape(v_his_word, -1, bsz)
        if v_doc is not None:
            v_doc = self._shape(v_doc, -1, bsz)

        if saved_state is not None:
            k_his_word, k_his_utt, v_his_word, v_doc, key_padding_mask = self._use_saved_state(k_his_word, k_his_utt, v_his_word, v_doc, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key_his_word": k_his_word.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_his_utt": k_his_utt.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value_his_word": v_his_word.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value_doc": v_doc.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        # q_doc and k_his_word cross_attn
        assert k_his_word is not None
        his_word_doc_attn = torch.bmm(q_doc, k_his_word.transpose(1, 2))
        assert his_word_doc_attn.size() == (bsz * self.num_heads, doc_len, his_word_len)

        if attn_mask is not None:
            his_word_doc_attn = his_word_doc_attn.view(bsz, self.num_heads, doc_len, his_word_len) + attn_mask
            his_word_doc_attn = his_word_doc_attn.view(bsz * self.num_heads, doc_len, his_word_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        his_word_mask = key_padding_mask[0]
        doc_mask = key_padding_mask[1]

        if doc_mask is not None and doc_mask.dim() == 0:
            doc_mask = None
        assert doc_mask is None or doc_mask.size()[:2] == (
            bsz,
            doc_len,
        )

        if his_word_mask is not None:  # don't attend to padding symbols
            his_word_doc_attn = his_word_doc_attn.view(bsz, self.num_heads, doc_len, his_word_len)
            reshaped = his_word_mask.unsqueeze(1).unsqueeze(2)
            his_word_doc_attn = his_word_doc_attn.masked_fill(reshaped, float("-inf"))
            his_word_doc_attn = his_word_doc_attn.view(bsz * self.num_heads, doc_len, his_word_len)

        his_word_doc_attn = F.softmax(his_word_doc_attn, dim=-1)        # (bs*num_heads, doc_len, his_word_len)

        # q_doc and k_his_utt cross_attn
        assert k_his_utt is not None
        his_utt_doc_attn = torch.bmm(q_doc, k_his_utt.transpose(1, 2))
        assert his_utt_doc_attn.size() == (bsz * self.num_heads, doc_len, his_utt_len)

        if attn_mask is not None:
            his_utt_doc_attn = his_utt_doc_attn.view(bsz, self.num_heads, doc_len, his_utt_len) + attn_mask
            his_utt_doc_attn = his_utt_doc_attn.view(bsz * self.num_heads, doc_len, his_utt_len)

        his_utt_doc_attn = F.softmax(his_utt_doc_attn, dim=-1)      # (bs*num_heads,doc_len,his_utt_len)

        his_utt_doc_attn = his_utt_doc_attn.reshape(bsz*self.num_heads, doc_len, his_utt_len, 1)\
            .repeat((1, 1, 1, one_his_len))\
            .reshape(bsz*self.num_heads, doc_len, his_utt_len*one_his_len)      # (bs*num_heads, doc_len, his_word_len)

        attn = his_word_doc_attn * his_utt_doc_attn                     # (bs*num_heads,doc_len, his_word_len)
        # re-mask attn
        if his_word_mask is not None:  # don't attend to padding symbols
            attn = attn.view(bsz, self.num_heads, doc_len, his_word_len)
            reshaped = his_word_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(reshaped, 0)
            attn = attn.view(bsz * self.num_heads, doc_len, his_word_len)

        scaler = (1/(attn.sum(-1) + 1e-18))
        scaler = scaler.reshape(bsz*self.num_heads, doc_len, 1).tile(1, 1, his_word_len)
        attn = attn * scaler

        # get doc2his attn
        doc_his_attn = attn
        doc_his_attn_probs = F.dropout(
            doc_his_attn,
            p=self.dropout,
            training=self.training,
        )

        # get his2doc attn
        his_doc_attn = attn
        his_doc_attn = torch.max(his_doc_attn, dim=-1)[0].unsqueeze(-1)  # (bs*num_heads, doc_len, 1)
        his_doc_attn = his_doc_attn.transpose(1, 2)     # (bs*num_heads, 1, doc_len)

        if doc_mask is not None:  # don't attend to padding symbols
            his_doc_attn = his_doc_attn.view(bsz, self.num_heads, 1, doc_len)
            reshaped = doc_mask.unsqueeze(1).unsqueeze(2)
            his_doc_attn = his_doc_attn.masked_fill(reshaped, 0)
            his_doc_attn = his_doc_attn.view(bsz * self.num_heads, 1, doc_len)

        scaler_doc = (1/(his_doc_attn.sum(-1) + 1e-18))
        scaler_doc = scaler_doc.reshape(bsz*self.num_heads, 1, 1).tile(1, 1, doc_len)
        his_doc_attn = his_doc_attn * scaler_doc            # (bs*num_heads, 1, doc_len)
        his_doc_attn_probs = F.dropout(
            his_doc_attn,
            p=self.dropout,
            training=self.training,
        )

        # K * V
        assert v_his_word is not None
        doc_his_attn_output = torch.bmm(doc_his_attn_probs, v_his_word)
        assert doc_his_attn_output.size() == (bsz * self.num_heads, doc_len, self.head_dim)
        doc_his_attn_output = doc_his_attn_output.transpose(0, 1).contiguous().view(doc_len, bsz, embed_dim)

        assert v_doc is not None
        his_doc_attn_output = torch.bmm(his_doc_attn_probs, v_doc)
        assert his_doc_attn_output.size() == (bsz * self.num_heads, 1, self.head_dim)
        his_doc_attn_output = his_doc_attn_output.transpose(0, 1).contiguous().view(1, bsz, embed_dim)
        his_doc_attn_output = his_doc_attn_output.repeat(doc_len, 1, 1)

        # combine doc2his and his2doc
        attn_output = torch.cat([doc_his_attn_output, his_doc_attn_output], dim=-1)  # (bs,doc_len,2*d)
        attn_output = self.mlp(attn_output)     # (bs,doc_len,d)
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights = attn.view(bsz, self.num_heads, doc_len, his_word_len)
        else:
            attn_weights = None

        return attn_output, attn_weights

    def _use_saved_state(self, k_his_word, k_his_utt, v_his_word, v_doc, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key_his_word" in saved_state:
            _prev_key_his_word = saved_state["prev_key_his_word"]
            assert _prev_key_his_word is not None
            prev_key_his_word = _prev_key_his_word.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k_his_word = prev_key_his_word
            else:
                assert k_his_word is not None
                k_his_word = torch.cat([prev_key_his_word, k_his_word], dim=1)
        if "prev_key_his_utt" in saved_state:
            _prev_key_his_utt = saved_state["prev_key_his_utt"]
            assert _prev_key_his_utt is not None
            prev_key_his_utt = _prev_key_his_utt.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k_his_utt = prev_key_his_utt
            else:
                assert k_his_utt is not None
                k_his_utt = torch.cat([prev_key_his_utt, k_his_utt], dim=1)
        if "prev_value_doc" in saved_state:
            _prev_value_doc = saved_state["prev_value_doc"]
            assert _prev_value_doc is not None
            prev_value_doc = _prev_value_doc.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v_doc = prev_value_doc
            else:
                assert v_doc is not None
                v_doc = torch.cat([prev_value_doc, v_doc], dim=1)
        if "prev_value_his_word" in saved_state:
            _prev_value_his_word = saved_state["prev_value_his_word"]
            assert _prev_value_his_word is not None
            prev_value_his_word = _prev_value_his_word.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v_his_word = prev_value_his_word
            else:
                assert v_his_word is not None
                v_his_word = torch.cat([prev_value_his_word, v_his_word], dim=1)
        assert k_his_word is not None and k_his_utt is not None and v_doc is not None and v_his_word is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = key_padding_mask
        return k_his_word, k_his_utt, v_his_word, v_doc, new_key_padding_mask

class His_Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.his_num = 3
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

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
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        his_word = k
        his_len = int(his_word.shape[1] / 3)
        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
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
        his_word_attn_weights = F.softmax(attn_weights, dim=-1)

        n_his_word_weights = torch.sum(his_word_attn_weights, dim=1).reshape(bsz*self.num_heads, self.his_num, his_len, 1)
        n_his_word = his_word.reshape(bsz*self.num_heads, self.his_num, his_len, -1)
        his_utt = n_his_word_weights * n_his_word
        his_utt = torch.sum(his_utt, dim=2)
        his_utt_attn_weights = torch.bmm(q, his_utt.transpose(1, 2))
        his_utt_attn_weights = F.softmax(his_utt_attn_weights, dim=-1)

        his_utt_attn_weight = his_utt_attn_weights.reshape(bsz*self.num_heads, tgt_len, self.his_num, 1).repeat((1, 1, 1, his_len)).reshape(bsz*self.num_heads, tgt_len, his_len*self.his_num)

        attn_weights = his_word_attn_weights * his_utt_attn_weight
        scaler = (1/(attn_weights.sum(-1) + 1e-18))
        scaler = scaler.reshape(bsz*self.num_heads, tgt_len, 1).tile(1, 1, his_len*self.his_num)
        attn_weights = attn_weights * scaler
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

class His_Attention_p(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
        his_num=3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.his_num = his_num
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
        his_utt=None
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
        if key is not None:
            his_len, bsz, embed_dim = key.size()
            his_word = key
            if his_utt is None:
                his_utt = key.reshape(self.his_num, int(his_len/self.his_num), bsz, embed_dim).mean(1)
            key = (his_word, his_utt)

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
                k2 = v2 = None
            else:
                k = self.k_proj(key[0])
                k2 = self.k2_proj(key[1])
                v = self.v_proj(key[0])
                v2 = self.v_proj(key[1])
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)
            k2 = v2 = None

        q = self._shape(q, tgt_len, bsz)
        if k is not None and k2 is not None:
            k = self._shape(k, -1, bsz)
            k2 = self._shape(k2, -1, bsz)
        if v is not None and v2 is not None:
            v = self._shape(v, -1, bsz)
            v2 = self._shape(v2, -1, bsz)

        if saved_state is not None:
            k, k2, v, v2, key_padding_mask = self._use_saved_state(k, k2, v, v2, saved_state, key_padding_mask, static_kv, bsz)

        his_word = k
        his_len = int(his_word.shape[1] / self.his_num)
        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key2": k2.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value2": v2.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
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
        his_word_attn_weights = F.softmax(attn_weights, dim=-1)

        assert k2 is not None
        src_len2 = k2.size(1)
        attn_weights2 = torch.bmm(q, k2.transpose(1, 2))
        assert attn_weights2.size() == (bsz * self.num_heads, tgt_len, src_len2)

        his_utt_attn_weights = F.softmax(attn_weights2, dim=-1)
        his_utt_attn_weights = his_utt_attn_weights.unsqueeze(-1).repeat((1, 1, 1, his_len)).reshape(bsz*self.num_heads, tgt_len, his_len*self.his_num)

        attn_weights = his_word_attn_weights * his_utt_attn_weights
        scaler = (1/(attn_weights.sum(-1) + 1e-18))
        scaler = scaler.reshape(bsz*self.num_heads, tgt_len, 1).tile(1, 1, his_len*self.his_num)
        attn_weights = attn_weights * scaler
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

    def _use_saved_state(self, k, k2, v, v2, saved_state, key_padding_mask, static_kv, bsz):
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
        if "prev_key2" in saved_state:
            _prev_key2 = saved_state["prev_key2"]
            assert _prev_key2 is not None
            prev_key2 = _prev_key2.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k2 = prev_key2
            else:
                assert k2 is not None
                k2 = torch.cat([prev_key2, k2], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        if "prev_value2" in saved_state:
            _prev_value2 = saved_state["prev_value2"]
            assert _prev_value2 is not None
            prev_value2 = _prev_value2.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v2 = prev_value2
            else:
                assert v2 is not None
                v2 = torch.cat([prev_value2, v2], dim=1)
        assert k is not None and v is not None and k2 is not None and v2 is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = key_padding_mask
        return k, k2, v, v2, new_key_padding_mask

class MHAEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn_his = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_his_layer_norm = LayerNorm(self.embed_dim)

        self.self_attn_doc = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_doc_layer_norm = LayerNorm(self.embed_dim)

        self.doc_his_attn = Doc_His_Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.doc_his_attn_layer_norm = LayerNorm(self.embed_dim)

        self.normalize_before = config.normalize_before
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
        his, doc = x[0], x[1]
        his_mask, doc_mask = encoder_padding_mask[0], encoder_padding_mask[1]
        # his_self_attn
        residual_his = his
        if self.normalize_before:
            his = self.self_attn_his_layer_norm(his)
        his, attn_weights = self.self_attn_his(
            query=his, key=his, key_padding_mask=his_mask, output_attentions=output_attentions
        )
        his = F.dropout(his, p=self.dropout, training=self.training)
        his = residual_his + his
        if not self.normalize_before:
            his = self.self_attn_his_layer_norm(his)

        # doc_self_attn
        residual_doc = doc
        if self.normalize_before:
            doc = self.self_attn_doc_layer_norm(doc)
        doc, attn_weights = self.self_attn_doc(
            query=doc, key=doc, key_padding_mask=doc_mask, output_attentions=output_attentions
        )
        doc = F.dropout(doc, p=self.dropout, training=self.training)
        doc = residual_doc + doc
        if not self.normalize_before:
            doc = self.self_attn_doc_layer_norm(doc)

        # his_doc cross attn
        residual = doc
        if self.normalize_before:
            doc = self.doc_his_attn_layer_norm(doc)
        n_doc, _ = self.doc_his_attn(
            query=doc,
            key=his,
            key_padding_mask=doc_mask    # mutates layer state
        )
        n_doc = F.dropout(n_doc, p=self.dropout, training=self.training)
        n_doc = residual + n_doc
        if not self.normalize_before:
            n_doc = self.encoder_attn_doc_layer_norm(n_doc)

        # FFN
        residual = n_doc
        if self.normalize_before:
            n_doc = self.final_layer_norm(n_doc)
        n_doc = self.activation_fn(self.fc1(n_doc))
        n_doc = F.dropout(n_doc, p=self.activation_dropout, training=self.training)
        n_doc = self.fc2(n_doc)
        n_doc = F.dropout(n_doc, p=self.dropout, training=self.training)
        n_doc = residual + n_doc
        if not self.normalize_before:
            n_doc = self.final_layer_norm(n_doc)

        return n_doc, attn_weights

class MHABartEncoder(nn.Module):
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
        self.layers = nn.ModuleList([MHAEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

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
            his_mask = invert_mask(attention_mask[0])
            doc_mask = invert_mask(attention_mask[1])
            attention_mask = (his_mask, doc_mask)

        his_ids, doc_ids = input_ids[0], input_ids[1]
        his_mask, doc_mask = attention_mask[0], attention_mask[1]

        his_embeds = self.embed_tokens(his_ids) * self.embed_scale
        embed_pos = self.embed_positions(his_ids)
        his = his_embeds + embed_pos
        his = self.layernorm_embedding(his)
        his = F.dropout(his, p=self.dropout, training=self.training)

        doc_embeds = self.embed_tokens(doc_ids) * self.embed_scale
        embed_pos = self.embed_positions(doc_ids)
        doc = doc_embeds + embed_pos
        doc = self.layernorm_embedding(doc)
        doc = F.dropout(doc, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        his = his.transpose(0, 1)
        doc = doc.transpose(0, 1)
        x = (his, doc)

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

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)

class HHDKSEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.encoder_attn = Doc_His_Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
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
        his_word, his_utt, doc = x[0], x[1], x[2]
        residual = doc
        x = doc
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        inputs = (his_word, his_utt)
        x, attn_weights = self.encoder_attn(
            query=x, key=inputs, key_padding_mask=encoder_padding_mask, output_attentions=output_attentions
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

class HHDKSBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens, layer_nums=0):
        super().__init__()
        if layer_nums == 0:
            layer_nums = config.enoder_layers
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
        self.layers = nn.ModuleList([HHDKSEncoderLayer(config) for _ in range(layer_nums)])  #config.enoder_layers
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

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
            his_mask, doc_mask = attention_mask[0], attention_mask[1]
            if his_mask is not None:
                his_mask = invert_mask(his_mask)
            if doc_mask is not None:
                doc_mask = invert_mask(doc_mask)
            attention_mask = (his_mask, doc_mask)

        his_word, his_utt, doc = input_ids[0], input_ids[1], input_ids[2]

        # B x T x C -> T x B x C
        his_word = his_word.transpose(0, 1)
        his_utt = his_utt.transpose(0, 1)
        doc = doc.transpose(0, 1)
        inputs = (his_word, his_utt, doc)

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(doc)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                doc, attn = encoder_layer(inputs, attention_mask, output_attentions=output_attentions)
                inputs = (his_word, his_utt, doc)
            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            doc = self.layer_norm(doc)
        if output_hidden_states:
            encoder_states.append(doc)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        doc = doc.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [doc, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=doc, hidden_states=encoder_states, attentions=all_attentions)

class HHDKS_BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MHAAttention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout)
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

class HHDKS_BartEncoder(nn.Module):
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
        self.layers = nn.ModuleList([HHDKS_BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

    def forward(
        self, input_ids, his_utt=None, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False
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
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        if his_utt is not None:
            x = torch.cat([his_utt, x], dim=1)
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 3).cuda(), attention_mask], dim=1)

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)
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

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)

class HHDKS_CA2_BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.config = config
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
        self.layers = nn.ModuleList([HHDKS_BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

    def forward(
        self, input_ids, his_utt=None, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False
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
        if his_utt is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            embed_pos = self.embed_positions(input_ids)
            x = inputs_embeds + embed_pos
        else:
            doc_emb = self.embed_tokens(input_ids)
            bos_emb = self.embed_tokens((torch.ones(doc_emb.shape[0], 1).cuda()*self.config.bos_token_id).int())
            inputs_embeds = torch.cat([bos_emb, his_utt, doc_emb], dim=1) * self.embed_scale
            embed_pos = self.embed_positions(torch.cat([torch.ones(doc_emb.shape[0], 4).cuda()*self.config.bos_token_id, input_ids], dim=1).int())
            x = inputs_embeds + embed_pos

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)
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

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)

class HHDKS_CA_DecoderLayer(nn.Module):
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

        self.encoder_attn_z = MHAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
            attention_type='z'
        )
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

class HHDKS_CA_BartDecoder(nn.Module):
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
            [HHDKS_CA_DecoderLayer(config) for _ in range(config.decoder_layers)]
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

class HHDKSDecoderLayer(nn.Module):
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
        self.encoder_attn_his = His_Attention(
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
        assert self.encoder_attn_his.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn_his(
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

class HHDKSBartDecoder(nn.Module):
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
            [HHDKSDecoderLayer(config) for _ in range(config.decoder_layers)]
        )   ##  type: List[DecoderLayer]
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

class HHDKSDecoderLayer_p(nn.Module):
    def __init__(self, config: BartConfig, his_num):
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
        self.encoder_attn_his = His_Attention_p(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
            his_num=his_num
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
        assert self.encoder_attn_his.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn_his(
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

class HHDKSBartDecoder_p(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding, his_num):
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
            [HHDKSDecoderLayer_p(config,his_num) for _ in range(config.decoder_layers)]
        )   ##  type: List[DecoderLayer]
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

class His_Attention_dyn(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
        his_num=3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.his_num = his_num
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
        his_utt=None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        ori_his_word = key
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

        if key is not None:
            his_len, bsz, embed_dim = key.size()
            one_his_len = int(his_len / self.his_num)
            if his_utt is None:  # (bs,tgt_len,768)
                his_word = key
                his_utt = key
                key = (his_word, his_utt)

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
                k2 = v2 = None
            else:
                k = self.k_proj(key[0])
                k2 = self.k2_proj(key[1])
                v = self.v_proj(key[0])
                v2 = self.v2_proj(key[1])

        else:
            k = self.k_proj(query)
            v = self.v_proj(query)
            k2 = self.k2_proj(key[1])
            v2 = self.v2_proj(key[1])

        q = self._shape(q, tgt_len, bsz)
        if k is not None and k2 is not None:
            k = self._shape(k, -1, bsz)
            k2 = self._shape(k2, -1, bsz)
        if v is not None and v2 is not None:
            v = self._shape(v, -1, bsz)
            v2 = self._shape(v2, -1, bsz)

        if saved_state is not None:
            k, k2, v, v2, key_padding_mask = self._use_saved_state(k, k2, v, v2, saved_state, key_padding_mask, static_kv, bsz)

        one_his_len = int(k.size(1) / self.his_num)
        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key2": k2.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value2": v2.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
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
        his_word_attn_weights = F.softmax(attn_weights, dim=-1)

        # dynamic build his_utt by his_word and tgt
        attn_word = attn_weights.reshape(bsz * self.num_heads, tgt_len, self.his_num, one_his_len)
        attn_word_weights = F.softmax(attn_word.mean(1), dim=-1).unsqueeze(-1)  # (bs, 3, his_len, 1)
        r_his_word = ori_his_word.reshape(bsz * self.num_heads, self.his_num, one_his_len, -1)  # (bs, 3, his_len, d)
        his_utt = attn_word_weights * r_his_word    # (bs*num_heads,tgt_len,3,one_his_len,d)
        his_utt = his_utt.mean(2).reshape(bsz, self.his_num, -1)   # (bs*num_heads,tgt_len,his_len,d)
        k2 = self.k2_proj(his_utt)
        k2 = self._shape(k2, -1, bsz)   # (bs*num_heads,tgt_len,)

        assert k2 is not None
        src_len2 = k2.size(1)
        attn_weights2 = torch.bmm(q, k2.transpose(1, 2))
        assert attn_weights2.size() == (bsz * self.num_heads, tgt_len, src_len2)

        his_utt_attn_weights = F.softmax(attn_weights2, dim=-1)
        his_utt_attn_weights = his_utt_attn_weights.unsqueeze(-1).repeat((1, 1, 1, one_his_len)).reshape(bsz*self.num_heads, tgt_len, one_his_len*self.his_num)

        attn_weights = his_word_attn_weights * his_utt_attn_weights
        scaler = (1/(attn_weights.sum(-1) + 1e-18))
        scaler = scaler.reshape(bsz*self.num_heads, tgt_len, 1).tile(1, 1, one_his_len*self.his_num)
        attn_weights = attn_weights * scaler
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

    def _use_saved_state(self, k, k2, v, v2, saved_state, key_padding_mask, static_kv, bsz):
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
        if "prev_key2" in saved_state:
            _prev_key2 = saved_state["prev_key2"]
            assert _prev_key2 is not None
            prev_key2 = _prev_key2.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k2 = prev_key2
            else:
                assert k2 is not None
                k2 = torch.cat([prev_key2, k2], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        if "prev_value2" in saved_state:
            _prev_value2 = saved_state["prev_value2"]
            assert _prev_value2 is not None
            prev_value2 = _prev_value2.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v2 = prev_value2
            else:
                assert v2 is not None
                v2 = torch.cat([prev_value2, v2], dim=1)
        assert k is not None and v is not None and k2 is not None and v2 is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = key_padding_mask
        return k, k2, v, v2, new_key_padding_mask

class HHDKSDecoderLayer_dyn(nn.Module):
    def __init__(self, config: BartConfig, his_num):
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
        self.encoder_attn_his = His_Attention_dyn(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
            his_num=his_num
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
        assert self.encoder_attn_his.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn_his(
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

class HHDKSBartDecoder_dyn(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding, his_num):
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
            [HHDKSDecoderLayer_dyn(config,his_num) for _ in range(config.decoder_layers)]
        )   ##  type: List[DecoderLayer]
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

class HHDKSDecoderLayer_p_avgemb(nn.Module):
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
        self.encoder_attn_his = His_Attention_p(
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
        encoder_attn_mask_doc=None,
        his_utt=None
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
        assert self.encoder_attn_his.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn_his(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
            his_utt=his_utt
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

class HHDKSBartDecoder_p_avgemb(nn.Module):
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
            [HHDKSDecoderLayer_p_avgemb(config) for _ in range(config.decoder_layers)]
        )   ##  type: List[DecoderLayer]
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
        his_utt=None,
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
                encoder_attn_mask_doc=encoder_attn_mask_doc,
                his_utt=his_utt
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
        )   ##  type: List[DecoderLayer]
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

class HHDKS_Enc_Dec_BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.mlp = nn.Linear(config.d_model * 2, config.d_model, bias=True)
        self.decoder = HHDKSBartDecoder(config, self.shared)

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

class HHDKS_Enc_Dec_BartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = HHDKS_Enc_Dec_BartModel(config)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
   
    @classmethod
    def from_pretrained_multi(
        cls, args, model_file_path, use_pretrain=True
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
                if 'encoder_attn_doc' in n:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                if not use_pretrain:
                    continue
                if 'encoder2' in n:
                    if 'encoder_attn' not in n:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'encoder' + name_split[1]
                        p.data.copy_(bart_state_dict[name[6:]].data)
                    else:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'decoder' + name_split[1]
                        if 'k2' or 'v2' in name:
                            name = name.replace('k2', 'k').replace('v2', 'v')
                        if 'mlp' in name:
                            continue
                        p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_his' in n:
                    name_split = n.split('encoder_attn_his')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                else:
                    print(n)
                    continue

            print(model)
            del bart_model

            return model

class HHDKS_Enc_BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.encoder2 = HHDKSBartEncoder(config, self.shared, layer_nums=1)
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

class HHDKS_Enc_BartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = HHDKS_Enc_BartModel(config)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, use_pretrain=True
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
                if 'encoder_attn_doc' in n:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                if not use_pretrain:
                    continue
                if 'encoder2' in n:
                    if 'encoder_attn' not in n:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'encoder' + name_split[1]
                        p.data.copy_(bart_state_dict[name[6:]].data)
                    else:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'decoder' + name_split[1]
                        if 'k2' or 'v2' in name:
                            name = name.replace('k2', 'k').replace('v2', 'v')
                        if 'mlp' in name:
                            continue
                        p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_his' in n:
                    name_split = n.split('encoder_attn_his')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)

            print(model)
            del bart_model

            return model

class HHDKS_Enc_np_BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.utt_mlp = nn.Linear(config.d_model, config.d_model, bias=True)
        self.mlp = nn.Linear(config.d_model * 2, config.d_model, bias=True)
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

class HHDKS_Enc_np_BartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = HHDKS_Enc_np_BartModel(config)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, use_pretrain=True
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
                if 'encoder_attn_doc' in n:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                if not use_pretrain:
                    continue
                if 'encoder2' in n:
                    if 'encoder_attn' not in n:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'encoder' + name_split[1]
                        p.data.copy_(bart_state_dict[name[6:]].data)
                    else:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'decoder' + name_split[1]
                        if 'k2' or 'v2' in name:
                            name = name.replace('k2', 'k').replace('v2', 'v')
                        if 'mlp' in name:
                            continue
                        p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_his' in n:
                    name_split = n.split('encoder_attn_his')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)

            print(model)
            del bart_model

            return model

class HHDKS_Enc_Dec_np_BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.mlp = nn.Linear(config.d_model * 2, config.d_model, bias=True)
        self.decoder = HHDKSBartDecoder(config, self.shared)

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

class HHDKS_Enc_Dec_np_BartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = HHDKS_Enc_Dec_np_BartModel(config)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, use_pretrain=True
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
                elif 'encoder_attn_doc' in n:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                elif not use_pretrain:
                    continue
                elif 'encoder2' in n:
                    if 'encoder_attn' not in n:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'encoder' + name_split[1]
                        p.data.copy_(bart_state_dict[name[6:]].data)
                    else:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'decoder' + name_split[1]
                        if 'k2' or 'v2' in name:
                            name = name.replace('k2', 'k').replace('v2', 'v')
                        if 'mlp' in name:
                            continue
                        p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_his' in n:
                    name_split = n.split('encoder_attn_his')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                else:
                    print(n)
                    continue
            print(model)
            del bart_model

            return model

class HHDKS_Enc_np_Dec_p_BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig, his_num):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.utt_mlp = nn.Linear(config.d_model, config.d_model, bias=True)
        self.mlp = nn.Linear(config.d_model * 2, config.d_model, bias=True)
        self.decoder = HHDKSBartDecoder_p(config, self.shared, his_num)

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

class HHDKS_Enc_np_Dec_p_BartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig, his_num):
        super().__init__(config)
        base_model = HHDKS_Enc_np_Dec_p_BartModel(config, his_num)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, use_pretrain=True, his_num=3
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config, his_num)

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
                elif 'encoder_attn_doc' in n:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                elif not use_pretrain:
                    continue
                elif 'encoder2' in n:
                    if 'encoder_attn' not in n:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'encoder' + name_split[1]
                        p.data.copy_(bart_state_dict[name[6:]].data)
                    else:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'decoder' + name_split[1]
                        if 'k2' or 'v2' in name:
                            name = name.replace('k2', 'k').replace('v2', 'v')
                        if 'mlp' in name:
                            continue
                        p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_his' in n:
                    name_split = n.split('encoder_attn_his')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    if 'k2' or 'v2' in name:
                        name = name.replace('k2', 'k').replace('v2', 'v')
                    if 'mlp' in name:
                        continue
                    p.data.copy_(bart_state_dict[name[6:]].data)
                else:
                    print(n)
                    continue
            print(model)
            del bart_model

            return model

class HHDKS_Enc_ca_Dec_ca_BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = HHDKS_BartEncoder(config, self.shared)
        self.mlp = nn.Linear(config.d_model, config.d_model, bias=True)
        self.decoder = HHDKS_CA_BartDecoder(config, self.shared)

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

class HHDKS_Enc_ca_Dec_ca_BartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = HHDKS_Enc_ca_Dec_ca_BartModel(config)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, use_pretrain=True
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
                elif 'encoder_attn_doc' in n:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_z' in n:
                    name_split = n.split('encoder_attn_z')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                elif not use_pretrain:
                    continue
                elif 'encoder2' in n:
                    if 'encoder_attn' not in n:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'encoder' + name_split[1]
                        p.data.copy_(bart_state_dict[name[6:]].data)
                    else:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'decoder' + name_split[1]
                        if 'k2' or 'v2' in name:
                            name = name.replace('k2', 'k').replace('v2', 'v')
                        if 'mlp' in name:
                            continue
                        p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_his' in n:
                    name_split = n.split('encoder_attn_his')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    if 'k2' or 'v2' in name:
                        name = name.replace('k2', 'k').replace('v2', 'v')
                    if 'mlp' in name:
                        continue
                    p.data.copy_(bart_state_dict[name[6:]].data)
                else:
                    print(n)
                    continue
            print(model)
            del bart_model

            return model

class HHDKS_Enc_ca2_Dec_ca2_BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = HHDKS_CA2_BartEncoder(config, self.shared)
        self.mlp = nn.Linear(config.d_model, config.d_model, bias=True)
        self.decoder = HHDKS_CA_BartDecoder(config, self.shared)

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

class HHDKS_Enc_ca2_Dec_ca2_BartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = HHDKS_Enc_ca2_Dec_ca2_BartModel(config)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, use_pretrain=True
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
                elif 'encoder_attn_doc' in n:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_z' in n:
                    name_split = n.split('encoder_attn_z')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                elif not use_pretrain:
                    continue
                elif 'encoder2' in n:
                    if 'encoder_attn' not in n:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'encoder' + name_split[1]
                        p.data.copy_(bart_state_dict[name[6:]].data)
                    else:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'decoder' + name_split[1]
                        if 'k2' or 'v2' in name:
                            name = name.replace('k2', 'k').replace('v2', 'v')
                        if 'mlp' in name:
                            continue
                        p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_his' in n:
                    name_split = n.split('encoder_attn_his')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    if 'k2' or 'v2' in name:
                        name = name.replace('k2', 'k').replace('v2', 'v')
                    if 'mlp' in name:
                        continue
                    p.data.copy_(bart_state_dict[name[6:]].data)
                else:
                    print(n)
                    continue
            print(model)
            del bart_model

            return model

class HHDKS_Enc_ca_BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = HHDKS_BartEncoder(config, self.shared)
        self.mlp = nn.Linear(config.d_model, config.d_model, bias=True)
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

class HHDKS_Enc_ca_BartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = HHDKS_Enc_ca_Dec_ca_BartModel(config)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, use_pretrain=True
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
                elif 'encoder_attn_doc' in n:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_z' in n:
                    name_split = n.split('encoder_attn_z')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                elif not use_pretrain:
                    continue
                elif 'encoder2' in n:
                    if 'encoder_attn' not in n:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'encoder' + name_split[1]
                        p.data.copy_(bart_state_dict[name[6:]].data)
                    else:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'decoder' + name_split[1]
                        if 'k2' or 'v2' in name:
                            name = name.replace('k2', 'k').replace('v2', 'v')
                        if 'mlp' in name:
                            continue
                        p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_his' in n:
                    name_split = n.split('encoder_attn_his')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    if 'k2' or 'v2' in name:
                        name = name.replace('k2', 'k').replace('v2', 'v')
                    if 'mlp' in name:
                        continue
                    p.data.copy_(bart_state_dict[name[6:]].data)
                else:
                    print(n)
                    continue
            print(model)
            del bart_model

            return model

class HHDKS_Enc_np_Dec_p_avgemb_BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.mlp = nn.Linear(config.d_model * 2, config.d_model, bias=True)
        self.decoder = HHDKSBartDecoder_p_avgemb(config, self.shared)

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
            his_utt=None,
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
            encoder_attn_mask_doc=attention_mask[1],
            his_utt=his_utt
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

class HHDKS_Enc_np_Dec_p_avgemb_BartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = HHDKS_Enc_np_Dec_p_avgemb_BartModel(config)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, use_pretrain=True
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
                elif 'encoder_attn_doc' in n:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                elif not use_pretrain:
                    continue
                elif 'encoder2' in n:
                    if 'encoder_attn' not in n:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'encoder' + name_split[1]
                        p.data.copy_(bart_state_dict[name[6:]].data)
                    else:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'decoder' + name_split[1]
                        if 'k2' or 'v2' in name:
                            name = name.replace('k2', 'k').replace('v2', 'v')
                        if 'mlp' in name:
                            continue
                        p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_his' in n:
                    name_split = n.split('encoder_attn_his')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    if 'k2' or 'v2' in name:
                        name = name.replace('k2', 'k').replace('v2', 'v')
                    if 'mlp' in name:
                        continue
                    p.data.copy_(bart_state_dict[name[6:]].data)
                else:
                    print(n)
                    continue
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
        his_utt=None,
        return_dict=None,
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

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            his_utt=his_utt,
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

class HHDKS_Enc_np_Dec_dyn_BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig, his_num):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.utt_mlp = nn.Linear(config.d_model, config.d_model, bias=True)
        self.mlp = nn.Linear(config.d_model * 2, config.d_model, bias=True)
        self.decoder = HHDKSBartDecoder_dyn(config, self.shared, his_num)

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

class HHDKS_Enc_np_Dec_dyn_BartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig, his_num):
        super().__init__(config)
        base_model = HHDKS_Enc_np_Dec_dyn_BartModel(config, his_num)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    @classmethod
    def from_pretrained_multi(
            cls, args, model_file_path, use_pretrain=True, his_num=3
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(args.model_name)
        model = cls(bart_config, his_num)

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
                elif 'encoder_attn_doc' in n:
                    name_split = n.split('encoder_attn_doc')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    p.data.copy_(bart_state_dict[name[6:]].data)
                elif not use_pretrain:
                    continue
                elif 'encoder2' in n:
                    if 'encoder_attn' not in n:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'encoder' + name_split[1]
                        p.data.copy_(bart_state_dict[name[6:]].data)
                    else:
                        name_split = n.split('encoder2')
                        name = name_split[0] + 'decoder' + name_split[1]
                        if 'k2' or 'v2' in name:
                            name = name.replace('k2', 'k').replace('v2', 'v')
                        if 'mlp' in name:
                            continue
                        p.data.copy_(bart_state_dict[name[6:]].data)
                elif 'encoder_attn_his' in n:
                    name_split = n.split('encoder_attn_his')
                    name = name_split[0] + 'encoder_attn' + name_split[1]
                    if 'k2' or 'v2' in name:
                        name = name.replace('k2', 'k').replace('v2', 'v')
                    if 'mlp' in name:
                        continue
                    p.data.copy_(bart_state_dict[name[6:]].data)
                else:
                    print(n)
                    continue
            print(model)
            del bart_model

            return model