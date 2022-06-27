# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from typing import Any, Dict, List, Optional, Tuple

from fairseq.models import FairseqEncoder, register_model, register_model_architecture
from fairseq.models.nat import NATransformerModel, FairseqNATDecoder, base_architecture, ensemble_decoder
from fairseq.modules import DynamicCRF
from fairseq import utils

from fairseq.iterative_refinement_generator import DecoderOut


from fairseq.modules import (
    AdaptiveSoftmax,
    DynamicConv,
    FairseqDropout,
    LayerNorm,
    LightweightConv,
    MultiheadAttention,
    PositionalEmbedding,
)

# The dynamicconvFunction in dynamicconv_layer.py is a method class. Should add staticmethod symbolic
#torch.ops.load_library("/root/workspace/lite_transformer_trt/fairseq/modules/dynamicconv_layer/build/lib.linux-x86_64-3.7/dynamicconv_cuda.cpython-37m-x86_64-linux-gnu.so")
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
@parse_args("v", "v", "i")
def symbolic_dynamicconv_forward(g, x, weights, padding_l):
    #args = [x, weights]
    #kwargs = {"padding_l_i": padding_l}
    #return g.op("DynamicconvSpace::dynamicconv_forward", *args, **kwargs)
    #return g.op("DynamicconvSpace::dynamicconv_forward",x, weights, padding_l)
    return g.op("ai.onnx.contrib::dynamicconv_forward",x, weights, padding_l)
# Register contrib  operation as a counterpart for  xx
register_custom_op_symbolic('DynamicconvSpace::dynamicconv_forward', symbolic_dynamicconv_forward, 11)
'''
'''

@register_model("conv_crf_transformer")
class NACovCrfTransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.crf_layer = DynamicCRF(
            num_embedding=len(self.tgt_dict),
            low_rank=args.crf_lowrank_approx,
            beam_size=args.crf_beam_approx,
        )

    @property
    def allow_ensemble(self):
        return False

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument(
            "--crf-lowrank-approx",
            type=int,
            help="the dimension of low-rank approximation of transition",
        )
        parser.add_argument(
            "--crf-beam-approx",
            type=int,
            help="the beam size for apporixmating the normalizing factor",
        )
        parser.add_argument(
            "--word-ins-loss-factor",
            type=float,
            help="weights on NAT loss used to co-training with CRF loss.",
        )

        # conv arg
        """Add model-specific arguments to the parser."""
       
        parser.add_argument(
            "--encoder-conv-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
    
        parser.add_argument(
            "--decoder-conv-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
       

        """LightConv and DynamicConv arguments"""
        parser.add_argument(
            "--encoder-kernel-size-list",
            nargs='*', default=[3, 7, 15, 31, 31, 31, 31], type=int,
            help='list of kernel size (default: "[3,7,15,31,31,31,31]")',
        )
        parser.add_argument(
            "--decoder-kernel-size-list",
            nargs='*', default=[3, 7, 15, 31, 31, 31, 31], type=int,
            help='list of kernel size (default: "[3,7,15,31,31,31]")',
        )
        parser.add_argument(
            "--encoder-glu", type=utils.eval_bool, help="glu after in proj"
        )
        parser.add_argument(
            "--decoder-glu", type=utils.eval_bool, help="glu after in proj"
        )
        parser.add_argument(
            "--encoder-conv-type",
            default="dynamic",
            type=str,
            choices=["dynamic", "lightweight"],
            help="type of convolution",
        )
        parser.add_argument(
            "--decoder-conv-type",
            default="dynamic",
            type=str,
            choices=["dynamic", "lightweight"],
            help="type of convolution",
        )
        parser.add_argument("--weight-softmax", default=True, type=utils.eval_bool)
        parser.add_argument(
            "--weight-dropout",
            type=float,
            metavar="D",
            help="dropout probability for conv weights",
        )

        parser.add_argument('--conv-linear', default=False, action='store_true')
        parser.add_argument('--share-input-output-embed', default=False, action='store_true', \
            help='share input and output embeddings (requires –decoder-out-embed-dim and –decoder-embed-dim to be equal)')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 1024
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError(
                    "--share-all-embeddings requires a joined dictionary"
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise RuntimeError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = LightConvNATEncoder(args, src_dict, encoder_embed_tokens)
        decoder = LightConvNATDecoder(args, tgt_dict, decoder_embed_tokens)
        return NACovCrfTransformerModel(args, encoder, decoder)
    
    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )
        # onnx转tensorRT engine会出现错误:
        #ArgMax_423: ITopKLayer cannot be used to compute a shape tensor) In node 673 (parseGraph): INVALID_NODE: Invalid Node - Range_429
        # 由于 torch.arange 接收的是一个tensor ，其range把这个tensor作为shape tensor，
        # 并且将这条路径上的tensor都标记成shape tensor，所以前面的layer作检查时报错了。
        # 办法是把导出代码中torch.arange输入 tensor改为 int
        #max_length = length_tgt.clamp_(min=2).max()
        max_length = src_tokens.shape[1]*2 #src_tokens.shape[1]*2 #length_tgt.clamp_(min=2).max().item() 这样值会固定?
        idx_length = utils.new_arange(src_tokens, max_length)
        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )
    
    #just for onnx export 
    def forward(
        self, src_tokens, **kwargs
    ):
        #print(f'src_tokens:{src_tokens} \nsrc_lengths:{src_lengths}')
        #forward_data = {'src_tokens':src_tokens.cpu().detach().numpy()}
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=None, **kwargs)
        decoder_out = self.initialize_output_tokens(encoder_out, src_tokens)
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores

        output_masks = output_tokens.ne(self.pad)
        word_ins_out = self.decoder(
            normalize=False, prev_output_tokens=output_tokens, encoder_out=encoder_out
        )
        #print(f'word_ins_out:{word_ins_out.shape}')
        # run viterbi decoding through CRF
        _scores, _tokens = self.crf_layer.forward_decoder(word_ins_out, output_masks)
        
        #print(f'_scores:{_scores.shape} output_scores:{output_scores.shape}')
        #output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        #output_scores.masked_scatter_(output_masks, _scores[output_masks])
        #output_tokens = _tokens[output_masks]
        #output_scores = _scores[output_masks]
        #print(f'output_tokens:{output_tokens.shape}\noutput_scores:{output_scores.shape}\noutput_scores:{output_scores}')
        #forward_data['output_tokens'] = _tokens.cpu().detach().numpy()
        #forward_data['output_scores'] = _scores.cpu().detach().numpy()
        #shape_ = src_tokens.shape
        
        #file_out = f'lite-cov-{shape_[0]}-{shape_[1]}.npz'
        #np.savez(file_out , **forward_data)
        #return (output_tokens, output_scores)
        return (_tokens, _scores)
        #return  output_scores

    def forward_train(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        
        #print(f'src_tokens:{src_tokens} \nsrc_lengths:{src_lengths}\nprev_output_tokens:{prev_output_tokens} \ntgt_tokens:{tgt_tokens}')
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        word_ins_tgt, word_ins_mask = tgt_tokens, tgt_tokens.ne(self.pad)

        # compute the log-likelihood of CRF
        crf_nll = -self.crf_layer(word_ins_out, word_ins_tgt, word_ins_mask)
        crf_nll = (crf_nll / word_ins_mask.type_as(crf_nll).sum(-1)).mean()

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": word_ins_tgt,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": self.args.word_ins_loss_factor,
            },
            "word_crf": {"loss": crf_nll},
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }
        
    
    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        #print(f'forward_decoder:------>:{output_scores}')
        # execute the decoder and get emission scores
        output_masks = output_tokens.ne(self.pad)
        word_ins_out = self.decoder(
            normalize=False, prev_output_tokens=output_tokens, encoder_out=encoder_out
        )

        # run viterbi decoding through CRF
        _scores, _tokens = self.crf_layer.forward_decoder(word_ins_out, output_masks)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

class LightConvNATEncoder(FairseqEncoder):
    """
    LightConv encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`LightConvNATEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        #for i in range(args.encoder_layers):
        #    print(f"encoder_kernel_size_list --- > {args.encoder_kernel_size_list[i]}")
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                LightConvNATEncoderLayer(
                    args, kernel_size=args.encoder_kernel_size_list[i]
                )
                for i in range(args.encoder_layers)
            ]
        )
        self.register_buffer("version", torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim,export=True)

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        #if self.layernorm_embedding is not None:
        #    x = self.dropout_module(x)
        #if self.quant_noise is not None:
        #    x = self.quant_noise(x)
        return x, embed

    def forward(self, src_tokens, 
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        #x = self.embed_scale * self.embed_tokens(src_tokens)
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        x = self.dropout_module(x)
        encoder_pos = self.embed_positions(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states = []
        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_pos": [encoder_pos],
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }
    
    '''
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out
    '''
    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_pos"]) == 0:
            new_encoder_pos = []
        else:
            new_encoder_pos = [
                encoder_out["encoder_pos"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_pos": new_encoder_pos,
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)


class LightConvNATEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        kernel_size: kernel size of the convolution
    """

    def __init__(self, args, kernel_size=0):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.conv_dim = args.encoder_conv_dim
        padding_l = (
            kernel_size // 2
            if kernel_size % 2 == 1
            else ((kernel_size - 1) // 2, kernel_size // 2)
        )
        #padding_l=kernel_size-1

        if args.encoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None
        if args.encoder_conv_type == "lightweight":
            self.conv = LightweightConv(
                self.conv_dim,
                kernel_size,
                padding_l=padding_l,
                weight_softmax=args.weight_softmax,
                num_heads=args.encoder_attention_heads,
                weight_dropout=args.weight_dropout, with_linear=args.conv_linear,
            )
        elif args.encoder_conv_type == "dynamic":
            self.conv = DynamicConv(
                self.conv_dim,
                kernel_size,
                padding_l=padding_l,
                weight_softmax=args.weight_softmax,
                num_heads=args.encoder_attention_heads,
                weight_dropout=args.weight_dropout, with_linear=args.conv_linear,
                glu=args.encoder_glu,
            )
        else:
            raise NotImplementedError
        self.linear2 = Linear(self.conv_dim, self.embed_dim)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.relu_dropout_module = FairseqDropout(
            args.relu_dropout, module_name=self.__class__.__name__
        )
        self.input_dropout_module = FairseqDropout(
            args.input_dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim,export=True) for _ in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x = self.input_dropout_module(x)
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
        x = self.conv(x)
        x = self.linear2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = self.relu_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    def extra_repr(self):
        return (
            "dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}".format(
                self.dropout_module.p,
                self.relu_dropout_module.p,
                self.input_dropout_module.p,
                self.normalize_before,
            )
        )

# light conv decoder
def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t

class LightConvNATDecoder(FairseqNATDecoder):
    """
    LightConv decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`LightConvNATDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__( self, args, dictionary, embed_tokens, no_encoder_attn=False, final_norm=True ):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        
        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.embed_length = Embedding(256, self.encoder_embed_dim, None)
        #self.embed_length = Embedding(getattr(args, "max_target_positions", 256), self.encoder_embed_dim, None)
        if self.src_embedding_copy:
            self.copy_attn = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
        # add decoder layer @Daniel
        self.layers = nn.ModuleList([])
        self.layers.extend([
            LightConvNATDecoderLayer(args, no_encoder_attn, kernel_size=args.decoder_kernel_size_list[i])
            for i in range(args.decoder_layers)
        ])

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, **unused):
        #print(f'))))))))))))))self.dictionary:{len(self.dictionary )} encoder_out:{encoder_out}')
        #print(f'encoder_out:{encoder_out["encoder_out"][0].shape}')
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )
        #print(f'embedding_copy:{embedding_copy}\nprev_output_tokens:{prev_output_tokens.shape}\nself.embed_positions:{self.embed_positions}\n')
        # embedding
        if embedding_copy:
            src_embd = encoder_out["encoder_embedding"][0]
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_mask = encoder_out["encoder_padding_mask"][0]
            else:
                src_mask = None
            bsz, seq_len = prev_output_tokens.size()
            #print(f'positions:{positions.shape} src_embd:{src_embd.shape} \n encoder_out[encoder_pos][0]:{encoder_out["encoder_pos"][0].shape}')
            #print(f'attn_score self.copy_attn(positions):{self.copy_attn(positions).shape} \
            #    (src_embd + encoder_out[\'encoder_pos\'][0]:{(src_embd + encoder_out["encoder_pos"][0]).shape}')
            attn_score = torch.bmm(self.copy_attn(positions),
                                   (src_embd + encoder_out['encoder_pos'][0]).transpose(1, 2))
            #print(f'bsz:{bsz} seq len:{seq_len} \nattn_score:{attn_score.shape}')
            if src_mask is not None:
                attn_score = attn_score.masked_fill(src_mask.unsqueeze(1).expand(-1, seq_len, -1), float('-inf'))
            attn_weight = F.softmax(attn_score, dim=-1)
            x = torch.bmm(attn_weight, src_embd)
            #print(f'attn_weight:{attn_weight.shape} src_embd:{src_embd.shape} x:{x.shape}')
            mask_target_x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)
            #print(f'mask_target_x:{mask_target_x.shape} decoder_padding_mask:{decoder_padding_mask.shape}')            
            output_mask = prev_output_tokens.eq(self.unk)
            cat_x = torch.cat([mask_target_x.unsqueeze(2), x.unsqueeze(2)], dim=2).view(-1, x.size(2))
            x = cat_x.index_select(dim=0, index=torch.arange(bsz * seq_len).cuda() * 2 +
                                                output_mask.view(-1).long()).reshape(bsz, seq_len, x.size(2))
        else:

            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        positions = positions.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            if positions is not None:
                x += positions
            x = self.dropout_module(x)

            #x, attn, _ = layer(
            x, attn = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed tokens
        if states is None:
            x = self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            #max_len_res = length_out.max(-1)
            #print(f'length_out:{length_out.shape} {type(max_len_res)} max:{max_len_res}')
            #pred_lengs = max_len_res[1]
            pred_lengs =  length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]


class LightConvNATDecoderLayer(nn.Module):
    """Decoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        kernel_size: kernel size of the convolution
    """

    def __init__(self, args, no_encoder_attn=False, kernel_size=0):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.conv_dim = args.decoder_conv_dim
        if args.decoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None

        if args.decoder_conv_type == "lightweight":
            self.conv = LightweightConv(
                self.conv_dim,
                kernel_size,
                padding_l=kernel_size - 1,
                weight_softmax=args.weight_softmax,
                num_heads=args.decoder_attention_heads,
                weight_dropout=args.weight_dropout, with_linear=args.conv_linear,
            )
        elif args.decoder_conv_type == "dynamic":
            self.conv = DynamicConv(
                self.conv_dim,
                kernel_size,
                padding_l=kernel_size - 1,
                weight_softmax=args.weight_softmax,
                num_heads=args.decoder_attention_heads,
                weight_dropout=args.weight_dropout, glu=args.decoder_glu, with_linear=args.conv_linear,
            )
        else:
            raise NotImplementedError
        self.linear2 = Linear(self.conv_dim, self.embed_dim)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.relu_dropout_module = FairseqDropout(
            args.relu_dropout, module_name=self.__class__.__name__
        )
        self.input_dropout_module = FairseqDropout(
            args.input_dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        self.conv_layer_norm = LayerNorm(self.embed_dim,export=True)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim,export=True)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim,export=True)
        self.need_attn = True

    def forward(
        self,
        x,
        encoder_out,
        encoder_padding_mask,
        incremental_state = None,
        prev_conv_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        conv_mask=None,
        conv_padding_mask=None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.conv_layer_norm, x, before=True)
        if prev_conv_state is not None:
            if incremental_state is None:
                incremental_state = {}
            self.conv._set_input_buffer(incremental_state, prev_conv_state)
        x = self.input_dropout_module(x)
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)
        x = self.conv(x, incremental_state=incremental_state)
        x = self.linear2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(self.conv_layer_norm, x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = self.dropout_module(x)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = self.relu_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def extra_repr(self):
        return (
            "dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}".format(
                self.dropout_module.p,
                self.relu_dropout_module.p,
                self.input_dropout_module.p,
                self.normalize_before,
            )
        )

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

@register_model_architecture("conv_crf_transformer", "conv_crf_transformer")
def nacrf_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 7)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.encoder_conv_dim = getattr(args, "encoder_conv_dim", args.encoder_embed_dim)
    args.decoder_conv_dim = getattr(args, "decoder_conv_dim", args.decoder_embed_dim)

    args.encoder_kernel_size_list = getattr(
        args, "encoder_kernel_size_list", [3, 7, 15, 31, 31, 31, 31]
    )
    args.decoder_kernel_size_list = getattr(
        args, "decoder_kernel_size_list", [3, 7, 15, 31, 31, 31]
    )
    if len(args.encoder_kernel_size_list) == 1:
        args.encoder_kernel_size_list = (
            args.encoder_kernel_size_list * args.encoder_layers
        )
    if len(args.decoder_kernel_size_list) == 1:
        args.decoder_kernel_size_list = (
            args.decoder_kernel_size_list * args.decoder_layers
        )
    assert (
        len(args.encoder_kernel_size_list) == args.encoder_layers
    ), "encoder_kernel_size_list doesn't match encoder_layers"
    assert (
        len(args.decoder_kernel_size_list) == args.decoder_layers
    ), "decoder_kernel_size_list doesn't match decoder_layers"
    args.encoder_glu = getattr(args, "encoder_glu", True)
    args.decoder_glu = getattr(args, "decoder_glu", True)
    args.input_dropout = getattr(args, "input_dropout", 0.1)
    args.weight_dropout = getattr(args, "weight_dropout", args.attention_dropout)
    
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)

    args.crf_lowrank_approx = getattr(args, "crf_lowrank_approx", 32)
    args.crf_beam_approx = getattr(args, "crf_beam_approx", 64)
    args.word_ins_loss_factor = getattr(args, "word_ins_loss_factor", 0.5)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    base_architecture(args)