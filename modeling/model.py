# Copyright (c) 2020, Salesforce.com, Inc.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import absolute_import, division, print_function, unicode_literals



import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel

from loss import FocalLoss


class BertFF(BertPreTrainedModel):
    def __init__(self, config):
        super(BertFF, self).__init__(config)
        self.num_labels = config.num_labels
        print("labels:", self.num_labels)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.label_classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        # run through bert
        bert_outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)

        # label classifier
        pooled_output = bert_outputs[1]
        pooled_output = self.dropout(pooled_output)
        label_logits = self.label_classifier(pooled_output)

        outputs = (label_logits,)+ bert_outputs[2:]

        loss_fct = FocalLoss(2)
        loss = loss_fct(label_logits, labels)

        outputs = (loss,) + outputs

        return outputs
