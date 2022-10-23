import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer
import transformers.trainer
from transformers.trainer import *
from transformers.modeling_outputs import SequenceClassifierOutput

def extractEuphIdx(tokenizer, input):
    """
    input is list of numbers
    """
    start_euph_idx = len(tokenizer)-2
    start_idx = (input==start_euph_idx).nonzero().squeeze()
    end_idx = (input==start_euph_idx+1).nonzero().squeeze()
    euph_idx = [idx for idx in range(start_idx+1, end_idx)]
    return euph_idx

class CLS_Layer(nn.Module):
    def __init__(self, pet_dim, device):
        super(CLS_Layer, self).__init__()
        self.pet_dim = pet_dim
        self.device = device
        self.linear = nn.Linear(pet_dim, 2)

    def forward(self, pooler_output):
        out = self.linear(pooler_output)
        return out

class PET_layer(nn.Module):
    def __init__(self, tokenizer, pet_dim, device):
        super(PET_layer, self).__init__()
        self.tokenizer = tokenizer
        self.pet_dim = pet_dim
        self.device = device
        self.linear1 = nn.Linear(pet_dim, pet_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(pet_dim, 2)

    def forward(self, inputs, input_ids):
        euph_tensor = torch.zeros([inputs.shape[0], inputs.shape[-1]]).to(self.device)
        for i in range(input_ids.shape[0]):
            idxs = extractEuphIdx(self.tokenizer, input_ids[i])
            for j in idxs:
                euph_tensor[i] += inputs[i][j]
        out = self.linear2(self.dropout(self.linear1(euph_tensor)))
        return out

class Sent_DAN_Simple(nn.Module):
    def __init__(self, tokenizer, pet_dim, device):
        super(Sent_DAN_Simple, self).__init__()
        self.tokenizer = tokenizer
        self.pet_dim = pet_dim
        self.device = device
        self.linear1 = nn.Linear(pet_dim, pet_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(pet_dim, 2)

    def forward(self, inputs, input_ids):
        euph_tensor = torch.zeros([inputs.shape[0], inputs.shape[-1]]).to(self.device)
        for i in range(input_ids.shape[0]):
            total_len = 0
            for j in range(input_ids.shape[0]):
                if j != self.tokenizer.pad_token_id:
                    euph_tensor[i] += inputs[i][j]
                    total_len += 1
                euph_tensor[i] /= total_len
        out = self.linear2(self.dropout(self.linear1(euph_tensor)))
        return out

class Sent_DAN(nn.Module):
    # This model (which is actually the canonical version I think) doesn't work too well for some reason, oh well
    def __init__(self, tokenizer, pet_dim, device, dropout_rate=0.2):
        super(Sent_DAN, self).__init__()
        self.tokenizer = tokenizer
        self.pet_dim = pet_dim
        self.device = device
        self.linear1 = self.linear1 = nn.Linear(pet_dim, 300)
        self.dropout_rate = dropout_rate
        self.linear2 = nn.Linear(300, 2)
        self.activation = torch.nn.functional.tanh
    
    def forward(self, inputs, input_ids):
        euph_tensor = torch.zeros([inputs.shape[0], inputs.shape[-1]]).to(self.device)
        
        for i in range(input_ids.shape[0]):
            try:
                end_ind = (input_ids[i] == 1).nonzero()[0][0].item()
            except:
                end_ind = input_ids.shape[0]
            bernoulli_mask = torch.bernoulli((1 - self.dropout_rate) * torch.ones_like(inputs[i][:, :1]))
            inputs_dropped = bernoulli_mask * inputs[i]
            for j in range(end_ind):
                euph_tensor[i] += inputs_dropped[i][j]
            euph_tensor[i] /= end_ind
            
        out = self.linear2(self.activation(self.linear1(euph_tensor)))
        return out


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        last_hidden_state = outputs['last_hidden_state']
        try:
            logits = model.pet(last_hidden_state, inputs['input_ids'])
        except:
            try:
                logits = model.module.pet(last_hidden_state, inputs['input_ids'])
            except:
                logits = model.cls_layer(outputs['pooler_output'])
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, inputs['labels'])
        outputs = SequenceClassifierOutput(loss=loss, logits=logits)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        labels = inputs['labels']
        return (loss, logits, labels)