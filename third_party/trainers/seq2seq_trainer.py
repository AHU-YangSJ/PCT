import copy
import os
from pathlib import Path

# from collections import OrderedDict

from packaging import version
import logging
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union, OrderedDict

from torch.cuda import amp
from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer #, Adafactor, AdamW, WEIGHTS_NAME, is_torch_tpu_available
# from transformers.deepspeed import is_deepspeed_zero3_enabled
# from transformers.modeling_utils import IS_SAGEMAKER_MP_POST_1_10
# from transformers.trainer_pt_utils import get_parameter_names
# from transformers.trainer_utils import ShardedDDPOption
# from transformers.utils import is_sagemaker_mp_enabled

from .trainer import BaseTrainer

from .trainer import BaseTrainer

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


class Seq2SeqTrainer(Seq2SeqTrainer, BaseTrainer):
    def __init__(self, train_dataset_sizes=None, shared=False, multiple_metrics=None, adapter_config=None,data_info=None,
                 # evaluation_metrics=None, multi_task_compute_metrics=None,
                 model_args=None, test_key=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_key = test_key
        self.adapter_config = adapter_config
        self.multiple_metrics = multiple_metrics
        self.train_dataset_sizes = train_dataset_sizes
        self.shared = shared
        self.model_args = model_args

        # self.evaluation_metrics = evaluation_metrics
        # self.multi_task_compute_metrics = multi_task_compute_metrics
        # self.data_info = data_info

        # self.best_metrics = OrderedDict({
        #     "best_epoch": 0,
        #     f"best_eval_{self.test_key}": 0,
        # })

        # """
        if self.model_args.prompt_transfer == 2:
            self.source_model = copy.deepcopy(self.model)  # 初始化模型

            model_dict = self.source_model.state_dict()  # model_dict保存初始化模型参数
            source_dict = torch.load(self.model_args.source_prompt, map_location='cuda')  # source_dict保存 源模型
            # if self.model_args.beta==0.1:
            #     initialized_dict = {k: v for k, v in source_dict.items() if (k in model_dict) and ('classifier' not in k)}
            # else:
            #     initialized_dict = {k: v for k, v in source_dict.items() if k in model_dict}  # initialized_dict 字典保存 source_dict保存 源模型参数

            self.source_model.prefix_shared.data = source_dict.data

            # model_dict.update(initialized_dict)  # model_dict更新参数
            # self.source_model.load_state_dict(model_dict)
            self.source_model = self._wrap_model(self.source_model)
            self.source_model.zero_grad()
            print()
            # for param in self.source_model.parameters():    # 原本是屏蔽的
            #     param.requires_grad = False

    """
    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def get_beta(self):
        warmup_epoch=int(self.state.num_train_epochs*0.1)
        other_epoch=self.state.num_train_epochs-warmup_epoch
        if self.state.epoch < warmup_epoch:
            beta=self.model_args.beta*(self.state.epoch/warmup_epoch)
        else:
            now_epoch=self.state.epoch-warmup_epoch
            beta=self.model_args.beta*(1-now_epoch/other_epoch)
        return beta

    def get_beta2(self):
        warmup_epoch=int(self.state.num_train_epochs*0.1)
        other_epoch=self.state.num_train_epochs-warmup_epoch
        if self.state.epoch < warmup_epoch:
            beta=self.model_args.beta
        else:
            now_epoch=self.state.epoch-warmup_epoch
            beta=self.model_args.beta*(1-now_epoch/other_epoch)
        return beta

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.model_args.prompt_transfer == 2:  ##### PANDA #####
            self.source_model.train()
            loss, loss_mse = self.compute_loss(model, inputs)
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.model_args.prompt_transfer == 2:  ##### PANDA #####
            beta = self.get_beta2()
            # if self.state.epoch > int(self.state.num_train_epochs*0.0):
            loss = loss + self.model_args.beta * loss_mse * 0.1   # 0.1 *
            # loss = loss+beta*loss_mse

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        if self.model_args.prompt_transfer == 2:
            self.source_model.zero_grad()

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # model_dict_student = self.source_model.state_dict().copy()
        # model_dict_teacher = model.state_dict().copy()
        # model_dict_student = {k: v for k, v in model_dict_student.items()}
        # model_dict_teacher = {k: v for k, v in model_dict_teacher.items()}
        # model_dict = {}
        # for k in model_dict_student.keys():
        #     parameters = 0.99 * model_dict_teacher.get(k) + 0.01 * model_dict_student.get(k)
        #     model_dict[k] = parameters
        # model.load_state_dict(model_dict)

        outputs = model(**inputs)
        if self.model_args.prompt_transfer == 2:        ##### PANDA #####
            outputs2= self.source_model(**inputs)
            logit, logit2=outputs["logits"], outputs2["logits"]
            # logit.requires_grad_()
            # logit2.requires_grad_()
            loss_mse=torch.nn.functional.mse_loss(logit, logit2)/logit.shape[0]   ## 1:/logit.shape[0], 2:/

            # logit=outputs["logits"]
            # hidden1, hidden2=outputs["hidden_states"], outputs2["hidden_states"]
            # hidden1.requires_grad_()
            # hidden2.requires_grad_()
            # loss_mse=torch.nn.functional.mse_loss(hidden1,hidden2)/logit.shape[0]

            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
                # loss2 = self.label_smoother(outputs, labels)
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                # loss2 = outputs2["loss"] if isinstance(outputs2, dict) else outputs2[0]

            return (loss, outputs) if return_outputs else (loss, loss_mse)


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    """


    def evaluate(
        self,
        eval_dataset: Optional[Dict[str, Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length
        self._num_beams = num_beams,
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "task": inputs["task"] if "task" in inputs else "all"
        }

        # gen_kwargs['num_beams'] = self.model.config.num_beams
        # self.use_amp = None
        # gen_kwargs['generation_config'] = self.model.generation_config


        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(
                        outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(
                labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    """
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "task": inputs["task"] if "task" in inputs else "all"
        }

        # #### lora 时解锁
        gen_kwargs['num_beams'] = self.model.config.num_beams
        # self.use_amp = None

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(
                        outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(
                labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)
    """

    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     """
    #     Will save the model, so you can reload it using `from_pretrained()`.
    #
    #     Will only save from the main process.
    #     """
    #
    #     if output_dir is None:
    #         output_dir = self.args.output_dir
    #
    #     if is_torch_tpu_available():
    #         self._save_tpu(output_dir)
    #     elif is_sagemaker_mp_enabled():
    #         # Calling the state_dict needs to be done on the wrapped model and on all processes.
    #         os.makedirs(output_dir, exist_ok=True)
    #         state_dict = self.model_wrapped.state_dict()
    #         if self.args.should_save:
    #             self._save(output_dir, state_dict=state_dict)
    #         if IS_SAGEMAKER_MP_POST_1_10:
    #             # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
    #             Path(os.path.join(output_dir, "user_content.pt")).touch()
    #     elif (
    #         # ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
    #         # or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
    #         # or
    #         self.fsdp is not None
    #         or self.is_fsdp_enabled
    #     ):
    #         if self.is_fsdp_enabled:
    #             self.accelerator.state.fsdp_plugin.save_model(self.accelerator, self.model, output_dir)
    #         else:
    #             state_dict = self.model.state_dict()
    #
    #             if self.args.should_save:
    #                 self._save(output_dir, state_dict=state_dict)
    #     elif self.is_deepspeed_enabled:
    #         # this takes care of everything as long as we aren't under zero3
    #         if self.args.should_save:
    #             self._save(output_dir)
    #
    #         if is_deepspeed_zero3_enabled():
    #             # It's too complicated to try to override different places where the weights dump gets
    #             # saved, so since under zero3 the file is bogus, simply delete it. The user should
    #             # either user deepspeed checkpoint to resume or to recover full weights use
    #             # zero_to_fp32.py stored in the checkpoint.
    #             if self.args.should_save:
    #                 file = os.path.join(output_dir, WEIGHTS_NAME)
    #                 if os.path.isfile(file):
    #                     # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
    #                     os.remove(file)
    #
    #             # now save the real model if stage3_gather_16bit_weights_on_model_save=True
    #             # if false it will not be saved.
    #             # This must be called on all ranks
    #             if not self.model_wrapped.save_16bit_model(output_dir, WEIGHTS_NAME):
    #                 logger.warning(
    #                     "deepspeed.save_16bit_model didn't save the model, since"
    #                     " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
    #                     " zero_to_fp32.py to recover weights"
    #                 )
    #                 self.model_wrapped.save_checkpoint(output_dir)
    #
    #     elif self.args.should_save:
    #         if self.model.config.prefix_tuning:
    #             my_params_list = ['prefix_shared','mul_prefix_emb', 'tasks_mul_prefix', 'shared_info']
    #             state_dict = self.model.state_dict()
    #             for k, v in self.model.state_dict().items():
    #                 if k not in my_params_list :
    #                     state_dict.pop(k)
    #             self._save(output_dir, state_dict=state_dict)
    #         else:
    #             self._save(output_dir)
    #     # Push to the Hub when `save_model` is called by the user.
    #     if self.args.push_to_hub and not _internal_call:
    #         self.push_to_hub(commit_message="Model save")


