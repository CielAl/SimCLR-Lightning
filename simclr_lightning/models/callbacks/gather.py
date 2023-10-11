from typing import Optional, Dict, List, Tuple, Union, Any, Sequence, Callable
import os
import torch
import warnings
import pytorch_lightning as L
from pytorch_lightning.strategies import Strategy
from torch.distributed import group as dist_group
from lightning_fabric.utilities.apply_func import convert_to_tensors
from lightning_utilities.core.apply_func import apply_to_collection
import pickle
# import operator

DEFAULT_REDUCE_OP = list.__add__  # operator.add


class AllGatherWriter(L.callbacks.BasePredictionWriter):
    """Collect the batch output and export to file in validation and testing.

    The leading dimension of all output data must be equal to batch size (beware batch of only one element)
    """
    export_dir: Optional[str]
    __prediction: Dict[str, Any]
    _trim_size: int
    _keep_in_memory: bool
    _reduce_ops: Optional[Callable]

    @property
    def prediction(self):
        return self.__prediction

    @staticmethod
    def path_invalid(export_dir):
        """Check if export_dir is not a str. Only as simple sanitization.

        Args:
            export_dir:

        Returns:

        """
        return export_dir is None or not isinstance(export_dir, str)

    def _init_export_dir(self):
        """Create the export folder after validation. Raise a warning if not a valid str.

        Returns:

        """
        if AllGatherWriter.path_invalid(self.export_dir):
            warnings.warn(f"export_dir is not set - ignore output")
            return
        os.makedirs(self.export_dir, exist_ok=True)

    def __init__(self, export_dir: Optional[str] = None, trim_size: Optional[int] = None,
                 keep_in_memory: bool = True,
                 reduce_ops: Optional[Callable] = DEFAULT_REDUCE_OP):
        """Callbacks to gather the prediction output with multi-gpu support.

        Support file exportation or retain the results in memory.

        Args:
            export_dir: Directory of output.
            trim_size: Cut off the prediction after the defined trim_size. No cutoff if set to None.
            keep_in_memory: Whether to retain the results in the `prediction` attribute
            reduce_ops: what operations to reduce the output collected from each device after gathering. By default
                use list concatenation. The ops must be a callable of reduce_ops(*args) wherein each element in args
                is the result collected from one device.
        """
        super().__init__(write_interval='epoch')
        self.export_dir = export_dir
        self._init_export_dir()
        self.__prediction = dict()
        self._trim_size = trim_size
        self._keep_in_memory = keep_in_memory
        self._reduce_ops = reduce_ops

    @staticmethod
    def export_data(data, export_dir: Optional[str], fname: str):
        """Export the results to file with given fname if the export_dir is set.

        Args:
            data:
            export_dir:
            fname:

        Returns:

        """
        if AllGatherWriter.path_invalid(export_dir):
            return
        dest = os.path.join(export_dir, fname)
        torch.save(data, dest)

    @staticmethod
    def stage_name(trainer):
        """Name of current trainer stage in lower cases.

        Args:
            trainer:

        Returns:

        """
        return trainer.state.stage.name.lower()

    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        prediction,
        batch_indices,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Callbacks to override in BasePredictionWriter. Defines how to export batch-level output.

        Write the batch-level output of each device. Specify the dataloader_idx and batch_idx as well as the
        rank of device.

        Args:
            trainer:
            pl_module:
            prediction:
            batch_indices:
            batch:
            batch_idx:
            dataloader_idx:

        Returns:

        """
        stage_name = AllGatherWriter.stage_name(trainer)
        AllGatherWriter.export_data(prediction,
                                    self.export_dir, fname=f"{stage_name}_batch_{trainer.global_rank}"
                                                           f"_{dataloader_idx}_{batch_idx}.pkl")

    @staticmethod
    def _not_empty_single_batch(data: Union[Dict, Sequence, torch.Tensor]):
        """Identify whether the current batch is empty.

        For tensor output, check its nelement - nonzero?.
        For Sequence, check its length - nonzero?.
        For Dict, recursively check all values.

        Args:
            data: data to inspect.

        Returns:
            True if nothing is empty in the data.
        """
        if isinstance(data, torch.Tensor):
            return data.nelement() > 0
        if isinstance(data, Sequence):
            return len(data) > 0
        data_length = [AllGatherWriter._not_empty_single_batch(x) for x in data.values()]
        return all(data_length)

    @staticmethod
    def _pad_helper(prediction: torch.Tensor, strategy: Strategy,
                    group: Optional[Any] = None, sync_grads: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad the tensor by the max length of corresponding value across all devices.

        This is mandatory as all_gather only supports tensor with same shape. Therefore if the device has tensor of
        different shape, e.g., the last non-full batch, padding is needed.

        Args:
            prediction: tensor to pad
            strategy: Current DDP strategy. See `Lightning Fabric`
            group: See `all_gather`
            sync_grads: See `all_gather`

        Returns:
            padded prediction and the corresponding list of original shapes (as tensor).
        """
        shape_tensor = torch.tensor(prediction.shape, device=prediction.device)

        # if world size 1 --> will be flattened. Use atleast_2d to add the leading dimension
        shape_gathered = strategy.all_gather(shape_tensor, group=group, sync_grads=sync_grads)
        strategy.barrier()
        shape_gathered = torch.atleast_2d(shape_gathered)
        # padding result part tensor to max length
        shape_max = shape_gathered[:, 0].max()
        pad_shape = shape_gathered.detach().clone()[0]
        pad_shape[0] = shape_max
        pred_padded = torch.zeros(*pad_shape, dtype=prediction.dtype, device=prediction.device)
        pred_padded[:prediction.shape[0]] = prediction

        return pred_padded, shape_gathered  # , pad_shape, shape_max

    @staticmethod
    def _data_unpad(pred_gathered: torch.Tensor, shape_gathered: torch.Tensor):
        """Use the original shape to retrieve the original results from padded tensor.

        Args:
            pred_gathered:
            shape_gathered:

        Returns:

        """
        part_list = []
        for recv, shape in zip(pred_gathered, shape_gathered):
            part_list.append(recv[:shape[0]])
        return part_list

    @staticmethod
    def _bytes_data(data: Any):
        """Serialize arbitrary objects to bytes stream in form of Byte Tensor (uint8).

        This is for convenience if the output contains python object that is not supported by torch.Tensor (e.g., str),
        but can be pickled.

        Args:
            data: data to serialize.

        Returns:
            Corresponding Byte Tensor.
        """
        return torch.tensor(bytearray(pickle.dumps(data)), dtype=torch.uint8, device='cuda')

    @staticmethod
    def _deserialized_from_bytes(chunk: torch.Tensor):
        return pickle.loads(chunk.cpu().numpy().tobytes())

    @staticmethod
    def data_preprocess(data, serialize_flag: bool):
        if not serialize_flag:
            assert isinstance(data, torch.Tensor)
            return data
        return AllGatherWriter._bytes_data(data)

    @staticmethod
    def data_postprocess(part_list: List, serialize_flag: bool) -> List:
        if not serialize_flag:
            return part_list
        return [AllGatherWriter._deserialized_from_bytes(chunk) for chunk in part_list]

    @staticmethod
    def data_reduce(part_list: List, reduce_ops: Optional[Callable] = None):
        if reduce_ops is None:
            return part_list
        assert isinstance(reduce_ops, Callable)
        return reduce_ops(*part_list)

    # noinspection PyUnusedLocal
    @staticmethod
    def _collect_results(prediction: torch.Tensor,
                         global_rank: int,
                         strategy: Strategy,
                         serialize_flag: bool,
                         reduce_ops: Optional[Callable] = None,
                         size: Optional[int] = None,
                         group: Optional[Any] = None, sync_grads: bool = False):
        """ Gather the results from all gpus

        Derived from https://github.com/open-mmlab/
        mmdetection/blob/482f60fe55c364e50e4fc4b50893a25d8cc261b0/mmdet/apis/test.py#L160

        Args:
            prediction: Prediction results to collect. If not tensor, then `serialize_flag` should be set to serialize
                pickled prediction as a stream of byte tensors.
            global_rank: global_rank of current device.
            strategy: strategy of trainer
            size: trim size
            group: see `all_gather`
            sync_grads: see `all_gather`
        Returns:
            list of batch-level outputs from each device, ordered by rank.
        """
        # , pad_shape, shape_max
        all_gather = strategy.all_gather
        # in case only one gpu is used
        prediction = AllGatherWriter.data_preprocess(prediction, serialize_flag)
        pred_padded, shape_gathered = AllGatherWriter._pad_helper(prediction, strategy, group, sync_grads)

        # gather all result part
        # world x pred_shape

        pred_gathered = all_gather(pred_padded, group=group, sync_grads=sync_grads)
        strategy.barrier()
        part_list = AllGatherWriter._data_unpad(pred_gathered, shape_gathered)
        part_list = AllGatherWriter.data_postprocess(part_list, serialize_flag)

        strategy.barrier()
        if global_rank == 0:
            ordered_results = AllGatherWriter.data_reduce(part_list, reduce_ops)
            ordered_results = ordered_results[:size]
        else:
            ordered_results = []
        return ordered_results

    # @staticmethod
    # def all_gather_list_helper(trainer: L.Trainer, pl_module: L.LightningModule,
    #                            data: Union[torch.Tensor, Dict, List, Tuple],
    #                            group: Optional[Any] = None, sync_grads: bool = False,
    #                            size: Optional[int] = None):
    #     trainer.strategy.barrier()
    #     if not isinstance(data, Sequence):
    #         return apply_to_collection(data, Sequence, AllGatherWriter._collect_results,
    #                                    global_rank=trainer.global_rank,
    #                                    strategy=trainer.strategy,
    #                                    serialize_flag=True,
    #                                    size=size,
    #                                    group=group, sync_grads=sync_grads)
    #     # if x is sequence - is it necessary here? can we pickle/unpickle the entire prediction list?
    #     return [apply_to_collection(x, Sequence, AllGatherWriter._collect_results,
    #                                 global_rank=trainer.global_rank,
    #                                 strategy=trainer.strategy,
    #                                 serialize_flag=True,
    #                                 size=size,
    #                                 group=group, sync_grads=sync_grads) for x in data]

    @staticmethod
    def all_gather_var_length(trainer: L.Trainer, pl_module: L.LightningModule,
                              data: Union[torch.Tensor, Dict, List, Tuple],
                              reduce_ops: Optional[Callable] = None,
                              group: Optional[Any] = None, sync_grads: bool = False,
                              size: Optional[int] = None):
        """Invoke the all_gather of specified strategy (e.g., DDP) but deal with data of various size.

        Serialization as byte tensor and padding.

        Args:
            trainer:
            pl_module:
            data:
            reduce_ops:
            group:
            sync_grads:
            size:

        Returns:

        """
        group = group if group is not None else dist_group.WORLD
        data = convert_to_tensors(data, pl_module.device)

        # 1d array edge case (e.g., if using `torch.atleast_2d` it will be 1xBatch).
        # Potential edge case: batch of single element
        # gather tensor first
        data_tensor_gathered = apply_to_collection(data, List, AllGatherWriter._collect_results,
                                                   global_rank=trainer.global_rank,
                                                   strategy=trainer.strategy,
                                                   serialize_flag=True,
                                                   reduce_ops=reduce_ops,
                                                   size=size,
                                                   group=group, sync_grads=sync_grads)
        return data_tensor_gathered

    def write_to_dict(self, key, predictions_all):
        """write to the `prediction` attribute if _keep_in_memory is set.

        Args:
            key:
            predictions_all:

        Returns:

        """
        if self._keep_in_memory:
            self.__prediction[key] = predictions_all

    def write_on_epoch_end(self, trainer: L.Trainer,
                           pl_module: L.LightningModule,
                           predictions: List[Any],
                           batch_indices: Optional[Sequence[Any]]):
        """Write the prediction result after the end of epoch.

        Note that it assumes the memory consumption of the epoch-level output is manageable. Otherwise, it could be
        better to save the batch-level results of all ranks and corresponding batch indices for merge later.

        Args:
            trainer:
            pl_module:
            predictions:
            batch_indices:

        Returns:

        """
        # training predicting etc.
        stage_name = AllGatherWriter.stage_name(trainer)
        # predictions: List with length N-batches of current rank
        predictions_non_empty = [x for x in predictions if AllGatherWriter._not_empty_single_batch(x)]
        AllGatherWriter.export_data(predictions_non_empty,
                                    self.export_dir,
                                    fname=f"{stage_name}_parts_{trainer.global_rank}.pkl")

        # list of batch out
        predictions_all = AllGatherWriter.all_gather_var_length(trainer, pl_module,
                                                                predictions_non_empty,
                                                                reduce_ops=self._reduce_ops,
                                                                size=self._trim_size)
        trainer.strategy.barrier()
        if trainer.global_rank == 0:
            AllGatherWriter.export_data(predictions_all, self.export_dir, f"{stage_name}_out.pkl")
            self.write_to_dict(stage_name, predictions_all)
