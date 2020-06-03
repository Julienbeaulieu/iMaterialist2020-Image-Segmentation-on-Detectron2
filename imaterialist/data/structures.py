import torch
from typing import Iterator, List, Tuple, Union
import torch.nn.functional as F

# Base Attribute holder
class Attributes:
    """
    This structure stores a list of attributes as a Nx13 torch.Tensor.
    It behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all attributes)
    """
    
    AttributeSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx14 matrix.  Each row is [attribute_1, attribute_2, ...].
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.int64, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, 295)).to(dtype=torch.int64, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 295, tensor.size()

        self.tensor = tensor


    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        """
        Returns:
            Attributes: Create a new :class:`Attributes` by indexing.
        The following usage are allowed:
        1. `new_attributes = attributes[3]`: return a `Attributes` which contains only one Attribute.
        2. `new_attributes = attributes[2:10]`: return a slice of attributes.
        3. `new_attributes = attributes[vector]`, where vector is a torch.BoolTensor
           with `length = len(attributes)`. Nonzero elements in the vector will be selected.
        Note that the returned Attributes might share storage with this Attributes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Attributes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Attributes with {} failed to return a matrix!".format(item)
        return Attributes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]
    
    def to(self, device: str) -> "Attributes":
        return Attributes(self.tensor.to(device))

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find attributes that are non-empty.
        An attribute is considered empty if its first attribute in the list is 999.
        Returns:
            Tensor:
                a binary vector which represents whether each attribute is empty
                (False) or non-empty (True).
        """
        attributes = self.tensor
        first_attr = attributes[:, 0]
        keep = (first_attr != 999)
        return keep

    def __repr__(self) -> str:
        return "Attributes(" + str(self.tensor) + ")"


    def remove_padding(self, attribute):
        pass

    @classmethod
    def cat(cls, attributes_list: List["Attributes"]) -> "Attributes":
        """
        Concatenates a list of Attributes into a single Attributes
        Arguments:
            Attributes_list (list[Attributes])
        Returns:
            Attributes: the concatenated Attributes
        """
        assert isinstance(attributes_list, (list, tuple))
        if len(attributes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(attribute, Attributes) for attribute in attributes_list)

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_attributes = cls(torch.cat([b.tensor for b in attributes_list], dim=0))
        return cat_attributes
    
    def size(self):
        'required in order to pass loss function assertions'
        return (len(self), 295)

    def numel(self):
        'required in order to pass loss function assertions'
        return len(self)

    @property
    def device(self) -> torch.device:
        return self.tensor.device
 
    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield attributes as a Tensor of shape (14,) at a time.
        """
        yield from self.tensor