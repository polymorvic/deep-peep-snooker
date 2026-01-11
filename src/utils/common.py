from abc import ABC, abstractmethod
from collections.abc import Hashable as SupportsHash
import json
from pathlib import Path
from typing import Any

import numpy as np

type array_like = np.ndarray | NumpyImage

class NumpyImage(np.ndarray):
    """
    A lightweight wrapper around `numpy.ndarray` for easier image shape handling.

    Provides convenient properties to access image dimensions:
    - `height`: number of rows
    - `width`: number of columns
    - `depth`: number of channels (defaults to 1 if not present)

    Fully compatible with OpenCV and other libraries that expect a standard
    NumPy array, since it is implemented as a view of the original array.
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
    
    @property
    def width(self):
        return self.shape[1] if len(self.shape) > 1 else 1
    
    @property
    def height(self):
        return self.shape[0]
    
    @property
    def depth(self):
        return self.shape[2] if len(self.shape) > 2 else 1
    
    def as_array(self):
        """Convert back to regular numpy array for compatibility"""
        return np.asarray(self)


class Hashable(ABC):
    @abstractmethod
    def _key_(self) -> SupportsHash:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self._key_())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._key_() == other._key_()


class Annotation(ABC):
    """Base class for all annotation types.
    
    Provides common functionality for loading, processing, and saving annotations.
    Subclasses must implement the `clean_annotations` property to define how
    raw annotation data is transformed into a cleaned format.
    
    Attributes:
        root_dir: Directory containing annotation files.
        raw_annotations: Raw annotation data loaded from files (list of dicts or None).
        cleaned_annotations: Processed annotation data (list of dicts or None).
    """

    def __init__(self, root_dir: Path) -> None:
        """Initialize annotation handler.
        
        Args:
            root_dir: Path to directory containing annotation files.
        """
        self.root_dir: Path = Path(root_dir)
        self.raw_annotations: list[dict[str, Any]] | None = None
        self.cleaned_annotations: list[dict[str, Any]] | None = None

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        """Get a cleaned annotation by index.
        
        Args:
            index: Index of the annotation to retrieve.
            
        Returns:
            Dictionary containing the annotation data, or None if not available.
        """
        if self.cleaned_annotations is None:
            return None
        return self.cleaned_annotations[index]

    def __len__(self) -> int:
        """Return the number of raw annotations.
        
        Returns:
            Number of raw annotations, or 0 if none loaded.
        """
        return len(self.raw_annotations) if self.raw_annotations is not None else 0

    def save(self, file_path: Path) -> None:
        """Save cleaned annotations to a JSON file if it doesn't already exist.
        
        Args:
            file_path: Path where the annotations should be saved.
            
        Note:
            If the file already exists, the save operation is skipped and a message
            is printed. Parent directories are created automatically if needed.
        """
        if self.cleaned_annotations is None:
            print(f"No cleaned annotations to save.")
            return
        if file_path.exists():
            print(f"File already exists: {file_path}. Skipping save.")
            return
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.cleaned_annotations, f, indent=4)
        print(f"Saved {len(self.cleaned_annotations)} annotations to {file_path}")

    def concat_files(self, extension: str = 'json') -> list[dict[str, Any]]:
        """Concatenate all annotation files from the root directory.
        
        Args:
            extension: File extension to search for (default: 'json').
            
        Returns:
            List of all annotation dictionaries from all matching files.
            
        Note:
            Loads and combines all annotation files found in root_dir with the
            specified extension. Sets self.raw_annotations to the combined result.
        """
        ground_truth_dir = self.root_dir
        all_ground_truth: list[dict[str, Any]] = []

        for json_file in sorted(ground_truth_dir.glob(f'*.{extension}')):
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_ground_truth.extend(data)  
        self.raw_annotations = all_ground_truth
        return all_ground_truth

    @property
    @abstractmethod
    def clean_annotations(self) -> list[dict[str, Any]]:
        """Process raw annotations into cleaned format.
        
        Returns:
            List of dictionaries containing cleaned annotation data.
            
        Note:
            This property should be implemented by subclasses to define the
            specific cleaning/transformation logic for each annotation type.
        """
        raise NotImplementedError



