from math import exp, floor
from enum import Enum, auto
from copy import deepcopy
from functools import partial
from itertools import accumulate, count, takewhile
from typing import Optional, Callable, Iterable, Tuple, Final, Union

import numpy as np
from numpy.typing import NDArray

from scipy.ndimage import binary_fill_holes, label, binary_dilation
from skimage.segmentation import find_boundaries

from minisom import MiniSom, _build_iteration_indexes, warn

type _winner_function = Callable[[NDArray], Tuple[int, int]]
type _update_function = Callable[[NDArray, Tuple[int, int], int, int], None]
type _mesh_construction_method = Callable[[int, int], None]

class Neighborhood(Enum):
    GAUSSIAN = 1
    # TOPOLOGY = 3


class Topology(Enum):
    RECTANGULAR = 1
    HEXAGONAL = 2


class Distance(Enum):
    EUCLIDEAN = 'euclidean'
    COSINE = 'cosine', 
    MANHATTAN = 'manhattan'
    CHEBYSHEV = 'chebyshev'


class Decay(Enum):
    CHI = '_chi_decay'
    DELTA = 'delta'
    SIGMA = 'sigma'


class Origin(Enum):
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()


class MeshSom(MiniSom):

    _topology_origin: Final[Origin] = Origin.TOP_LEFT
    """The origin used for the `_xx` and `_yy` arrays."""

    _fixed_weights_origin: Final[Origin] = Origin.TOP_LEFT
    """The origin used for the `_fixed_weights_mask` array."""

    _disabled_weights_origin: Final[Origin] = Origin.TOP_LEFT
    """The origin used for the `_disabled_weights_mask` array."""

    _enabled_weights_origin: Final[Origin] = Origin.TOP_LEFT
    """The origin used for the `_enabled_weights_mask` array."""

    _weights_origin: Final[Origin] = Origin.TOP_LEFT
    """The origin used for the `_weights` array."""


    def __init__(
        self, 
        row_dimension: int, 
        col_dimension: int, 
        input_dimension: int = 2, 
        sigma: float = 25, 
        sigma_last: float = 1.0,
        learning_rate: float = 0.5,
        decay_function: str = 'composite',
        neighborhood_function='gaussian', 
        topology='rectangular', 
        activation_distance='euclidean', 
        random_seed=None, 
        sigma_decay_function='composite'
    ) -> None:
        super().__init__(
            row_dimension, col_dimension, input_dimension, sigma, 
            learning_rate, 
            self._composite_decay if decay_function == "composite" else \
                decay_function, 
            neighborhood_function, 
            topology, activation_distance, random_seed, 
            'asymptotic_decay' if sigma_decay_function == "composite" else \
                sigma_decay_function
        )
        self._row_dimension = row_dimension
        """The number of nodes along the first axes."""

        self._col_dimension = col_dimension
        """The number of nodes along the second axes."""

        self._weights_shape = (row_dimension, col_dimension, input_dimension)
        """The shape of the network weights. 
        Equals `(row_dimension, col_dimension, input_dimension)`."""

        self._map_shape = (row_dimension, col_dimension)
        """The shape of the network grid. 
        Equals `(row_dimension, col_dimension)`."""

        self._neighbor_structure: NDArray[np.bool]
        """The neighbor_structure"""

        self._neighbor_index_offsets: NDArray[np.int_]
        """The neighbor index offsets"""

        self._random_choice_generator = np.random.default_rng(random_seed)


        if topology == "rectangular":
            self._neighbor_structure = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ], dtype=bool)

            self._neighbor_index_offsets = np.array([
                [-1,  0],
                [ 1,  0],
                [ 0, -1],
                [ 0,  1]
            ])

        # Hexagonal topology
        else:
            self._neighbor_structure = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ], dtype=bool)

            self._neighbor_index_offsets = np.array([
                [-1,  0],
                [ 1,  0],
                [ 0, -1],
                [ 0,  1]
            ])

        self._sigma_last = sigma_last

        if sigma_decay_function == "composite":
            self._sigma_decay_function = self._composite_sigma_decay

        INTERIOR_INDICES = slice(1, -1)
        MASKED, UNMASKED = (0, 1)
        EXPLODED = np.inf
        self._interior_masked = np.ones((row_dimension, col_dimension))
        self._boundary_masked = np.zeros((row_dimension, col_dimension))
        self._interior_exploded = np.zeros((row_dimension, col_dimension))
        self._interior_masked[INTERIOR_INDICES, INTERIOR_INDICES] = MASKED
        self._boundary_masked[INTERIOR_INDICES, INTERIOR_INDICES] = UNMASKED
        self._interior_exploded[INTERIOR_INDICES, INTERIOR_INDICES] = EXPLODED

        self._complete_eta = self._learning_rate_decay_function
        """Aliases the learning rate decay function, without any masking."""

        self._original_activation_distance = self._activation_distance
        """Aliases the activation distance function, without any explosion."""

        self.reset_disabled_weights_mask()
        self.reset_fixed_weights_mask()


    def _find_sparse_boundaries(self) -> Tuple[NDArray, int]:
        """
        Identifies and labels sparse boundaries along enabled weights.

        This method pads the `_enabled_weights_mask` to handle edge cases,
        finds the inner boundaries of enabled regions, labels these boundaries
        as individual connected components, and then extracts the sparse 
        boundary map within the original (unpadded) dimensions.

        Returns:
            Tuple: A tuple of (`sparse_boundaries`, `num_boundaries`) values.
        """
        padded_enabled_weights_mask = np.pad(
            self._enabled_weights_mask, pad_width=1, constant_values=0
        )

        padded_enabled_boundaries = find_boundaries(
            padded_enabled_weights_mask, connectivity=2, mode="inner"
        )

        padded_labeled_boundaries, num_boundaries = \
            label(padded_enabled_boundaries)
        
        sparse_boundaries = padded_labeled_boundaries[1:-1, 1:-1]

        return sparse_boundaries, num_boundaries


    def _find_dense_boundaries(
        self, 
        sparse_boundaries: NDArray, 
        num_boundaries: int
    ) -> NDArray[np.bool]:
        """
        Converts a sparse boundary representation into a dense, one-hot 
        encoded format.

        This method takes a 2D sparse boundary map (where each non-zero value
        is a boundary label) and transforms it into a 3D boolean array.
        In the output, `output[i, j, k]` is True if the pixel at `(i, j)`
        belongs to boundary `k+1`, and False otherwise.

        Args:
            sparse_boundaries (NDArray): A 2D NumPy array containing labeled
                boundary segments, as returned by `_find_sparse_boundaries`.
                Non-zero values represent boundary labels.
            num_boundaries (int): The total number of unique boundary segments,
                as returned by `_find_sparse_boundaries`.

        Returns:
            NDArray[np.bool]: A 3D NumPy array of boolean values. The shape
                will be `(height, width, num_boundaries)`.
                `output[:, :, k]` represents a binary mask for the (k+1)-th 
                boundary.
        """
        bounds = sparse_boundaries[:, :, None]
        labels = np.arange(1, num_boundaries + 1)[None, None, :]

        return bounds == labels

    # def _boundary_eta(
    #     self, 
    #     learning_rate: float, 
    #     t: int, 
    #     max_iter: int
    # ) -> NDArray:
    #     return (
    #         self._interior_masked * \
    #         self._complete_eta(learning_rate, t, max_iter)
    #     )
    

    # def _interior_eta(
    #     self, 
    #     learning_rate: float, 
    #     t: int, 
    #     max_iter: int
    # ) -> NDArray:
    #     return (
    #         self._boundary_masked * \
    #         self._complete_eta(learning_rate, t, max_iter)
    #     )
    

    # def _boundary_activation_distance(self, x, w) -> NDArray:
    #     return (
    #         self._interior_exploded +
    #         self._original_activation_distance(x, w)
    #     )

    def _composite_decay(
        self, 
        learning_rate: float, 
        t: int, 
        max_iter: int) -> float:
        """The decay function for the paper.

        Args:
            learning_rate (float): The learning rate parameter (unused).
            t (int): The current iteration index.
            max_iter (int): The maximum number of iterations.

        Returns:
            float: The delta(t) learning rate.
        """
        return t**(-0.2) * (1 - exp(5 * (t - max_iter) / max_iter))
    

    def _composite_sigma_decay(self, sigma, t, max_iter) -> float:
        """Decay function of sigma that asymptotically approaches one.
        """
        return \
            self._sigma_last + \
            (1 - exp(5 * (t - max_iter) / max_iter)) * \
            (sigma * (0.005 ** (t / max_iter)) - self._sigma_last) * \
            (t ** -0.25)


    def _phi(
        self,
        first_value: int = 0,
        first_input: int = 0,
        input_step: int = 1
    ) -> Iterable[int]:
        """
        Generates the sequence of Phi(s) values.

        The sequence is an accumulation of the phi(k) function's values.

        Args:
            first_value (int): The Phi_0 initial value of the accumulation.
                Defaults to 0.
            first_input (int): The starting argument for the phi(k) function.
                Defaults to 0.
            input_step (int): The step size for the input. Defaults to 1.

        Returns:
            Iterable[int]: An iterable yielding the values of the Phi(s) 
                function.
        """
        function = lambda input: floor(500 * (1 - exp(-2 * input / 15)))
        return self._accumulate(function, first_value, first_input, input_step)
    

    # def _phi_until_exceed(
    #     self,
    #     max_value: int,
    #     first_value: int = 0,
    #     first_input: int = 0,
    #     input_step: int = 1
    # ) -> Iterable[int]:
    #     phi = self._phi(first_value, first_input, input_step)
    #     return self._until_exceed(max_value, phi)
    

    def _psi(
        self, 
        first_value: int = 0,
        first_input: int = 0,
        input_step: int = 1
    ) -> Iterable[int]:
        """
        Generates the sequence of Psi(s) values.

        The sequence is an accumulation of the psi(k) function's values.

        Args:
            first_value (int): The Phi_0 initial value of the accumulation.
                Defaults to 0.
            first_input (int): The starting argument for the psi(k) function.
                Defaults to 0.
            input_step (int): The step size for the input. Defaults to 1.

        Returns:
            Iterable[int]: An iterable yielding the values of the Psi(s) 
                function.
        """
        function = lambda input: floor(500 * exp(input**2 / -200))
        return self._accumulate(function, first_value, first_input, input_step)


    def _accumulate(
        self, 
        function: Callable[[int], int],
        first_value: int = 0,
        first_input: int = 0,
        input_step: int = 1
    ) -> Iterable[int]:
        """
        Accumulates values generated by a given function.

        Args:
            function (Callable): The integer to integer function to apply.
            first_value (int): The initial value for accumulation.
            first_input (int): The starting input for the function.
            input_step (int): The step size for the input.

        Returns:
            Iterable[int]: An iterable yielding the accumulated values.
        """
        inputs = count(first_input, input_step)
        function_values = map(function, inputs)
        return accumulate(function_values, initial=first_value)
    

    def _until_exceed(
        self, 
        max_value: int, 
        iterable: Iterable[int]
    ) -> Iterable[int]:
        """
        Takes elements from an iterable until a value exceeds `max_value`.

        Args:
            max_value (int): The maximum allowed value in the yielded sequence.
            iterable (Iterable[int]): The iterable to draw values from.

        Returns:
            Iterable[int]: An iterable yielding values from the input iterable
                up to (and including) `max_value`.
        """
        return takewhile(lambda value: value <= max_value, iterable)


    def _row_direction_top_bottom(self) -> Iterable[Origin]:
        """
        Origins where the row index increas from top to bottom.
        
        Returns:
            Iterable: An iterable of origin members.
        """
        yield Origin.TOP_LEFT
        yield Origin.TOP_RIGHT


    def _col_direction_left_right(self) -> Iterable[Origin]:
        """
        Origins where the column index increas from left to right.

        Returns:
            Iterable: An iterable of origin members.
        """
        yield Origin.TOP_LEFT
        yield Origin.BOTTOM_LEFT


    def _transform_array_origin(
        self,
        array: NDArray, 
        input_origin: Origin, 
        output_origin: Origin,
        copy: bool = True
    ) -> NDArray:
        """
        Transforms an N-dimensional array's first two (row, column) dimensions 
        based on origin changes.

        Args:
            array (NDArray): The N-dimensional input array with shape 
                `(row, col, d3, ... dN)`.
            input_origin (Origin): The initial origin of the input array.
            output_origin (Origin): The target origin for the input array.
            copy (bool, optional): If `True` a deep copy of the input array
                is transformed and returned. Defaults to `True`.

        Returns:
            NDArray: The transformed array. If `copy == False`, than it is a
                view (not a deep copy) of the input array.
        """
        if copy:
            array = deepcopy(array)

        if input_origin == output_origin:
            return array
        
        input_row_dir  = input_origin  in self._row_direction_top_bottom()
        input_col_dir  = input_origin  in self._col_direction_left_right()
        output_row_dir = output_origin in self._row_direction_top_bottom()
        output_col_dir = output_origin in self._col_direction_left_right()

        if input_row_dir != output_row_dir:
            array = array[::-1, :, ...]

        if input_col_dir != output_col_dir:
            array = array[:, ::-1, ...]

        return array


    def _validate_array_shape(
        self, 
        data: NDArray, 
        expected_shape: Tuple
    ) -> None:
        """
        Validates the shape of the data array.

        Args:
            data (NDArray): The array to be validated.
            expected_shape (Tuple): The expected shape of the data array.

        Raises:
            ValueError: If the shape of the provided data is incorrect.
        """
        if data.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, "
                f"actual shape {data.shape}"
            )


    def grid_weights_init(
        self, 
        x_min: float, 
        x_max: float, 
        y_min: float, 
        y_max: float
    ) -> None:
        """
        Initializes the weights as a Cartesian grid, within the 
        specified bounding box.

        The X-coordinates are stored in `self._weights[..., 0]` with the 
        following layout.

        >>> [[x_min, ..., x_max],
        >>>  [...,   ...,   ...],
        >>>  [x_min, ..., x_max]]

        The Y-coordinates are stored in `self._weights[..., 1]` with the
        following layout.
            
        >>> [[y_min, ..., y_min],
        >>>  [...,   ...,   ...],
        >>>  [y_max, ..., y_max]]

        Note that the origin of the gird is in the top left corner.

        Args:
            x_min (float): The minimum x-coordinate for the grid.
            x_max (float): The maximum x-coordinate for the grid.
            y_min (float): The minimum y-coordinate for the grid.
            y_max (float): The maximum y-coordinate for the grid.
        """
        x = np.linspace(x_min, x_max, self._col_dimension)
        y = np.linspace(y_min, y_max, self._row_dimension)
        xy = np.meshgrid(x, y, indexing="xy")
        self._weights = np.stack(xy, axis=-1)
        

    def fix_weights(self, mask: NDArray[np.bool], origin: Origin) -> None:
        """Fixes the training weights given by the mask.

        Args:
            mask (NDArray): A boolean array, where `True` values are 
                associated with the positions of fixed weights.
            origin (Origin): The origin's position in the mask array.

        Raises:
            ValueError: If the shape of the provided mask is incorrect.
        """
        self._validate_array_shape(mask, self._map_shape)
        self._fixed_weights_mask = self._transform_array_origin(
            mask, origin, self._fixed_weights_origin, copy=True
        )


    def fixed_weights_mask(self, origin: Origin) -> NDArray[np.bool]:
        """Returns the fixed weights mask in the requested orientation.

        Args:
            origin (Origin): The origin's position in the output mask array.

        Returns:
            NDArray[bool]: A boolean array, where `True` values are associated 
                with the positions of fixed weights.
        """
        return self._transform_array_origin(
            self._fixed_weights_mask, self._fixed_weights_origin, origin, 
            copy=True
        )
    

    def reset_fixed_weights_mask(self) -> None:
        """Enables all the training weights to be updated during training.

        This method only resets the fixed weights mask for the training 
        weights. The weights would not be changed.
        """
        self._fixed_weights_mask = np.zeros(self._map_shape, dtype=np.bool)
    

    def disable_weights(self, mask: NDArray[np.bool], origin: Origin) -> None:
        """Disable the training weights given my the mask.

        Currently only exterior weight disable is supported.

        Args:
            mask (NDArray): A boolean array, where `True` values are 
                associated with the positions of disabled weights.
            origin (Origin): The origin's position in the mask array.

        Raises:
            ValueError: If the shape of the provided mask is incorrect.
        """
        # mask = mask[::-1, :]     # Convert to the computational domain.
        self._validate_array_shape(mask, self._map_shape)
        self._disabled_weights_mask = self._transform_array_origin(
            mask, origin, self._disabled_weights_origin, copy=True
        )
        self._enabled_weights_mask = self._transform_array_origin(
            ~mask, origin, self._enabled_weights_origin, copy=True
        )

    
    def reset_disabled_weights_mask(self) -> None:
        """Enables all the training weights in the map.

        This method only resets the enabled and disabled masks for the training 
        weights. The weights would not be changed.
        """
        self._disabled_weights_mask = np.zeros(self._map_shape, dtype=np.bool)
        self._enabled_weights_mask  = np.ones( self._map_shape, dtype=np.bool)



    def enabled_weights_mask(self, origin: Origin) -> NDArray[np.bool]:
        """Returns the enabled weights mask in the requested orientation.

        Args:
            origin (Origin): The origin's position in the output mask array.

        Returns:
            NDArray[bool]: A boolean array, where `True` values are associated 
                with the positions of enabled weights.
        """
        return self._transform_array_origin(
            self._enabled_weights_mask, self._enabled_weights_origin, origin,
            copy=True
        )
    

    def disabled_weights_mask(self, origin: Origin) -> NDArray[np.bool]:
        """Returns the disabled weights mask in the requested orientation.

        Args:
            origin (Origin): The origin's position in the output mask array.

        Returns:
            NDArray[bool]: A boolean array, where `True` values are associated 
                with the positions of disabled weights.
        """
        return self._transform_array_origin(
            self._disabled_weights_mask, self._disabled_weights_origin, origin,
            copy=True
        )
    

    def data_weights_init(self, data: NDArray, origin: Origin) -> None:
        """
        Initializes the SOM's weights directly with provided data.

        Args:
            data (NDArray): The initial weights.
            origin (Origin): The origin's position in the mask array.

        Raises:
            ValueError: If the shape of the provided mask is incorrect.
        """
        self._validate_array_shape(data, self._weights_shape)
        self._weights = self._transform_array_origin(
            data, origin, self._weights_origin, copy=True
        )


    def get_weights(
        self, 
        origin: Origin = Origin.BOTTOM_LEFT,
        disable: bool = True,
        disable_value: float = np.nan
    ) -> NDArray[np.float32]:
        """
        Returns the weights of the neural network.

        Args:
            origin (Origin, optional): The origin's position in the returned 
                array. Defaults to `Origin.BOTTOM_LEFT`.
            disable (bool, optional): If `True`, the disabled weights will be
                masked with the provided mask value. Defaults to `True`.
            disable_value (float, optional): Used to mask the disabled weights
                if `disable == True`. Defaults to `numpy.nan`.
        
        Returns:
            NDArray[float]: The weights of the neural network.
        """
        weights = deepcopy(self._weights)

        if disable:
            weights[self._disabled_weights_mask] = disable_value

        return self._transform_array_origin(
            weights, self._weights_origin, origin, copy=False
        )



    # def _masked_learning_rate_decay(
    #     self, 
    #     learning_rate: float, 
    #     t: int, 
    #     max_iter: int
    # ) -> float:
    #     pass


    # def update(self, x: NDArray, win, t: int, max_iteration: int) -> None:
    #     t += self.start_iter
    #     return super().update(x, win, t, self.max_iteration)


    def _activate_masked(self, x: NDArray, mask: NDArray) -> None:
        """
        Updates the activation map by calculating distances and applying a 
        mask.

        The mask is added to the activation distances, effectively increasing
        the distance for masked neurons, making them less likely to be winners.

        Args:
            x (NDArray): The input vector.
            mask (NDArray): A two-dimensional array used to modify activation 
                distances.
        """
        self._activation_map: NDArray = \
            self._activation_distance(x, self._weights) + mask


    def winner_masked(self, x: NDArray, mask: NDArray) -> Tuple[int, int]:
        """
        Computes the coordinates of the winning neuron for the sample, 
        considering a mask.

        The mask modifies the activation distances before finding the minimum,
        allowing specific neurons to be excluded or penalized.

        Args:
            x (NDArray): The input pattern.
            mask (NDArray): A two-dimensional array used to modify activation 
                distances.

        Returns:
            Tuple: The (row, column) coordinates of the winning neuron.
        """
        self._activate_masked(x, mask)
        return np.unravel_index(self._activation_map.argmin(),
                                self._activation_map.shape)
    

    def update_masked(
        self, 
        x: NDArray, 
        win: Tuple[int, int], 
        t: int, 
        max_iteration: int, 
        mask: NDArray
    ) -> None:
        """
        Updates the weights of the neurons in the masked neighborhood.

        Args:
            x (NDArray): Current pattern to learn.
            win (Tuple[int, int]): Position of the winning neuron for x.
            t (int): Current iteration number.
            max_iteration (int): Maximum number of iterations for the decay 
                functions.
            mask (NDArray): A two-dimensional array which maskes the 
                neighborhood.
        """
        eta = self._learning_rate_decay_function(
            self._learning_rate, t, max_iteration
        )
        sig = self._sigma_decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig) * eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += \
            np.einsum('ij, ijk->ijk', g, x-self._weights) * mask[:, :, None]


    def _mesh_construction_setup_winner(
        self,
        winner_mask: Optional[NDArray]
    ) :
        """
        Configures the winner selection function based on an optional mask.

        If a `winner_mask` is provided, some neurons would be masked during the
        winner selection.

        Args:
            winner_mask (NDArray, optional): A mask to be applied during winner 
                selection.

        Returns:
            Callable: The appropriate winner selection function 
                (`winner_masked` or `winner`).
        """
        if winner_mask is not None:
            winner = partial(self.winner_masked, mask=winner_mask)
        else:
            winner = self.winner
        return winner
    

    def _mesh_construction_setup_update(
        self,
        adjust_mask: Optional[NDArray]
    ):
        """
        Configures the weight update function based on an optional mask.

        If an `adjust_mask` is provided, some neurons would be masked during 
        the neuron update.

        Args:
            adjust_mask (NDArray, optional): A mask to be applied during weight
                updates.

        Returns:
            Callable: The appropriate weight update function 
                (`update_masked` or `update`).
        """
        if adjust_mask is not None:
            update = partial(self.update_masked, mask=adjust_mask)
        else:
            update = self.update
        return update
    

    def _mesh_construction_setup_samples(
        self,
        input_data: NDArray,
        start_iter: int,
        final_iter: int,
        first_index: int = 1
    ) -> Iterable[NDArray]:
        """
        Prepares a random sample sequence for mesh construction.

        Args:
            input_data (NDArray): The input data to be used for training.
            start_iter (int): The starting iteration number for this phase.
            final_iter (int): The final iteration number for this phase.
            first_index (int): The first index to consider for iterations.

        Returns:
            NDArray: An array of samples for the mesh construction.
        """
        num_iteration = final_iter - start_iter + first_index

        self._check_iteration_number(num_iteration)
        self._check_input_len(input_data)

        rng = np.random.default_rng()

        return rng.choice(input_data, num_iteration, axis=0)
    

    def _mesh_construction_setup_iterations(
        self,
        input_data: NDArray,
        start_iter: int,
        final_iter: int,
        first_index: int = 1,
        random_order: bool = True,
        verbose: bool = False
    ) -> Iterable[int]:
        """
        Prepares the iteration sequence for mesh construction.

        Args:
            input_data (NDArray): The input data to be used for training.
            start_iter (int): The starting iteration number for this phase.
            final_iter (int): The final iteration number for this phase.
            first_index (int): The first index to consider for iterations.
            random_order (bool): If `True`, iterations are randomly ordered.
            verbose (bool): If `True`, print verbose output.

        Returns:
            Iterable[int]: An iterable of data indices to be used in each 
                iteration.
        """
        num_iteration = final_iter - start_iter + first_index

        self._check_iteration_number(num_iteration)
        self._check_input_len(input_data)

        if random_order:
            random_generator = self._random_generator
        else:
            random_generator = None

        return _build_iteration_indexes(
            len(input_data), num_iteration, verbose, random_generator
        )
    

    def _mesh_construction_setup_max_iteration(
        self,
        final_iter: int,
        max_iteration: Optional[int],
    ) -> int:
        """
        Determines the effective `max_iteration` for the decay functions.

        Ensures `max_iteration` is at least `final_iter`.

        Args:
            final_iter (int): The final iteration of the current training 
                phase.
            max_iteration (int, optional): The explicitly provided maximum 
                iteration, or `None` to default to `final_iter`.

        Returns:
            int: The maximum iteration value to be used for decay calculations.
        """
        if max_iteration is None:
            max_iteration = final_iter

        elif max_iteration < final_iter:
            warn("Updated max iterations to final_iter")
            max_iteration = final_iter
    
        return max_iteration


    def _mesh_construction_setup(
        self,
        input_data: NDArray,
        winner_mask: Optional[NDArray] = None,
        adjust_mask: Optional[NDArray] = None,
        max_iteration: Optional[int] = None,
        first_index: int = 1,
    ) -> _mesh_construction_method:
        """
        Performs a generic mesh construction (training) phase.

        This method is a generalized training loop that allows for custom
        winner selection and weight adjustment masks.

        Args:
            input_data (NDArray): The data to train the SOM with.
            first_index (int, optional): The first index to consider for 
                iterations. Defaults to 1.
            max_iteration (Optional[int]): The total number of iterations for 
                decay calculations. If `None`, defaults to `final_iter`.
            winner_mask (Optional[NDArray]): A mask to apply during winner 
                selection. If `None`, all neurons are considered.
            adjust_mask (Optional[NDArray]): A mask to apply to the learning 
                rate during weight adjustment. If `None`, all neurons adjust 
                equally.
            random_order (bool): If `True`, iterate through data in a random 
                order.
            verbose (bool): If `True`, print progress and quantization error.
        """
        winner = self._mesh_construction_setup_winner(winner_mask)
        update = self._mesh_construction_setup_update(adjust_mask)

        return partial(
            self._mesh_construction, 
            input_data, winner, update, first_index, max_iteration
        )


    def _mesh_construction(
        self,
        input_data: NDArray[np.float32],
        winner: _winner_function,
        update: _update_function,
        first_index: int,
        max_iteration: int,
        start_iter: int,
        final_iter: int
    ) -> None:
        """
        Performs a generic mesh construction (training) phase.

        This method is a generalized training loop that allows for custom
        winner selection and weight adjustment masks.

        Args:
            input_data (NDArray): The data to train the SOM with.
            start_iter (int): The starting iteration number for this phase.
            final_iter (int): The final iteration number for this phase.
            first_index (int): The first index to consider for iterations. 
                Defaults to 1.
            max_iteration (Optional[int]): The total number of iterations for 
                decay calculations. If `None`, defaults to `final_iter`.
            winner_mask (Optional[NDArray]): A mask to apply during winner 
                selection. If `None`, all neurons are considered.
            adjust_mask (Optional[NDArray]): A mask to apply to the learning 
                rate during weight adjustment. If `None`, all neurons adjust 
                equally.
            random_order (bool): If `True`, iterate through data in a random 
                order.
            verbose (bool): If `True`, print progress and quantization error.
        """
        samples = self._mesh_construction_setup_samples(
            input_data, start_iter, final_iter, first_index
        )

        max_iteration = self._mesh_construction_setup_max_iteration(
            final_iter, max_iteration
        )

        for t, sample in enumerate(samples, start_iter):
            update(sample, winner(sample), t, max_iteration)



    def mesh_construction(
        self,
        input_data: NDArray,
        start_iter: int,
        final_iter: int,
        first_index: int = 1,
        max_iteration: Optional[int] = None,
        winner_mask: Optional[NDArray] = None,
        adjust_mask: Optional[NDArray] = None,
        random_order: bool = True,
        verbose: bool = False
    ) -> None:
        """
        Performs a generic mesh construction (training) phase.

        This method is a generalized training loop that allows for custom
        winner selection and weight adjustment masks.

        Args:
            input_data (NDArray): The data to train the SOM with.
            start_iter (int): The starting iteration number for this phase.
            final_iter (int): The final iteration number for this phase.
            first_index (int): The first index to consider for iterations. 
                Defaults to 1.
            max_iteration (Optional[int]): The total number of iterations for 
                decay calculations. If `None`, defaults to `final_iter`.
            winner_mask (Optional[NDArray]): A mask to apply during winner 
                selection. If `None`, all neurons are considered.
            adjust_mask (Optional[NDArray]): A mask to apply to the learning 
                rate during weight adjustment. If `None`, all neurons adjust 
                equally.
            random_order (bool): If `True`, iterate through data in a random 
                order.
            verbose (bool): If `True`, print progress and quantization error.
        """
        winner = self._mesh_construction_setup_winner(winner_mask)
        update = self._mesh_construction_setup_update(adjust_mask)

        # iterations = self._mesh_construction_setup_iterations(
        #     input_data, start_iter, final_iter, first_index, 
        #     random_order, verbose
        # )

        samples = self._mesh_construction_setup_samples(
            input_data, start_iter, final_iter, first_index
        )

        max_iteration = self._mesh_construction_setup_max_iteration(
            final_iter, max_iteration
        )

        for t, sample in enumerate(samples, start_iter):
            update(sample, winner(sample), t, max_iteration)


        # for t, iteration in enumerate(iterations, start_iter):
        #     sample = input_data[iteration]
        #     update(sample, winner(sample), t, max_iteration)
        if verbose:
            print(f"Iteration: [{start_iter} .. {final_iter}]")


    


    def initial_mesh_construction(
        self,
        complete_data: NDArray,
        final_iter: int,
        max_iteration: int,
        random_order: bool = True,
        verbose: bool = False
    ) -> None:
        """
        Performs an initial mesh construction phase using all available data.

        In this phase all neurons are learned by random points from the whole 
        domain G during the given number of iterations Phi(0).

        Args:
            complete_data (NDArray): The entire dataset to train the SOM with.
            final_iter (int): The final iteration number for this initial 
                phase.
            max_iteration (int): The total number of iterations for decay 
                calculations.
            random_order (bool): If `True`, iterate through data in a random 
                order.
            verbose (bool): If `True`, print progress and quantization error.
        """
        self.mesh_construction(
            complete_data, 
            start_iter=1, final_iter=final_iter, max_iteration=max_iteration, 
            winner_mask=self._default_winner_mask(), 
            adjust_mask=self._default_adjust_mask(),
            random_order=random_order, verbose=verbose
        )
    

    def _create_float_mask(
        self,
        mask: NDArray[np.bool],
        mask_fixed_weights: bool,
        mask_disabled_weights: bool,
        mask_value: float,
        other_value: float,
    ) -> NDArray[np.float32]:
        """Creates a float mask from a boolean mask, considering fixed and 
        disabled weights.

        Args:
            mask (NDArray[numpy.bool]): A boolean array representing the base 
                mask.
            mask_fixed_weights (bool): If `True`, positions where weights are 
                fixed **will** have `mask_value` applied.
            mask_disabled_weights (bool): If `True`, positions where weights 
                are disabled **will** have `mask_value` applied.
            mask_value (float): The value to apply where the final processed 
                mask is `True`.
            other_value (float): The value to apply where the final processed 
                mask is `False`.

        Returns:
            NDArray: A float array where values are either `mask_value` or 
                `other_value` based on the input mask and filtering conditions.

        Raises:
            ValueError: If the shape of the input mask is incorrect.
        """
        self._validate_array_shape(mask, self._map_shape)

        if mask_fixed_weights:
            mask = mask | self._fixed_weights_mask

        if mask_disabled_weights:
            mask = mask | self._disabled_weights_mask

        return np.where(mask, mask_value, other_value)
    

    def _create_explosion_mask(
        self,
        mask: NDArray[np.bool],
        mask_fixed_weights: bool = False,
        mask_disabled_weights: bool = False
    ) -> NDArray[np.float32]:
        """Creates an additive explosion mask for winner calculations.

        This method generates an array where `np.inf` is applied at mask
        specific positions, and `0.0` elsewhere.

        Args:
            mask (NDArray[numpy.bool]): A boolean array representing the base 
                mask.
            mask_fixed_weights (bool, optional): If `True`, positions where 
                weights are fixed **will** have `np.inf` applied. Defaults to 
                `False`.
            mask_disabled_weights (bool, optional): If `True`, positions where 
                weights are disabled **will** have `np.inf` applied. Defaults
                to `False`.

        Returns:
            NDArray: A float array where values are either `np.inf` or `0.0`
                based on the input mask and filtering conditions.

        Raises:
            ValueError: If the shape of the input mask is incorrect.
        """
        MASK_OUT_WITH_INFINITY = np.inf
        KEEP_ORIGINAL_VALUE = 0.0
        return self._create_float_mask(
            mask, mask_fixed_weights, mask_disabled_weights, 
            MASK_OUT_WITH_INFINITY, KEEP_ORIGINAL_VALUE
        )
    

    def _create_binary_mask(
        self,
        mask: NDArray[np.bool],
        mask_fixed_weights: bool = False,
        mask_disabled_weights: bool = False
    ) -> NDArray[np.float32]:
        """Creates a multiplicative binary mask for update calculations.

        This method generates an array where `0.0` is applied at mask
        specific positions, and `1.0` elsewhere.

        Args:
            mask (NDArray[numpy.bool]): A boolean array representing the base 
                mask.
            mask_fixed_weights (bool, optional): If `True`, positions where 
                weights are fixed **will** have `0.0` applied. Defaults to 
                `False`.
            mask_disabled_weights (bool, optional): If `True`, positions where 
                weights are disabled **will** have `0.0` applied. Defaults to 
                `False`.

        Returns:
            NDArray: A float array where values are either `0.0` or `1.0`
                based on the input mask and filtering conditions.

        Raises:
            ValueError: If the shape of the input mask is incorrect.
        """
        MASK_OUT_WITH_ZERO = 0.0
        KEEP_ORIGINAL_VALUE = 1.0
        return self._create_float_mask(
            mask, mask_fixed_weights, mask_disabled_weights, 
            MASK_OUT_WITH_ZERO, KEEP_ORIGINAL_VALUE
        )
    

    def _create_winner_mask(
        self, 
        mask: NDArray[np.bool]
    ) -> NDArray[np.float32]:
        """Generates a winner mask for winner calculation.

        By default, this mask is configured such that:
            - `mask` controls which nodes to mask (these cannot be winners),
            - disabled weights are exploded (disabled nodes cannot be winners),
            - fixed weights are unchanged (fixed nodes can be winners).

        Args:
            mask (NDArray[numpy.bool]): A boolean array representing the base 
                mask.

        Returns:
            NDArray: A float array where disabled weights have `np.inf` values, 
                all other values are `0.0`.
        """
        DISABLED_CANNOT_BE_WINNER = True
        FIXED_CAN_BE_WINNER = False
        return self._create_explosion_mask(
            mask, 
            mask_disabled_weights=DISABLED_CANNOT_BE_WINNER,
            mask_fixed_weights=FIXED_CAN_BE_WINNER
        )
    

    def _create_adjust_mask(
        self, 
        mask: NDArray[np.bool]
    ) -> NDArray[np.float32]:
        """Generates an adjust mask for weight update.

        By default, this mask is configured such that:
            - `mask` controls which nodes to mask (these cannot be adjusted),
            - disabled and fixed weights are masked (these cannot be adjusted),
            - all other nodes are unchanged (these can be adjusted).

        Args:
            mask (NDArray[numpy.bool]): A boolean array representing the base 
                mask.
            
        Returns:
            NDArray: A float array where values are either `0.0` or `1.0`
                based on the input mask and filtering conditions.

        Raises:
            ValueError: If the shape of the input mask is incorrect.
        """
        DISABLED_CANNOT_BE_ADJUSTED = True
        FIXED_CANNOT_BE_ADJUSTED = True
        return self._create_binary_mask(
            mask, 
            mask_disabled_weights=DISABLED_CANNOT_BE_ADJUSTED,
            mask_fixed_weights=FIXED_CANNOT_BE_ADJUSTED
        )
    

    def _default_winner_mask(self) -> NDArray[np.float32]:
        """Generates the default explosion mask for winner calculation.

        By default, this mask is configured such that:
            - disabled weights are exploded (disabled nodes cannot be winners),
            - other weights are unchanged (these can be winners).

        Returns:
            NDArray: A float array where disabled weights have `np.inf` values, 
                all other values are `0.0`.
        """
        DISABLED_ALREADY_MASKED = False
        FIXED_CAN_BE_WINNER = False
        return self._create_explosion_mask(
            mask=self._disabled_weights_mask,
            mask_disabled_weights=DISABLED_ALREADY_MASKED,
            mask_fixed_weights=FIXED_CAN_BE_WINNER
        )
      

    def _default_adjust_mask(self) -> NDArray[np.float32]:
        """Generates the default adjust mask for weight update.

        By default, this mask is configured such that:
            - disabled and fixed weights are masked (these cannot be adjusted),
            - other weights are unchanged (these can be adjusted).

        Returns:
            NDArray: A float array where disabled and fixed weights have 
                `0.0` values, all other values are `1.0`.
        """
        DISABLED_ALREADY_MASKED = False
        FIXED_CANNOT_BE_ADJUSTED = True
        return self._create_binary_mask(
            mask=self._disabled_weights_mask,
            mask_disabled_weights=DISABLED_ALREADY_MASKED,
            mask_fixed_weights=FIXED_CANNOT_BE_ADJUSTED
        )


    # def _create_interior_masked(self, ) -> None:
    #     self._interior_masked, _ = self._find_sparse_boundaries()
    #     print("Interior masked:\n", self._interior_masked.astype(int))


    # def _create_boundary_masked(self) -> None:
    #     boundary, _ = self._find_sparse_boundaries()
    #     self._boundary_masked = ~np.astype(boundary, np.bool)
    #     self._boundary_masked[self._disabled_weights_mask] = 0
    #     self._boundary_masked = np.astype(self._boundary_masked, np.float32)
    #     print("Boundary masked:\n", self._boundary_masked.astype(int))


    # def _create_interior_exploded(self) -> None:
    #     boundary, _ = self._find_sparse_boundaries()
    #     not_boundary = ~np.astype(boundary, np.bool)
    #     self._interior_exploded = np.zeros(self._map_shape, np.float32)
    #     self._interior_exploded[not_boundary] = np.inf
    #     print("Interior exploded:\n", self._interior_exploded)


    def _boundary_mesh_construction_setup(
        self,
        interior: NDArray[np.bool],
        boundary_data: NDArray,
        max_iteration: int
    ) -> _mesh_construction_method:
        return self._mesh_construction_setup(
            boundary_data, 
            self._create_winner_mask(mask=interior), # Winnerset = boundary
            self._create_adjust_mask(mask=interior), # Adjustset = boundary
            max_iteration
        )
    

    def _interior_mesh_construction_setup(
        self,
        boundary: NDArray[np.bool],
        complete_data: NDArray,
        max_iteration: int
    ) -> _mesh_construction_method:
        return self._mesh_construction_setup(
            complete_data, 
            self._default_winner_mask(),             # Winnerset = complete
            self._create_adjust_mask(mask=boundary), # Adjustset = interior
            max_iteration
        )


    def _boundary_mesh_construction(
        self,
        interior: NDArray[np.bool],
        boundary: NDArray[np.bool],
        boundary_data: NDArray,
        start_iter: int,
        final_iter: int,
        max_iteration: int,
        random_order: bool = True,
        verbose: bool = False
    ) -> None:
        """
        Constructs/refines the mesh focusing on boundary nodes.

        This method uses a mask to ensure that only boundary nodes are 
        considered for both winner selection and weight adjustment.

        Args:
            boundary_data (NDArray): The data representing boundary points.
            start_iter (int): The starting iteration number for this phase.
            final_iter (int): The final iteration number for this phase.
            max_iteration (int): The total number of iterations for decay 
                calculations.
            random_order (bool): If `True`, iterate through data in a random 
                order.
            verbose (bool): If `True`, print progress and quantization error.
        """
        # print("winner\n", self._create_winner_mask(mask=interior))
        # print("adjust\n", self._create_adjust_mask(mask=interior))
        self.mesh_construction(
            boundary_data, start_iter, final_iter,
            max_iteration=max_iteration, 
            # winner_mask=self._interior_exploded,    # winner_set = boundary
            # adjust_mask=self._interior_masked,      # adjust_set = boundary
            winner_mask=self._create_winner_mask(mask=interior),
            adjust_mask=self._create_adjust_mask(mask=interior),
            random_order=random_order, 
            verbose=verbose
        )

    
    def _interior_mesh_construction(
        self,
        interior: NDArray[np.bool],
        boundary: NDArray[np.bool],
        complete_data: NDArray,
        start_iter: int,
        final_iter: int,
        max_iteration: int,
        random_order: bool = True,
        verbose: bool = False
    ) -> None:
        """
        Constructs/refines the mesh focusing on interior nodes.

        This method considers all data for winner selection but only adjusts
        the weights of interior nodes.

        Args:
            complete_data (NDArray): The entire dataset (including interior 
                and boundary points).
            start_iter (int): The starting iteration number for this phase.
            final_iter (int): The final iteration number for this phase.
            max_iteration (int): The total number of iterations for decay 
                calculations.
            random_order (bool): If True, iterate through data in a random 
                order.
            verbose (bool): If True, print progress and quantization error.
        """
        # print("winner\n", self._default_winner_mask())
        # print("adjust\n", self._create_adjust_mask(mask=boundary))

        self.mesh_construction(
            complete_data, start_iter, final_iter,
            max_iteration=max_iteration, 
            # winner_mask=None,                       # winner_set = complete
            # adjust_mask=self._boundary_masked,      # adjust_set = interior
            winner_mask=self._default_winner_mask(),
            adjust_mask=self._create_adjust_mask(mask=boundary),
            random_order=random_order, 
            verbose=verbose
        )


    # def _boundary_interior_bounds(
    #     self, 
    #     start_constant: int,
    #     max_iteration: int
    # ) -> Iterable[Tuple[int, int, int, int]]:
    #     """
    #     Generates iteration bounds for alternating boundary and interior 
    #     training phases.

    #     This method uses `_phi` and `_psi` functions to determine the starting
    #     and ending iteration numbers for subsequent boundary and interior 
    #     training steps.

    #     Args:
    #         start_constant (int): The initial constant used to offset 
    #             iteration numbers.
    #         max_iteration (int): The overall maximum iteration for the 
    #             training process.

    #     Returns:
    #         Iterable: An iterable yielding tuples of (`boundary_start`, 
    #             `boundary_final`, `interior_start`, `interior_final`) 
    #             for each training cycle.
    #     """
    #     START_WITH_FIRST  = 0
    #     START_WITH_SECOND = 1
    #     PLUS_ONE_INCREMENT = 1

    #     start_constant = start_constant + PLUS_ONE_INCREMENT
    #     final_constant = start_constant

    #     phi_start = self._phi(start_constant, first_input=START_WITH_FIRST)
    #     phi_final = self._phi(final_constant, first_input=START_WITH_SECOND)
    #     phi_final = self._until_exceed(max_iteration, phi_final)

    #     psi_start = self._psi(start_constant, first_input=START_WITH_FIRST)
    #     psi_final = self._psi(final_constant, first_input=START_WITH_SECOND)

    #     return zip(psi_start, psi_final, phi_start, phi_final)
    

    def _boundary_interior_bounds(
        self, 
        start_constant: int,
        max_iteration: int,
        first_input: int = 0,
        input_step: int = 1
    ) -> Iterable[Tuple[int, int, int, int]]:
        """
        Generates iteration bounds for alternating boundary and interior 
        training phases.

        This method uses `_phi` and `_psi` functions to determine the starting
        and ending iteration numbers for subsequent boundary and interior 
        training steps.

        Args:
            start_constant (int): The initial constant used to offset 
                iteration numbers.
            max_iteration (int): The overall maximum iteration for the 
                training process.

        Returns:
            Iterable: An iterable yielding tuples of (`boundary_start`, 
                `boundary_final`, `interior_start`, `interior_final`) 
                for each training cycle.
        """
        INCREMENT_WITH_ONE = 1

        start_constant = start_constant + INCREMENT_WITH_ONE
        final_constant = start_constant

        phi_start = self._phi(start_constant, first_input, input_step)
        phi_final = self._phi(final_constant, first_input, input_step)
        psi_start = self._psi(start_constant, first_input, input_step)
        psi_final = self._psi(final_constant, first_input, input_step)

        # Skip the first elements
        next(phi_final)
        next(psi_final)

        phi_final = self._until_exceed(max_iteration, phi_final)

        return zip(psi_start, psi_final, phi_start, phi_final)
    

    def find_boundary_neighborhood(
        self, 
        dense_boundaries: NDArray[np.bool]
    ) -> NDArray[np.bool]:
        """
        Applies binary dilation to each (num_rows, num_cols) slice along the
        'num_items' dimension of dense_boundaries and then masks the result.

        Args:
            dense_boundaries (NDArray[np.bool_]): A boolean array of shape
                                                  (num_rows, num_cols, num_items).

        Returns:
            NDArray[np.bool_]: The dilated and masked boolean array,
                                also of shape (num_rows, num_cols, num_items).
        """
        neighbors = binary_dilation(dense_boundaries, self._neighbor_structure)
        neighbors &= self._enabled_weights_mask[..., None]
        return neighbors
        

    def find_neighbors(self, boundary: NDArray[np.bool]) -> NDArray[np.bool]:
        neighbors = binary_dilation(boundary, self._neighbor_structure)
        neigbhors &= self._enabled_weights_mask
        return neigbhors
    

    def random_boundary_neighbor_index(
        self, 
        index: Tuple[int, int], 
        boundary: NDArray[np.bool],
        size: int = 1
    ) -> NDArray[np.int_]:
        candidate_neighbors = self._neighbor_index_offsets + index

        candidate_row_index = candidate_neighbors[:, 0]
        candidate_col_index = candidate_neighbors[:, 1]

        on_boundary = boundary[candidate_row_index, candidate_col_index]

        return self._random_choice_generator.choice(
            candidate_neighbors[on_boundary], size, axes=0
        )
    

    def train_multiconnected(
        self,  
        max_iteration: int,
        initial_final_iter: int,
        interior_data: NDArray[np.float32], 
        outer_boundary_data: NDArray[np.float32],
        *inner_boundary_data: NDArray[np.float32],
        complete_data: Optional[NDArray] = None,
        macro_first_input: int = 0,
        macro_input_step: int = 1,
        random_order: bool = True,
        verbose: bool = False,
    ) -> None:
        if complete_data is None:
            complete_data = np.concatenate((
                interior_data, outer_boundary_data, *inner_boundary_data
            ))

        sparse_boundary, num_boundaries = self._find_sparse_boundaries()

        print("num boundaries:", num_boundaries)
        print("sparse boundary\n", sparse_boundary.astype(int))

        dense_boundaries = \
            self._find_dense_boundaries(sparse_boundary, num_boundaries)

        interior = ~dense_boundaries
        boundary = np.sum(dense_boundaries, axis=-1, dtype=bool)

        for i in range(interior.shape[2]):
            print("interior\n", interior[..., i].astype(int))

        print("boundary\n", boundary.astype(int))

        self.initial_mesh_construction(
            complete_data, initial_final_iter, max_iteration,
            random_order, verbose
        )

        OUTER_BOUNDARY_INDEX = 0
        INNER_BOUNDARY_START = 1

        boundary_mesh_construction = self._boundary_mesh_construction_setup(
            interior[..., OUTER_BOUNDARY_INDEX], outer_boundary_data, max_iteration
        )
        interior_mesh_construction = self._interior_mesh_construction_setup(
            boundary, complete_data, max_iteration
        )

        inner_boundary_mesh_constructions = []

        for index, data in enumerate(inner_boundary_data, INNER_BOUNDARY_START):
            inner_boundary_mesh_constructions.append(
                self._boundary_mesh_construction_setup(
                    interior[..., index], data, max_iteration
                )
            )

        boundary_interior_bounds = self._boundary_interior_bounds(
            initial_final_iter, max_iteration, 
            macro_first_input, macro_input_step
        )

        for boundary_start, boundary_final, interior_start, interior_final in \
            boundary_interior_bounds:

            boundary_mesh_construction(boundary_start, boundary_final)
            interior_mesh_construction(interior_start, interior_final)

            for inner_boundary_mesh_construction in \
                inner_boundary_mesh_constructions:

                inner_boundary_mesh_construction(boundary_start, boundary_final)


    def boundary_mesh_construction(
        self,  
        max_iteration: int,
        initial_final_iter: int,
        boundary_data: NDArray, 
        interior_data: NDArray,
        complete_data: Optional[NDArray] = None,
        macro_first_input: int = 0,
        macro_input_step: int = 1,
        random_order: bool = True,
        verbose=False,
    ) -> None:
        """
        The composite training method for simply connected domains.

        This method performs an initial training phase with all data, then 
        iteratively refines the mesh by focusing on boundary regions and then 
        interior regions. Note that this method doesn't handle interior holes.

        Args:
            max_iteration (int): The maximum iterations index for the entire 
                training process.
            initial_final_iter (int): The final iteration for the initial 
                training phase.
            boundary_data (NDArray): The data representing the boundary points 
                of the mesh.
            interior_data (NDArray): The data representing the interior points 
                of the mesh.
            complete_data (NDArray, optional): The combined complete dataset. 
                If `None`, than the union of `interior_data` and 
                `boundary_data` form this set.
            random_order (bool): If `True`, data samples are presented in a 
                random order during training.
            fixed_weights (bool, optional): If `True`, fixed weights will be
                unchanged during the training process. Defaults to `True`.
            disabled_weights (bool, optional): If `True`, disabled weights will 
                be ignored during the training process. Defaults to `True`.
            verbose (bool): If `True`, print progress and quantization error 
                for each phase.
        """
        if complete_data is None:
            complete_data = np.concatenate((interior_data, boundary_data))

        boundary, num_boundaries = self._find_sparse_boundaries()

        if num_boundaries > 1:
            raise Exception(f"Found {num_boundaries} different boundaries.")

        # self._create_boundary_masked()
        # self._create_interior_masked()
        # self._create_interior_exploded()

        boundary =  boundary.astype(bool)
        interior = ~boundary

        print("boundary\n", boundary.astype(int))
        print("interior\n", interior.astype(int))

        self.initial_mesh_construction(
            complete_data, initial_final_iter, max_iteration,
            random_order, verbose
        )

        boundary_mesh_construction = self._boundary_mesh_construction_setup(
            interior, boundary_data, max_iteration
        )

        boundary_mesh_construction(initial_final_iter, max_iteration)



    def train_composite(
        self,  
        max_iteration: int,
        initial_final_iter: int,
        boundary_data: NDArray, 
        interior_data: NDArray,
        complete_data: Optional[NDArray] = None,
        macro_first_input: int = 0,
        macro_input_step: int = 1,
        random_order: bool = True,
        verbose=False,
    ) -> None:
        """
        The composite training method for simply connected domains.

        This method performs an initial training phase with all data, then 
        iteratively refines the mesh by focusing on boundary regions and then 
        interior regions. Note that this method doesn't handle interior holes.

        Args:
            max_iteration (int): The maximum iterations index for the entire 
                training process.
            initial_final_iter (int): The final iteration for the initial 
                training phase.
            boundary_data (NDArray): The data representing the boundary points 
                of the mesh.
            interior_data (NDArray): The data representing the interior points 
                of the mesh.
            complete_data (NDArray, optional): The combined complete dataset. 
                If `None`, than the union of `interior_data` and 
                `boundary_data` form this set.
            random_order (bool): If `True`, data samples are presented in a 
                random order during training.
            fixed_weights (bool, optional): If `True`, fixed weights will be
                unchanged during the training process. Defaults to `True`.
            disabled_weights (bool, optional): If `True`, disabled weights will 
                be ignored during the training process. Defaults to `True`.
            verbose (bool): If `True`, print progress and quantization error 
                for each phase.
        """
        if complete_data is None:
            complete_data = np.concatenate((interior_data, boundary_data))

        boundary, num_boundaries = self._find_sparse_boundaries()

        if num_boundaries > 1:
            raise Exception(f"Found {num_boundaries} different boundaries.")

        # self._create_boundary_masked()
        # self._create_interior_masked()
        # self._create_interior_exploded()

        boundary =  boundary.astype(bool)
        interior = ~boundary

        print("boundary\n", boundary.astype(int))
        print("interior\n", interior.astype(int))

        self.initial_mesh_construction(
            complete_data, initial_final_iter, max_iteration,
            random_order, verbose
        )

        boundary_mesh_construction = self._boundary_mesh_construction_setup(
            interior, boundary_data, max_iteration
        )
        interior_mesh_construction = self._interior_mesh_construction_setup(
            boundary, complete_data, max_iteration
        )

        boundary_interior_bounds = self._boundary_interior_bounds(
            initial_final_iter, max_iteration, 
            macro_first_input, macro_input_step
        )

        for boundary_start, boundary_final, interior_start, interior_final in \
            boundary_interior_bounds:

            boundary_mesh_construction(boundary_start, boundary_final)
            interior_mesh_construction(interior_start, interior_final)

    # def train_composite(
    #     self,  
    #     max_iteration: int, 
    #     boundary_data: NDArray, 
    #     interior_data: NDArray,
    #     complete_data: Optional[NDArray] = None,
    #     verbose=False,
    # ) -> None:
    #     if complete_data is None:
    #         complete_data = np.concatenate((interior_data, boundary_data))

    #     CONSTANT = 10

    #     self.start_iter = 0
    #     self.final_iter = CONSTANT

    #     self.max_iteration = max_iteration

    #     # Initial training with the whole dataset and network.
    #     self.train(
    #         complete_data, 
    #         num_iteration=CONSTANT, random_order=True, verbose=verbose
    #     )

    #     start_constant = CONSTANT + 1
    #     final_constant = CONSTANT

    #     phi_start = self._phi(start_constant, first_input=0)
    #     phi_final = self._phi(final_constant, first_input=1)
    #     phi_final = self._until_exceed(max_iteration, phi_final)

    #     psi_start = self._psi(start_constant, first_input=0)
    #     psi_final = self._psi(final_constant, first_input=1)

    #     # FINAL_ITER = -1
    #     # max_iteration = max(psi_final[FINAL_ITER], phi_final[FINAL_ITER])

    #     bounds = zip(psi_start, psi_final, phi_start, phi_final)

    #     for boundary_start, boundary_final, interior_start, interior_final in bounds:

    #         self.start_iter = boundary_start
    #         self.final_iter = boundary_final

    #         FIRST_INDEX = 1

    #         # Calculate the number of iterations.
    #         num_iterations = boundary_final - boundary_start + FIRST_INDEX

    #         # Boundary update
    #         self._activation_distance = self._boundary_activation_distance
    #         self._learning_rate_decay_function = self._boundary_eta
        
    #         self.train(
    #             boundary_data, num_iterations, random_order=True, verbose=verbose
    #         )

    #     # Interior update
    #     self._activation_distance = self._original_activation_distance
    #     self._learning_rate_decay_function = self._interior_eta

    #     self.train(
    #         complete_data, num_iterations, random_order=True, verbose=verbose
    #     )

    #     # Reset to original state.
    #     self._learning_rate_decay_function = self._complete_eta