from enum import Enum, auto

# auto() automatically assigns a simple integer value, which is perfect
# when the actual value doesn't matter, only the identity.

class StepSizeMode(Enum):
    """Defines the available modes for calculating agent step size."""
    SAMPLE_AVERAGE = auto()
    CONSTANT = auto()
    UNBIASED_TRICK = auto() # Example for future expansion

class BaselineMode(Enum):
    """Defines the available modes for the gradient agent's baseline."""
    NONE = auto()
    SAMPLE_AVERAGE = auto()
    CONSTANT = auto()